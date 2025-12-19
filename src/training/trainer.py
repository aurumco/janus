"""Training module for Mamba regressor."""

import gc
import signal
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler as AmpGradScaler, autocast as amp_autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from src.utils.logger import logger
except:
    from ..utils.logger import logger

TENSORBOARD_AVAILABLE = False
SummaryWriter = None


class Trainer:
    """Handles model training with early stopping and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scheduler: Optional[_LRScheduler] = None,
        gradient_clip: Optional[float] = None,
        checkpoint_dir: Optional[Path] = None,
        log_dir: Optional[Path] = None,
        early_stopping_patience: int = 10,
        early_stopping_min_delta: float = 0.0001,
        use_amp: bool = False,
        warmup_epochs: int = 0,
        accumulation_steps: int = 1,
        checkpoint_interval: int = 10,
    ) -> None:
        """Initialize trainer.

        Args:
            model: Model to train.
            optimizer: Optimizer for training.
            criterion: Loss function.
            device: Device to train on.
            scheduler: Learning rate scheduler.
            gradient_clip: Maximum gradient norm for clipping.
            checkpoint_dir: Directory to save checkpoints.
            log_dir: Directory for TensorBoard logs.
            early_stopping_patience: Epochs to wait before early stopping.
            early_stopping_min_delta: Minimum change to qualify as improvement.
            use_amp: Whether to use automatic mixed precision.
            warmup_epochs: Number of warmup epochs with linear LR increase.
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.warmup_epochs = warmup_epochs
        self.initial_lr = optimizer.param_groups[0]['lr']
        self.use_amp = use_amp
        self.accumulation_steps = max(1, accumulation_steps)

        # Pre-calculate AMP dtype to avoid overhead in training loop
        self.amp_dtype = torch.float16
        if self.use_amp:
            self.scaler = AmpGradScaler(device="cuda")
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
                self.amp_dtype = torch.bfloat16
        else:
            self.scaler = None
        self.gradient_clip = gradient_clip
        self.checkpoint_dir = checkpoint_dir
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.checkpoint_interval = checkpoint_interval

        if checkpoint_dir:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.writer = None
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=str(log_dir))
            except (ImportError, AttributeError, ValueError) as e:
                print(f"TensorBoard unavailable: {e}. Training will continue without logging.")

        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
        }
        self.start_epoch = 1
        self.interrupted = False
        
        # Register signal handler for graceful interruption
        signal.signal(signal.SIGINT, self._signal_handler)

    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freeze or unfreeze backbone layers for fine-tuning.

        Args:
            freeze: If True, freeze backbone. If False, unfreeze all.
        """
        backbone_keywords = [
            'input_projection',
            'input_norm',
            'mamba_layers',
            'layer_norms',
        ]

        for name, param in self.model.named_parameters():
            if any(keyword in name for keyword in backbone_keywords):
                param.requires_grad = not freeze
            else:
                param.requires_grad = True

        status = "frozen" if freeze else "unfrozen"
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Backbone {status}. Trainable parameters: {trainable:,}")

    def _signal_handler(self, sig, frame):
        """Handle Ctrl+C gracefully by saving checkpoint."""
        print("\n\n⚠️  Training interrupted! Saving checkpoint...")
        self.interrupted = True

    def save_checkpoint(self, epoch: int, is_best: bool = False, is_interrupted: bool = False) -> None:
        """Save training checkpoint.
        
        Args:
            epoch: Current epoch number.
            is_best: Whether this is the best model so far.
            is_interrupted: Whether saving due to interruption.
        """
        if self.checkpoint_dir is None:
            return
            
        actual_model = self.model.module if hasattr(self.model, 'module') else self.model
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': actual_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter,
            'history': self.history,
        }
        
        # Save latest checkpoint (always)
        latest_path = self.checkpoint_dir / 'checkpoint_latest.pt'
        torch.save(checkpoint, latest_path)
        
        # Save periodic full checkpoint every N epochs
        if epoch % self.checkpoint_interval == 0:
            epoch_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, epoch_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'checkpoint_best.pt'
            torch.save(checkpoint, best_path)
        
        # Save interrupted checkpoint
        if is_interrupted:
            interrupted_path = self.checkpoint_dir / 'checkpoint_interrupted.pt'
            torch.save(checkpoint, interrupted_path)

    def load_checkpoint(self, checkpoint_path: Optional[Path] = None) -> bool:
        """Load training checkpoint to resume training.
        
        Args:
            checkpoint_path: Path to checkpoint file. If None, tries to load latest.
            
        Returns:
            True if checkpoint was loaded successfully, False otherwise.
        """
        # Normalize to Path if a string was provided
        if isinstance(checkpoint_path, str):
            checkpoint_path = Path(checkpoint_path)

        if checkpoint_path is None and self.checkpoint_dir:
            # Try interrupted first, then latest
            for name in ['checkpoint_interrupted.pt', 'checkpoint_latest.pt']:
                potential_path = self.checkpoint_dir / name
                if potential_path.exists():
                    checkpoint_path = potential_path
                    break
        
        if checkpoint_path is None or not checkpoint_path.exists():
            return False
        
        try:
            checkpoint = torch.load(str(checkpoint_path), map_location=self.device, weights_only=False)
            
            actual_model = self.model.module if hasattr(self.model, 'module') else self.model
            actual_model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if self.scheduler and checkpoint['scheduler_state_dict']:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            if self.scaler and checkpoint['scaler_state_dict']:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_val_loss = checkpoint['best_val_loss']
            self.patience_counter = checkpoint['patience_counter']
            self.history = checkpoint['history']
            
            logger.success(f"Resumed from epoch {checkpoint['epoch']}", indent=1)
            logger.metric("Best validation loss", f"{self.best_val_loss:.6f}", indent=1)
            
            return True
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}", indent=1)
            return False

    def train_epoch(self, train_loader: DataLoader, epoch: int, epochs: int = 100) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            train_loader: Training data loader.
            epoch: Current epoch number.

        Returns:
            Dictionary with training metrics.
        """
        self.model.train()
        total_loss = 0.0
        loss_components = {}

        progress = tqdm(
            train_loader, 
            total=len(train_loader), 
            desc=f"Epoch {epoch}/{epochs} [Train]",
            leave=False,
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        for batch_idx, batch_data in enumerate(progress):
            # Handle both tuple (inputs, targets) and dict formats
            # Note: For multi-task finetuning, the dataset returns a dictionary
            # containing 'input_sequence', 'targets', and optional 'asset_id'.
            if isinstance(batch_data, dict):
                inputs = batch_data["input_sequence"]
                batch = batch_data
            else:
                inputs, targets = batch_data
                batch = {"targets": targets}
            # Move data to device (non_blocking allows overlap)
            if isinstance(batch, dict):
                batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                inputs = batch["input_sequence"] if "input_sequence" in batch else inputs
            else:
                inputs = inputs.to(self.device, non_blocking=True)
                batch["targets"] = batch["targets"].to(self.device, non_blocking=True)

            if self.use_amp:
                with amp_autocast(device_type="cuda", dtype=self.amp_dtype):
                    outputs = self.model(inputs, batch.get("asset_id") if isinstance(batch, dict) and "asset_id" in batch else None)
                    loss_output = self.criterion(outputs, batch.get("targets") if not isinstance(outputs, dict) else batch)
                    
                    if isinstance(loss_output, dict):
                        loss = loss_output["total_loss"]
                        # Defer .item() calls to avoid GPU synchronization
                        with torch.no_grad():
                            for key, val in loss_output.items():
                                if key != "total_loss":
                                    # Initialize tensor on device if needed
                                    if key not in loss_components:
                                        loss_components[key] = torch.tensor(0.0, device=self.device)
                                    loss_components[key] += val.detach()
                    else:
                        loss = loss_output
                
                # Keep original unscaled loss for metrics
                # Since loss is about to be divided by accumulation_steps for backward
                metrics_loss = loss.detach()

                loss = loss / self.accumulation_steps
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    if self.gradient_clip:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.gradient_clip
                        )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(inputs, batch.get("asset_id") if isinstance(batch, dict) and "asset_id" in batch else None)
                loss_output = self.criterion(outputs, batch.get("targets") if not isinstance(outputs, dict) else batch)
                
                if isinstance(loss_output, dict):
                    loss = loss_output["total_loss"]
                    with torch.no_grad():
                        for key, val in loss_output.items():
                            if key != "total_loss":
                                if key not in loss_components:
                                    loss_components[key] = torch.tensor(0.0, device=self.device)
                                loss_components[key] += val.detach()
                else:
                    loss = loss_output

                # FIX: Added missing backward/step for non-AMP training
                loss = loss / self.accumulation_steps
                loss.backward()

                if (batch_idx + 1) % self.accumulation_steps == 0:
                    if self.gradient_clip:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.gradient_clip
                        )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # Check for finite loss
            if not torch.isfinite(loss):
                loss_value = loss.item()
                print(f"\nWARNING: Non-finite loss detected at batch {batch_idx}")
                print(f"  Loss value: {loss_value}")
                print(f"  Skipping this batch...")
                continue
            
            # Accumulate total loss as tensor to avoid per-batch sync
            if isinstance(total_loss, float):
                 total_loss = torch.tensor(0.0, device=self.device)

            # Correctly accumulate loss. In AMP block, we captured metrics_loss before division.
            # In Non-AMP block, loss was divided by accumulation_steps.
            if self.use_amp:
                 total_loss += metrics_loss
            else:
                 # In non-AMP path, loss was divided by accumulation_steps, so we multiply back
                 total_loss += loss.detach() * self.accumulation_steps
            
        num_batches = len(train_loader)

        # Convert accumulated tensors to float at the end of epoch
        if isinstance(total_loss, torch.Tensor):
            total_loss = total_loss.item()

        if total_loss == 0 and num_batches > 0:
            print("\nWARNING: All batches had zero loss!")
            avg_loss = float('nan')
        else:
            avg_loss = total_loss / max(num_batches, 1)
        
        progress.close()
        
        metrics = {'loss': avg_loss}
        
        for key, val in loss_components.items():
            metrics[key] = val.item() / len(train_loader)

        return metrics

    def validate(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Validate the model.

        Args:
            val_loader: Validation data loader.
            epoch: Current epoch number.

        Returns:
            Dictionary with validation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        loss_components = {}

        with torch.no_grad():
            vprogress = tqdm(
                val_loader, 
                total=len(val_loader), 
                desc=f"Epoch {epoch} [Val]",
                leave=False,
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]'
            )
            for batch_idx, batch_data in enumerate(vprogress):
                if isinstance(batch_data, dict):
                    inputs = batch_data["input_sequence"]
                    batch = batch_data
                else:
                    inputs, targets = batch_data
                    batch = {"targets": targets}

                if isinstance(batch, dict):
                    batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    inputs = batch["input_sequence"] if "input_sequence" in batch else inputs
                else:
                    inputs = inputs.to(self.device, non_blocking=True)
                    batch["targets"] = batch["targets"].to(self.device, non_blocking=True)

                outputs = self.model(inputs, batch.get("asset_id") if isinstance(batch, dict) and "asset_id" in batch else None)
                loss_output = self.criterion(outputs, batch.get("targets") if not isinstance(outputs, dict) else batch)
                
                if isinstance(loss_output, dict):
                    loss = loss_output["total_loss"]
                    # Defer item()
                    for key, val in loss_output.items():
                        if key != "total_loss":
                            if key not in loss_components:
                                loss_components[key] = torch.tensor(0.0, device=self.device)
                            loss_components[key] += val.detach()
                else:
                    loss = loss_output

                # Check finite
                if not torch.isfinite(loss):
                    print(f"\nWARNING: Non-finite loss in validation batch {batch_idx}")
                    continue
                
                if isinstance(total_loss, float):
                    total_loss = torch.tensor(0.0, device=self.device)
                total_loss += loss.detach()

        if isinstance(total_loss, torch.Tensor):
            total_loss = total_loss.item()

        avg_loss = total_loss / len(val_loader)
        vprogress.close()
        
        metrics = {'loss': avg_loss}
        
        for key, val in loss_components.items():
            metrics[key] = val.item() / len(val_loader)

        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        log_interval: int = 10,
        freeze_backbone_epochs: int = 0,
    ) -> Dict[str, list]:
        """Train the model for multiple epochs.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            epochs: Number of epochs to train.
            log_interval: Interval for logging to TensorBoard.

        Returns:
            Training history dictionary.
        """
        actual_model = self.model.module if hasattr(self.model, 'module') else self.model
        params = actual_model.get_num_parameters()
        
        logger.training_start(
            epochs=epochs,
            device=str(self.device),
            mixed_precision=self.use_amp,
            params=params
        )

        if self.writer:
            params = actual_model.get_num_parameters()
            self.writer.add_text('model/parameters', f"total: {params['total']}, trainable: {params['trainable']}", 0)

        for epoch in range(self.start_epoch, epochs + 1):
            if self.interrupted:
                self.save_checkpoint(epoch - 1, is_interrupted=True)
                logger.success("Checkpoint saved. Training stopped gracefully.", indent=1)
                break
                
            epoch_start_time = time.time()
            
            if freeze_backbone_epochs > 0:
                if epoch == 1:
                    self.freeze_backbone(freeze=True)
                elif epoch == freeze_backbone_epochs + 1:
                    self.freeze_backbone(freeze=False)

            train_metrics = self.train_epoch(train_loader, epoch, epochs)
            
            if self.interrupted:
                self.save_checkpoint(epoch, is_interrupted=True)
                logger.success("Checkpoint saved. Training stopped gracefully.", indent=1)
                break
                
            val_metrics = self.validate(val_loader, epoch)

            if epoch <= self.warmup_epochs:
                warmup_lr = self.initial_lr * (epoch / self.warmup_epochs)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = warmup_lr
            elif self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']

            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['learning_rate'].append(current_lr)

            if self.writer:
                self.writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
                self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
                self.writer.add_scalar('Learning_Rate', current_lr, epoch)
                
                for key in train_metrics:
                    if key not in ['loss']:
                        self.writer.add_scalar(f'Loss/train_{key}', train_metrics[key], epoch)
                for key in val_metrics:
                    if key not in ['loss']:
                        self.writer.add_scalar(f'Loss/val_{key}', val_metrics[key], epoch)

            epoch_time = time.time() - epoch_start_time
            
            summary_metrics = {
                "Train Loss": train_metrics['loss'],
                "Val Loss": val_metrics['loss'],
                "Learning Rate": current_lr,
            }

            is_best = val_metrics['loss'] < self.best_val_loss - self.early_stopping_min_delta
            if is_best:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            logger.epoch_summary(epoch, epochs, summary_metrics, epoch_time, is_best)
            
            if not is_best:
                logger.info(f"Patience: {self.patience_counter}/{self.early_stopping_patience}", indent=1)
            
            if self.checkpoint_dir:
                self.save_checkpoint(epoch, is_best=is_best)
                if is_best:
                    logger.success("New best model saved", indent=1)

            if self.patience_counter >= self.early_stopping_patience:
                logger.blank_line()
                logger.warning(f"Early stopping at epoch {epoch}", indent=1)
                logger.metric("Best validation loss", f"{self.best_val_loss:.6f}", indent=1)
                break

        if self.writer:
            self.writer.close()

        logger.blank_line()
        logger.success("Training completed!", indent=0)
        logger.metric("Best validation loss", f"{self.best_val_loss:.6f}", indent=1)

        return self.history
