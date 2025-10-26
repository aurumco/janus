"""Training module for Mamba regressor."""

import gc
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
try:
    from torch.amp import GradScaler as AmpGradScaler, autocast as amp_autocast  # type: ignore
except Exception:
    from torch.cuda.amp import GradScaler as AmpGradScaler, autocast as amp_autocast  # type: ignore
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

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
        aggressive_cleanup: bool = True,
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
        if self.use_amp:
            try:
                self.scaler = AmpGradScaler(device_type="cuda")  # type: ignore[arg-type]
            except TypeError:
                self.scaler = AmpGradScaler()
        else:
            self.scaler = None
        self.gradient_clip = gradient_clip
        self.checkpoint_dir = checkpoint_dir
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta

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
        self.aggressive_cleanup = aggressive_cleanup
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
        }

    def _cleanup_memory(self) -> None:
        """Aggressively clean up memory."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Force garbage collection of cached tensors
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

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

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
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

        for batch_idx, batch_data in enumerate(train_loader):
            # Handle both tuple (inputs, targets) and dict formats
            if isinstance(batch_data, dict):
                inputs = batch_data["input_sequence"]
                batch = batch_data
            else:
                inputs, targets = batch_data
                batch = {"targets": targets}
            # Move data to device
            if isinstance(batch, dict):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                inputs = batch["input_sequence"] if "input_sequence" in batch else inputs
            else:
                inputs = inputs.to(self.device)
                batch["targets"] = batch["targets"].to(self.device)

            if self.use_amp:
                try:
                    ctx = amp_autocast(device_type="cuda")
                except TypeError:
                    ctx = amp_autocast()
                with ctx:
                    outputs = self.model(inputs)
                    loss_output = self.criterion(outputs, batch.get("targets") if not isinstance(outputs, dict) else batch)
                    
                    if isinstance(loss_output, dict):
                        loss = loss_output["total_loss"]
                        for key, val in loss_output.items():
                            if key != "total_loss":
                                loss_components[key] = loss_components.get(key, 0.0) + val.item()
                    else:
                        loss = loss_output
                
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
                outputs = self.model(inputs)
                loss_output = self.criterion(outputs, batch.get("targets") if not isinstance(outputs, dict) else batch)
                
                if isinstance(loss_output, dict):
                    loss = loss_output["total_loss"]
                    for key, val in loss_output.items():
                        if key != "total_loss":
                            loss_components[key] = loss_components.get(key, 0.0) + val.item()
                else:
                    loss = loss_output
                
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

            total_loss += loss.item()
            
            # Periodic memory cleanup every 50 batches (more aggressive)
            if self.aggressive_cleanup and batch_idx > 0 and batch_idx % 50 == 0:
                self._cleanup_memory()
            elif batch_idx > 0 and batch_idx % 100 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        avg_loss = total_loss / len(train_loader)
        
        # Cleanup after epoch
        if self.aggressive_cleanup:
            self._cleanup_memory()
        else:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        metrics = {'loss': avg_loss}
        
        # Add averaged loss components
        for key, val in loss_components.items():
            metrics[key] = val / len(train_loader)

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
            for batch_idx, batch_data in enumerate(val_loader):
                # Handle both tuple and dict formats
                if isinstance(batch_data, dict):
                    inputs = batch_data["input_sequence"]
                    batch = batch_data
                else:
                    inputs, targets = batch_data
                    batch = {"targets": targets}

                # Move data to device
                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    inputs = batch["input_sequence"] if "input_sequence" in batch else inputs
                else:
                    inputs = inputs.to(self.device)
                    batch["targets"] = batch["targets"].to(self.device)

                outputs = self.model(inputs)
                loss_output = self.criterion(outputs, batch.get("targets") if not isinstance(outputs, dict) else batch)
                
                # Handle dictionary loss output
                if isinstance(loss_output, dict):
                    loss = loss_output["total_loss"]
                    for key, val in loss_output.items():
                        if key != "total_loss":
                            loss_components[key] = loss_components.get(key, 0.0) + val.item()
                else:
                    loss = loss_output

                total_loss += loss.item()

        avg_loss = total_loss / len(val_loader)
        
        # Cleanup after validation
        if self.aggressive_cleanup:
            self._cleanup_memory()
        else:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        metrics = {'loss': avg_loss}
        
        # Add averaged loss components
        for key, val in loss_components.items():
            metrics[key] = val / len(val_loader)

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
        print(f"\nStarting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.use_amp}")
        print(f"Model parameters: {actual_model.get_num_parameters()}")

        # Log parameter counts once to TensorBoard for visibility
        if self.writer:
            params = actual_model.get_num_parameters()
            self.writer.add_text('model/parameters', f"total: {params['total']}, trainable: {params['trainable']}", 0)

        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            
            if freeze_backbone_epochs > 0:
                if epoch == 1:
                    self.freeze_backbone(freeze=True)
                elif epoch == freeze_backbone_epochs + 1:
                    self.freeze_backbone(freeze=False)

            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader, epoch)

            # Warmup phase: linearly increase LR
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
                
                # Log additional loss components if present
                for key in train_metrics:
                    if key not in ['loss']:
                        self.writer.add_scalar(f'Loss/train_{key}', train_metrics[key], epoch)
                for key in val_metrics:
                    if key not in ['loss']:
                        self.writer.add_scalar(f'Loss/val_{key}', val_metrics[key], epoch)

            epoch_time = time.time() - epoch_start_time
            
            print(f"\nEpoch {epoch}/{epochs} ({epoch_time:.1f}s)")
            print(f"  • Train Loss:    {train_metrics['loss']:.6f}")
            print(f"  • Val Loss:      {val_metrics['loss']:.6f}")
            print(f"  • Learning Rate: {current_lr:.6f}")

            if val_metrics['loss'] < self.best_val_loss - self.early_stopping_min_delta:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                if self.checkpoint_dir:
                    self.save_checkpoint(epoch, val_metrics, is_best=True)
                    print(f"  ✓ New best model saved")
            else:
                self.patience_counter += 1
                print(f"  ⏳ Patience: {self.patience_counter}/{self.early_stopping_patience}")

            if self.patience_counter >= self.early_stopping_patience:
                print(f"\n{'='*50}")
                print(f"Early stopping at epoch {epoch}")
                print(f"Best validation loss: {self.best_val_loss:.6f}")
                print(f"{'='*50}")
                break

        if self.writer:
            self.writer.close()

        print(f"\n{'='*50}")
        print(f"Training completed")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        print(f"{'='*50}\n")

        return self.history

    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
    ) -> None:
        """Save model checkpoint in multiple formats.

        Args:
            epoch: Current epoch number.
            metrics: Validation metrics.
            is_best: Whether this is the best model so far.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'history': self.history,
            'best_val_loss': self.best_val_loss,
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if is_best:
            best_full = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_full)
            
            state_dict_path = self.checkpoint_dir / "best_model_state_dict.pth"
            torch.save(self.model.state_dict(), state_dict_path)
            
            print(f"  ✓ Saved best checkpoint: {best_full}")
        
        latest_path = self.checkpoint_dir / "latest_checkpoint.pt"
        torch.save(checkpoint, latest_path)
        
        if epoch % 10 == 0:
            milestone_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint, milestone_path)

    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.

        Returns:
            Checkpoint dictionary.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        return checkpoint
