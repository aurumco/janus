"""Training module for Mamba regressor."""

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
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
        }

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

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                try:
                    ctx = amp_autocast(device_type="cuda")
                except TypeError:
                    ctx = amp_autocast()
                with ctx:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                self.scaler.scale(loss).backward()
                
                if self.gradient_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                loss.backward()
                
                if self.gradient_clip:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )
                
                self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        return {'loss': avg_loss}

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

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()

        avg_loss = total_loss / len(val_loader)

        return {'loss': avg_loss}

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        log_interval: int = 10,
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
                if self.checkpoint_dir:
                    self.save_checkpoint(epoch, val_metrics, is_best=False)
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
        """Save model checkpoint.

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
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)

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
