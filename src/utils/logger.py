"""Professional logging utility for clean, organized console output."""

import sys
from typing import Dict, Any, Optional
from datetime import datetime


class TrainingLogger:
    """Clean, professional logger for training progress."""
    
    def __init__(self, width: int = 80):
        self.width = width
        self.separator_width = 80  # Fixed width for all separators
        self.last_was_progress = False
    
    def _clear_line(self):
        """Clear the current line."""
        if self.last_was_progress:
            sys.stdout.write('\r' + ' ' * self.width + '\r')
            sys.stdout.flush()
            self.last_was_progress = False
    
    def header(self, title: str):
        """Print a main header."""
        self._clear_line()
        print(f"\n{'═' * self.separator_width}")
        print(f"{title.upper().center(self.separator_width)}")
        print(f"{'═' * self.separator_width}\n")
    
    def section(self, title: str):
        """Print a section header."""
        self._clear_line()
        print(f"\n{title}")
        print(f"{'─' * self.separator_width}")
    
    def info(self, message: str, indent: int = 0):
        """Print an info message."""
        self._clear_line()
        prefix = "  " * indent
        print(f"{prefix}{message}")
    
    def success(self, message: str, indent: int = 0):
        """Print a success message."""
        self._clear_line()
        prefix = "  " * indent
        print(f"{prefix}✓ {message}")
    
    def warning(self, message: str, indent: int = 0):
        """Print a warning message."""
        self._clear_line()
        prefix = "  " * indent
        print(f"{prefix}⚠ {message}")
    
    def error(self, message: str, indent: int = 0):
        """Print an error message."""
        self._clear_line()
        prefix = "  " * indent
        print(f"{prefix}✗ {message}")
    
    def metric(self, name: str, value: Any, indent: int = 0, width: int = 20):
        """Print a metric in key-value format."""
        self._clear_line()
        prefix = "  " * indent
        print(f"{prefix}• {name:<{width}}: {value}")
    
    def config_section(self, title: str, config: Dict[str, Any], indent: int = 0):
        """Print a configuration section."""
        self._clear_line()
        prefix = "  " * indent
        print(f"\n{prefix}{title}:")
        for key, value in config.items():
            print(f"{prefix}  • {key:<24}: {value}")
    
    def progress_inline(self, message: str):
        """Print a progress message on the same line (overwritable)."""
        sys.stdout.write('\r' + ' ' * self.width + '\r')
        sys.stdout.write(f"{message}")
        sys.stdout.flush()
        self.last_was_progress = True
    
    def blank_line(self):
        """Print a blank line for spacing."""
        self._clear_line()
        print()
    
    def separator(self, char: str = "─"):
        """Print a separator line."""
        self._clear_line()
        print(char * self.separator_width)
    
    def data_info(self, info: Dict[str, Any]):
        """Print dataset information in organized format."""
        self._clear_line()
        print("\nDataset Information:")
        print("  Samples:")
        if 'total_samples' in info:
            print(f"    • Total                : {info['total_samples']:,}")
        if 'train_batches' in info:
            print(f"    • Training batches     : {info['train_batches']:,}")
        if 'val_batches' in info:
            print(f"    • Validation batches   : {info['val_batches']:,}")
        
        print("  Configuration:")
        if 'sequence_length' in info:
            print(f"    • Sequence length      : {info['sequence_length']}")
        if 'num_features' in info:
            print(f"    • Features             : {info['num_features']}")
        if 'batch_size' in info:
            print(f"    • Batch size           : {info['batch_size']}")
    
    def training_start(self, epochs: int, device: str, mixed_precision: bool, params: Dict[str, int]):
        """Print training start information."""
        self._clear_line()
        self.section("Training Configuration")
        print(f"  Target:")
        print(f"    • Epochs               : {epochs}")
        print(f"  ")
        print(f"  Hardware:")
        print(f"    • Device               : {device}")
        print(f"    • Mixed precision      : {'Enabled' if mixed_precision else 'Disabled'}")
        print(f"  ")
        print(f"  Model:")
        print(f"    • Total parameters     : {params['total']:,}")
        print(f"    • Trainable parameters : {params['trainable']:,}")
        print()
    
    def epoch_summary(self, epoch: int, total: int, metrics: Dict[str, float], epoch_time: float, is_best: bool = False):
        """Print epoch summary."""
        self._clear_line()
        print(f"\n┌─ Epoch {epoch}/{total} {'(Best!)' if is_best else ''} ─ {epoch_time:.1f}s")
        for key, value in metrics.items():
            if 'loss' in key.lower():
                print(f"│  • {key:<18}: {value:.6f}")
            else:
                print(f"│  • {key:<18}: {value:.6e}")
        print("└" + "─" * 50)


# Global logger instance
logger = TrainingLogger()
