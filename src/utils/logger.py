"""Professional logging utility for clean, organized console output."""

import sys
from typing import Dict, Any, Optional, TextIO
from datetime import datetime
from pathlib import Path


class TrainingLogger:
    """Clean, professional logger for training progress."""
    
    def __init__(self, width: int = 80, log_file: Optional[str] = None):
        self.width = width
        self.separator_width = 80  # Fixed width for all separators
        self.last_was_progress = False
        self.log_file: Optional[TextIO] = None
        
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_file = open(log_path, 'w', encoding='utf-8')
            self._log_to_file(f"Log started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    def _log_to_file(self, message: str):
        """Write message to log file if enabled."""
        if self.log_file:
            # Strip ANSI codes and progress indicators for clean file logs
            clean_msg = message.replace('\r', '').replace('✓', '[OK]').replace('⚠', '[WARN]').replace('✗', '[ERROR]')
            self.log_file.write(clean_msg)
            self.log_file.flush()
    
    def _clear_line(self):
        """Clear the current line."""
        if self.last_was_progress:
            sys.stdout.write('\r' + ' ' * self.width + '\r')
            sys.stdout.flush()
            self.last_was_progress = False
    
    def close(self):
        """Close the log file if open."""
        if self.log_file:
            self._log_to_file(f"\nLog ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.log_file.close()
            self.log_file = None
    
    def header(self, title: str):
        """Print a main header."""
        self._clear_line()
        msg = f"\n{'═' * self.separator_width}\n{title.upper().center(self.separator_width)}\n{'═' * self.separator_width}\n"
        print(msg, end='')
        self._log_to_file(msg)
    
    def section(self, title: str):
        """Print a section header."""
        self._clear_line()
        msg = f"\n{title}\n{'─' * self.separator_width}\n"
        print(msg, end='')
        self._log_to_file(msg)
    
    def info(self, message: str, indent: int = 0):
        """Print an info message."""
        self._clear_line()
        prefix = "  " * indent
        msg = f"{prefix}{message}\n"
        print(msg, end='')
        self._log_to_file(msg)
    
    def success(self, message: str, indent: int = 0):
        """Print a success message."""
        self._clear_line()
        prefix = "  " * indent
        msg = f"{prefix}✓ {message}\n"
        print(msg, end='')
        self._log_to_file(msg)
    
    def warning(self, message: str, indent: int = 0):
        """Print a warning message."""
        self._clear_line()
        prefix = "  " * indent
        msg = f"{prefix}⚠ {message}\n"
        print(msg, end='')
        self._log_to_file(msg)
    
    def error(self, message: str, indent: int = 0):
        """Print an error message."""
        self._clear_line()
        prefix = "  " * indent
        msg = f"{prefix}✗ {message}\n"
        print(msg, end='')
        self._log_to_file(msg)
    
    def metric(self, name: str, value: Any, indent: int = 0, width: int = 20):
        """Print a metric in key-value format."""
        self._clear_line()
        prefix = "  " * indent
        msg = f"{prefix}• {name:<{width}}: {value}\n"
        print(msg, end='')
        self._log_to_file(msg)
    
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
        self._log_to_file("\n")
    
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
