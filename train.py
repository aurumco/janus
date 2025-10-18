"""Main training script for Mamba Bitcoin trend classifier."""

import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import warnings

from src.config.config_loader import ConfigLoader
from src.data.data_loader import DataLoaderFactory
from src.data.sequence_strategy import SequenceProcessingStrategy
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.visualizer import MetricsVisualizer
from src.models.mamba_classifier import MambaClassifier
from src.training.trainer import Trainer
from src.utils.helpers import get_device, save_model_architecture, set_seed


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Train Mamba classifier for Bitcoin trend prediction'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Override data path from config'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Override output directory from config'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training'
    )

    return parser.parse_args()


def main() -> None:
    """Main training function."""
    args = parse_args()

    config = ConfigLoader(args.config)

    set_seed(config.get('seed', 42))

    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module=r"mamba_ssm\..*",
    )
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=r"`torch\.cuda\.amp\.(autocast|custom_fwd|custom_bwd).* is deprecated",
    )

    device = get_device(
        use_cuda=config.get('device.use_cuda', True),
        device_id=config.get('device.device_id', 0)
    )
    if device.type == 'cuda':
        try:
            _ = torch.tensor(0.0, device=device)
            torch.cuda.synchronize()
        except Exception:
            pass

    data_path = args.data_path or config.get('paths.data_path')
    output_dir = Path(args.output_dir or config.get('paths.output_dir'))

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = output_dir / config.get('paths.results_dir', 'results') / timestamp
    checkpoint_dir = output_dir / config.get('paths.checkpoint_dir', 'checkpoints')
    log_dir = output_dir / config.get('paths.logs_dir', 'logs') / timestamp

    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("MAMBA BITCOIN TREND CLASSIFIER - TRAINING")
    print("="*70)
    print(f"Configuration: {args.config}")
    print(f"Data path: {data_path}")
    print(f"Output directory: {output_dir}")
    print(f"Results directory: {results_dir}")
    print("="*70 + "\n")

    data_factory = DataLoaderFactory(
        data_path=data_path,
        processing_strategy=SequenceProcessingStrategy(
            feature_columns=config.get('data.feature_columns'),
            target_column=config.get('data.target_column'),
            sequence_length=config.get('data.input_window'),
        ),
        train_ratio=config.get('data.train_ratio'),
        val_ratio=config.get('data.val_ratio'),
        test_ratio=config.get('data.test_ratio'),
        batch_size=config.get('data.batch_size'),
        num_workers=config.get('data.num_workers'),
        shuffle_train=config.get('data.shuffle_train'),
        random_seed=config.get('seed', 42),
        oversample_smote=config.get('data.oversample_smote', False),
        smote_k_neighbors=config.get('data.smote_k_neighbors', 5),
    )

    print("Creating data loaders...")
    data_loaders = data_factory.create_data_loaders()

    train_loader = data_loaders['train']
    val_loader = data_loaders['val']
    test_loader = data_loaders['test']

    enable_weighted_sampling = config.get('data.weighted_sampling', False)
    num_classes = config.get('model.num_classes')

    if enable_weighted_sampling:
        y_list = []
        for _, yb in train_loader:
            y_list.append(yb.cpu().numpy())
        y_all = np.concatenate(y_list, axis=0)
        class_counts = np.bincount(y_all, minlength=num_classes)
        sample_weights = 1.0 / (class_counts[y_all] + 1e-8)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_dataset = train_loader.dataset
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.get('data.batch_size'),
            sampler=sampler,
            num_workers=config.get('data.num_workers'),
            pin_memory=True,
        )
        data_loaders['train'] = train_loader

    dataset_info = data_factory.get_dataset_info()

    print(f"Dataset info: {dataset_info}")
    print(f"Train batches: {len(data_loaders['train'])}")
    print(f"Val batches: {len(data_loaders['val'])}")
    print(f"Test batches: {len(data_loaders['test'])}\n")

    model = MambaClassifier(
        input_dim=config.get('data.num_features'),
        d_model=config.get('model.d_model'),
        d_state=config.get('model.d_state'),
        d_conv=config.get('model.d_conv'),
        n_layers=config.get('model.n_layers'),
        num_classes=config.get('model.num_classes'),
        dropout=config.get('model.dropout'),
    )
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    
    model = model.to(device)

    actual_model = model.module if hasattr(model, 'module') else model
    print("Model created successfully")
    print(f"Parameters: {actual_model.get_num_parameters()}\n")

    save_model_architecture(model, results_dir / 'model_architecture.txt')

    class_weights_cfg = config.get('loss.class_weights')
    class_weights_tensor = None
    if class_weights_cfg:
        class_weights_tensor = torch.tensor(class_weights_cfg, dtype=torch.float32, device=device)
    else:
        y_list = []
        for _, yb in train_loader:
            y_list.append(yb.cpu().numpy())
        y_all = np.concatenate(y_list, axis=0) if y_list else np.array([], dtype=np.int64)
        if y_all.size > 0:
            counts = np.bincount(y_all, minlength=num_classes)
            inv_freq = 1.0 / (counts + 1e-8)
            weights = inv_freq / inv_freq.sum() * len(counts)
            class_weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

    use_focal = config.get('loss.type', 'cross_entropy').lower() == 'focal'

    class FocalLoss(nn.Module):
        def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = 'mean') -> None:
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.reduction = reduction
        def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            ce = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
            pt = torch.exp(-ce)
            loss = (1 - pt) ** self.gamma * ce
            if self.reduction == 'mean':
                return loss.mean()
            return loss.sum()

    if use_focal:
        criterion = FocalLoss(alpha=class_weights_tensor, gamma=config.get('loss.gamma', 2.0))
    else:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights_tensor,
            label_smoothing=config.get('loss.label_smoothing', 0.0)
        )

    optimizer = AdamW(
        model.parameters(),
        lr=config.get('training.learning_rate'),
        weight_decay=config.get('training.weight_decay'),
    )

    total_epochs = config.get('training.epochs')
    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        gradient_clip=config.get('training.gradient_clip', 1.0),
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir if config.get('logging.tensorboard', True) else None,
        early_stopping_patience=config.get('training.early_stopping_patience', 10),
        early_stopping_min_delta=config.get('training.early_stopping_min_delta', 0.0001),
        use_amp=config.get('device.mixed_precision', True),
    )

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.get('training.epochs', 100),
        log_interval=config.get('logging.log_interval', 10),
    )

    print("\nTraining completed!")

    visualizer = MetricsVisualizer(
        class_names=config.get('evaluation.class_names')
    )

    print("Plotting training curves...")
    visualizer.plot_training_curves(history, results_dir)

    print("Loading best model for evaluation...")
    best_checkpoint = checkpoint_dir / 'best_model.pt'
    if best_checkpoint.exists():
        trainer.load_checkpoint(str(best_checkpoint))

    evaluator = ModelEvaluator(
        model=model,
        device=device,
        class_names=config.get('evaluation.class_names'),
    )

    print("\nEvaluating on test set...")
    test_metrics = evaluator.evaluate(test_loader)

    evaluator.print_metrics(test_metrics)
    evaluator.save_metrics(test_metrics, results_dir / 'evaluation_metrics.txt')

    print("Creating visualizations...")
    visualizer.plot_confusion_matrix(
        test_metrics['confusion_matrix'],
        results_dir / 'confusion_matrix.png'
    )

    visualizer.plot_roc_curves(
        test_metrics['roc_curves'],
        results_dir / 'roc_curves.png'
    )

    # Export artifacts
    export_dir = results_dir / 'exports'
    export_dir.mkdir(parents=True, exist_ok=True)

    # Save state_dict
    try:
        sd_path = export_dir / 'model_state_dict.pth'
        torch.save((actual_model if 'actual_model' in locals() else model).state_dict(), sd_path)
    except Exception as e:
        print(f"[warn] State dict export failed: {e}")

    # TorchScript via trace (more robust for third-party Python code)
    try:
        example = torch.randn(1, config.get('data.input_window'), config.get('data.num_features')).to(device)
        (actual_model if 'actual_model' in locals() else model).eval()
        with torch.no_grad():
            traced = torch.jit.trace((actual_model if 'actual_model' in locals() else model), example, strict=False)
        ts_path = export_dir / 'model_traced.pt'
        traced.save(str(ts_path))
    except Exception as e:
        print(f"[warn] TorchScript trace export failed: {e}")

    # ONNX export (best-effort; may not support mamba-ssm custom ops)
    try:
        onnx_path = export_dir / 'model.onnx'
        torch.onnx.export(
            (actual_model if 'actual_model' in locals() else model),
            example,
            str(onnx_path),
            input_names=['input'],
            output_names=['logits'],
            dynamic_axes={'input': {0: 'batch', 1: 'seq'}, 'logits': {0: 'batch'}},
            opset_version=17,
            do_constant_folding=False,
        )
    except Exception as e:
        print(f"[warn] ONNX export skipped: {e}")
    else:
        print(f"Exports saved: {export_dir}")

    print(f"\nAll results saved to: {results_dir}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print("\nTraining pipeline completed successfully!")


if __name__ == '__main__':
    main()
