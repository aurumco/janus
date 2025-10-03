"""Main training script for Mamba Bitcoin trend classifier."""

import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

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

    device = get_device(
        use_cuda=config.get('device.use_cuda', True),
        device_id=config.get('device.device_id', 0)
    )

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

    processing_strategy = SequenceProcessingStrategy(
        feature_columns=config.get('data.feature_columns'),
        target_column=config.get('data.target_column'),
        sequence_length=config.get('data.input_window'),
    )

    data_factory = DataLoaderFactory(
        data_path=data_path,
        processing_strategy=processing_strategy,
        train_ratio=config.get('data.train_ratio'),
        val_ratio=config.get('data.val_ratio'),
        test_ratio=config.get('data.test_ratio'),
        batch_size=config.get('data.batch_size'),
        num_workers=config.get('data.num_workers'),
        shuffle_train=config.get('data.shuffle_train'),
        random_seed=config.get('seed', 42),
    )

    print("Creating data loaders...")
    data_loaders = data_factory.create_data_loaders()
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
    ).to(device)

    print("Model created successfully")
    print(f"Parameters: {model.get_num_parameters()}\n")

    save_model_architecture(model, results_dir / 'model_architecture.txt')

    criterion = nn.CrossEntropyLoss(
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
        gradient_clip=config.get('training.gradient_clip'),
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir if config.get('logging.tensorboard') else None,
        early_stopping_patience=config.get('training.early_stopping_patience'),
        early_stopping_min_delta=config.get('training.early_stopping_min_delta'),
    )

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    history = trainer.fit(
        train_loader=data_loaders['train'],
        val_loader=data_loaders['val'],
        epochs=total_epochs,
        log_interval=config.get('logging.log_interval'),
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
    test_metrics = evaluator.evaluate(data_loaders['test'])

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

    print(f"\nAll results saved to: {results_dir}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print("\nTraining pipeline completed successfully!")


if __name__ == '__main__':
    main()
