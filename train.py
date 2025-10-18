"""Main training script for Mamba Bitcoin price change regressor."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import warnings

from src.config.config_loader import ConfigLoader
from src.data.data_loader import DataLoaderFactory
from src.data.sequence_strategy import SequenceProcessingStrategy
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.visualizer import MetricsVisualizer
from src.models.mamba_regressor import MambaRegressor
from src.training.trainer import Trainer
from src.training.losses import (
    DirectionalMSELoss,
    AsymmetricMSELoss,
    SignWeightedMSELoss,
)
from src.training.multitask_losses import (
    MultiTaskRegressionLoss,
    AdaptiveMultiTaskLoss,
)
from src.models.mamba_multitask import MambaMultitaskRegressor
from src.evaluation.multitask_evaluator import MultiTaskEvaluator
from src.utils.helpers import get_device, save_model_architecture, set_seed


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Train Mamba regressor for Bitcoin price prediction'
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
    print("MAMBA BITCOIN PRICE REGRESSOR - TRAINING")
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
    )

    print("Creating data loaders...")
    data_loaders = data_factory.create_data_loaders()

    train_loader = data_loaders['train']
    val_loader = data_loaders['val']
    test_loader = data_loaders['test']


    dataset_info = data_factory.get_dataset_info()

    print(f"Dataset info: {dataset_info}")
    print(f"Train batches: {len(data_loaders['train'])}")
    print(f"Val batches: {len(data_loaders['val'])}")
    print(f"Test batches: {len(data_loaders['test'])}\n")

    model_type = config.get('model.type', 'standard').lower()
    
    if model_type == 'multitask':
        model = MambaMultitaskRegressor(
            d_model=config.get('model.d_model'),
            d_state=config.get('model.d_state'),
            d_conv=config.get('model.d_conv'),
            n_layers=config.get('model.n_layers'),
            dropout=config.get('model.dropout'),
            num_features=config.get('data.num_features'),
            sequence_length=config.get('data.input_window'),
        )
        print(f"Using Multi-Task Mamba Regressor")
    else:
        model = MambaRegressor(
            input_dim=config.get('data.num_features'),
            d_model=config.get('model.d_model'),
            d_state=config.get('model.d_state'),
            d_conv=config.get('model.d_conv'),
            n_layers=config.get('model.n_layers'),
            output_dim=config.get('model.num_classes', 1),
            dropout=config.get('model.dropout'),
        )
        print(f"Using Standard Mamba Regressor")
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    
    model = model.to(device)

    actual_model = model.module if hasattr(model, 'module') else model
    print("Model created successfully")
    print(f"Parameters: {actual_model.get_num_parameters()}\n")

    save_model_architecture(model, results_dir / 'model_architecture.txt')

    loss_type = config.get('loss.type', 'huber').lower()
    if loss_type == 'mse':
        criterion = nn.MSELoss()
        print("Using MSE Loss for regression")
    elif loss_type == 'mae':
        criterion = nn.L1Loss()
        print("Using MAE Loss for regression")
    elif loss_type == 'directional_mse':
        direction_weight = config.get('loss.direction_weight', 2.0)
        variance_weight = config.get('loss.variance_weight', 0.5)
        criterion = DirectionalMSELoss(
            direction_weight=direction_weight,
            variance_weight=variance_weight,
        )
        print(f"Using Directional MSE Loss (dir_w={direction_weight}, var_w={variance_weight})")
    elif loss_type == 'sign_weighted_mse':
        sign_penalty = config.get('loss.sign_penalty_multiplier', 5.0)
        criterion = SignWeightedMSELoss(sign_penalty_multiplier=sign_penalty)
        print(f"Using Sign-Weighted MSE Loss (penalty={sign_penalty}x for wrong sign)")
    elif loss_type == 'asymmetric_mse':
        penalty = config.get('loss.underestimate_penalty', 1.5)
        criterion = AsymmetricMSELoss(underestimate_penalty=penalty)
        print(f"Using Asymmetric MSE Loss (penalty={penalty})")
    elif loss_type == 'multitask':
        base_loss = MultiTaskRegressionLoss(
            direction_weight=config.get('loss.direction_weight', 2.0),
            magnitude_weight=config.get('loss.magnitude_weight', 1.0),
            confidence_weight=config.get('loss.confidence_weight', 0.5),
            regression_weight=config.get('loss.regression_weight', 1.5),
            focal_gamma=config.get('loss.focal_gamma', 2.0),
        )
        if config.get('loss.use_adaptive_weights', False):
            criterion = AdaptiveMultiTaskLoss(base_loss)
            print(f"Using Adaptive Multi-Task Loss (auto-weighted)")
        else:
            criterion = base_loss
            print(f"Using Multi-Task Loss (dir={config.get('loss.direction_weight')}, " +
                  f"mag={config.get('loss.magnitude_weight')}, " +
                  f"conf={config.get('loss.confidence_weight')})")
    else:
        huber_delta = config.get('loss.huber_delta', 0.5)
        criterion = nn.HuberLoss(delta=huber_delta)
        print(f"Using Huber Loss (delta={huber_delta}) for regression")

    optimizer = AdamW(
        model.parameters(),
        lr=config.get('training.learning_rate'),
        weight_decay=config.get('training.weight_decay'),
    )

    scheduler_type = config.get('training.scheduler', 'reduce_on_plateau')
    if scheduler_type == 'cosine':
        total_epochs = config.get('training.epochs')
        scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs)
        print("Using CosineAnnealingLR scheduler")
    else:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=config.get('training.scheduler_patience', 5),
            factor=config.get('training.scheduler_factor', 0.5),
            min_lr=config.get('training.scheduler_min_lr', 1e-6),
            verbose=True,
        )
        print("Using ReduceLROnPlateau scheduler")

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
        warmup_epochs=config.get('training.warmup_epochs', 0),
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

    visualizer = MetricsVisualizer()

    print("Plotting training curves...")
    visualizer.plot_training_curves(history, results_dir)

    print("Loading best model for evaluation...")
    best_checkpoint = checkpoint_dir / 'best_model.pt'
    if best_checkpoint.exists():
        trainer.load_checkpoint(str(best_checkpoint))

    if model_type == 'multitask':
        evaluator = MultiTaskEvaluator(device=device)
        print("\nEvaluating on test set...")
        test_metrics = evaluator.evaluate(model, test_loader)
        evaluator.print_metrics(test_metrics)
    else:
        evaluator = ModelEvaluator(model=model, device=device)
        print("\nEvaluating on test set...")
        test_metrics = evaluator.evaluate(test_loader)
        evaluator.print_metrics(test_metrics)
        evaluator.save_metrics(test_metrics, results_dir / 'evaluation_metrics.txt')

    print("Creating visualizations...")
    if config.get('evaluation.plot_predictions', True):
        if model_type == 'multitask':
            model.eval()
            all_preds = []
            all_targets = []
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    preds = outputs['regression'].cpu().numpy()
                    all_preds.append(preds)
                    all_targets.append(targets.numpy())
            
            y_pred = np.concatenate(all_preds, axis=0).squeeze()
            y_true = np.concatenate(all_targets, axis=0).squeeze()
        else:
            y_pred = test_metrics['y_pred']
            y_true = test_metrics['y_true']
        
        visualizer.plot_predictions(y_true, y_pred, results_dir / 'predictions.png')
    
    if config.get('evaluation.plot_residuals', True):
        if model_type == 'multitask':
            residuals = y_true - y_pred
        else:
            residuals = test_metrics['residuals']
        visualizer.plot_residuals(residuals, results_dir / 'residuals.png')

    export_dir = results_dir / 'exports'
    export_dir.mkdir(parents=True, exist_ok=True)

    try:
        sd_path = export_dir / 'model_state_dict.pth'
        torch.save((actual_model if 'actual_model' in locals() else model).state_dict(), sd_path)
    except Exception:
        pass

    try:
        example = torch.randn(1, config.get('data.input_window'), config.get('data.num_features')).to(device)
        (actual_model if 'actual_model' in locals() else model).eval()
        with torch.no_grad():
            traced = torch.jit.trace((actual_model if 'actual_model' in locals() else model), example, strict=False)
        ts_path = export_dir / 'model_traced.pt'
        traced.save(str(ts_path))
    except Exception:
        pass

    try:
        onnx_path = export_dir / 'model.onnx'
        torch.onnx.export(
            (actual_model if 'actual_model' in locals() else model),
            example,
            str(onnx_path),
            input_names=['input'],
            output_names=['prediction'],
            dynamic_axes={'input': {0: 'batch', 1: 'seq'}, 'prediction': {0: 'batch'}},
            opset_version=17,
            do_constant_folding=False,
        )
    except Exception:
        pass
    
    print(f"\nExports saved: {export_dir}")

    print(f"\nAll results saved to: {results_dir}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print("\nTraining pipeline completed successfully!")


if __name__ == '__main__':
    main()
