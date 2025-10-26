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
from src.evaluation.pretrain_evaluator import PretrainEvaluator
from src.evaluation.visualizer import MetricsVisualizer
from src.models.mamba_regressor import MambaRegressor
from src.models.mamba_pretrain import MambaPretrainModel
from src.training.trainer import Trainer
from src.training.losses import (
    DirectionalMSELoss,
    AsymmetricMSELoss,
    SignWeightedMSELoss,
    ConfidenceWeightedLoss,
    AdaptiveSignLoss,
)
from src.training.pretrain_losses import PretrainLoss
from src.utils.helpers import get_device, save_model_architecture, set_seed


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Train Mamba regressor for cryptocurrency price prediction'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['pretrain', 'finetune'],
        required=True,
        help='Training mode: pretrain or finetune'
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
        '--load-checkpoint',
        type=str,
        default=None,
        help='Path to pretrained checkpoint for fine-tuning'
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

    # Load full config
    full_config = ConfigLoader(args.config)

    set_seed(full_config.get('seed', 42))

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
        use_cuda=full_config.get('device.use_cuda', True),
        device_id=full_config.get('device.device_id', 0)
    )
    if device.type == 'cuda':
        try:
            _ = torch.tensor(0.0, device=device)
            torch.cuda.synchronize()
        except Exception:
            pass

    # Get mode-specific config section
    mode = args.mode
    config_prefix = mode
    
    # Create a view into the mode-specific config
    class ModeConfig:
        def __init__(self, full_cfg, prefix):
            self.full_cfg = full_cfg
            self.prefix = prefix
        
        def get(self, key, default=None):
            # Try mode-specific first, then fall back to global
            mode_key = f"{self.prefix}.{key}"
            if self.full_cfg.config.get(self.prefix) and key in str(self.full_cfg.config.get(self.prefix, {})):
                return self.full_cfg.get(mode_key, default)
            # Fall back to global config
            return self.full_cfg.get(key, default)
    
    config = ModeConfig(full_config, config_prefix)

    data_path = args.data_path or config.get('data.path')
    output_dir = Path(args.output_dir or full_config.get('paths.output_dir'))

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    mode_suffix = f"{mode}_{timestamp}"
    results_dir = output_dir / full_config.get('paths.results_dir', 'results') / mode_suffix
    checkpoint_dir = output_dir / full_config.get('paths.checkpoint_dir', 'checkpoints') / mode
    log_dir = output_dir / full_config.get('paths.logs_dir', 'logs') / mode_suffix

    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print(f"MAMBA CRYPTOCURRENCY FORECASTING - {mode.upper()} MODE")
    print("="*70)
    print(f"Configuration: {args.config}")
    print(f"Mode: {mode}")
    print(f"Data path: {data_path}")
    print(f"Output directory: {output_dir}")
    print(f"Results directory: {results_dir}")
    if args.load_checkpoint:
        print(f"Loading pretrained weights: {args.load_checkpoint}")
    print("="*70 + "\n")

    # Prepare data loader based on mode
    if mode == 'pretrain':
        # Pre-training mode: no target column needed
        data_factory = DataLoaderFactory(
            data_path=data_path,
            processing_strategy=SequenceProcessingStrategy(
                feature_columns=None,  # Will use all features
                target_column=None,
                sequence_length=config.get('data.sequence_length'),
            ),
            mode='pretrain',
            train_ratio=config.get('data.train_ratio'),
            val_ratio=config.get('data.val_ratio'),
            test_ratio=config.get('data.test_ratio', 0.0),
            batch_size=config.get('data.batch_size'),
            num_workers=config.get('data.num_workers'),
            shuffle_train=config.get('data.shuffle_train'),
            random_seed=full_config.get('seed', 42),
            masking_ratio=config.get('data.masking_ratio', 0.15),
            volatility_lookahead=config.get('data.volatility_lookahead', 60),
            sequence_length=config.get('data.sequence_length'),
            smart_masking_prob=config.get('data.smart_masking_prob', 0.4),
            cross_asset_masking_prob=config.get('data.cross_asset_masking_prob', 0.3),
        )
    else:
        # Fine-tuning mode
        data_factory = DataLoaderFactory(
            data_path=data_path,
            processing_strategy=SequenceProcessingStrategy(
                feature_columns=config.get('data.feature_columns'),
                target_column=config.get('data.target_column'),
                sequence_length=config.get('data.sequence_length'),
            ),
            mode='finetune',
            train_ratio=config.get('data.train_ratio'),
            val_ratio=config.get('data.val_ratio'),
            test_ratio=config.get('data.test_ratio'),
            batch_size=config.get('data.batch_size'),
            num_workers=config.get('data.num_workers'),
            shuffle_train=config.get('data.shuffle_train'),
            random_seed=full_config.get('seed', 42),
            sequence_length=config.get('data.sequence_length'),
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

    # Create model based on mode
    if mode == 'pretrain':
        model = MambaPretrainModel(
            input_dim=config.get('data.num_features'),
            d_model=config.get('model.d_model'),
            d_state=config.get('model.d_state'),
            d_conv=config.get('model.d_conv'),
            n_layers=config.get('model.n_layers'),
            reconstruction_head_dim=config.get('model.reconstruction_head_dim'),
            volatility_head_dim=config.get('model.volatility_head_dim', 1),
            dropout=config.get('model.dropout'),
            num_assets=config.get('model.num_assets', 15),
            asset_embedding_dim=config.get('model.asset_embedding_dim', 16),
            use_gradient_checkpointing=config.get('model.use_gradient_checkpointing', False),
        )
    else:
        model = MambaRegressor(
            input_dim=config.get('data.num_features'),
            d_model=config.get('model.d_model'),
            d_state=config.get('model.d_state'),
            d_conv=config.get('model.d_conv'),
            n_layers=config.get('model.n_layers'),
            output_dim=config.get('model.output_dim', 1),
            dropout=config.get('model.dropout'),
            pretrained_checkpoint_path=args.load_checkpoint,
        )
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    
    model = model.to(device)

    actual_model = model.module if hasattr(model, 'module') else model
    print("Model created successfully")
    print(f"Parameters: {actual_model.get_num_parameters()}\n")

    save_model_architecture(model, results_dir / 'model_architecture.txt')

    # Create loss function based on mode
    if mode == 'pretrain':
        criterion = PretrainLoss(
            masked_price_weight=config.get('loss.masked_price_weight', 1.0),
            volatility_weight=config.get('loss.volatility_weight', 0.5),
            contrastive_weight=config.get('loss.contrastive_weight', 0.2),
            temporal_consistency_weight=config.get('loss.temporal_consistency_weight', 0.1),
            temperature=config.get('loss.temperature', 0.07),
        )
        print(f"Using Enhanced Pre-training Loss:")
        print(f"  - Masked Price Weight      : {config.get('loss.masked_price_weight', 1.0)}")
        print(f"  - Volatility Weight        : {config.get('loss.volatility_weight', 0.5)}")
        print(f"  - Contrastive Weight       : {config.get('loss.contrastive_weight', 0.2)}")
        print(f"  - Temporal Consistency     : {config.get('loss.temporal_consistency_weight', 0.1)}")
        print(f"  - Temperature              : {config.get('loss.temperature', 0.07)}")
    else:
        loss_type = config.get('loss.type', 'huber').lower()
        if loss_type == 'mse':
            criterion = nn.MSELoss()
            print("Using MSE Loss for regression")
        elif loss_type == 'mae':
            criterion = nn.L1Loss()
            print("Using MAE Loss for regression")
        elif loss_type == 'confidence_weighted':
            wrong_sign_penalty = config.get('loss.wrong_sign_penalty', 3.0)
            magnitude_weight = config.get('loss.magnitude_weight', 0.3)
            criterion = ConfidenceWeightedLoss(
                wrong_sign_penalty=wrong_sign_penalty,
                magnitude_weight=magnitude_weight,
            )
            print(f"Using Confidence-Weighted Loss (sign_penalty={wrong_sign_penalty}, mag_w={magnitude_weight})")
        elif loss_type == 'adaptive_sign':
            base_penalty = config.get('loss.base_penalty', 2.5)
            magnitude_threshold = config.get('loss.magnitude_threshold', 0.005)
            criterion = AdaptiveSignLoss(
                base_penalty=base_penalty,
                magnitude_threshold=magnitude_threshold,
            )
            print(f"Using Adaptive Sign Loss (base_penalty={base_penalty}, threshold={magnitude_threshold})")
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
        else:
            huber_delta = config.get('loss.huber_delta', 0.5)
            criterion = nn.HuberLoss(delta=huber_delta)
            print(f"Using Huber Loss (delta={huber_delta}) for regression")

    if mode == 'finetune':
        pretrained_params = []
        new_params = []
        
        for name, param in model.named_parameters():
            if 'regression_head' in name or 'adapter' in name or 'output' in name:
                new_params.append(param)
            else:
                pretrained_params.append(param)
        
        base_lr = config.get('training.learning_rate')
        head_lr_multiplier = config.get('training.head_lr_multiplier', 10.0)
        
        optimizer = AdamW([
            {'params': pretrained_params, 'lr': base_lr, 'name': 'pretrained'},
            {'params': new_params, 'lr': base_lr * head_lr_multiplier, 'name': 'head'}
        ], weight_decay=config.get('training.weight_decay'))
        
        print(f"Using layered LR: backbone={base_lr:.2e}, head={base_lr*head_lr_multiplier:.2e}")
    else:
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
        log_dir=log_dir if full_config.get('logging.tensorboard', True) else None,
        early_stopping_patience=config.get('training.early_stopping_patience', 10),
        early_stopping_min_delta=config.get('training.early_stopping_min_delta', 0.0001),
        use_amp=full_config.get('device.mixed_precision', True),
        warmup_epochs=config.get('training.warmup_epochs', 0),
        accumulation_steps=config.get('training.accumulation_steps', 1),
    )
    
    if mode == 'finetune' and config.get('training.freeze_backbone_epochs', 0) > 0:
        freeze_epochs = config.get('training.freeze_backbone_epochs')
        print(f"\nWill freeze backbone for first {freeze_epochs} epochs")

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.get('training.epochs', 100),
        log_interval=full_config.get('logging.log_interval', 10),
        freeze_backbone_epochs=config.get('training.freeze_backbone_epochs', 0) if mode == 'finetune' else 0,
    )

    print("\nTraining completed!")

    visualizer = MetricsVisualizer()

    print("Plotting training curves...")
    visualizer.plot_training_curves(history, results_dir)

    print("Loading best model for evaluation...")
    best_checkpoint = checkpoint_dir / 'best_model.pt'
    if best_checkpoint.exists():
        trainer.load_checkpoint(str(best_checkpoint))

    # Run mode-specific evaluation
    if mode == 'finetune':
        evaluator = ModelEvaluator(
            model=model,
            device=device,
        )

        print("\nEvaluating on test set...")
        test_metrics = evaluator.evaluate(test_loader)

        evaluator.print_metrics(test_metrics)
        evaluator.save_metrics(test_metrics, results_dir / 'evaluation_metrics.txt')

        print("Creating visualizations...")
        if full_config.get('evaluation.plot_predictions', True):
            visualizer.plot_predictions(
                test_metrics['y_true'],
                test_metrics['y_pred'],
                results_dir / 'predictions.png'
            )
        
        if full_config.get('evaluation.plot_residuals', True):
            visualizer.plot_residuals(
                test_metrics['residuals'],
                results_dir / 'residuals.png'
            )
    else:
        print("\nEvaluating SSL pre-training metrics...")
        pretrain_evaluator = PretrainEvaluator(model=model, device=device)
        ssl_metrics = pretrain_evaluator.evaluate(val_loader)
        pretrain_evaluator.print_metrics(ssl_metrics)

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
