"""Main training script for Mamba Bitcoin price change regressor."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['ABSL_LOGGING_MIN_LOG_LEVEL'] = '3'

import argparse
from datetime import datetime
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module=r"mamba_ssm\..*")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"torch\.nn\.parallel\.parallel_apply")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"torch\.utils\.checkpoint")

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
try:
    import absl.logging as absl_logging  # type: ignore
    absl_logging.set_verbosity(absl_logging.ERROR)
    absl_logging.use_python_logging()
except Exception:
    pass

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
from src.utils.logger import logger


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
            """Fetch config with mode prefix first, then fall back to global.

            Example: if prefix='pretrain' and key='data.path', this checks
            'pretrain.data.path' first, then 'data.path'.
            """
            mode_key = f"{self.prefix}.{key}"
            mode_val = self.full_cfg.get(mode_key, None)
            if mode_val is not None:
                return mode_val
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

    logger.header(f"Janus Cryptocurrency Forecasting - {mode.upper()} Mode")
    
    logger.config_section("Configuration", {
        "Config file": args.config,
        "Training mode": mode,
        "Timestamp": timestamp,
    })
    
    logger.config_section("Paths", {
        "Data": data_path,
        "Output": output_dir,
        "Results": results_dir,
        "Checkpoints": checkpoint_dir,
    })
    
    if args.load_checkpoint:
        logger.info(f"Pretrained weights: {args.load_checkpoint}", indent=1)
    
    logger.blank_line()

    # Prepare data loader based on mode
    use_gpu_pre = config.get('data.use_gpu_preprocess')
    if use_gpu_pre is None:
        use_gpu_pre = full_config.get('device.use_gpu_preprocess', True)
    if mode == 'pretrain':
        # Pre-training mode: no target column needed
        use_streaming_fallback = bool(config.get('data.use_streaming_fallback', False))
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
            use_gpu_preprocess=use_gpu_pre,
            use_streaming_fallback=use_streaming_fallback,
            verbose=True,
            stride=config.get('data.stride', 4),
        )
    else:
        # Fine-tuning mode
        use_streaming_fallback = bool(config.get('data.use_streaming_fallback', False))
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
            use_gpu_preprocess=use_gpu_pre,
            use_streaming_fallback=use_streaming_fallback,
        )

    logger.section("Data Loading")
    try:
        start_dl = datetime.now()
        data_loaders = data_factory.create_data_loaders()
        end_dl = datetime.now()
        logger.success(f"Data loaders ready in {(end_dl - start_dl).total_seconds():.2f}s", indent=1)
    except MemoryError as me:
        logger.error("MemoryError! Reduce batch size or sequence length.", indent=1)
        logger.info(f"Details: {me}", indent=2)
        raise
    except Exception as e:
        import traceback
        logger.error(f"Failed to create data loaders: {type(e).__name__}", indent=1)
        logger.info(str(e), indent=2)
        traceback.print_exc()
        raise

    train_loader = data_loaders['train']
    val_loader = data_loaders['val']
    test_loader = data_loaders['test']


    dataset_info = data_factory.get_dataset_info()
    dataset_info['train_batches'] = len(data_loaders['train'])
    dataset_info['val_batches'] = len(data_loaders['val'])
    dataset_info['batch_size'] = config.get('data.batch_size')
    logger.data_info(dataset_info)
    logger.blank_line()

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
        checkpoint_path = None
        if args.load_checkpoint:
            checkpoint_path_obj = Path(args.load_checkpoint)
            
            if checkpoint_path_obj.exists():
                checkpoint_path = args.load_checkpoint
                logger.success(f"Found checkpoint: {checkpoint_path}", indent=1)
            else:
                possible_paths = [
                    Path("/kaggle/input") / args.load_checkpoint,
                    Path("/kaggle/working") / args.load_checkpoint,
                    Path("checkpoints/pretrain") / args.load_checkpoint,
                ]
                
                for alt_path in possible_paths:
                    if alt_path.exists():
                        checkpoint_path = str(alt_path)
                        logger.success(f"Found checkpoint at alternative path: {checkpoint_path}", indent=1)
                        break
                
                if checkpoint_path is None:
                    logger.warning(f"Checkpoint not found: {args.load_checkpoint}", indent=1)
                    logger.info("Checked paths:", indent=1)
                    logger.info(f"- {checkpoint_path_obj}", indent=2)
                    for p in possible_paths:
                        logger.info(f"- {p}", indent=2)
                    logger.warning("Falling back to random initialization", indent=1)
        
        try:
            model = MambaRegressor(
                input_dim=config.get('data.num_features'),
                d_model=config.get('model.d_model'),
                d_state=config.get('model.d_state'),
                d_conv=config.get('model.d_conv'),
                n_layers=config.get('model.n_layers'),
                output_dim=config.get('model.output_dim', 1),
                dropout=config.get('model.dropout'),
                pretrained_checkpoint_path=checkpoint_path,
            )
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}", indent=1)
            logger.warning("Falling back to random initialization", indent=1)
            model = MambaRegressor(
                input_dim=config.get('data.num_features'),
                d_model=config.get('model.d_model'),
                d_state=config.get('model.d_state'),
                d_conv=config.get('model.d_conv'),
                n_layers=config.get('model.n_layers'),
                output_dim=config.get('model.output_dim', 1),
                dropout=config.get('model.dropout'),
                pretrained_checkpoint_path=None,
            )
    
    logger.section("Model Initialization")
    
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel", indent=1)
        model = nn.DataParallel(model)
    
    model = model.to(device)

    if not hasattr(model, 'module'):
        try:
            model = torch.compile(model, mode='reduce-overhead')
            logger.success("Model compiled successfully with torch.compile", indent=1)
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}", indent=1)
    else:
        logger.info("Skipping torch.compile due to DataParallel incompatibility", indent=1)

    actual_model = model.module if hasattr(model, 'module') else model
    params = actual_model.get_num_parameters()
    logger.success("Model created successfully", indent=1)
    logger.metric("Total parameters", f"{params['total']:,}", indent=1)
    logger.metric("Trainable parameters", f"{params['trainable']:,}", indent=1)
    logger.blank_line()

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
        logger.config_section("Loss Function: Enhanced Pre-training", {
            "Masked price weight": config.get('loss.masked_price_weight', 1.0),
            "Volatility weight": config.get('loss.volatility_weight', 0.5),
            "Contrastive weight": config.get('loss.contrastive_weight', 0.2),
            "Temporal consistency": config.get('loss.temporal_consistency_weight', 0.1),
            "Temperature": config.get('loss.temperature', 0.07),
        })
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
        logger.info(f"Scheduler: CosineAnnealingLR (T_max={total_epochs})", indent=1)
    else:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=config.get('training.scheduler_patience', 5),
            factor=config.get('training.scheduler_factor', 0.5),
            min_lr=config.get('training.scheduler_min_lr', 1e-6),
            verbose=False,
        )
        logger.info("Scheduler: ReduceLROnPlateau", indent=1)
    
    logger.blank_line()

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
        checkpoint_interval=full_config.get('logging.checkpoint_interval', 10),
    )
    
    if mode == 'finetune' and config.get('training.freeze_backbone_epochs', 0) > 0:
        freeze_epochs = config.get('training.freeze_backbone_epochs')
        logger.info(f"Will freeze backbone for first {freeze_epochs} epochs", indent=1)

    # Try to resume training from checkpoint
    if args.resume:
        resume_path = Path(args.resume)
        if trainer.load_checkpoint(resume_path):
            logger.success(f"Resumed training from: {args.resume}", indent=1)
        else:
            logger.warning(f"Could not load checkpoint: {args.resume}", indent=1)
    else:
        # Auto-resume from latest if exists
        if trainer.load_checkpoint(None):
            logger.success("Auto-resumed from latest checkpoint", indent=1)

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.get('training.epochs', 100),
        log_interval=full_config.get('logging.log_interval', 10),
        freeze_backbone_epochs=config.get('training.freeze_backbone_epochs', 0) if mode == 'finetune' else 0,
    )

    logger.success("Training completed!", indent=0)
    
    # Check if training produced valid losses
    import math
    if history['train_loss'] and history['val_loss']:
        last_train_loss = history['train_loss'][-1]
        last_val_loss = history['val_loss'][-1]
        if math.isnan(last_train_loss) or math.isnan(last_val_loss):
            logger.warning("Training produced NaN losses - model may not have trained properly", indent=1)
            logger.info("This could be due to:", indent=1)
            logger.info("  - Learning rate too high", indent=2)
            logger.info("  - Numerical instability in loss computation", indent=2)
            logger.info("  - Model initialization issues", indent=2)

    visualizer = MetricsVisualizer()

    logger.section("Visualization")
    logger.info("Plotting training curves...", indent=1)
    visualizer.plot_training_curves(history, results_dir)

    logger.info("Loading best model for evaluation...", indent=1)
    best_checkpoint = checkpoint_dir / 'checkpoint_best.pt'
    if best_checkpoint.exists():
        logger.success("Found best checkpoint, loading...", indent=2)
        trainer.load_checkpoint(str(best_checkpoint))
    else:
        logger.warning("No best checkpoint found - using current model state", indent=2)

    # Run mode-specific evaluation
    if mode == 'finetune':
        evaluator = ModelEvaluator(
            model=model,
            device=device,
        )

        logger.section("Evaluation (Test)")
        test_metrics = evaluator.evaluate(test_loader)

        evaluator.print_metrics(test_metrics)
        evaluator.save_metrics(test_metrics, results_dir / 'evaluation_metrics.txt')

        logger.info("Creating visualizations...", indent=1)
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
        logger.section("Evaluation (Pre-training)")
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
    
    logger.blank_line()
    logger.success(f"Exports saved: {export_dir}", indent=0)
    logger.success(f"Results saved: {results_dir}", indent=0)
    logger.success(f"Checkpoints saved: {checkpoint_dir}", indent=0)
    logger.blank_line()
    logger.header("Training Pipeline Completed Successfully")


if __name__ == '__main__':
    main()
