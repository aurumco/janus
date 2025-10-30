"""Standalone evaluation script for post-training analysis."""

import sys
import argparse
from pathlib import Path
from datetime import datetime

import torch
import warnings

try:
    _here = Path(__file__).resolve().parent
except NameError:
    _here = Path.cwd()

_candidates = {
    _here,
    _here.parent,
    _here / 'janus',
    Path('/kaggle/working'),
    Path('/kaggle/working/janus'),
}
for _p in list(_candidates):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

try:
    from src.config.config_loader import ConfigLoader
    from src.data.data_loader import DataLoaderFactory
    from src.data.sequence_strategy import SequenceProcessingStrategy
    from src.models.mamba_pretrain import MambaPretrainModel
    from src.evaluation.pretrain_evaluator import PretrainEvaluator
    from src.evaluation.visualizer import MetricsVisualizer
    from src.utils.logger import TrainingLogger
except Exception as e:
    sys.stderr.write(f"Import error: {e}\n")
    sys.exit(1)


def main():
    warnings.filterwarnings("ignore", category=FutureWarning, message="`torch.cuda.amp.custom_fwd")
    warnings.filterwarnings("ignore", category=FutureWarning, message="`torch.cuda.amp.custom_bwd")

    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint file')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for results')
    parser.add_argument('--max-batches', type=int, default=100, help='Max batches to evaluate')
    
    raw_args = sys.argv[1:]
    filtered_args = [a for a in raw_args if not a.startswith('-f') and 'kernel' not in a.lower()]
    args = parser.parse_args(filtered_args)
    
    if args.checkpoint is None:
        candidates = [
            Path('/kaggle/working/checkpoints/pretrain/checkpoint_best.pt'),
            Path('/kaggle/working/checkpoints/pretrain/checkpoint_latest.pt'),
            Path('checkpoints/pretrain/checkpoint_best.pt'),
            Path('checkpoints/pretrain/checkpoint_latest.pt'),
        ]
        checkpoint_path = next((p for p in candidates if p.exists()), None)
        if checkpoint_path is None:
            logger.error("No checkpoint provided and none found under checkpoints/pretrain")
            sys.exit(1)
    else:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            sys.exit(1)
    
    config_path = Path(args.config) if args.config else Path('config.yaml')
    if not config_path.exists():
        alt_candidates = [
            Path('./config.yaml'),
            Path('./janus/config.yaml'),
            Path('/kaggle/working/janus/config.yaml'),
            Path('/kaggle/working/config.yaml'),
        ]
        config_path = next((p for p in alt_candidates if p.exists()), None)
        if config_path is None:
            logger.error("Config file not found (tried config.yaml and janus/config.yaml)")
            sys.exit(1)
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path('evaluation_results') / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger with file output
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"evaluate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = TrainingLogger(log_file=str(log_file))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.header(f"Janus Model Evaluation")
    logger.info(f"Checkpoint: {checkpoint_path}", indent=1)
    logger.info(f"Config: {config_path}", indent=1)
    logger.info(f"Output: {output_dir}", indent=1)
    
    config_loader = ConfigLoader(str(config_path))
    full_config = config_loader.config
    pget = lambda k, d=None: config_loader.get(f'pretrain.{k}', d)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}", indent=1)
    
    logger.section("Loading Data")
    processing_strategy = SequenceProcessingStrategy(
        feature_columns=None,
        target_column=None,
        sequence_length=pget('data.sequence_length', 72),
    )
    data_factory = DataLoaderFactory(
        data_path=pget('data.path'),
        processing_strategy=processing_strategy,
        mode='pretrain',
        batch_size=pget('data.batch_size', 256),
        sequence_length=pget('data.sequence_length', 72),
        num_workers=pget('data.num_workers', 4),
        train_ratio=pget('data.train_ratio', 0.7),
        val_ratio=pget('data.val_ratio', 0.15),
        test_ratio=pget('data.test_ratio', 0.15),
        masking_ratio=pget('data.masking_ratio', 0.15),
        volatility_lookahead=pget('data.volatility_lookahead', 60),
        stride=pget('data.stride', 4),
        use_gpu_preprocess=pget('data.use_gpu_preprocess', True),
        verbose=True,
    )
    
    data_loaders = data_factory.create_data_loaders()
    val_loader = data_loaders['val']
    test_loader = data_loaders['test']

    try:
        train_loader = data_loaders['train']
        logger.info(
            f"Datasets: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, test={len(test_loader.dataset)}",
            indent=1,
        )
    except Exception:
        pass
    
    logger.section("Loading Model")
    model = MambaPretrainModel(
        input_dim=pget('data.num_features', 16),
        d_model=pget('model.d_model', 320),
        d_state=pget('model.d_state', 16),
        d_conv=pget('model.d_conv', 4),
        n_layers=pget('model.n_layers', 8),
        reconstruction_head_dim=pget('data.num_features', 16),
        volatility_head_dim=pget('model.volatility_head_dim', 1),
        dropout=pget('model.dropout', 0.1),
        num_assets=len(full_config.get('assets', [])),
        asset_embedding_dim=pget('model.asset_embedding_dim', 32),
        use_gradient_checkpointing=pget('model.use_gradient_checkpointing', False),
        enable_direction_head=pget('model.enable_direction_head', True),
    ).to(device)
    
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.success(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}", indent=1)
        logger.metric("Best val loss", f"{checkpoint.get('best_val_loss', 'unknown'):.6f}", indent=1)
    else:
        model.load_state_dict(checkpoint)
        logger.success("Loaded model state dict", indent=1)
    
    logger.section("Evaluation on Validation Set")
    evaluator = PretrainEvaluator(model=model, device=device)
    val_metrics = evaluator.evaluate(val_loader, max_batches=args.max_batches)
    evaluator.print_metrics(val_metrics)
    
    logger.section("Evaluation on Test Set")
    if len(test_loader) == 0:
        logger.warning("Test loader is empty; skipping test evaluation", indent=1)
        test_metrics = {
            'masked_reconstruction_mse': 0.0,
            'volatility_mse': 0.0,
            'embedding_silhouette_score': 0.0,
            'volatility_correlation': 0.0,
            'temporal_consistency': 0.0,
        }
    else:
        test_metrics = evaluator.evaluate(test_loader, max_batches=args.max_batches)
        evaluator.print_metrics(test_metrics)
    
    logger.section("Saving Results")
    results_file = output_dir / 'evaluation_results.txt'
    with open(results_file, 'w') as f:
        f.write(f"Evaluation Results\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Validation Metrics:\n")
        for key, value in val_metrics.items():
            f.write(f"  {key}: {value}\n")
        
        f.write("\nTest Metrics:\n")
        for key, value in test_metrics.items():
            f.write(f"  {key}: {value}\n")
    
    logger.success(f"Results saved to {results_file}", indent=1)
    
    if 'history' in checkpoint:
        logger.info("Generating training curves from checkpoint history", indent=1)
        visualizer = MetricsVisualizer()
        curves_file = output_dir / 'training_curves.png'
        logger.info(f"Saving curves to {curves_file}", indent=1)
        visualizer.save_training_curves(checkpoint, str(curves_file))
        logger.success(f"Training curves saved to {curves_file}", indent=1)
    
    logger.success(f"Evaluation complete! Results saved to {output_dir}", indent=1)

    try:
        export_dir = output_dir / 'exports'
        export_dir.mkdir(parents=True, exist_ok=True)

        seq_len = pget('data.sequence_length', 72)
        in_features = pget('data.num_features', 16)
        batch_dim = 1

        dummy_seq = torch.randn(batch_dim, seq_len, in_features, device=device)
        dummy_asset = torch.zeros(batch_dim, dtype=torch.long, device=device)

        model.eval()
        onnx_path = export_dir / 'pretrain_model.onnx'
        torch.onnx.export(
            model,
            (dummy_seq, dummy_asset),
            str(onnx_path),
            input_names=['input_sequence', 'asset_ids'],
            output_names=['reconstructed_sequence', 'predicted_volatility', 'predicted_direction'],
            dynamic_axes={'input_sequence': {0: 'batch', 1: 'seq'},
                          'asset_ids': {0: 'batch'},
                          'reconstructed_sequence': {0: 'batch', 1: 'seq'},
                          'predicted_volatility': {0: 'batch'},
                          'predicted_direction': {0: 'batch'}},
            opset_version=17,
            do_constant_folding=False,
        )
        logger.success(f"ONNX exported to {onnx_path}", indent=1)
    except Exception as e:
        logger.warning(f"ONNX export failed: {e}", indent=1)

    logger.info(f"Log file: {log_file}", indent=1)
    logger.close()


if __name__ == '__main__':
    main()
