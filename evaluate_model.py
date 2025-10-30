"""Standalone evaluation script for post-training analysis."""

import sys
import argparse
from pathlib import Path
from datetime import datetime

import torch

try:
    from src.config.config_loader import ConfigLoader
    from src.data.data_loader import DataLoaderFactory
    from src.models.mamba_pretrain import MambaPretrainModel
    from src.evaluation.pretrain_evaluator import PretrainEvaluator
    from src.evaluation.visualizer import MetricsVisualizer
    from src.utils.logger import logger
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for results')
    parser.add_argument('--max-batches', type=int, default=100, help='Max batches to evaluate')
    
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path('evaluation_results') / f'eval_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.section("Model Evaluation")
    logger.info(f"Checkpoint: {checkpoint_path}", indent=1)
    logger.info(f"Config: {config_path}", indent=1)
    logger.info(f"Output: {output_dir}", indent=1)
    
    config_loader = ConfigLoader(str(config_path))
    full_config = config_loader.get_full_config()
    config = config_loader.get_config('pretrain')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}", indent=1)
    
    logger.section("Loading Data")
    data_factory = DataLoaderFactory(
        data_path=config.get('data.path'),
        mode='pretrain',
        batch_size=config.get('data.batch_size', 256),
        sequence_length=config.get('data.sequence_length', 72),
        num_workers=config.get('data.num_workers', 4),
        train_ratio=config.get('data.train_ratio', 0.7),
        val_ratio=config.get('data.val_ratio', 0.15),
        test_ratio=config.get('data.test_ratio', 0.15),
        masking_ratio=config.get('data.masking_ratio', 0.15),
        volatility_lookahead=config.get('data.volatility_lookahead', 60),
        stride=config.get('data.stride', 4),
        use_gpu_preprocess=config.get('data.use_gpu_preprocess', True),
        verbose=True,
    )
    
    data_loaders = data_factory.create_data_loaders()
    val_loader = data_loaders['val']
    test_loader = data_loaders['test']
    
    logger.section("Loading Model")
    model = MambaPretrainModel(
        input_dim=config.get('data.num_features', 16),
        d_model=config.get('model.d_model', 256),
        n_layers=config.get('model.n_layers', 8),
        d_state=config.get('model.d_state', 16),
        d_conv=config.get('model.d_conv', 4),
        expand=config.get('model.expand', 2),
        num_assets=len(full_config.get('assets', [])),
        asset_embedding_dim=config.get('model.asset_embedding_dim', 32),
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
        visualizer.plot_training_curves(checkpoint['history'], output_dir)
        logger.success(f"Training curves saved to {output_dir / 'training_curves.png'}", indent=1)
    
    logger.blank_line()
    logger.success("Evaluation complete!")


if __name__ == '__main__':
    main()
