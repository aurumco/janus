# Janus V5 - Multi-Asset Cryptocurrency Forecasting

**State-of-the-Art Two-Phase Training with Mamba-SSM Architecture**

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

Janus V5 is a two-phase deep learning system for multi-asset cryptocurrency forecasting using Mamba State Space Models. The system learns market dynamics through self-supervised pre-training, then fine-tunes for directional price prediction.

### Training Pipeline

**Phase 1: Self-Supervised Pre-training**
- 5-minute multi-asset data (15 crypto pairs)
- Three SSL tasks: masked reconstruction, volatility prediction, direction classification
- Model: `d_model=320`, `n_layers=8`, ~5M parameters
- Enhanced contrastive learning for asset separation

**Phase 2: Supervised Fine-tuning**
- 30-minute data with technical indicators
- Transfer learning from pretrained backbone
- Regression task: directional price change prediction
- Preserves asset embeddings for multi-asset awareness

### Key Features

- 🚀 **Mamba-SSM**: Linear-time sequence modeling with selective state spaces
- 🎯 **Multi-Task SSL**: Reconstruction + volatility + direction prediction
- 🧠 **Robust Targets**: Volatility with tail-safe fallback, scaled for stable gradients
- 📊 **Rich Metrics**: Reconstruction MSE, volatility correlation, direction accuracy, Silhouette score
- ⚡ **Optimized Training**: AMP, cosine scheduling, early stopping, gradient clipping
- 🔧 **Transfer Learning**: Asset embeddings preserved across pre-train → fine-tune

## Architecture

### Pre-training Model (MambaPretrainModel)

```
Input (B, 72, 16) 
  → Asset Embedding (15 assets → 32-dim)
  → Input Projection (16+32 → 320)
  → LayerNorm
  → 8× Mamba Blocks (Pre-Norm, d_model=320)
  ├─→ Reconstruction Head (320 → 16)
  ├─→ Volatility Head (320 → 1)
  └─→ Direction Head (320 → 2)
```

**SSL Tasks:**
1. **Masked Reconstruction**: Predict original values at masked positions (MSE loss)
2. **Volatility Prediction**: Forecast future volatility, scaled ×100 (Huber loss)
3. **Direction Classification**: Binary prediction of next candle direction (CrossEntropy)
4. **Contrastive Learning**: Asset separation via InfoNCE (temperature=0.05)

**Parameters:** ~5.0M (`d_model=320`, `n_layers=8`)

### Fine-tuning Model (MambaRegressor)

```
Input (B, 96, 16)
  → [Pretrained] Asset Embedding (15 assets → 32-dim)
  → [Pretrained] Input Projection (16+32 → 320)
  → [Pretrained] LayerNorm
  → [Pretrained] 8× Mamba Blocks (Pre-Norm, d_model=320)
  → [New] Regression Head (320 → 1)
```

**Transfer Learning:**
- Pretrained: `asset_embedding`, `input_projection`, `input_norm`, `mamba_layers`, `layer_norms`
- New: `regression_head`
- First 3 epochs: Backbone frozen
- Remaining epochs: Full fine-tuning

**Parameters:** ~5.0M (`d_model=320`, `n_layers=8`)



### Setup

```bash
# Clone repository
git clone https://github.com/aurumco/janus.git
cd Janus

# Install as package
pip install -e .

# Or install dependencies only
pip install -r requirements.txt
```

## Usage

### Pre-training

```bash
python train.py --mode pretrain --config config.yaml
```

**Outputs:**
- `checkpoints/pretrain/checkpoint_best.pt` - Best model by validation loss
- `logs/pretrain_<timestamp>.log` - Training logs
- `results/pretrain_<timestamp>/` - Metrics and curves

**Flags:**
- `--config` - Path to config file (default: `config.yaml`)
- `--mode` - Training mode: `pretrain` or `finetune`

### Fine-tuning

```bash
python train.py \
  --mode finetune \
  --config config.yaml \
  --load-checkpoint checkpoints/pretrain/checkpoint_best.pt
```

**Outputs:**
- `checkpoints/finetune/checkpoint_best.pt`
- `logs/finetune_<timestamp>.log`
- `results/finetune_<timestamp>/`

**Flags:**
- `--load-checkpoint` - Path to pretrained checkpoint

### Evaluation

```bash
python evaluate_model.py
```

Auto-discovers checkpoint and config. Alternatively:

```bash
python evaluate_model.py \
  --checkpoint checkpoints/pretrain/checkpoint_best.pt \
  --config config.yaml \
  --max-batches 200
```

**Outputs:**
- `evaluation_results/eval_<timestamp>/evaluation_results.txt`
- `evaluation_results/eval_<timestamp>/training_curves.png`
- `logs/evaluate_<timestamp>.log`

## Project Structure

```
Janus/
├── config.yaml                   # Training configuration
├── pyproject.toml                # Package configuration
├── requirements.txt              # Dependencies
├── train.py                      # Main training script
│
├── src/
│   ├── data/
│   │   ├── pretrain_dataset.py   # SSL dataset with masking
│   │   ├── finetune_dataset.py   # Regression dataset
│   │   ├── data_loader.py        # Mode-aware data factory
│   │   └── sequence_strategy.py  # Processing strategy
│   │
│   ├── models/
│   │   ├── mamba_block.py        # Mamba SSM block
│   │   ├── mamba_pretrain.py     # Pre-training model
│   │   └── mamba_regressor.py    # Fine-tuning model
│   │
│   ├── training/
│   │   ├── trainer.py            # Training loop
│   │   ├── losses.py             # Regression losses
│   │   └── pretrain_losses.py    # SSL combined loss
│   │
│   └── evaluation/
│       ├── evaluator.py          # Regression evaluation
│       ├── pretrain_evaluator.py # SSL metrics
│       └── visualizer.py         # Plotting utilities
│
├── tests/
│   ├── test_pretrain_dataset.py  # Dataset tests
│   └── test_models.py            # Model tests
│
└── checkpoints/
    ├── pretrain/                 # Pre-training checkpoints
    └── finetune/                 # Fine-tuning checkpoints
```

## References

- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
- [mamba-ssm GitHub Repository](https://github.com/state-spaces/mamba)
- [Understanding State Space Models](https://srush.github.io/annotated-s4/)

## Contact

For questions or issues, please open an issue on GitHub.

---

**Built with ❤️ by Aurum**
