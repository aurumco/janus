# Janus V5 - Multi-Asset Cryptocurrency Forecasting

**State-of-the-Art Two-Phase Training with Mamba-SSM Architecture**

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

Janus V5 implements a cutting-edge two-phase training system for multi-asset cryptocurrency price forecasting using the Mamba State Space Model architecture.

### Two-Phase Training Pipeline

**Phase 1: Self-Supervised Pre-training (SSL)**
- Foundation model trained on 5-minute multi-asset data
- Masked reconstruction task (15% masking ratio)
- Future volatility prediction task
- Learns market dynamics across 15 cryptocurrency pairs
- Larger model capacity (d_model=256, 8 layers)

**Phase 2: Supervised Fine-tuning**
- Task-specific regression on 30-minute data
- Transfer learning from pretrained backbone
- Directional price change prediction
- Optimized for inference (d_model=128, 4 layers)

### Key Features

- 🚀 **Mamba-SSM Architecture**: Linear-time sequence modeling with selective state spaces
- 🔄 **Two-Phase Training**: Foundation model + task-specific fine-tuning
- 🎯 **Multi-Asset Learning**: Simultaneous training on 15 cryptocurrency pairs
- ⚡ **Advanced Optimization**:
  - Gradient accumulation (4x for pre-train, 2x for fine-tune)
  - Gradient checkpointing for memory efficiency
  - Backbone freezing during early fine-tuning epochs
  - Mixed precision training (AMP)
- 📊 **Comprehensive Monitoring**: SSL metrics + regression performance
- 🔧 **Production Ready**: Multi-format checkpoints, robust weight loading

## Architecture

### Phase 1: Pre-training Model (MambaPretrainModel)

```
Input (B, 256, 16) 
  → Asset Embedding (15 assets → 32-dim)
  → Input Projection (16+32 → 256)
  → LayerNorm
  → 8× Mamba Blocks (Pre-Norm, d_model=256)
  ├─→ Reconstruction Head (256 → 16)
  └─→ Volatility Head (256 → 1)
```

**SSL Tasks:**
1. **Masked Reconstruction**: Predict original values at 15% randomly masked positions
2. **Volatility Prediction**: Forecast future 60-step price volatility

**Parameters:** ~3.5M (d_model=256, n_layers=8)

### Phase 2: Fine-tuning Model (MambaRegressor)

```
Input (B, 96, 16)
  → [Pretrained] Input Projection (16 → 128)
  → [Pretrained] LayerNorm
  → [Pretrained] 4× Mamba Blocks (Pre-Norm, d_model=128)
  → [New] Regression Head (128 → 1)
```

**Transfer Learning:**
- Pretrained layers: `input_projection`, `input_norm`, `mamba_layers`, `layer_norms`
- Randomly initialized: `regression_head`
- First 3 epochs: Backbone frozen, only regression head trains
- Remaining epochs: Full model fine-tuning

**Parameters:** ~3.5M (d_model=256, n_layers=8)



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

## Quick Start

### 1. Dataset Preparation

Datasets are pre-created with scalers:

**Pre-training Dataset:**
```
outputs/datasets/pre-train/parquet/
├── janus_pretrain_5min_dataset.parquet
└── janus_pretrain_5min_scaler.joblib
```

**Fine-tuning Dataset:**
```
outputs/datasets/fine-tune/parquet/
├── janus_finetune_30min_dataset.parquet
└── janus_finetune_30min_scaler.joblib
```

### 2. Phase 1: Pre-training

```bash
python train.py --mode pretrain --config config.yaml
```

**Outputs:**
- `checkpoints/pretrain/best_model.pt`
- `checkpoints/pretrain/best_model_state_dict.pth`
- `checkpoints/pretrain/latest_checkpoint.pt`

### 3. Phase 2: Fine-tuning

```bash
python train.py \
  --mode finetune \
  --config config.yaml \
  --load-checkpoint checkpoints/pretrain/best_model.pt
```

**Outputs:**
- `checkpoints/finetune/best_model.pt`
- `checkpoints/finetune/best_model_state_dict.pth`
- `results/finetune_<timestamp>/`

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
