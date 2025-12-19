# Janus V5 - Multi-Asset Cryptocurrency Forecasting

**State-of-the-Art Two-Phase Training with Mamba-SSM Architecture**

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

Janus V5 is a two-phase deep learning system for multi-asset cryptocurrency forecasting using Mamba State Space Models. The system learns market dynamics through self-supervised pre-training, then fine-tunes for various downstream tasks like direction classification, volatility bounds prediction, and price forecasting.

### Training Pipeline

**Phase 1: Self-Supervised Pre-training**
- 5-minute multi-asset data (15 crypto pairs)
- Three SSL tasks: masked reconstruction, volatility prediction, direction classification
- Enhanced contrastive learning for asset separation

**Phase 2: Supervised Fine-tuning (Multi-Task)**
- 30-minute data with technical indicators
- Transfer learning from pretrained backbone
- Tasks:
    1.  **Directional Classification**: Predict next candle direction (Up/Down).
    2.  **Volatility Bounds (Dynamic TP/SL)**: Predict next 12-candle High/Low returns.
    3.  **Price Forecasting**: Predict next candle log return.

## Quick Start (Kaggle / Local)

### 1. Generate Datasets

First, generate the task-specific datasets from your processed parquet file.

```bash
# Assuming your main processed data is at /kaggle/input/janus/processed.parquet
python -m src.data.generate_finetune_tasks \
  --input "/path/to/your/input_dataset.parquet" \
  --output "/kaggle/working/tasks"
```
This will create:
- `/kaggle/working/tasks/finetune_task1_direction.parquet`
- `/kaggle/working/tasks/finetune_task2_volatility.parquet`
- `/kaggle/working/tasks/finetune_task3_price.parquet`

### 2. Fine-tune Models

You can fine-tune for each task separately using the provided configuration files.

#### Task 1: Directional Classification (Binary)
Learns to predict if the next close will be higher (1) or lower (0) than current.

```bash
python train.py \
  --mode finetune \
  --config config/finetune_direction.yaml \
  --load-checkpoint checkpoints/pretrain/checkpoint_best.pt \
  --data-path /kaggle/working/tasks/finetune_task1_direction.parquet
```

#### Task 2: Dynamic TP/SL (Volatility Regression)
Learns to predict the maximum upside (High) and maximum downside (Low) over the next 12 candles.
Output is 2-dimensional: `[Max_Return, Min_Return]`.

```bash
python train.py \
  --mode finetune \
  --config config/finetune_volatility.yaml \
  --load-checkpoint checkpoints/pretrain/checkpoint_best.pt \
  --data-path /kaggle/working/tasks/finetune_task2_volatility.parquet
```

#### Task 3: Price Forecasting (Scalar Regression)
Learns to predict the exact log return of the next candle.

```bash
python train.py \
  --mode finetune \
  --config config/finetune_price.yaml \
  --load-checkpoint checkpoints/pretrain/checkpoint_best.pt \
  --data-path /kaggle/working/tasks/finetune_task3_price.parquet
```

### 3. Model Export

The training script automatically exports models in both **PyTorch State Dict** (`.pth`) and **ONNX** (`.onnx`) formats at the end of training.

Location: `results/<mode>_<timestamp>/exports/`

- `model_state_dict.pth`: PyTorch weights.
- `model.onnx`: ONNX graph for deployment.
- `model_traced.pt`: TorchScript traced model.

## Configuration

We adhere to Separation of Concerns by using specific config files for each task.

- **`config/finetune_direction.yaml`**: Sets loss to `bce` (Binary Cross Entropy) and output dim to 1.
- **`config/finetune_volatility.yaml`**: Sets loss to `mse` and output dim to 2. Handles multi-column targets.
- **`config/finetune_price.yaml`**: Sets loss to `confidence_weighted` and output dim to 1.

You can modify these files or override settings via CLI (limited support) or by editing the YAMLs directly.

## Project Structure

```
Janus/
├── config.yaml                   # Global configuration
├── config/                       # Task-specific configurations
│   ├── finetune_direction.yaml
│   ├── finetune_volatility.yaml
│   └── finetune_price.yaml
├── src/
│   ├── data/
│   │   ├── generate_finetune_tasks.py # Dataset generation script
│   │   ├── finetune_dataset.py        # Dataset class (dict output)
│   │   └── ...
│   ├── models/
│   │   └── ...
│   └── training/
│       └── ...
├── train.py                      # Main training script
└── ...
```

## References

- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
