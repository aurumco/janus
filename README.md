# HRM Timeseries Training (Janus V4)

A minimal, production-ready training pipeline for hierarchical reasoning on financial timeseries with hybrid 5-class labels.

## Quick Start

- Install deps (CPU-friendly):
  ```bash
  pip install -r requirements.txt
  ```
- Train (CPU, minimal console):
  ```bash
  python pretrain.py
  ```

## Data
- Expected parquet at: `../Dataset/janus_m15_dataset.parquet`
- Index: DatetimeIndex (UTC). Columns: feature columns + `target` (int in [0..4]).
- Window length: 27 (M15). Labels taken at t+27.
- Split:
  - Train/Val: 2023-01-01 .. 2025-06-30 (90/10 chrono)
  - Test:      >= 2025-07-01

## Model
- File: `models/hrm/hrm_act_v1.py`
- Numeric-only embedding → hierarchical blocks → last-timestep logits (5 classes).
- Light dropout (0.1) to reduce overfitting.

## Training
- File: `pretrain.py`
- Batch size, epochs, LR from `config/cfg_pretrain.yaml`.
- Optimizer: AdamW; Scheduler: CosineAnnealingLR.
- Loss: focal loss with class weights, last-timestep classification.
- Metrics: accuracy, macro-F1, and per-class (0,4) precision/recall/F1.
- Logging: wandb (layers/params via `wandb.watch`), console kept minimal.

## Outputs
- Checkpoints: `checkpoints/janus_v4/`
  - `latest.pt` per epoch
  - `best.pt` (state_dict)
  - `best_full.pth` (full model)
  - `best_scripted.pt` (TorchScript)
  - `best.onnx` (opset 17, dynamic batch)
- Results: `results/<timestamp>/`
  - `epoch_####.txt` per-epoch JSON report (metrics, timing, optimizer, config)
  - `loss_curves.png`, `metrics_curves.png` (dark minimal theme)
  - `history.json`, `final_summary.txt`

## Configuration
- File: `config/cfg_pretrain.yaml`
  - `global_batch_size`: default 1280
  - `epochs`: default 150
  - `eval_interval`: default 5
  - `lr`, `weight_decay`, warmup, etc.
- Architecture: `config/arch/hrm_v1.yaml`

## Performance & Memory
- CPU-first by default (to avoid noisy CUDA warnings). If using CUDA, set `CUDA_VISIBLE_DEVICES` accordingly.
- Per-batch cleanup and (optional) `torch.cuda.empty_cache()` reduce peak memory.
- DataLoader: tuned `num_workers` and `pin_memory` based on device.

## License
Proprietary. All rights reserved.
