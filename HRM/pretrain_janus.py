import os
# Silence CUDA probing and WANDB verbosity before importing torch/wandb
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("WANDB_SILENT", "true")
os.environ.setdefault("WANDB_CONSOLE", "off")

import math
import yaml
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import wandb

from utils.functions import load_model_class
from janus_dataset_loader import create_dataloaders, INPUT_WINDOW_CANDLES
from models.losses import IGNORE_LABEL_ID


@dataclass
class TrainConfig:
    data_path: str
    arch_name: str
    loss_name: str
    global_batch_size: int = 1024
    epochs: int = 150
    eval_interval: int = 5
    lr: float = 3e-4
    weight_decay: float = 1e-2
    lr_min_ratio: float = 0.1
    seed: int = 0
    checkpoint_path: str = "checkpoints/janus_v4"
    project_name: str = "Janus-V4"
    run_name: str = "hrm_v1"


def build_batch(batch_inputs: torch.Tensor, batch_labels: torch.Tensor) -> Dict[str, torch.Tensor]:
    # inputs: [B, Seq, F], labels: [B]
    B, S, _ = batch_inputs.shape
    labels_seq = torch.full((B, S), IGNORE_LABEL_ID, dtype=torch.long)
    labels_seq[:, -1] = batch_labels
    return {
        "inputs": batch_inputs,
        "labels": labels_seq,
    }


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, scheduler: CosineAnnealingLR, device: torch.device) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_count = 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)
        batch = build_batch(X, y)
        carry = model.initial_carry(batch)  # type: ignore
        optimizer.zero_grad(set_to_none=True)
        carry, loss, metrics, _, _ = model(carry=carry, batch=batch, return_keys=[])  # type: ignore
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.detach().item()
        total_count += 1

    return {"train/loss": total_loss / max(1, total_count)}


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    correct = 0
    total = 0

    with torch.inference_mode():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)
            batch = build_batch(X, y)
            carry = model.initial_carry(batch)  # type: ignore
            carry, loss, metrics, outputs, _ = model(carry=carry, batch=batch, return_keys=["logits"])  # type: ignore
            total_loss += loss.detach().item()
            total_count += 1
            # Last-timestep logits
            logits = outputs["logits"][:, -1, :]
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == y).sum().item()
            total += y.numel()

    return {
        "val/loss": total_loss / max(1, total_count),
        "val/accuracy": correct / max(1, total),
    }


def main():
    # Load YAML config for arch and training
    cfg_path = os.path.join(os.path.dirname(__file__), "config", "cfg_pretrain.yaml")
    with open(cfg_path, "r") as f:
        raw = yaml.safe_load(f)
    data_path = raw.get("data_path", "../Dataset/janus_m15_dataset.parquet")
    arch = raw.get("defaults", [{"arch": "hrm_v1"}])[0]["arch"]
    arch_cfg_path = os.path.join(os.path.dirname(__file__), "config", "arch", f"{arch}.yaml")
    with open(arch_cfg_path, "r") as f:
        arch_cfg = yaml.safe_load(f)

    # Coerce numeric types from YAML (in case they are loaded as strings)
    def _i(x, d):
        try:
            return int(x)
        except Exception:
            return int(d)
    def _f(x, d):
        try:
            return float(x)
        except Exception:
            return float(d)

    cfg = TrainConfig(
        data_path=str(data_path),
        arch_name=str(arch_cfg["name"]),
        loss_name=str(arch_cfg["loss"]["name"]),
        global_batch_size=_i(raw.get("global_batch_size", 1024), 1024),
        epochs=_i(raw.get("epochs", 150), 150),
        eval_interval=_i(raw.get("eval_interval", 5), 5),
        lr=_f(raw.get("lr", 3e-4), 3e-4),
        weight_decay=_f(raw.get("weight_decay", 1e-2), 1e-2),
        project_name="Janus-V4",
        run_name="HRM_ACTV1",
    )

    # Robust device selection: prefer CUDA only if it can execute a simple kernel; otherwise fallback to CPU
    device = torch.device("cpu")
    if torch.cuda.is_available():
        try:
            _t = torch.randn(1, device="cuda")
            _t = _t * 1.0  # trigger a trivial kernel
            device = torch.device("cuda")
        except Exception:
            device = torch.device("cpu")

    # Data
    train_loader, val_loader, test_loader, num_features = create_dataloaders(
        parquet_path=cfg.data_path,
        seq_len=INPUT_WINDOW_CANDLES,
        batch_size=cfg.global_batch_size,
    )

    # Model init
    model_cls = load_model_class(cfg.arch_name)
    loss_head_cls = load_model_class(cfg.loss_name)

    model_cfg = dict(
        batch_size=cfg.global_batch_size,
        seq_len=INPUT_WINDOW_CANDLES,
        num_features=num_features,
        H_cycles=2,
        L_cycles=2,
        H_layers=2,
        L_layers=6,
        hidden_size=256,
        expansion=4,
        num_heads=8,
        pos_encodings="rope",
        halt_max_steps=1,
        halt_exploration_prob=0.0,
        forward_dtype="bfloat16" if torch.cuda.is_available() else "float32",
    )

    base_model: nn.Module = model_cls(model_cfg)
    model: nn.Module = loss_head_cls(base_model, loss_type="focal_loss")
    model = model.to(device)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    T_max = max(1, (cfg.epochs // cfg.eval_interval) * len(train_loader))
    scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=cfg.lr * cfg.lr_min_ratio)

    # Logging
    wandb.init(project=cfg.project_name, name=cfg.run_name)
    wandb.config.update({"seq_len": INPUT_WINDOW_CANDLES, "num_features": num_features})

    # Early stopping
    best_val = math.inf
    patience = 10
    bad_epochs = 0

    # Training loop
    for epoch in range(1, cfg.epochs + 1):
        metrics = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        wandb.log({"epoch": epoch, **metrics})

        if epoch % cfg.eval_interval == 0:
            val_metrics = evaluate(model, val_loader, device)
            wandb.log({"epoch": epoch, **val_metrics})
            # Early stopping
            if val_metrics["val/loss"] + 1e-6 < best_val:
                best_val = val_metrics["val/loss"]
                bad_epochs = 0
                # Save best
                os.makedirs(cfg.checkpoint_path, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(cfg.checkpoint_path, "best.pt"))
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"Early stopping at epoch {epoch} (best val loss={best_val:.4f})")
                    break

    # Final evaluation on test set
    model.load_state_dict(torch.load(os.path.join(cfg.checkpoint_path, "best.pt"), map_location=device))
    test_metrics = evaluate(model, test_loader, device)
    wandb.log({"final/test_loss": test_metrics["val/loss"], "final/test_accuracy": test_metrics["val/accuracy"]})
    print(f"Done. Final test: loss={test_metrics['val/loss']:.4f}, acc={test_metrics['val/accuracy']:.4f}")


if __name__ == "__main__":
    main()
