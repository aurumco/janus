import os
# Silence WANDB verbosity before importing torch/wandb
os.environ.setdefault("WANDB_SILENT", "true")
os.environ.setdefault("WANDB_CONSOLE", "off")

import math
import time
import json
from datetime import datetime
import yaml
from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.functions import load_model_class
from dataset_loader import create_dataloaders, INPUT_WINDOW_CANDLES
from models.losses import IGNORE_LABEL_ID


@dataclass
class TrainConfig:
    data_path: str
    arch_name: str
    loss_name: str
    global_batch_size: int = 512
    epochs: int = 150
    eval_interval: int = 5
    lr: float = 3e-4
    weight_decay: float = 1e-2
    lr_min_ratio: float = 0.1
    seed: int = 0
    checkpoint_path: str = "checkpoints/janus_v4"
    project_name: str = "Janus-V4"
    run_name: str = "HRM_ACTV1"


def build_batch(batch_inputs: torch.Tensor, batch_labels: torch.Tensor) -> Dict[str, torch.Tensor]:
    # inputs: [B, Seq, F], labels: [B]
    B, S, _ = batch_inputs.shape
    labels_seq = torch.full((B, S), IGNORE_LABEL_ID, dtype=torch.long, device=batch_inputs.device)
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
        carry, loss, _, _, _ = model(carry=carry, batch=batch, return_keys=[])  # type: ignore
        loss.backward()
        # Aggressive cleanup to avoid RAM spikes on Kaggle
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        else:
            import gc; gc.collect()
        # XLA-aware optimizer step if running on TPU
        if device.type == "xla":
            try:
                import torch_xla.core.xla_model as xm
                xm.optimizer_step(optimizer, barrier=True)
                xm.mark_step()
            except Exception:
                optimizer.step()
        else:
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += float(loss.detach().item())
        total_count += 1
        # Free per-batch tensors
        del X, y, batch, carry, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    import gc; gc.collect()

    return {"train/loss": total_loss / max(1, total_count)}


def _compute_metrics_from_conf(conf: torch.Tensor) -> Dict[str, float]:
    num_classes = conf.shape[0]
    per_class = {}
    f1s = []
    total = conf.sum().item()
    correct = int(conf.diag().sum().item())
    for c in range(num_classes):
        tp = int(conf[c, c].item())
        fp = int(conf[:, c].sum().item() - tp)
        fn = int(conf[c, :].sum().item() - tp)
        prec = tp / max(1, tp + fp)
        rec  = tp / max(1, tp + fn)
        f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
        per_class[f"prec_{c}"] = prec
        per_class[f"rec_{c}"] = rec
        per_class[f"f1_{c}"] = f1
        f1s.append(f1)
    macro_f1 = sum(f1s) / num_classes
    acc = correct / max(1, total)
    out = {"val/accuracy": acc, "val/macro_f1": macro_f1}
    for k in ("prec_0","rec_0","f1_0","prec_4","rec_4","f1_4"):
        out[f"val/{k}"] = per_class[k]
    return out


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    correct = 0
    total = 0

    conf = torch.zeros((5, 5), dtype=torch.int64)
    with torch.inference_mode():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)
            batch = build_batch(X, y)
            carry = model.initial_carry(batch)  # type: ignore
            carry, loss, metrics, outputs, _ = model(carry=carry, batch=batch, return_keys=["logits"])  # type: ignore
            total_loss += loss.detach().item()
            total_count += 1
            logits = outputs["logits"]  # [B, C]
            preds = torch.argmax(logits, dim=-1)
            # Update confusion matrix on CPU to keep memory low
            yt = y.detach().cpu().to(torch.long)
            pt = preds.detach().cpu().to(torch.long)
            for t, p in zip(yt.tolist(), pt.tolist()):
                if 0 <= t < 5 and 0 <= p < 5:
                    conf[t, p] += 1
            correct += int((preds == y).sum().item())
            total += int(y.numel())
            # Free per-batch tensors
            del X, y, batch, carry, outputs, logits, preds
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if device.type == "xla":
                try:
                    import torch_xla.core.xla_model as xm
                    xm.mark_step()
                except Exception:
                    pass
    import gc; gc.collect()
    base = {"val/loss": total_loss / max(1, total_count)}
    if conf.sum().item() > 0:
        base.update(_compute_metrics_from_conf(conf))
    return base


def main():
    # Load YAML config for arch and training
    cfg_path = os.path.join(os.path.dirname(__file__), "config", "cfg_pretrain.yaml")
    with open(cfg_path, "r") as f:
        raw = yaml.safe_load(f)
    # Allow overriding data path via environment (e.g., Kaggle)
    data_path = os.environ.get("DATA_PATH", raw.get("data_path", "../Dataset/janus_m15_dataset.parquet"))
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

    # If OUTPUT_DIR is set, redirect checkpoints there
    output_dir = os.environ.get("OUTPUT_DIR")
    if output_dir:
        try:
            os.makedirs(os.path.join(output_dir, "checkpoints", "janus_v4"), exist_ok=True)
            cfg.checkpoint_path = os.path.join(output_dir, "checkpoints", "janus_v4")
        except Exception:
            pass

    # Device selection: prefer TPU (XLA), else GPU. Abort if neither is available.
    device = None
    # Try TPU (XLA)
    try:
        import torch_xla.core.xla_model as xm  # type: ignore
        device = xm.xla_device()
        # Tag device.type for downstream checks
        device.type = "xla"  # type: ignore
    except Exception:
        device = None
    # Else try CUDA
    if device is None:
        # Extra diagnostics
        try:
            cuda_available = torch.cuda.is_available()
            cuda_count = torch.cuda.device_count() if cuda_available else 0
            print(f"[info] torch.cuda.is_available={cuda_available}, device_count={cuda_count}, torch.version.cuda={getattr(torch.version, 'cuda', None)}")
        except Exception:
            pass
        if torch.cuda.device_count() > 0 or torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            raise RuntimeError("No TPU or GPU detected. Please enable TPU or GPU in your environment.")

    # Prepare results directory (timestamped)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_base = os.environ.get("OUTPUT_DIR", os.path.dirname(__file__))
    results_dir = os.path.join(results_base, "results", run_id)
    os.makedirs(results_dir, exist_ok=True)
    # Data
    # Kaggle-safe DataLoader settings: no workers, no pin_memory to avoid extra copies/forks
    pin_mem = False
    train_loader, val_loader, test_loader, num_features = create_dataloaders(
        parquet_path=cfg.data_path,
        seq_len=INPUT_WINDOW_CANDLES,
        batch_size=min(cfg.global_batch_size, 512),
        num_workers=0,
        pin_memory=pin_mem,
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
    model: nn.Module = loss_head_cls(
        base_model,
        loss_type="focal_loss",
        alpha=0.25,
        gamma=2.0,
        class_weights=[1.5, 1.2, 0.5, 1.2, 1.5],
        q_loss_weight=0.1,
    )
    model = model.to(device)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    T_max = max(1, (cfg.epochs // cfg.eval_interval) * len(train_loader))
    scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=cfg.lr * cfg.lr_min_ratio)

    # Logging
    wandb.init(project=cfg.project_name, name=cfg.run_name, settings=wandb.Settings(_disable_stats=True))
    wandb.config.update({"seq_len": INPUT_WINDOW_CANDLES, "num_features": num_features})
    wandb.watch(model, log="all", log_freq=100)

    # Early stopping
    best_val = math.inf
    patience = 10
    bad_epochs = 0

    # History tracking for plots and reports
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
        "time_sec": [],
        "cumulative_sec": [],
    }
    t0 = time.time()

    # Training loop
    for epoch in range(1, cfg.epochs + 1):
        ep_start = time.time()
        metrics = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        wandb.log({"epoch": epoch, **metrics})
        # Always save latest for resilience
        os.makedirs(cfg.checkpoint_path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(cfg.checkpoint_path, "latest.pt"))

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
                torch.save(model, os.path.join(cfg.checkpoint_path, "best_full.pth"))
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"Early stopping at epoch {epoch} (best val loss={best_val:.4f})")
                    break

        # Update history and write per-epoch report
        ep_time = time.time() - ep_start
        cum_time = time.time() - t0
        history["epoch"].append(epoch)
        history["train_loss"].append(float(metrics.get("train/loss", 0.0)))
        # Use last computed val metrics or empty defaults
        vloss = float(val_metrics.get("val/loss", float('nan'))) if 'val_metrics' in locals() else float('nan')
        vacc = float(val_metrics.get("val/accuracy", float('nan'))) if 'val_metrics' in locals() else float('nan')
        vf1  = float(val_metrics.get("val/macro_f1", float('nan'))) if 'val_metrics' in locals() else float('nan')
        history["val_loss"].append(vloss)
        history["val_accuracy"].append(vacc)
        history["val_macro_f1"].append(vf1)
        history["time_sec"].append(ep_time)
        history["cumulative_sec"].append(cum_time)

        # Per-epoch text report (minimal, complete info)
        report = {
            "epoch": epoch,
            "train_loss": history["train_loss"][-1],
            "val": {k: val_metrics.get(k) if 'val_metrics' in locals() else None for k in [
                "val/loss", "val/accuracy", "val/macro_f1", "val/prec_0", "val/rec_0", "val/f1_0", "val/prec_4", "val/rec_4", "val/f1_4"
            ]},
            "timing": {"epoch_sec": ep_time, "cumulative_sec": cum_time},
            "optimizer": {"lr": optimizer.param_groups[0]["lr"], "weight_decay": optimizer.param_groups[0].get("weight_decay", None)},
            "config": {
                "batch_size": cfg.global_batch_size,
                "epochs": cfg.epochs,
                "eval_interval": cfg.eval_interval,
                "lr": cfg.lr,
                "weight_decay": cfg.weight_decay,
                "seq_len": INPUT_WINDOW_CANDLES,
                "num_features": num_features,
            },
        }
        # Results dir structure
        run_id = os.path.basename(results_dir)
        with open(os.path.join(results_dir, f"epoch_{epoch:04d}.txt"), "w") as f:
            f.write(json.dumps(report, indent=2))

        # Update plots with dark theme
        try:
            plt.style.use('dark_background')
            # Loss curves
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(history["epoch"], history["train_loss"], label="train_loss", color="#4FC3F7")
            ax.plot(history["epoch"], history["val_loss"], label="val_loss", color="#FFB74D")
            ax.set_xlabel("epoch")
            ax.set_ylabel("loss")
            ax.legend()
            ax.grid(True, alpha=0.2)
            fig.tight_layout()
            fig.savefig(os.path.join(results_dir, "loss_curves.png"), dpi=140)
            plt.close(fig)

            # Accuracy & F1
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(history["epoch"], history["val_accuracy"], label="val_acc", color="#81C784")
            ax.plot(history["epoch"], history["val_macro_f1"], label="val_macro_f1", color="#E57373")
            ax.set_ylim(0, 1)
            ax.set_xlabel("epoch")
            ax.set_ylabel("score")
            ax.legend()
            ax.grid(True, alpha=0.2)
            fig.tight_layout()
            fig.savefig(os.path.join(results_dir, "metrics_curves.png"), dpi=140)
            plt.close(fig)
        except Exception:
            pass

    # Final evaluation on test set
    model.load_state_dict(torch.load(os.path.join(cfg.checkpoint_path, "best.pt"), map_location=device))
    test_metrics = evaluate(model, test_loader, device)
    wandb.log({"final/test_loss": test_metrics["val/loss"], "final/test_accuracy": test_metrics["val/accuracy"]})
    print(f"Done. Final test: loss={test_metrics['val/loss']:.4f}, acc={test_metrics['val/accuracy']:.4f}")

    # Export formats for broad compatibility
    class InferenceWrapper(nn.Module):
        def __init__(self, mdl: nn.Module):
            super().__init__()
            self.mdl = mdl
        def forward(self, x: torch.Tensor):  # x: [B, S, F]
            B = x.shape[0]
            dummy_labels = torch.zeros((B,), dtype=torch.long, device=x.device)
            batch = build_batch(x, dummy_labels)
            carry = self.mdl.initial_carry(batch)  # type: ignore
            _, _, _, outputs, _ = self.mdl(carry=carry, batch=batch, return_keys=["logits"])  # type: ignore
            return outputs["logits"]  # [B, C]

    iw = InferenceWrapper(model).to(device).eval()
    dummy = torch.zeros((1, INPUT_WINDOW_CANDLES, num_features), dtype=torch.float32, device=device)
    # TorchScript
    try:
        scripted = torch.jit.trace(iw, (dummy,))
        scripted.save(os.path.join(cfg.checkpoint_path, "best_scripted.pt"))
    except Exception:
        pass
    # ONNX
    try:
        onnx_path = os.path.join(cfg.checkpoint_path, "best.onnx")
        torch.onnx.export(
            iw, (dummy,), onnx_path,
            input_names=["inputs"], output_names=["logits"],
            opset_version=17,
            dynamic_axes={"inputs": {0: "batch"}, "logits": {0: "batch"}},
        )
    except Exception:
        pass

    # Dump final history and summary into results directory
    with open(os.path.join(results_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(results_dir, "final_summary.txt"), "w") as f:
        f.write(json.dumps({"test": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
