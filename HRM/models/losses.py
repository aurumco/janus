from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn


IGNORE_LABEL_ID = -100


def s(x, epsilon=1e-30):
    return torch.where(
        x<0,
        1/(1-x+ epsilon),
        x + 1
    )


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x/torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    valid_mask = labels != ignore_index
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100, weight: torch.Tensor | None = None):
    # logits: [..., C], labels: [...]
    return F.cross_entropy(
        logits.to(torch.float32).view(-1, logits.shape[-1]),
        labels.to(torch.long).view(-1),
        ignore_index=ignore_index,
        reduction="none",
        weight=weight,
    ).view(labels.shape)


def focal_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100, alpha: float = 0.25, gamma: float = 2.0, class_weights: torch.Tensor | None = None):
    """Multi-class Focal Loss (per-element, no reduction).
    - logits: [..., C]
    - labels: [...]
    - class_weights: optional tensor of shape [C]
    """
    ce = softmax_cross_entropy(logits, labels, ignore_index=ignore_index, weight=class_weights)
    pt = torch.exp(-ce)
    return alpha * (1.0 - pt) ** gamma * ce


class ACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str, alpha: float = 0.25, gamma: float = 2.0, class_weights: Sequence[float] | None = None, q_loss_weight: float = 0.1):
        super().__init__()
        self.model = model
        self.loss_type = loss_type
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = None
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.q_loss_weight = q_loss_weight
        # When training with DataParallel, returning only a scalar tensor loss avoids gather issues
        self.return_loss_only: bool = False
        
    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        # Ensure carry is created per-replica when using DataParallel
        if "batch" in model_kwargs and "carry" not in model_kwargs:
            try:
                model_kwargs["carry"] = self.model.initial_carry(model_kwargs["batch"])  # type: ignore
            except Exception:
                # Fallback: rely on model to handle missing carry if supported
                pass
        # Model logits
        # B x SeqLen x D
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]
        # Ensure labels shape is [B] (last-timestep label per sequence)
        if labels.ndim == 2:
            labels = labels[:, -1]
        elif labels.ndim > 2:
            labels = labels.reshape(labels.shape[0], -1)[:, -1]
        else:
            labels = labels.view(-1)

        # Correctness
        with torch.no_grad():
            logits_last = outputs["logits"]  # [B, C] or [C]
            if logits_last.ndim == 1:
                logits_last = logits_last.unsqueeze(0)
            # Align devices
            if labels.device != logits_last.device:
                labels = labels.to(logits_last.device)
            preds = torch.argmax(logits_last, dim=-1)
            if preds.ndim == 0:
                preds = preds.unsqueeze(0)
            preds = preds.view(-1)
            labels = labels.view(-1)
            # Align lengths defensively
            n = min(preds.numel(), labels.numel())
            preds = preds[:n]
            labels = labels[:n]
            valid = (labels != IGNORE_LABEL_ID).view(-1)
            if valid.numel() != n:
                valid = torch.ones((n,), dtype=torch.bool, device=labels.device)
            acc = (preds[valid] == labels[valid]).float().mean() if valid.any() else torch.tensor(0.0, device=labels.device)
            metrics = {
                "count": valid.sum(),
                "accuracy": acc * valid.sum(),  # scaled for later reduction
                "exact_accuracy": (preds[valid] == labels[valid]).sum(),
                "steps": new_carry.steps.sum(),
            }

        # Losses (last-timestep classification)
        logits_last = outputs["logits"]  # [B, C] or [C]
        if logits_last.ndim == 1:
            logits_last = logits_last.unsqueeze(0)
        if labels.device != logits_last.device:
            labels = labels.to(logits_last.device)
        weight = None
        if self.class_weights is not None:
            weight = self.class_weights.to(logits_last.device)
        if self.loss_type == "focal_loss":
            per_example = focal_loss(logits_last, labels, ignore_index=IGNORE_LABEL_ID, alpha=self.alpha, gamma=self.gamma, class_weights=weight)
        else:
            per_example = softmax_cross_entropy(logits_last, labels, ignore_index=IGNORE_LABEL_ID, weight=weight)
        lm_loss = per_example.sum()
        # Halting auxiliary loss (down-weighted)
        # Use correct/incorrect as target
        target_is_correct = (torch.argmax(logits_last.detach(), dim=-1) == labels).to(outputs["q_halt_logits"].dtype)
        q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], target_is_correct, reduction="sum")

        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": (self.q_loss_weight * q_halt_loss).detach(),
        })

        # Q continue (bootstrapping target loss)
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")

            metrics["q_continue_loss"] = q_continue_loss.detach()

        total_loss = lm_loss + self.q_loss_weight * (q_halt_loss + q_continue_loss)

        if self.return_loss_only:
            # DP-friendly: return a 1D tensor so DataParallel gathers without scalar warning
            return total_loss.view(1)

        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        # Return a 1D loss tensor and a 1D halted flag to keep DataParallel gather consistent
        halted_flag = new_carry.halted.all().view(1)
        return new_carry, total_loss.view(1), metrics, detached_outputs, halted_flag
