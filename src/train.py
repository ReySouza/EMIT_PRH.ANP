from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import yaml

from dataset import EMITPatchDataset
from unet import build_unet


# ------------------------------ Utils ------------------------------

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(pref: str | None) -> torch.device:
    pref = (pref or "").lower()
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if pref == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if pref == "cpu" or pref == "" or not torch.cuda.is_available():
        return torch.device("cpu")
    # fallback
    return torch.device("cpu")


def sigmoid_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits)


# --------------------------- Loss functions ---------------------------

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.float()
        dims = (1, 2, 3) if probs.ndim == 4 else tuple(range(1, probs.ndim))
        intersection = torch.sum(probs * targets, dim=dims)
        union = torch.sum(probs, dim=dims) + torch.sum(targets, dim=dims)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class TverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1.0):
        super().__init__()
        self.alpha, self.beta, self.smooth = alpha, beta, smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.float()
        dims = (1, 2, 3) if probs.ndim == 4 else tuple(range(1, probs.ndim))
        tp = torch.sum(probs * targets, dim=dims)
        fp = torch.sum(probs * (1 - targets), dim=dims)
        fn = torch.sum((1 - probs) * targets, dim=dims)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky.mean()


class FocalLossBinary(nn.Module):
    """Binary focal loss with logits for stability (alpha optional)."""
    def __init__(self, gamma: float = 2.0, alpha: float | None = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
        p = torch.sigmoid(logits)
        pt = targets * p + (1 - targets) * (1 - p)
        loss = ((1 - pt) ** self.gamma) * bce
        if self.alpha is not None:
            loss = self.alpha * targets * loss + (1 - self.alpha) * (1 - targets) * loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight: float = 1.0, dice_weight: float = 0.5, bce_weight: float = 0.5):
        super().__init__()
        self.pos_weight = torch.tensor([pos_weight], dtype=torch.float32)
        self.dice = DiceLoss()
        self.dice_w = dice_weight
        self.bce_w = bce_weight

    def to(self, device):
        self.pos_weight = self.pos_weight.to(device)
        return super().to(device)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets.float(), pos_weight=self.pos_weight
        )
        dice = self.dice(logits, targets)
        return self.bce_w * bce + self.dice_w * dice


def build_loss(loss_cfg: Dict, pos_weight_boost: float) -> nn.Module:
    t = (loss_cfg.get("type") or "bce_dice").lower()
    if t == "bce_dice":
        return BCEDiceLoss(pos_weight=float(pos_weight_boost))
    if t == "focal":
        gamma = float(loss_cfg.get("focal_gamma", 2.0))
        # map pos_weight to alpha (approximate) if provided
        alpha = None
        if pos_weight_boost and pos_weight_boost > 1.0:
            # Convert pos_weight to alpha in [0,1]; stronger pos_weight -> higher alpha
            alpha = float(min(0.95, max(0.05, pos_weight_boost / (pos_weight_boost + 1.0))))
        return FocalLossBinary(gamma=gamma, alpha=alpha)
    if t == "tversky":
        a = float(loss_cfg.get("tversky_alpha", 0.7))
        b = float(loss_cfg.get("tversky_beta", 0.3))
        return TverskyLoss(alpha=a, beta=b)
    if t == "focal_tversky":
        a = float(loss_cfg.get("tversky_alpha", 0.7))
        b = float(loss_cfg.get("tversky_beta", 0.3))
        gamma = float(loss_cfg.get("focal_gamma", 2.0))
        class FocalTversky(nn.Module):
            def __init__(self, a, b, g):
                super().__init__(); self.t = TverskyLoss(a, b); self.g = g
            def forward(self, logits, targets):
                ft = self.t(logits, targets)
                return torch.pow(ft, self.g)
        return FocalTversky(a, b, gamma)
    # default
    return BCEDiceLoss(pos_weight=float(pos_weight_boost))


# --------------------------- Metrics ---------------------------

@torch.no_grad()
def binarize(probs: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    return (probs >= thr).float()


@torch.no_grad()
def confusion_counts(preds: torch.Tensor, targets: torch.Tensor) -> Tuple[int,int,int]:
    """Return TP, FP, FN counts across batch (tensors of 0/1)."""
    preds = preds.view(-1)
    targets = targets.view(-1)
    tp = int(torch.sum(preds * targets).item())
    fp = int(torch.sum(preds * (1 - targets)).item())
    fn = int(torch.sum((1 - preds) * targets).item())
    return tp, fp, fn


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    denom = 2*tp + fp + fn
    return 0.0 if denom == 0 else (2.0 * tp) / denom


def iou_from_counts(tp: int, fp: int, fn: int) -> float:
    denom = tp + fp + fn
    return 0.0 if denom == 0 else tp / denom


# --------------------------- Training step ---------------------------

def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total = 0
    total_loss = 0.0
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total += bs
        total_loss += loss.item() * bs
    return total_loss / max(1, total)


@torch.no_grad()
def evaluate_epoch(model, loader, device, thr: float = 0.5) -> Dict[str, float]:
    model.eval()
    total = 0
    total_loss_bce = 0.0  # for logging only (approximate)
    total_dice_soft = 0.0  # 1 - soft dice
    # confusion
    sum_tp = sum_fp = sum_fn = 0

    bce_for_log = nn.BCEWithLogitsLoss()

    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        probs = torch.sigmoid(logits)

        # logging helpers
        total += x.size(0)
        total_loss_bce += bce_for_log(logits, y).item() * x.size(0)

        # soft dice (for trend only)
        dims = (1, 2, 3) if probs.ndim == 4 else tuple(range(1, probs.ndim))
        inter = torch.sum(probs * y, dim=dims)
        union = torch.sum(probs, dim=dims) + torch.sum(y, dim=dims)
        dice_soft = 1.0 - ((2.0 * inter + 1.0) / (union + 1.0))
        total_dice_soft += dice_soft.mean().item() * x.size(0)

        # hard metrics at 0.5
        preds = (probs >= thr).float()
        tp, fp, fn = confusion_counts(preds, y)
        sum_tp += tp; sum_fp += fp; sum_fn += fn

    f1 = f1_from_counts(sum_tp, sum_fp, sum_fn)
    iou = iou_from_counts(sum_tp, sum_fp, sum_fn)
    return {
        "val_bce": total_loss_bce / max(1, total),
        "val_dice": total_dice_soft / max(1, total),  # (1 - soft dice)
        "val_f1": f1,
        "val_iou": iou,
    }


# --------------------------- Main ---------------------------

@dataclass
class Config:
    paths: Dict
    processing: Dict
    masking: Dict
    model: Dict
    training: Dict
    inference: Dict


def build_loaders(cfg: Config):
    # Select which patch dirs to use
    use_pca = bool(cfg.processing.get("apply_pca", False))
    if use_pca:
        img_dir = Path(cfg.paths["patches_pca_images"])
        mask_dir = Path(cfg.paths["patches_pca_masks"])
    else:
        img_dir = Path(cfg.paths["patches_raw_images"])
        mask_dir = Path(cfg.paths["patches_raw_masks"])

    dataset = EMITPatchDataset(
        img_dir=img_dir,
        mask_dir=mask_dir,
        include_mag1c=bool(cfg.processing.get("include_mag1c", False) or cfg.__dict__.get("include_mag1c", False)),
        mag1c_dir=Path(cfg.paths.get("patches_mag1c", cfg.paths.get("patches_raw_images","")))
                  if (cfg.processing.get("include_mag1c", False) or cfg.__dict__.get("include_mag1c", False)) else None,
        normalization=bool(cfg.processing.get("normalization", True)),
        min_valid_fraction=float(cfg.processing.get("min_valid_fraction", 0.6)),
        seed=int(cfg.training.get("seed", 42)),
    )

    # Split
    val_split = float(cfg.training.get("val_split", 0.2))
    n_total = len(dataset)
    n_val = max(1, int(round(val_split * n_total)))
    n_train = max(1, n_total - n_val)
    gen = torch.Generator().manual_seed(int(cfg.training.get("seed", 42)))
    train_ds, val_ds = torch.utils.data.random_split(dataset, lengths=[n_train, n_val], generator=gen)

    # Sampler weights computed on full dataset but we map to subset indices
    min_pos_frac = float(cfg.training.get("min_pos_frac_patch", 0.0))
    target_pos_ratio = float(cfg.training.get("target_pos_ratio", 0.5))
    boost = float(cfg.training.get("weight_pos_boost", 8.0))

    pos_frac = np.array(dataset.pos_frac)
    pos_flags = np.array(dataset.pos_flags, dtype=bool)

    base_w = np.ones(len(dataset), dtype=np.float32)
    base_w[np.where(pos_frac >= min_pos_frac)[0]] *= boost

    eps = 1e-6
    # fraction of positives meeting min_pos_frac
    mask_pos = (pos_frac >= min_pos_frac)
    p = float(mask_pos.mean()) if mask_pos.size > 0 else 0.0
    p = min(max(p, eps), 1 - eps)
    w_pos = target_pos_ratio / p
    w_neg = (1 - target_pos_ratio) / (1 - p)
    class_w = np.where(mask_pos, w_pos, w_neg).astype(np.float32)

    all_w = base_w * class_w

    train_indices = train_ds.indices if hasattr(train_ds, "indices") else train_ds.dataset.indices
    val_indices = val_ds.indices if hasattr(val_ds, "indices") else val_ds.dataset.indices

    train_weights = torch.as_tensor(all_w[train_indices], dtype=torch.float32)
    sampler = WeightedRandomSampler(train_weights, num_samples=len(train_indices), replacement=True)

    batch_size = int(cfg.training.get("batch_size", 16))
    num_workers = int(cfg.training.get("num_workers", 0))

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              pin_memory=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            pin_memory=True, num_workers=num_workers)
    return train_loader, val_loader, dataset.n_channels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=str, help="Path to config.yaml")
    ap.add_argument("--checkpoint-out", type=str, default=None, help="Optional override for model_dir")
    args = ap.parse_args()

    cfg_dict = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg = Config(**cfg_dict)

    set_seed(int(cfg.training.get("seed", 42)))
    device = get_device(cfg.training.get("device", "cpu"))
    print(f"[device] Using {device}")

    # Data
    train_loader, val_loader, in_channels = build_loaders(cfg)

    # Model
    out_channels = int(cfg.model.get("out_channels", 1))
    base_ch = int(cfg.model.get("base_channels", 64))
    dropout = float(cfg.model.get("dropout", 0.1))
    model = build_unet(in_channels=in_channels, out_channels=out_channels,
                       base_channels=base_ch, dropout=dropout)
    model.to(device)

    # Loss
    pos_boost = float(cfg.training.get("weight_pos_boost", 8.0))
    loss_fn = build_loss(cfg.training.get("loss", {}), pos_boost).to(device)

    # Optim / Scheduler
    lr = float(cfg.training.get("learning_rate", 1e-3))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler_patience = int(cfg.training.get("scheduler_patience", 3))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                                                     patience=max(1, scheduler_patience))

    # Early stopping
    early_stop_patience = int(cfg.training.get("early_stop_patience", 10))
    best_metric = -1.0
    best_epoch = -1
    epochs_without_improve = 0

    # Paths
    model_dir = Path(cfg.paths.get("model_dir", "./models/checkpoints"))
    out_dir = Path(cfg.paths.get("output_dir", "./outputs"))
    plots_dir = out_dir / "plots"
    model_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    max_epochs = int(cfg.training.get("max_epochs", 100))
    history: List[Dict] = []

    for epoch in range(1, max_epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_metrics = evaluate_epoch(model, val_loader, device=device, thr=0.5)

        # log
        entry = {"epoch": epoch, "train_loss": tr_loss, **val_metrics,
                 "lr": float(optimizer.param_groups[0]["lr"])}
        history.append(entry)
        print(f"[{epoch:03d}] train_loss={tr_loss:.4f} | "
              f"val_bce={val_metrics['val_bce']:.4f} val_dice={val_metrics['val_dice']:.4f} "
              f"val_f1={val_metrics['val_f1']:.4f} val_iou={val_metrics['val_iou']:.4f} | "
              f"lr={entry['lr']:.2e}")

        # model selection on IoU
        cur_metric = val_metrics["val_iou"]
        scheduler.step(cur_metric)

        if cur_metric > best_metric:
            best_metric = cur_metric
            best_epoch = epoch
            epochs_without_improve = 0
            # save best
            best_path = model_dir / "best.pt"
            torch.save({"epoch": epoch,
                        "state_dict": model.state_dict(),
                        "in_channels": int(in_channels),
                        "out_channels": int(cfg.model.get('out_channels', 1)),
                        "base_channels": int(cfg.model.get('base_channels', 64)),
                        "dropout": float(cfg.model.get('dropout', 0.1))},
                       best_path)
        else:
            epochs_without_improve += 1

        # periodic save
        save_every = int(cfg.training.get("save_every", 5))
        if save_every > 0 and (epoch % save_every == 0):
            path = model_dir / f"epoch_{epoch:03d}.pt"
            torch.save({"epoch": epoch, "state_dict": model.state_dict()}, path)

        # early stopping
        if early_stop_patience > 0 and epochs_without_improve >= early_stop_patience:
            print(f"[early-stop] No improvement in {early_stop_patience} epochs. "
                  f"Best IoU={best_metric:.4f} @ epoch {best_epoch}.")
            break

    # save history
    (plots_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    # simple curves
    ep = [h["epoch"] for h in history]
    v_iou = [h["val_iou"] for h in history]
    v_f1 = [h["val_f1"] for h in history]
    t_loss = [h["train_loss"] for h in history]
    plt.figure(figsize=(10,4.2))
    plt.subplot(1,2,1); plt.plot(ep, t_loss, label="train loss"); plt.xlabel("epoch"); plt.legend(); plt.title("Loss")
    plt.subplot(1,2,2); plt.plot(ep, v_iou, label="val IoU"); plt.plot(ep, v_f1, label="val F1"); plt.xlabel("epoch"); plt.legend(); plt.title("Val metrics")
    plt.tight_layout(); plt.savefig(plots_dir / "training_curves.png", dpi=140); plt.close()

    print(f"[done] Best IoU={best_metric:.4f} @ epoch {best_epoch}. Checkpoints in: {model_dir}")


if __name__ == "__main__":
    main()
