from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml

from dataset import EMITPatchDataset
from unet import build_unet


@torch.no_grad()
def confusion_counts(preds: torch.Tensor, targets: torch.Tensor) -> Tuple[int,int,int]:
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


def get_device(pref: str | None) -> torch.device:
    pref = (pref or "").lower()
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if pref == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=str, help="Path to config.yaml")
    ap.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (default: best.pt in model_dir)")
    ap.add_argument("--save", type=str, default=None, help="Override output JSON path")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    device = get_device(cfg.get("training",{}).get("device","cpu"))

    # Dataset choice consistent with training split
    use_pca = bool(cfg.get("processing",{}).get("apply_pca", False))
    if use_pca:
        img_dir = Path(cfg["paths"]["patches_pca_images"])
        mask_dir = Path(cfg["paths"]["patches_pca_masks"])
    else:
        img_dir = Path(cfg["paths"]["patches_raw_images"])
        mask_dir = Path(cfg["paths"]["patches_raw_masks"])

    dataset = EMITPatchDataset(
        img_dir=img_dir,
        mask_dir=mask_dir,
        include_mag1c=bool(cfg.get("processing",{}).get("include_mag1c", False) or cfg.get("include_mag1c", False)),
        mag1c_dir=Path(cfg["paths"].get("patches_mag1c", cfg["paths"].get("patches_raw_images","")))
                  if (cfg.get("processing",{}).get("include_mag1c", False) or cfg.get("include_mag1c", False)) else None,
        normalization=bool(cfg.get("processing",{}).get("normalization", True)),
        min_valid_fraction=float(cfg.get("processing",{}).get("min_valid_fraction", 0.6)),
        seed=int(cfg.get("training",{}).get("seed", 42)),
    )

    # Recreate val split
    val_split = float(cfg.get("training",{}).get("val_split", 0.2))
    n_total = len(dataset)
    n_val = max(1, int(round(val_split * n_total)))
    n_train = max(1, n_total - n_val)
    gen = torch.Generator().manual_seed(int(cfg.get("training",{}).get("seed", 42)))
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val], generator=gen)

    # Build model
    in_channels = dataset.n_channels
    out_channels = int(cfg.get("model",{}).get("out_channels", 1))
    base_ch = int(cfg.get("model",{}).get("base_channels", 64))
    dropout = float(cfg.get("model",{}).get("dropout", 0.1))
    model = build_unet(in_channels=in_channels, out_channels=out_channels,
                       base_channels=base_ch, dropout=dropout).to(device)

    model_dir = Path(cfg["paths"].get("model_dir", "./models/checkpoints"))
    ckpt_path = Path(args.checkpoint) if args.checkpoint else (model_dir / "best.pt")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["state_dict"] if "state_dict" in state else state)

    loader = torch.utils.data.DataLoader(val_ds, batch_size=int(cfg.get("training",{}).get("batch_size",16)),
                                         shuffle=False, pin_memory=True, num_workers=int(cfg.get("training",{}).get("num_workers",0)))

    # Collect all predictions and masks
    model.eval()
    all_probs: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu()
            all_probs.append(probs)
            all_targets.append(y.cpu())

    probs = torch.cat(all_probs, dim=0)
    targets = torch.cat(all_targets, dim=0)

    # Threshold sweep
    thrs = np.linspace(0.05, 0.95, 19)
    best = {"threshold": 0.5, "f1": -1.0, "iou": 0.0}
    for thr in thrs:
        preds = (probs >= thr).float()
        tp, fp, fn = confusion_counts(preds, targets)
        f1 = f1_from_counts(tp, fp, fn)
        iou = iou_from_counts(tp, fp, fn)
        if f1 > best["f1"]:
            best = {"threshold": float(thr), "f1": float(f1), "iou": float(iou)}

    result = {
        "best_threshold_f1": best["threshold"],
        "best_f1": best["f1"],
        "iou_at_best_f1": best["iou"],
        "val_size": int(targets.shape[0]),
    }

    out_dir = Path(cfg["paths"].get("output_dir", "./outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = Path(args.save) if args.save else (out_dir / "eval_val_metrics.json")
    save_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"[saved] {save_path}")


if __name__ == "__main__":
    main()
