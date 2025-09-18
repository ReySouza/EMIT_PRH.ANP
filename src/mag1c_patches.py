
# src/make_mag1c_patches.py
from __future__ import annotations

import argparse, json
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import yaml

# Reuse helper functions from mask_generation.py (already in the repo)
from mask_generation import find_latest_pair, read_mag1c_any, read_manifest_rows

def tprint(msg: str): print(f"[mag1c_patches] {msg}")

def load_cfg(cfg_path: Path) -> Dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def crop_pad(arr: np.ndarray, y: int, x: int, patch: int) -> np.ndarray:
    """Crop [y:y+patch, x:x+patch] from arr; zero-pad if window exceeds bounds."""
    H, W = arr.shape[:2]
    y2, x2 = y + patch, x + patch
    y1c, y2c = max(0, y), min(H, y2)
    x1c, x2c = max(0, x), min(W, x2)
    out = np.zeros((patch, patch), dtype=arr.dtype)
    hh, ww = y2c - y1c, x2c - x1c
    if hh > 0 and ww > 0:
        out[:hh, :ww] = arr[y1c:y2c, x1c:x2c]
    return out

def save_mag1c_patches(cfg: Dict, scene_stem: str, manifest_csv: Path) -> Tuple[int, Path]:
    paths = cfg["paths"]; proc = cfg["processing"]
    patch = int(proc.get("patch_size", 128))

    mag_dir = Path(paths["mag1c_dir"])
    out_dir = Path(paths["patches_mag1c"])
    out_dir.mkdir(parents=True, exist_ok=True)

    arr, prof, src = read_mag1c_any(mag_dir, scene_stem)
    tprint(f"MAG1C fonte: {src}  shape={arr.shape}")

    rows = read_manifest_rows(manifest_csv)
    if not rows:
        raise RuntimeError(f"Manifesto vazio: {manifest_csv}")

    n_saved = 0
    for i, r in enumerate(rows):
        y, x = int(r["y"]), int(r["x"])
        m = crop_pad(arr, y, x, patch)  # (patch, patch)
        # Save with the same filename pattern as image patches
        np.save(out_dir / f"{scene_stem}_{i:06d}.npy", m.astype(np.float32))
        n_saved += 1

    tprint(f"Salvos {n_saved} patches MAG1C em: {out_dir}")
    return n_saved, out_dir

def main(config_path: str, scene: Optional[str] = None):
    cfg = load_cfg(Path(config_path))
    outputs_dir = Path(cfg["paths"]["output_dir"])

    # Find matching stats/manifest produced by preprocess
    stats_path, manifest_path = find_latest_pair(outputs_dir, scene_hint=scene)
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    scene_stem = stats["scene"]
    tprint(f"Scene: {scene_stem}")

    n, out_dir = save_mag1c_patches(cfg, scene_stem, Path(manifest_path))
    tprint("Concluído.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Gera patches MAG1C (1 canal) alinhados ao manifest do preprocess.")
    ap.add_argument("--config", type=str, required=True, help="configs/config.yaml")
    ap.add_argument("--scene", type=str, default=None, help="(opcional) prefixo do arquivo de cena (stem)")
    args = ap.parse_args()
    main(config_path=args.config, scene=args.scene)
