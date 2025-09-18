# tools/inspect_patches.py (enhanced + robust manifest/paths)
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import yaml
import random


# ------------------------- config & io helpers -------------------------------

def load_config(cfg_path: Path) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _find_scene_stats(outputs_dir: Path, scene: str) -> Optional[Path]:
    pats = [
        f"{scene}*_raw_stats.json",
        f"{scene}*_pca_stats.json",
        f"{scene}*_stats.json",
    ]
    for pat in pats:
        found = sorted(outputs_dir.glob(pat))
        if found:
            return found[0]
    return None

def _read_kept_wavelengths(outputs_dir: Path, scene: str) -> List[float]:
    stats_file = _find_scene_stats(outputs_dir, scene)
    if not stats_file:
        return []
    try:
        with open(stats_file, "r", encoding="utf-8") as f:
            jd = json.load(f)
        return [float(x) for x in jd.get("kept_bands_nm", [])]
    except Exception:
        return []

def _find_manifest(outputs_dir: Path, scene: str) -> Optional[Path]:
    # cobre *_raw_manifest.csv, *_pca_manifest.csv e *_manifest.csv
    patterns = [
        f"{scene}*_raw_manifest.csv",
        f"{scene}*_pca_manifest.csv",
        f"{scene}*_manifest.csv",
    ]
    for pat in patterns:
        found = sorted(outputs_dir.glob(pat))
        if found:
            return found[0]
    # fallback: procurar por nome que começa com scene
    found = sorted(outputs_dir.glob(f"{scene}*.csv"))
    for f in found:
        if "manifest" in f.stem.lower():
            return f
    return None

def _load_manifest_rows(manifest_csv: Path) -> List[dict]:
    rows: List[dict] = []
    with manifest_csv.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def _infer_run_tag_from_manifest(manifest_csv: Path, cfg: dict) -> str:
    name = manifest_csv.stem.lower()
    if "pca" in name:
        return "pca"
    if "raw" in name:
        return "raw"
    # fallback: usar config.processing.apply_pca
    try:
        apply_pca = bool(cfg.get("processing", {}).get("apply_pca", False))
    except Exception:
        apply_pca = False
    return "pca" if apply_pca else "raw"

def _candidate_base_dirs(cfg: dict, run_tag: str) -> List[Path]:
    cand: List[Path] = []
    paths_cfg = cfg.get("paths", {})
    # diretórios do config
    if run_tag == "pca" and "patches_pca_images" in paths_cfg:
        cand.append(Path(paths_cfg["patches_pca_images"]))
    if run_tag == "raw" and "patches_raw_images" in paths_cfg:
        cand.append(Path(paths_cfg["patches_raw_images"]))
    # heurísticas extras (caso o config não tenha chaves completas)
    for guess in ["data/patches/images", "data/patches/pca_images", "patches/images"]:
        cand.append(Path(guess))
    return [p for p in cand if p is not None]

def _resolve_patch_files(rows: Sequence[dict],
                         base_dirs: Sequence[Path],
                         scene: str) -> List[Path]:
    """
    Resolve paths de patches .npy a partir do manifesto.
    Estratégia:
      1) Se houver coluna com caminho, usa (relativo ou absoluto).
      2) Se houver 'patch_idx', usa <scene>_<idx:06d>.npy nos base_dirs.
      3) Caso contrário, assume que a ordem das linhas = índice do arquivo.
    """
    out: List[Path] = []

    # 1) detectar coluna de caminho, se existir
    path_keys = ("patch_path", "path", "img_path", "npy", "file")
    has_path_col = rows and any(k in rows[0] for k in path_keys)

    if has_path_col:
        for r in rows:
            p = None
            for key in path_keys:
                if key in r and r[key]:
                    p = Path(r[key])
                    break
            if p is None:
                continue
            if not p.is_absolute():
                # tente resolver em cada base_dir
                chosen = None
                for bd in base_dirs:
                    cand = (bd / p).resolve()
                    if cand.exists():
                        chosen = cand; break
                if chosen is None:
                    # tente como relativo ao CWD
                    cand = p.resolve()
                    if cand.exists():
                        chosen = cand
                p = chosen if chosen is not None else (base_dirs[0] / p).resolve()
            out.append(p)
        uniq = sorted({p for p in out if p and p.exists()})
        return uniq

    # 2) se houver 'patch_idx'
    if rows and "patch_idx" in rows[0]:
        for r in rows:
            try:
                idx = int(r["patch_idx"])
            except Exception:
                continue
            name = f"{scene}_{idx:06d}.npy"
            chosen = None
            for bd in base_dirs:
                cand = (bd / name).resolve()
                if cand.exists():
                    chosen = cand; break
            if chosen:
                out.append(chosen)
        return sorted({p for p in out})

    # 3) fallback: ordem das linhas = índice
    for i, _ in enumerate(rows):
        name = f"{scene}_{i:06d}.npy"
        chosen = None
        for bd in base_dirs:
            cand = (bd / name).resolve()
            if cand.exists():
                chosen = cand; break
        if chosen:
            out.append(chosen)

    return sorted({p for p in out})


# ------------------------- viz helpers ---------------------------------------

def _auto_rgb_bands(n_bands: int) -> Tuple[int, int, int]:
    if n_bands < 3:
        return (0, 0, max(0, n_bands - 1))
    return (0, n_bands // 2, n_bands - 1)

def _compose_rgb(x: np.ndarray, bands: Tuple[int, int, int]) -> np.ndarray:
    r, g, b = [np.clip(x[..., i], 0, 1) for i in bands]
    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.power(rgb, 1/1.8)
    return np.clip(rgb, 0, 1)


# ------------------------- statistics ----------------------------------------

def _valid_mask_from_values(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return np.isfinite(x) & (x > eps)

def _stats_from_array(x: np.ndarray, valid: Optional[np.ndarray] = None) -> Dict[str, float]:
    if valid is None:
        valid = _valid_mask_from_values(x)
    vals = x[valid]
    if vals.size == 0:
        keys = ["n","min","p01","p05","p10","p25","p50","p75","p90","p95","p99","max","mean","std","frac_le_0","frac_ge_1"]
        return {k: float("nan") for k in keys}
    q = np.quantile(vals, [0.01,0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.99])
    return {
        "n": float(vals.size),
        "min": float(np.min(vals)),
        "p01": float(q[0]), "p05": float(q[1]), "p10": float(q[2]), "p25": float(q[3]),
        "p50": float(q[4]), "p75": float(q[5]), "p90": float(q[6]), "p95": float(q[7]), "p99": float(q[8]),
        "max": float(np.max(vals)),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "frac_le_0": float(np.mean(x <= 0)),
        "frac_ge_1": float(np.mean(x >= 1)),
    }

def _write_json(p: Path, obj: dict) -> None:
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def _write_csv(p: Path, rows: Sequence[dict]) -> None:
    ensure_dir(p.parent)
    if not rows: return
    fieldnames = list(rows[0].keys())
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(rows)


# ------------------------- plotting (idem versão anterior) -------------------

def per_patch_histograms(patch_files: Sequence[Path], sel_idx: Sequence[int], out_png: Path, bins: int = 50) -> None:
    datas = []; titles = []
    for i in sel_idx:
        if i < 0 or i >= len(patch_files): continue
        x = np.load(patch_files[i])
        valid = _valid_mask_from_values(x); vals = x[valid]
        datas.append(vals); titles.append(f"idx {i} (min={vals.min():.2f}, max={vals.max():.2f})")
    if not datas: return
    n = len(datas); cols = min(6, n); rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.2*rows)); axes = np.array(axes).reshape(rows, cols)
    for d, t, ax in zip(datas, titles, axes.ravel()):
        ax.hist(d, bins=bins, range=(0,1)); ax.set_title(t, fontsize=9); ax.set_xlim(0,1)
    for ax in axes.ravel()[len(datas):]: ax.axis("off")
    fig.tight_layout(); ensure_dir(out_png.parent); fig.savefig(out_png, dpi=180); plt.close(fig)

def _auto_rgb_bands(n_bands: int) -> Tuple[int, int, int]:
    if n_bands < 3: return (0, 0, max(0, n_bands - 1))
    return (0, n_bands // 2, n_bands - 1)

def _compose_rgb(x: np.ndarray, bands: Tuple[int, int, int]) -> np.ndarray:
    r, g, b = [np.clip(x[..., i], 0, 1) for i in bands]
    rgb = np.stack([r, g, b], axis=-1); rgb = np.power(rgb, 1/1.8)
    return np.clip(rgb, 0, 1)

def quicklook_grid(patch_files: Sequence[Path], out_png: Path, stride: int, rgb_bands: Tuple[int, int, int]) -> None:
    n = len(patch_files); 
    if n == 0: return
    cols = int(np.ceil(np.sqrt(n))); rows = int(np.ceil(n / cols))
    sel = list(range(min(n, rows*cols)))
    fig_w = max(10, min(20, cols*2)); fig_h = max(10, min(20, rows*2))
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h)); axes = np.array(axes).reshape(rows, cols)
    for ax in axes.ravel(): ax.axis("off")
    for k, ax in zip(sel, axes.ravel()):
        x = np.load(patch_files[k]); rgb = _compose_rgb(x, rgb_bands); ax.imshow(rgb)
    fig.suptitle(f"Quicklook | grid {cols}x{rows} (stride={stride})"); fig.tight_layout()
    ensure_dir(out_png.parent); fig.savefig(out_png, dpi=180); plt.close(fig)

def per_patch_gallery(patch_files: Sequence[Path], sel_idx: Sequence[int], out_png: Path, rgb_bands: Tuple[int, int, int]) -> None:
    imgs = []; titles = []
    for i in sel_idx:
        if i < 0 or i >= len(patch_files): continue
        x = np.load(patch_files[i]); vfrac = float(np.mean(_valid_mask_from_values(x)))
        imgs.append(_compose_rgb(x, rgb_bands)); titles.append(f"idx {i} | v={vfrac:.2f}")
    if not imgs: return
    n = len(imgs); cols = min(6, n); rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2*cols, 3.2*rows)); axes = np.array(axes).reshape(rows, cols)
    for ax in axes.ravel(): ax.axis("off")
    for img, title, ax in zip(imgs, titles, axes.ravel()):
        ax.imshow(img); ax.set_title(title, fontsize=9)
    fig.tight_layout(); ensure_dir(out_png.parent); fig.savefig(out_png, dpi=180); plt.close(fig)

def global_histogram_sample(patch_files: Sequence[Path], out_png: Path, max_patches: int = 100, bins: int = 100) -> None:
    n = len(patch_files); idx = list(range(n)); random.shuffle(idx); idx = idx[:min(max_patches, n)]
    vals = []
    for i in idx:
        x = np.load(patch_files[i]); v = _valid_mask_from_values(x); vals.append(x[v])
    if not vals: return
    vals = np.concatenate(vals, axis=0)
    fig, ax = plt.subplots(figsize=(10,5)); ax.hist(vals, bins=bins, range=(0,1))
    ax.set_title(f"Histograma global (~{len(idx)} patches)"); ax.set_xlim(0,1)
    ensure_dir(out_png.parent); fig.savefig(out_png, dpi=180); plt.close(fig)


# ------------------------- stats saving --------------------------------------

def save_patch_stats(patch_files: Sequence[Path], out_csv: Path) -> None:
    rows = []
    for i, p in enumerate(patch_files):
        x = np.load(p); valid = _valid_mask_from_values(x); s = _stats_from_array(x, valid)
        s.update({"patch_idx": i, "file": str(p), "valid_frac": float(np.mean(valid))})
        rows.append(s)
    _write_csv(out_csv, rows)

def save_global_stats(patch_files: Sequence[Path], out_json: Path, wav: Sequence[float], sample: Optional[int]) -> None:
    n = len(patch_files); idx = list(range(n))
    if sample is not None:
        random.shuffle(idx); idx = idx[:min(sample, n)]
    vals = []; B = None
    sum_b = None; sumsq_b = None; n_b = None
    for i in idx:
        x = np.load(patch_files[i])  # (H,W,B)
        if B is None: B = x.shape[-1]
        v = _valid_mask_from_values(x); vals.append(x[v])
        xi = x.reshape(-1, B); vi = v.reshape(-1, B)
        if sum_b is None:
            sum_b = np.zeros(B, dtype=np.float64); sumsq_b = np.zeros(B, dtype=np.float64); n_b = np.zeros(B, dtype=np.int64)
        for b in range(B):
            vb = vi[:, b]; xb = xi[vb, b]
            n_b[b] += xb.size; sum_b[b] += float(np.sum(xb)); sumsq_b[b] += float(np.sum(xb**2))
    if not vals: return
    vals = np.concatenate(vals, axis=0); gstats = _stats_from_array(vals)
    per_band = []
    for b in range(len(sum_b)):
        n = int(n_b[b])
        if n == 0: mean = float("nan"); std = float("nan")
        else:
            mean = sum_b[b] / n; var = max(sumsq_b[b]/n - mean**2, 0.0); std = var**0.5
        per_band.append({
            "band_index": b,
            "wavelength_nm": float(wav[b]) if b < len(wav) else float("nan"),
            "n": n, "mean": float(mean), "std": float(std),
        })
    _write_json(out_json, {"sampled_patches": len(idx), "overall": gstats, "per_band_summary": per_band})


# ------------------------- main ---------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Inspeção visual/estatística dos patches.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--global-max-patches", type=int, default=100)
    ap.add_argument("--save-stats", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)

    cfg = load_config(Path(args.config))
    outputs_dir = Path(cfg["paths"]["output_dir"]).resolve()

    scene = args.scene
    manifest_csv = _find_manifest(outputs_dir, scene)
    if not manifest_csv:
        raise FileNotFoundError(f"Manifesto não encontrado para {scene} em {outputs_dir}")
    rows = _load_manifest_rows(manifest_csv)

    run_tag = _infer_run_tag_from_manifest(manifest_csv, cfg)
    base_dirs = _candidate_base_dirs(cfg, run_tag)
    patch_files = _resolve_patch_files(rows, base_dirs, scene)
    if len(patch_files) == 0:
        raise RuntimeError(f"Nenhum patch encontrado para {scene}.\n"
                           f"Manifesto: {manifest_csv}\n"
                           f"Base dirs testados: {', '.join(str(b) for b in base_dirs)}")

    # diagnóstico
    stats_path = _find_scene_stats(outputs_dir, scene)
    if stats_path and stats_path.exists():
        with open(stats_path, "r", encoding="utf-8") as f:
            jd = json.load(f)
        shp = jd.get("shape_hw_b", None)
        print("==== DIAGNÓSTICO ====")
        print(f"Scene: {jd.get('scene', scene)}")
        if shp: print(f"Dimensão cena (H×W×B): {shp[0]}×{shp[1]}×{shp[2]}")
    print(f"Patches (manifest): {len(rows)} | (arquivos): {len(patch_files)}")
    print(f"Manifesto: {manifest_csv}")
    print("Bases de busca:", ", ".join(str(b) for b in base_dirs))

    # bandas p/ viz
    sample = np.load(patch_files[0]); H, W, B = sample.shape
    rgb_bands = (0, B//2, B-1)
    print(f"Bandas para visualização (auto): {rgb_bands}")

    # quicklook / galerias / hist
    quick_png = outputs_dir / f"quicklook_{scene}.png"
    quicklook_grid(patch_files, quick_png, stride=int(args.stride), rgb_bands=rgb_bands)
    print(f"Quicklook salvo em: {quick_png}")

    N = len(patch_files)
    sel_idx = sorted(set(np.linspace(0, N-1, num=min(12, N), dtype=int).tolist()))
    gallery_png = outputs_dir / f"patch_gallery_{scene}.png"
    per_patch_gallery(patch_files, sel_idx, gallery_png, rgb_bands=rgb_bands)
    print(f"Galeria salva em: {gallery_png}")

    h_png = outputs_dir / f"patch_histograms_{scene}.png"
    per_patch_histograms(patch_files, sel_idx, h_png, bins=50)
    print(f"Histogramas (um por patch) salvos em: {h_png}")

    g_png = outputs_dir / f"global_hist_{scene}.png"
    global_histogram_sample(patch_files, g_png, max_patches=int(args.global_max_patches), bins=100)
    print(f"Histograma global salvo em: {g_png}")

    if args.save_stats:
        wav = _read_kept_wavelengths(outputs_dir, scene)
        patch_stats_csv = outputs_dir / f"patch_stats_{scene}.csv"
        save_patch_stats(patch_files, patch_stats_csv)
        print(f"Estatísticas por patch salvas em: {patch_stats_csv}")

        global_stats_json = outputs_dir / f"global_stats_{scene}.json"
        save_global_stats(patch_files, global_stats_json, wav=wav, sample=int(args.global_max_patches))
        print(f"Estatísticas globais salvas em: {global_stats_json}")


if __name__ == "__main__":
    main()
