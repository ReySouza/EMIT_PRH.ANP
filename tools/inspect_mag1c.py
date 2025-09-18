# tools/inspect_mag1c_patches.py — inspeção dos patches de MAG1C (1 canal) alinhados ao manifest
from __future__ import annotations

import argparse, csv, json, random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

def tprint(*a): print("[inspect_mag1c]", *a)

# ------------------------- config & io helpers -------------------------------

def load_config(cfg_path: Path) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p

def _find_scene_stats(outputs_dir: Path, scene: str) -> Optional[Path]:
    pats = [
        f"{scene}*_raw_stats.json",
        f"{scene}*_pca_stats.json",
        f"{scene}*_stats.json",
    ]
    for pat in pats:
        found = sorted(outputs_dir.glob(pat))
        if found: return found[0]
    return None

def _find_manifest(outputs_dir: Path, scene: str) -> Optional[Path]:
    pats = [
        f"{scene}*_raw_manifest.csv",
        f"{scene}*_pca_manifest.csv",
        f"{scene}*_manifest.csv",
    ]
    for pat in pats:
        found = sorted(outputs_dir.glob(pat))
        if found: return found[0]
    # fallback
    for f in sorted(outputs_dir.glob(f"{scene}*.csv")):
        if "manifest" in f.stem.lower(): return f
    return None

def _load_manifest_rows(manifest_csv: Path) -> List[dict]:
    rows: List[dict] = []
    with manifest_csv.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r: rows.append(row)
    return rows

def _resolve_mag1c_patches(cfg: dict, scene: str) -> List[Path]:
    mag_dir = Path(cfg["paths"]["patches_mag1c"])
    files = sorted(mag_dir.glob(f"{scene}_*.npy"))
    return files

def _resolve_mask_patches(cfg: dict, scene: str, run_tag: str) -> List[Path]:
    key = "patches_pca_masks" if run_tag == "pca" else "patches_raw_masks"
    if key not in cfg["paths"]: return []
    md = Path(cfg["paths"][key])
    return sorted(md.glob(f"{scene}_*.npy"))

def _infer_run_tag(outputs_dir: Path, scene: str, cfg: dict) -> str:
    st = _find_scene_stats(outputs_dir, scene)
    if st and "pca" in st.stem.lower(): return "pca"
    if st and "raw" in st.stem.lower(): return "raw"
    return "pca" if bool(cfg.get("processing",{}).get("apply_pca", False)) else "raw"

# ------------------------------ stats ----------------------------------------

def _valid_values(x: np.ndarray) -> np.ndarray:
    return x[np.isfinite(x)]

def _robust_minmax(x: np.ndarray, qlo: float=1.0, qhi: float=99.0) -> Tuple[float,float]:
    v = _valid_values(x)
    if v.size == 0: return 0.0, 1.0
    lo, hi = np.percentile(v, [qlo, qhi])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
    return float(lo), float(hi)

def _patch_stats(x: np.ndarray) -> Dict[str, float]:
    v = _valid_values(x)
    if v.size == 0:
        return {k: float("nan") for k in ["n","min","p25","p50","p75","p90","p95","p99","max","mean","std"]}
    q = np.quantile(v, [0.25,0.50,0.75,0.90,0.95,0.99])
    return dict(n=float(v.size), min=float(np.min(v)), p25=float(q[0]), p50=float(q[1]), p75=float(q[2]),
                p90=float(q[3]), p95=float(q[4]), p99=float(q[5]), max=float(np.max(v)),
                mean=float(np.mean(v)), std=float(np.std(v)))

def _write_csv(p: Path, rows: Sequence[dict]) -> None:
    ensure_dir(p.parent)
    if not rows: return
    keys = list(rows[0].keys())
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

def _write_json(p: Path, obj: dict) -> None:
    ensure_dir(p.parent)
    with p.open("w", "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

# ------------------------------ plots ----------------------------------------

def _gallery_mag1c(mfiles: Sequence[Path], sel: Sequence[int], out_png: Path, masks: Optional[Sequence[Path]]=None):
    if not sel: return
    rows = len(sel); cols = 3 if masks else 1
    fig, axes = plt.subplots(rows, cols, figsize=(4.0*cols, 3.2*rows)); axes = np.array(axes).reshape(rows, cols)
    for ax in axes.ravel(): ax.axis("off")
    for i, idx in enumerate(sel):
        x = np.load(mfiles[idx]).astype(np.float32)
        lo, hi = _robust_minmax(x, 1.0, 99.0)
        ax = axes[i,0]; im = ax.imshow(x, vmin=lo, vmax=hi, cmap="viridis"); ax.set_title(f"idx {idx} | MAG1C"); ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if masks:
            if idx < len(masks) and masks[idx].exists():
                m = np.load(masks[idx]).astype(np.uint8)
                ax2 = axes[i,1]; ax2.imshow(m, cmap="gray"); ax2.set_title("Máscara (1=pluma)"); ax2.axis("off")
                ax3 = axes[i,2]; im2 = ax3.imshow(x, vmin=lo, vmax=hi, cmap="viridis"); 
                edge = (m.astype(bool) ^ (np.pad(m.astype(bool),1)[1:-1,1:-1] & m.astype(bool)))  # cheap outline
                ax3.imshow(np.ma.masked_where(~edge, edge), cmap="Reds", alpha=0.35, interpolation="none")
                ax3.set_title("Overlay"); ax3.axis("off"); fig.colorbar(im2, ax=ax3, fraction=0.046, pad=0.04)
    fig.tight_layout(); ensure_dir(out_png.parent); fig.savefig(out_png, dpi=180); plt.close(fig)

def _global_hist(mfiles: Sequence[Path], out_png: Path, bins: int=120, sample: int=200):
    n = len(mfiles); idx = list(range(n)); random.shuffle(idx); idx = idx[:min(n, sample)]
    vals = []
    for i in idx:
        x = np.load(mfiles[i]).astype(np.float32)
        v = _valid_values(x); 
        if v.size: vals.append(v)
    if not vals: return
    vals = np.concatenate(vals, axis=0)
    plt.figure(figsize=(10,5)); plt.hist(vals, bins=bins)
    plt.title(f"Histograma global de MAG1C (~{len(idx)} patches)"); plt.tight_layout()
    ensure_dir(out_png.parent); plt.savefig(out_png, dpi=160); plt.close()

def _per_patch_hists(mfiles: Sequence[Path], sel: Sequence[int], out_png: Path, bins: int=60):
    if not sel: return
    rows = int(np.ceil(len(sel) / 6)); cols = min(6, len(sel))
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.2*rows)); axes = np.array(axes).reshape(rows, cols)
    for ax in axes.ravel(): ax.axis("off")
    for ax, idx in zip(axes.ravel(), sel):
        x = np.load(mfiles[idx]).astype(np.float32); v = _valid_values(x)
        if v.size == 0: continue
        ax.hist(v, bins=bins); ax.set_title(f"idx {idx} | min={v.min():.2f} max={v.max():.2f}", fontsize=8)
    fig.tight_layout(); ensure_dir(out_png.parent); fig.savefig(out_png, dpi=160); plt.close(fig)

# ------------------------------ main -----------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Inspeção dos patches de MAG1C (1 canal).")
    ap.add_argument("--config", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--num", type=int, default=12, help="Nº de patches na galeria")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bins", type=int, default=60)
    ap.add_argument("--save-stats", action="store_true")
    ap.add_argument("--with-masks", action="store_true", help="Mostra também a máscara binária por patch, se existir.")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)

    cfg = load_config(Path(args.config))
    outputs_dir = Path(cfg["paths"]["output_dir"]).resolve()

    scene = args.scene
    manifest_csv = _find_manifest(outputs_dir, scene)
    if not manifest_csv:
        raise FileNotFoundError(f"Manifesto não encontrado para {scene} em {outputs_dir}")
    rows = _load_manifest_rows(manifest_csv); N = len(rows)

    mfiles = _resolve_mag1c_patches(cfg, scene)
    if len(mfiles) == 0:
        raise RuntimeError(f"Nenhum patch MAG1C encontrado para {scene} em {cfg['paths']['patches_mag1c']}")

    run_tag = _infer_run_tag(outputs_dir, scene, cfg)
    masks = _resolve_mask_patches(cfg, scene, run_tag) if args.with_masks else None

    # seleção de índices
    sel = sorted(set(np.linspace(0, len(mfiles)-1, num=min(args.num, len(mfiles)), dtype=int).tolist()))

    # figuras
    gal_png = outputs_dir / f"mag1c_patch_gallery_{scene}.png"
    _gallery_mag1c(mfiles, sel, gal_png, masks=masks)
    tprint("Galeria salva em:", gal_png)

    g_png = outputs_dir / f"mag1c_global_hist_{scene}.png"
    _global_hist(mfiles, g_png, bins=int(args.bins))
    tprint("Histograma global salvo em:", g_png)

    h_png = outputs_dir / f"mag1c_patch_hists_{scene}.png"
    _per_patch_hists(mfiles, sel, h_png, bins=int(args.bins))
    tprint("Histogramas por patch salvos em:", h_png)

    # stats (opcional)
    if args.save_stats:
        rows_csv = []
        for i, p in enumerate(mfiles):
            x = np.load(p).astype(np.float32)
            s = _patch_stats(x); s.update({"patch_idx": i, "file": str(p)})
            rows_csv.append(s)
        csv_path = outputs_dir / f"mag1c_patch_stats_{scene}.csv"
        _write_csv(csv_path, rows_csv); tprint("Estatísticas por patch salvas em:", csv_path)

        # simples consistência: contagem
        report = {
            "scene": scene,
            "mag1c_patches": len(mfiles),
            "manifest_rows": N,
            "has_masks": bool(masks) and len(masks) == len(mfiles)
        }
        json_path = outputs_dir / f"mag1c_patch_report_{scene}.json"
        with open(json_path, "w", encoding="utf-8") as f: json.dump(report, f, indent=2)
        tprint("Relatório salvo em:", json_path)


if __name__ == "__main__":
    main()
