# tools/inspect_masks.py — cena/patches + componentes conectados com anotações
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy import ndimage as ndi
from matplotlib.patches import Rectangle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

try:
    import rasterio
except Exception as e:
    raise RuntimeError("Instale 'rasterio' (pip install rasterio).") from e

# ENVI (opcional, só se MAG1C vier em .hdr)
try:
    import spectral as spy
    from spectral import io as spio  # spio.envi.open(...)
    _HAVE_SPECTRAL = True
except Exception:
    spy = None
    spio = None
    _HAVE_SPECTRAL = False


# --------------------------------- utils -------------------------------------

def tprint(*a): print("[inspect_masks]", *a)

def load_config(cfg_path: Path) -> Dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def choose_indices(total: int, num: int, given: Optional[List[int]]=None, seed: int=0) -> List[int]:
    if given:
        idx = [i for i in given if 0 <= i < total]
        return sorted(set(idx))
    import random
    num = min(max(num, 0), total)
    rng = random.Random(seed)
    return sorted(rng.sample(range(total), k=num))

def auto_rgb_bands(n_bands: int) -> Tuple[int, int, int]:
    if n_bands <= 1:  return (0, 0, 0)
    if n_bands == 2: return (0, 1, 1)
    return (0, n_bands // 2, n_bands - 1)

def rgb_from_patch(p: np.ndarray, bands_sel: Tuple[int, int, int]) -> np.ndarray:
    C = p.shape[-1]
    b0, b1, b2 = [min(max(b, 0), C-1) for b in bands_sel]
    rgb = p[..., [b0, b1, b2]]
    if rgb.ndim == 2: rgb = np.repeat(rgb[..., None], 3, axis=-1)
    return np.clip(rgb, 0.0, 1.0)

def outline_from_mask(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    er = ndi.binary_erosion(m, structure=np.ones((3,3), np.uint8))
    edge = m ^ er
    return edge.astype(np.uint8)

def robust_limits(v: np.ndarray, qlo=1.0, qhi=99.0) -> Tuple[float,float]:
    # usa apenas valores válidos e >0 (evita -9999 e zeros)
    vv = v[np.isfinite(v) & (v > 0)]
    if vv.size == 0:
        vv = v[np.isfinite(v)]
    if vv.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(vv, [qlo, qhi])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(vv)), float(np.nanmax(vv))
    return float(lo), float(hi)

def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows: return
    keys = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


# --------------------------- ler stats/manifest ------------------------------

def find_latest_pair(outputs_dir: Path, scene_hint: Optional[str]=None) -> Tuple[Path, Path]:
    if scene_hint:
        for tag in ("raw","pca"):
            sp = outputs_dir / f"{scene_hint}_{tag}_stats.json"
            mp = outputs_dir / f"{scene_hint}_{tag}_manifest.csv"
            if sp.exists() and mp.exists(): return sp, mp
    stats = sorted([p for p in outputs_dir.glob("*_stats.json")
                    if p.name.endswith("_raw_stats.json") or p.name.endswith("_pca_stats.json")],
                   key=lambda p: p.stat().st_mtime)
    if not stats: raise FileNotFoundError(f"Nenhum *_stats.json em {outputs_dir}")
    st = stats[-1]
    manifest = st.with_name(st.name.replace("_stats.json", "_manifest.csv"))
    if not manifest.exists():
        # tentativa: *_raw_manifest.csv / *_pca_manifest.csv
        if st.name.endswith("_raw_stats.json"):
            manifest = st.with_name(st.name.replace("_raw_stats.json","_raw_manifest.csv"))
        else:
            manifest = st.with_name(st.name.replace("_pca_stats.json","_pca_manifest.csv"))
    if not manifest.exists():
        raise FileNotFoundError(f"Manifesto para {st.name} não encontrado.")
    return st, manifest

def read_scene_stats(stats_json: Path) -> Dict:
    return json.loads(stats_json.read_text(encoding="utf-8"))

def read_manifest_rows(manifest_csv: Path) -> List[dict]:
    rows: List[dict] = []
    with manifest_csv.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r: rows.append(row)
    return rows


# --------------------------- ler MAG1C & máscara -----------------------------

def _find_envi_band_index_for_mf(hdr_meta: dict) -> int:
    meta = hdr_meta or {}
    names = (meta.get("band names") or meta.get("band_names") or "")
    if isinstance(names, str):
        names_l = names.lower()
        for key in ("mag1c","mf","ch4"):
            if key in names_l:
                # nesse caso não temos índice direto — só sinaliza que existe
                break
    # fallback: 1 banda só → índice 0
    return 0

def read_mag1c_any(mag1c_dir: Path, scene_stem: str) -> Tuple[np.ndarray, Optional[dict], Path]:
    sub = mag1c_dir / scene_stem

    # 1) Tenta GeoTIFF primeiro
    tifs = list(sub.rglob("*.tif")) + list(mag1c_dir.glob(f"{scene_stem}*.tif"))
    for p in sorted(tifs):
        name = p.name.lower()
        if any(k in name for k in ("mag", "mf", "ch4")):
            with rasterio.open(p) as ds:
                arr = ds.read(1).astype(np.float32)
                prof = ds.profile.copy()
            return arr, prof, p

    # 2) Tenta ENVI (.hdr)
    hdrs = list(sub.rglob("*.hdr")) + list(mag1c_dir.glob(f"{scene_stem}*.hdr"))
    for h in sorted(hdrs):
        if not _HAVE_SPECTRAL:
            continue
        # preferir spectral.open_image; se falhar, usar io.envi.open
        try:
            img = spy.open_image(str(h))
        except Exception:
            img = spio.envi.open(str(h))

        # escolhe banda (se for multibanda); para MAG1C geralmente é 1 banda
        try:
            meta = getattr(img, "metadata", {}) or {}
        except Exception:
            meta = {}
        idx = _find_envi_band_index_for_mf(meta)

        # ler somente a banda desejada (linhas x colunas)
        try:
            band = img.read_band(idx)   # (rows, cols)
            arr = np.asarray(band, dtype=np.float32)
        except Exception:
            # fallback genérico
            arr = np.asarray(img[:, :, idx], dtype=np.float32)

        return arr, None, h

    raise FileNotFoundError(f"Não achei MAG1C para '{scene_stem}' em {mag1c_dir} (nem .tif nem .hdr).")

def read_scene_mask_tif(outputs_dir: Path, scene_stem: str) -> Tuple[np.ndarray, dict, Path]:
    mask_tif = outputs_dir / f"{scene_stem}_mask.tif"
    if not mask_tif.exists():
        cands = sorted(outputs_dir.glob("*_mask.tif"), key=lambda p: p.stat().st_mtime)
        if not cands: raise FileNotFoundError(f"Nenhum *_mask.tif em {outputs_dir}")
        mask_tif = cands[-1]
    with rasterio.open(mask_tif) as ds:
        m = ds.read(1).astype(np.uint8)
        prof = ds.profile.copy()
    return m, prof, mask_tif


# ---------------------------- cenas & galeria --------------------------------

def plot_scene_panels(mag: np.ndarray, mask: np.ndarray, threshold: float,
                      out_png: Path, cmap: str="viridis", alpha: float=0.35):
    lo, hi = robust_limits(mag, 1.0, 99.0)
    edge = outline_from_mask(mask)

    plt.figure(figsize=(15, 7))

    ax1 = plt.subplot(2, 2, 1)
    im1 = ax1.imshow(mag, vmin=lo, vmax=hi, cmap=cmap)
    ax1.set_title(f"MAG1C (robust 1–99%: [{lo:.1f}, {hi:.1f}])"); ax1.axis("off")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = plt.subplot(2, 2, 2)
    ax2.imshow(mask, cmap="gray"); ax2.set_title("Máscara binária (1=pluma)"); ax2.axis("off")

    ax3 = plt.subplot(2, 2, 3)
    im3 = ax3.imshow(mag, vmin=lo, vmax=hi, cmap=cmap)
    ax3.imshow(np.ma.masked_where(edge == 0, edge), cmap="Reds", alpha=alpha, interpolation="none")
    ax3.set_title("Overlay: contorno da máscara sobre MAG1C"); ax3.axis("off")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    ax4 = plt.subplot(2, 2, 4)
    vals = mag[np.isfinite(mag) & (mag > 0)]
    ax4.hist(vals, bins=200)
    ax4.axvline(threshold, color="crimson", linestyle="--", label=f"threshold={threshold:.1f}")
    ax4.set_title("Histograma de MAG1C (válidos>0)")
    ax4.legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def show_patch_gallery(scene_stem: str,
                       rows_manifest: List[dict],
                       img_dir: Path,
                       mask_dir: Path,
                       mag_full: np.ndarray,
                       patch_size: int,
                       bands_sel: Tuple[int,int,int],
                       out_png: Path,
                       num: int = 12,
                       indices: Optional[List[int]] = None,
                       seed: int = 0,
                       cmap: str = "viridis",
                       alpha: float = 0.35):
    n = len(rows_manifest)
    if indices:
        sel = choose_indices(n, len(indices), given=indices)
    else:
        sel = choose_indices(n, num, given=None, seed=seed)
    if not sel:
        tprint("Nenhum índice selecionado para galeria."); return

    rows = len(sel)        # uma linha por patch
    cols = 3               # 3 colunas: RGB, máscara, MAG1C+contorno
    plt.figure(figsize=(4.2 * cols, 3.8 * rows))
    k = 1

    for idx in sel:
        p_img = img_dir  / f"{scene_stem}_{idx:06d}.npy"
        p_msk = mask_dir / f"{scene_stem}_{idx:06d}.npy"

        r = rows_manifest[idx]; y, x = int(r["y"]), int(r["x"])
        patch_mf = mag_full[y:y+patch_size, x:x+patch_size]

        # 1) RGB do patch (auto bandas)
        ax1 = plt.subplot(rows, cols, k); k += 1
        if p_img.exists():
            arr = np.load(p_img); rgb = rgb_from_patch(arr, bands_sel)
            ax1.imshow(rgb); ax1.set_title(f"idx {idx} | RGB"); ax1.axis("off")
        else:
            ax1.text(0.5, 0.5, "patch RGB ausente", ha="center", va="center"); ax1.axis("off")

        # 2) máscara (1=pluma)
        ax2 = plt.subplot(rows, cols, k); k += 1
        if p_msk.exists():
            m = np.load(p_msk).astype(np.uint8)
            ax2.imshow(m, cmap="gray"); ax2.set_title("Mask (1=pluma)"); ax2.axis("off")
        else:
            ax2.text(0.5, 0.5, "mask ausente", ha="center", va="center"); ax2.axis("off")

        # 3) MAG1C com contorno da máscara
        ax3 = plt.subplot(rows, cols, k); k += 1
        lo, hi = robust_limits(patch_mf, 1.0, 99.0)
        edge = outline_from_mask(m if p_msk.exists() else np.zeros_like(patch_mf))
        im = ax3.imshow(patch_mf, vmin=lo, vmax=hi, cmap=cmap)
        ax3.imshow(np.ma.masked_where(edge == 0, edge), cmap="Reds", alpha=alpha, interpolation="none")
        ax3.set_title("MAG1C + contorno"); ax3.axis("off")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# --------------------------- componentes conectados --------------------------

def component_metrics(mask: np.ndarray, mag: np.ndarray, min_area: int=1) -> Tuple[np.ndarray, List[dict]]:
    """Label 8-viz, e métricas por componente (área, bbox, centróide, alongamento, orientação, stats MAG1C)."""
    lab, n = ndi.label(mask.astype(bool), structure=np.ones((3,3), np.uint8))
    if n == 0:
        return lab, []
    objs = ndi.find_objects(lab)
    rows: List[dict] = []
    for i in range(n):
        sl = objs[i]
        if sl is None: continue
        y0, y1 = sl[0].start, sl[0].stop
        x0, x1 = sl[1].start, sl[1].stop
        sub_lab = lab[y0:y1, x0:x1]
        comp = (sub_lab == (i+1))
        area = int(comp.sum())
        if area < min_area: 
            continue
        # centróide (em coords da cena)
        cy, cx = ndi.center_of_mass(comp)
        cy = float(cy + y0); cx = float(cx + x0)
        # bbox & alongamento
        h = y1 - y0; w = x1 - x0
        elong = float(max(h, w) / max(1, min(h, w)))
        # orientação/eccentricidade aproximadas por autovetores da covariância
        ys, xs = np.nonzero(comp)
        if ys.size >= 2:
            X = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=0)
            C = np.cov(X)
            evals, evecs = np.linalg.eigh(C)
            order = np.argsort(evals)[::-1]
            evals = evals[order]; evecs = evecs[:, order]
            angle = math.degrees(math.atan2(evecs[1,0], evecs[0,0]))
            ecc = float(np.sqrt(max(0.0, 1.0 - evals[1]/max(evals[0], 1e-9))))
        else:
            angle = 0.0; ecc = 0.0
        # stats de MAG1C na região
        mf_vals = mag[y0:y1, x0:x1][comp]
        mf_vals = mf_vals[np.isfinite(mf_vals)]
        if mf_vals.size == 0:
            q01=q05=q25=q50=q75=q95=q99=mn=mx=mu=sd=so=0.0
        else:
            q01,q05,q25,q50,q75,q95,q99 = np.quantile(mf_vals, [0.01,0.05,0.25,0.50,0.75,0.95,0.99]).tolist()
            mn, mx = float(np.min(mf_vals)), float(np.max(mf_vals))
            mu, sd = float(np.mean(mf_vals)), float(np.std(mf_vals))
            so = float(np.sum(mf_vals))
        rows.append(dict(
            comp_id=i+1, area_px=area,
            y0=int(y0), y1=int(y1), x0=int(x0), x1=int(x1),
            cy=cy, cx=cx, height=h, width=w, elongation=elong,
            orientation_deg=float(angle), eccentricity=ecc,
            mf_min=mn, mf_p01=q01, mf_p05=q05, mf_p25=q25, mf_p50=q50, mf_p75=q75, mf_p95=q95, mf_p99=q99,
            mf_max=mx, mf_mean=mu, mf_std=sd, mf_sum=so
        ))
    rows.sort(key=lambda r: r["area_px"], reverse=True)
    return lab, rows

def plot_components_annotated(mag: np.ndarray, comps: List[dict], out_png: Path,
                              top_k: int = 20, cmap: str="viridis"):
    lo, hi = robust_limits(mag, 1.0, 99.0)
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(mag, vmin=lo, vmax=hi, cmap=cmap)
    ax.set_title(f"Componentes conectados (top {top_k} por área)"); ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for r in comps[:max(0, top_k)]:
        x0, y0, w, h = r["x0"], r["y0"], r["width"], r["height"]
        rect = Rectangle((x0, y0), w, h, fill=False, linewidth=1.5, edgecolor="white")
        ax.add_patch(rect)
        label = f"#{r['comp_id']} · A={r['area_px']}"
        ax.text(r["cx"], r["cy"], label, ha="center", va="center",
                fontsize=7, color="white",
                bbox=dict(facecolor="black", alpha=0.35, pad=1.0))
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)

def inside_outside_stats(mag: np.ndarray, mask: np.ndarray) -> Dict:
    v = np.isfinite(mag) & (mag > 0)
    inside = mag[v & (mask > 0)]
    outside = mag[v & (mask == 0)]
    def stats(vals: np.ndarray) -> Dict[str, float]:
        if vals.size == 0:
            return {k: float("nan") for k in ["n","min","p25","p50","p75","p90","p95","p99","max","mean","std"]}
        q = np.quantile(vals, [0.25,0.50,0.75,0.90,0.95,0.99])
        return dict(n=float(vals.size), min=float(np.min(vals)), p25=float(q[0]), p50=float(q[1]),
                    p75=float(q[2]), p90=float(q[3]), p95=float(q[4]), p99=float(q[5]),
                    max=float(np.max(vals)), mean=float(np.mean(vals)), std=float(np.std(vals)))
    return {"inside": stats(inside), "outside": stats(outside)}

# --------- threshold (reuso do que você já tinha, com fallback robusto) ------

def resolve_threshold(mask_stats_json: Path, cfg: Dict, mag_arr: np.ndarray) -> float:
    thr = None
    if mask_stats_json.exists():
        try:
            st = json.loads(mask_stats_json.read_text(encoding="utf-8"))
            if "thr" in st and np.isfinite(st["thr"]): thr = float(st["thr"])
            elif "threshold_value" in st and np.isfinite(st["threshold_value"]): thr = float(st["threshold_value"])
        except Exception:
            pass
    if thr is None or not np.isfinite(thr):
        vals = mag_arr[np.isfinite(mag_arr) & (mag_arr > 0)]
        if vals.size:
            method = (cfg.get("masking", {}).get("method", "percentile") or "percentile").lower()
            if method == "percentile":
                p = float(cfg.get("masking", {}).get("percentile", 99.0))
                thr = float(np.percentile(vals, p))
            else:
                thr = float(np.percentile(vals, 99.0))
        else:
            thr = 0.0
    return float(thr)

def save_histogram_only(mag_arr: np.ndarray, thr: float, out_png: Path, bins: int=200):
    vals = mag_arr[np.isfinite(mag_arr) & (mag_arr > 0)]
    plt.figure(figsize=(9,4))
    plt.hist(vals, bins=bins)
    plt.axvline(thr, color="crimson", linestyle="--", label=f"threshold={thr:.1f}")
    plt.title("Histograma de MAG1C (válidos>0)"); plt.legend(); plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160); plt.close()


# ---------------------------------- main -------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Inspeção visual das máscaras de pluma (cena e patches) com componentes conectados.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--scene", default=None)
    ap.add_argument("--num", type=int, default=12)
    ap.add_argument("--indices", type=int, nargs="*", default=None)
    ap.add_argument("--bands", nargs="*", default=["auto"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cmap", type=str, default="viridis")
    ap.add_argument("--alpha", type=float, default=0.35)
    # histogram-only (debug)
    ap.add_argument("--hist-only", action="store_true", help="Gera apenas o histograma (sem overview/galeria).")
    ap.add_argument("--hist-out", type=str, default=None, help="PNG do histograma; default: outputs/scene_hist_<scene>.png")
    ap.add_argument("--bins", type=int, default=200)
    # NOVO: componentes conectados
    ap.add_argument("--annotate-top", type=int, default=20, help="Quantos componentes anotar no overlay (ordenado por área).")
    ap.add_argument("--min-area", type=int, default=None, help="Filtrar componentes com área mínima (px). Default: masking.min_area do config, se existir.")
    ap.add_argument("--save-stats", action="store_true", help="Salva CSV de componentes e JSON inside/outside.")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    paths = cfg.get("paths", {})
    outputs_dir = Path(paths["output_dir"]).resolve()
    mag1c_dir   = Path(paths.get("mag1c_dir", outputs_dir)).resolve()  # seu log já apontou data/mag1c

    # stats/manifest
    scene_hint = args.scene
    stats_json, manifest_csv = find_latest_pair(outputs_dir, scene_hint=scene_hint)
    sstats = read_scene_stats(stats_json)
    scene_stem = sstats.get("scene") or (args.scene or manifest_csv.stem.split("_")[0])

    kept_nm = sstats.get("kept_bands_nm", [])
    B = int(sstats.get("shape_hw_b", [0,0,3])[2])
    bands_sel = auto_rgb_bands(B)

    # patch dirs (conforme preprocess)
    run_tag = "pca" if stats_json.name.endswith("_pca_stats.json") else "raw"
    img_dir = Path(paths.get("patches_pca_images" if run_tag=="pca" else "patches_raw_images", outputs_dir))
    mask_dir = Path(paths.get("patches_pca_masks"  if run_tag=="pca" else "patches_raw_masks",  outputs_dir))
    patch_size = int(cfg.get("processing", {}).get("patch_size", 128))

    # MAG1C + máscara da cena
    mag_arr, mag_prof, mag_src = read_mag1c_any(mag1c_dir, scene_stem)
    mask_arr, mask_prof, mask_tif = read_scene_mask_tif(outputs_dir, scene_stem)
    H, W = mag_arr.shape

    # threshold para exibição
    mask_stats_json = outputs_dir / f"{scene_stem}_mask_stats.json"
    thr = resolve_threshold(mask_stats_json, cfg, mag_arr)

    # hist-only
    if args.hist_only:
        out = Path(args.hist_out) if args.hist_out else (outputs_dir / f"scene_hist_{scene_stem}.png")
        save_histogram_only(mag_arr, thr, out_png=out, bins=int(args.bins))
        tprint("Histograma salvo em:", out)
        return

    # ---- Relatórios padrão (overview + galeria) ----
    tprint("Bandas (auto):", bands_sel)
    tprint("Scene:", scene_stem)
    tprint("MAG1C:", mag_src)
    tprint("Mask: ", mask_tif)
    tprint("Threshold usado/exibido:", f"{thr:.4f}")
    tprint("Dimensão cena (H×W×B):", f"{H}×{W}×{B}")
    if kept_nm:
        example = str(kept_nm[:min(6,len(kept_nm))])
        tprint(f"Bandas mantidas (nm): {len(kept_nm)}; exemplo: {example}{'...' if len(kept_nm)>6 else ''}")

    scene_png   = outputs_dir / f"scene_mask_overview_{scene_stem}.png"
    gallery_png = outputs_dir / f"patch_mask_gallery_{scene_stem}.png"

    plot_scene_panels(mag_arr, mask_arr, threshold=thr, out_png=scene_png, cmap=args.cmap, alpha=args.alpha)
    tprint("Overview salvo em:", scene_png)

    rows_manifest = read_manifest_rows(manifest_csv)
    show_patch_gallery(scene_stem, rows_manifest, img_dir, mask_dir,
                       mag_arr, patch_size, bands_sel, out_png=gallery_png,
                       num=int(args.num), indices=args.indices, seed=int(args.seed),
                       cmap=args.cmap, alpha=args.alpha)
    tprint("Galeria de patches salva em:", gallery_png)

    # ---- NOVO: componentes conectados (estatísticas + overlay anotado) ----
    min_area_cfg = int(cfg.get("masking", {}).get("min_area", 0))
    min_area = int(args.min_area) if args.min_area is not None else min_area_cfg
    lab, comp_rows = component_metrics(mask_arr, mag_arr, min_area=min_area)
    tprint(f"Componentes encontrados (>= {min_area}px):", len(comp_rows))

    ann_png = outputs_dir / f"scene_components_{scene_stem}.png"
    plot_components_annotated(mag_arr, comp_rows, out_png=ann_png,
                              top_k=int(args.annotate_top), cmap=args.cmap)
    tprint("Overlay de componentes salvo em:", ann_png)

    if args.save_stats:
        comps_csv = outputs_dir / f"{scene_stem}_mask_components.csv"
        write_csv(comps_csv, comp_rows)
        tprint("Tabela de componentes salva em:", comps_csv)

        io_json = outputs_dir / f"{scene_stem}_inside_outside_stats.json"
        write_json(io_json, inside_outside_stats(mag_arr, mask_arr))
        tprint("Inside/outside stats salvas em:", io_json)


if __name__ == "__main__":
    main()
