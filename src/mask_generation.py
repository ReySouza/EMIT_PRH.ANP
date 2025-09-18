from __future__ import annotations
import argparse, json, csv
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import yaml
from scipy import ndimage as ndi

try:
    import rasterio
    from rasterio.transform import Affine
except Exception as e:
    raise RuntimeError("Instale 'rasterio' (pip install rasterio).") from e

_HAVE_SPECTRAL = False
try:
    import spectral.io.envi as sp_envi  # type: ignore
    _HAVE_SPECTRAL = True
except Exception:
    _HAVE_SPECTRAL = False

def tprint(*a): print("[mask]", *a)

def load_cfg(p: Path) -> Dict:
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# -------- leitura do MAG1C (tif ou ENVI) --------

def _read_mag1c_from_tif(path: Path) -> Tuple[np.ndarray, dict]:
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype(np.float32)
        prof = ds.profile.copy()
    return arr, prof

def _find_envi_band_index_for_mf(hdr) -> int:
    bn = hdr.get('band names')
    if isinstance(bn, str):
        names = [s.strip() for s in bn.strip('{}').split(',')]
    else:
        names = list(bn) if bn is not None else []
    names_lower = [s.lower() for s in names]
    prefer = ["unmasked matched filter results","matched filter results","mf","mag","ch4"]
    for pref in prefer:
        for i, s in enumerate(names_lower):
            if pref in s: return i
    return max(0, len(names_lower)-1)

def _read_mag1c_from_envi(hdr_path: Path) -> Tuple[np.ndarray, dict]:
    if not _HAVE_SPECTRAL:
        raise RuntimeError("Para ler ENVI (*.hdr) instale 'spectral' (pip install spectral).")
    img = sp_envi.open(str(hdr_path))
    mf_idx = _find_envi_band_index_for_mf(img.metadata)
    arr = np.asarray(img.open_memmap(interleave='bip')[:, :, mf_idx], dtype=np.float32)
    prof = dict(driver="GTiff", width=arr.shape[1], height=arr.shape[0], count=1, dtype="float32",
                transform=Affine.identity(), crs=None)
    return arr, prof

def read_mag1c_any(mag1c_dir: Path, scene_stem: str) -> Tuple[np.ndarray, dict, Path]:
    """Retorna MAG1C *apenas* da cena indicada.
    Preferência: arquivos dentro de mag1c_dir/scene_stem; se não existir subpasta, busca por prefixo scene_stem*. 
    Nunca faz fallback para "qualquer" arquivo (para evitar misturar cenas!).
    """
    sub = mag1c_dir / scene_stem
    # 1) dentro da subpasta da cena
    if sub.exists():
        tifs = sorted(sub.rglob("*.tif"))
        hdrs = sorted(sub.rglob("*.hdr"))
        # prioriza nomes com pistas
        for p in tifs:
            if any(k in p.name.lower() for k in ("mag","mf","ch4")):
                arr, prof = _read_mag1c_from_tif(p); return arr, prof, p
        if tifs:
            arr, prof = _read_mag1c_from_tif(tifs[0]); return arr, prof, tifs[0]
        for h in hdrs:
            try:
                arr, prof = _read_mag1c_from_envi(h); return arr, prof, h
            except Exception:
                continue
    # 2) por prefixo na raiz mag1c_dir
    tifs = sorted(mag1c_dir.glob(f"{scene_stem}*.tif"))
    hdrs = sorted(mag1c_dir.glob(f"{scene_stem}*.hdr"))
    for p in tifs:
        arr, prof = _read_mag1c_from_tif(p); return arr, prof, p
    for h in hdrs:
        try:
            arr, prof = _read_mag1c_from_envi(h); return arr, prof, h
        except Exception:
            continue
    raise FileNotFoundError(f"MAG1C não encontrado para '{scene_stem}' em {mag1c_dir}.")

# -------- limiar & morfologia --------

def compute_threshold(values: np.ndarray, method: str, mcfg: Dict):
    vraw = values[np.isfinite(values) & (values > 0)]
    if vraw.size == 0:
        return float(mcfg.get("fixed_threshold", 0.0)), {"note": "no positive values"}
    method = (method or "mag1c_sigma").lower(); dbg = {}
    if method == "mag1c_sigma":
        k = float(mcfg.get("k_sigma", 3.0))
        floor_val = float(mcfg.get("abs_floor", 60.0))
        qlo = float(mcfg.get("sigma_clip_lo_q", 1.0)); qhi = float(mcfg.get("sigma_clip_hi_q", 99.5))
        capq = float(mcfg.get("cap_percentile", 99.9))
        lo, hi = np.percentile(vraw, [qlo, qhi])
        v = vraw[(vraw >= lo) & (vraw <= hi)]
        mu = float(np.mean(v)); sd = float(np.std(v) + 1e-6)
        thr = mu + k * sd; cap = float(np.percentile(vraw, capq))
        thr = max(min(thr, cap), floor_val)
        dbg = dict(method="mag1c_sigma", k=k, floor=floor_val, qlo=qlo, qhi=qhi, capq=capq,
                   mu=mu, sd=sd, cap=cap, thr=thr)
        return float(thr), dbg
    if method == "percentile":
        p = float(mcfg.get("percentile", 50.0))
        thr = float(np.percentile(vraw, p))
        return thr, dict(method="percentile", p=p, thr=thr)
    if method == "fixed":
        thr = float(mcfg.get("fixed_threshold", 0.0))
        return thr, dict(method="fixed", thr=thr)
    if method == "otsu_log":
        vv = np.log1p(vraw); hist, edges = np.histogram(vv, bins=256)
        w1 = np.cumsum(hist); w2 = np.cumsum(hist[::-1])[::-1]
        m1 = np.cumsum(hist * (edges[:-1] + edges[1:]) / 2) / np.maximum(w1, 1)
        m2 = (np.cumsum((hist * (edges[:-1] + edges[1:]) / 2)[::-1]) / np.maximum(w2[::-1],1))[::-1]
        icv = w1[:-1] * w2[1:] * (m1[:-1] - m2[1:])**2
        k = int(np.nanargmax(icv)); thr_log = (edges[k] + edges[k+1]) / 2.0
        thr = float(np.expm1(thr_log)); thr = max(thr, float(mcfg.get("abs_floor", 60.0)))
        return thr, dict(method="otsu_log", thr=thr)
    thr = float(mcfg.get("fixed_threshold", 0.0))
    return thr, dict(method="fallback_fixed", thr=thr)

def postprocess_mask(mask: np.ndarray, mcfg: Dict) -> np.ndarray:
    k = max(1, int(mcfg.get("morph_kernel", 3)))
    open_iter  = int(mcfg.get("open_iter", 1))
    close_iter = int(mcfg.get("close_iter", 1))
    min_area   = int(mcfg.get("min_area", 128))
    st = np.ones((k,k), dtype=np.uint8)
    m = mask.astype(bool)
    for _ in range(open_iter):  m = ndi.binary_opening(m, structure=st)
    for _ in range(close_iter): m = ndi.binary_closing(m, structure=st)
    if min_area > 1:
        lbl, n = ndi.label(m)
        if n > 0:
            sizes = ndi.sum(np.ones_like(m, dtype=np.int32), lbl, index=np.arange(1, n+1))
            keep = np.zeros(n+1, dtype=bool); keep[0] = False; keep[1:][sizes>=min_area] = True
            m = keep[lbl]
    return m.astype(np.uint8)

# -------- manifesto, I/O --------

def read_manifest_rows(csv_path: Path) -> List[Dict[str,str]]:
    rows: List[Dict[str,str]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd: rows.append(r)
    return rows

def write_mask_geotiff(path: Path, mask: np.ndarray, prof: dict):
    prof2 = prof.copy(); prof2.update(count=1, dtype="uint8", nodata=0, compress="deflate")
    with rasterio.open(path, "w", **prof2) as ds:
        ds.write(mask.astype(np.uint8), 1)

# -------- localizar par stats/manifest DO PREPROCESS --------

def find_latest_pair(outputs_dir: Path, scene_hint: Optional[str]=None) -> Tuple[Path, Path]:
    # só aceitar *_raw_stats.json ou *_pca_stats.json (evita *_mask_stats.json)
    def _pick(sp: Path) -> bool:
        return sp.name.endswith("_raw_stats.json") or sp.name.endswith("_pca_stats.json")
    if scene_hint:
        for tag in ("raw","pca"):
            sp = outputs_dir / f"{scene_hint}_{tag}_stats.json"
            mp = outputs_dir / f"{scene_hint}_{tag}_manifest.csv"
            if sp.exists() and mp.exists():
                return sp, mp
    stats = sorted([p for p in outputs_dir.glob("*_stats.json") if _pick(p)], key=lambda p: p.stat().st_mtime)
    if not stats:
        raise FileNotFoundError(f"Nenhum *_raw_stats.json ou *_pca_stats.json em {outputs_dir}")
    sp = stats[-1]; stem = sp.name.replace("_stats.json","")
    mp = outputs_dir / f"{stem}_manifest.csv"
    if not mp.exists():
        raise FileNotFoundError(f"Manifesto não encontrado para {sp.name}")
    return sp, mp

# -------- pipeline --------

def generate_scene_mask(cfg: Dict, scene_stem: str, manifest_csv: Path, outputs_dir: Path):
    paths, proc, mcfg = cfg["paths"], cfg["processing"], cfg["masking"]
    patch = int(proc.get("patch_size", 128))

    arr, prof, mag_src = read_mag1c_any(Path(paths["mag1c_dir"]), scene_stem)
    tprint(f"MAG1C: {mag_src}  shape={arr.shape}")

    valid = np.isfinite(arr) & (arr > 0)
    vpos = arr[valid]
    thr, dbg = compute_threshold(vpos, str(mcfg.get("method","mag1c_sigma")), mcfg)
    tprint(f"threshold ({dbg.get('method')}): {thr:.2f}")

    raw_mask = (arr >= thr).astype(np.uint8)
    mask_pp = postprocess_mask(raw_mask, mcfg)

    out_dir = outputs_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_tif = out_dir / f"{scene_stem}_mask.tif"
    write_mask_geotiff(mask_tif, mask_pp, prof)
    tprint(f"Máscara da cena salva: {mask_tif}")

    # máscaras por patch
    pm_dir = Path(paths["patches_pca_masks"] if bool(proc.get("apply_pca", False)) else paths["patches_raw_masks"])
    pm_dir.mkdir(parents=True, exist_ok=True)
    used_manifest = None
    if manifest_csv and manifest_csv.exists():
        rows = read_manifest_rows(manifest_csv)
        for i, r in enumerate(rows):
            y, x = int(r["y"]), int(r["x"])
            m = mask_pp[y:y+patch, x:x+patch]
            if m.shape != (patch, patch):
                mm = np.zeros((patch, patch), dtype=np.uint8)
                hh, ww = m.shape
                mm[:hh, :ww] = m
                m = mm
            np.save(pm_dir / f"{scene_stem}_{i:06d}.npy", m.astype(np.uint8))
        used_manifest = manifest_csv

    stats = dict(scene=scene_stem, thr=float(thr), dbg=dbg, method=str(mcfg.get("method")),
                 n_pos=int(mask_pp.sum()), frac_pos=float(mask_pp.mean()),
                 used_manifest=str(used_manifest) if used_manifest else None,
                 mask_tif=str(mask_tif))
    (out_dir / f"{scene_stem}_mask_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    tprint("Concluído.")


def main(config_path: str, scene: Optional[str]=None):
    cfg = load_cfg(Path(config_path))
    outputs_dir = Path(cfg["paths"]["output_dir"])  # contém *_raw_stats.json do preprocess
    if scene:
        stats_path, manifest_path = find_latest_pair(outputs_dir, scene_hint=scene)
    else:
        stats_path, manifest_path = find_latest_pair(outputs_dir)
    stats = json.loads(Path(stats_path).read_text(encoding="utf-8"))
    scene_stem = stats["scene"]
    generate_scene_mask(cfg, scene_stem, Path(manifest_path), outputs_dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/config.yaml")
    ap.add_argument("--scene", type=str, default=None, help="Se omitido, usa a última cena do outputs.")
    args = ap.parse_args()
    main(config_path=args.config, scene=args.scene)
