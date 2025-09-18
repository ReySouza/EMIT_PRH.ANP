# src/emit_reader.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np

# Preferir rasterio; cair para spectral se necessário
try:
    import rasterio
    RASTERIO_OK = True
except Exception:
    RASTERIO_OK = False

try:
    from spectral import io as spectral_io
    SPECTRAL_OK = True
except Exception:
    SPECTRAL_OK = False
try:
    from spectral.io import envi as sp_envi
    SPECTRAL_ENVI_OK = True
except Exception:
    SPECTRAL_ENVI_OK = False

def read_envi_wavelengths(hdr_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Lê wavelength/fwhm de um HDR ENVI (suporta listas multilinha típicas do EMIT)."""
    if not SPECTRAL_ENVI_OK:
        return np.array([]), np.array([])
    try:
        hdr = sp_envi.read_envi_header(str(hdr_path))
        wav = np.array([float(x) for x in hdr.get("wavelength", [])], dtype=np.float32)
        fwhm = np.array([float(x) for x in hdr.get("fwhm", [])], dtype=np.float32)
        return wav, fwhm
    except Exception:
        return np.array([]), np.array([])

def parse_envi_header_text(hdr_path: Path) -> Dict[str, str]:
    """Lê o .hdr como texto e monta um dicionário bruto {chave:valor} (sem depender de libs)."""
    txt = hdr_path.read_text(encoding="utf-8", errors="ignore")
    d = {}
    for line in txt.replace("\r","").split("\n"):
        if "=" in line:
            k, v = line.split("=", 1)
            d[k.strip().lower()] = v.strip()
    return d


def parse_float_list(envi_field: str) -> np.ndarray:
    """Converte um campo ENVI como '{1.0, 2.0, ...}' ou '{1,2,3}' em np.array(float)."""
    if envi_field is None:
        return np.array([])
    s = envi_field.strip().lstrip("{").rstrip("}")
    if not s:
        return np.array([])
    # aceita separadores vírgula e/ou espaço
    parts = [p for p in s.replace(",", " ").split() if p]
    arr = np.array([float(p) for p in parts], dtype=np.float32)
    return arr


def _read_with_rasterio(hdr_path: Path) -> Tuple[np.ndarray, Dict]:
    # tenta achar o bin (.img/.dat/.bsq/.bil/.bip)
    for suf in (".img", ".dat", ".raw", ".bsq", ".bil", ".bip"):
        bin_path = hdr_path.with_suffix(suf)
        if bin_path.exists():
            break
    else:
        raise FileNotFoundError(f"Não encontrei o binário ENVI para {hdr_path}")

    with rasterio.open(bin_path) as ds:
        cube = ds.read().astype(np.float32)       # (B,H,W)
        cube = np.moveaxis(cube, 0, -1)           # (H,W,B)
        meta = {
            "rows": ds.height, "cols": ds.width, "bands": ds.count,
            "dtype": ds.dtypes[0], "driver": ds.driver,
            "crs": str(ds.crs) if ds.crs else None,
            "transform": tuple(ds.transform) if ds.transform else None,
        }
    return cube, meta


def _read_with_spectral(hdr_path: Path) -> Tuple[np.ndarray, Dict]:
    img = spectral_io.envi.open(str(hdr_path), str(hdr_path.with_suffix(".img")))
    a = img.open_memmap(writeable=False)          # (R,C,B)
    cube = np.array(a, dtype=np.float32)
    if cube.ndim == 3 and cube.shape[-1] != 1:
        pass
    else:
        # fallback para (R,C,B)
        cube = img.load().astype(np.float32)
    # spectral costuma entregar (R,C,B) -> queremos (H,W,B)
    meta = {"rows": cube.shape[0], "cols": cube.shape[1], "bands": cube.shape[2]}
    return cube, meta


def read_emit_l1b_envi(hdr_path: Path) -> Tuple[np.ndarray, Dict]:
    """Lê um cubo EMIT L1B (ENVI .hdr + bin) e retorna (cube(H,W,B) float32, meta dict).
       Tenta rasterio e cai para spectral se necessário."""
    if RASTERIO_OK:
        try:
            cube, meta = _read_with_rasterio(hdr_path)
        except Exception:
            if not SPECTRAL_OK:
                raise
            cube, meta = _read_with_spectral(hdr_path)
    else:
        if not SPECTRAL_OK:
            raise RuntimeError("Instale rasterio ou spectral para ler ENVI")
        cube, meta = _read_with_spectral(hdr_path)

    # ler info espectral do hdr (wavelength, fwhm, data ignore)
    hdr = parse_envi_header_text(hdr_path)
    wav_nm  = parse_float_list(hdr.get("wavelength"))
    fwhm_nm = parse_float_list(hdr.get("fwhm"))
    if wav_nm.size == 0:  # fallback robusto para header multilinha
        wav_nm2, fwhm_nm2 = read_envi_wavelengths(hdr_path)
        if wav_nm2.size:
            wav_nm, fwhm_nm = wav_nm2, fwhm_nm2
    ignore_val = None
    for k in ("data ignore value", "data_ignore_value", "data value to ignore"):
        if k in hdr:
            try:
                ignore_val = float(hdr[k].strip().strip("{}"))
            except Exception:
                pass
            break

    meta.update({
        "wavelength_nm": wav_nm,
        "fwhm_nm": fwhm_nm,
        "data_ignore_value": ignore_val,
    })
    return cube, meta


def valid_mask_from_cube(cube: np.ndarray, ignore_val: Optional[float]=None, eps: float=1e-6) -> np.ndarray:
    """Pixels válidos: (i) não são ignore_val, (ii) têm algum valor finito > eps em pelo menos 1 banda."""
    v = np.isfinite(cube)
    if ignore_val is not None:
        v &= (cube != ignore_val)
    v &= (cube > eps)
    return np.any(v, axis=2)


def select_bands_by_nm(cube: np.ndarray, wav_nm: np.ndarray,
                       nm_min: float, nm_max: float) -> Tuple[np.ndarray, np.ndarray]:
    """Seleciona bandas por faixa espectral (ex.: 2122–2488 nm para CH₄)."""
    if wav_nm.size == 0:
        # se não houver wavelengths, retorna tudo
        return cube, np.arange(cube.shape[-1], dtype=np.int32)
    keep = np.where((wav_nm >= nm_min) & (wav_nm <= nm_max))[0]
    if keep.size == 0:
        return cube, np.arange(cube.shape[-1], dtype=np.int32)
    return cube[..., keep], keep


def drop_bad_bands(cube: np.ndarray, valid: np.ndarray,
                   max_nan_ratio: float=0.4, min_variance: float=1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """Remove bandas com muito NaN/ignore ou variância ~0 nos pixels válidos."""
    H, W, B = cube.shape
    keep = []
    for b in range(B):
        band = cube[..., b]
        mask = valid & np.isfinite(band)
        if mask.sum() == 0:
            continue
        nan_ratio = 1.0 - (mask.sum() / valid.sum())
        if nan_ratio > max_nan_ratio:
            continue
        var = float(np.nanvar(band[mask]))
        if not np.isfinite(var) or var < min_variance:
            continue
        keep.append(b)
    if not keep:
        raise ValueError("Todas as bandas foram descartadas — verifique os dados.")
    keep = np.array(keep, dtype=np.int32)
    return cube[..., keep], keep


def robust_minmax_01(cube: np.ndarray, valid: np.ndarray,
                     q_low: float=1.0, q_high: float=99.0) -> np.ndarray:
    """Escala 0..1 por banda usando percentis em pixels válidos; pixels inválidos viram 0."""
    H, W, B = cube.shape
    out = np.zeros((H, W, B), dtype=np.float32)
    for b in range(B):
        band = cube[..., b]
        vals = band[valid & np.isfinite(band)]
        if vals.size == 0:
            continue
        lo, hi = np.percentile(vals, [q_low, q_high])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                continue
        scaled = (band - lo) / max(hi - lo, 1e-6)
        scaled = np.clip(scaled, 0.0, 1.0)
        scaled[~valid] = 0.0
        out[..., b] = scaled.astype(np.float32)
    return out
