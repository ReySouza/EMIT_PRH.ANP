# src/preprocess.py
from __future__ import annotations
import os, json, argparse
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import yaml

from emit_reader import (
    read_emit_l1b_envi, valid_mask_from_cube, select_bands_by_nm,
    drop_bad_bands, robust_minmax_01, parse_envi_header_text
)

try:
    from tqdm import tqdm
    _HAVE_TQDM = True
except Exception:
    _HAVE_TQDM = False


def tprint(msg: str): print(f"[preprocess] {msg}")


# ----------------- config -----------------
def load_config(cfg_path: Optional[str]) -> Dict:
    if cfg_path is None:
        raise ValueError("Passe --config <configs/config.yaml>")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # valida mínimas
    for key in ["paths", "processing"]:
        if key not in cfg:
            raise ValueError(f"Faltando seção '{key}' no {cfg_path}")
    return cfg


# ----------------- tiling -----------------
def gen_patches(arr: np.ndarray, valid: np.ndarray,
                patch: int, stride: int, min_valid_frac: float) -> Tuple[List[Tuple[int,int]], np.ndarray]:
    H, W, C = arr.shape
    idxs, out = [], []
    for y in range(0, H - patch + 1, stride):
        for x in range(0, W - patch + 1, stride):
            v = valid[y:y+patch, x:x+patch]
            frac = float(v.mean())
            if frac < min_valid_frac:
                continue
            idxs.append((y, x))
            out.append(arr[y:y+patch, x:x+patch])
    if not out:
        return [], np.zeros((0, patch, patch, arr.shape[-1]), dtype=arr.dtype)
    return idxs, np.stack(out, axis=0)


def save_patches_npy(patches: np.ndarray, out_dir: Path, stem: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(patches.shape[0]):
        np.save(out_dir / f"{stem}_{i:06d}.npy", patches[i])


# ----------------- pipeline por cena -----------------
def preprocess_scene(hdr_path: Path, cfg: Dict, out_root: Path):
    proc = cfg["processing"]

    # 1) Ler cubo EMIT
    tprint(f"Carregando: {hdr_path.name}")
    cube, meta = read_emit_l1b_envi(hdr_path)
    stem = hdr_path.stem  # <<< definir cedo (usado em vários lugares)

    # 2) Máscara de válidos
    ignore_val = meta.get("data_ignore_value", None)
    valid = valid_mask_from_cube(cube, ignore_val=ignore_val, eps=1e-6)

    # 3) (Opcional) clip negativos (EMIT L1B é radiância; negativos geralmente = ruído/fora-FOV)
    if bool(proc.get("clip_negative", True)):
        cube = np.where(np.isfinite(cube), np.maximum(cube, 0.0), cube)

    # 4) Selecionar faixa SWIR para CH4 (default 2122–2488 nm)
    wav = meta.get("wavelength_nm", np.array([]))
    swir_range = proc.get("swir_range_nm", [2122.0, 2488.0])
    cube, kept_swir_idx = select_bands_by_nm(cube, wav, float(swir_range[0]), float(swir_range[1]))
    wav_swir = wav[kept_swir_idx] if wav.size else np.array([])

    # 5) Remover bandas ruins (NaN demais / variância ~0)
    cube, kept_good_idx = drop_bad_bands(
        cube, valid, max_nan_ratio=float(proc.get("max_nan_ratio", 0.4)),
        min_variance=float(proc.get("min_variance", 1e-8))
    )
    wav_keep = wav_swir[kept_good_idx] if wav_swir.size else np.array([])

    # 6) Normalização robusta 0..1 por banda (usando apenas válidos)
    cube_01 = robust_minmax_01(
        cube, valid,
        q_low=float(proc.get("q_low", 1.0)),
        q_high=float(proc.get("q_high", 99.0))
    )

    # 7) Gerar patches com overlap
    patch = int(proc.get("patch_size", 128))
    stride = int(proc.get("stride", patch // 2))  # overlap 50% por padrão
    min_valid_frac = float(proc.get("min_valid_fraction", 0.50))

    tprint(f"Gerando patches: size={patch}, stride={stride}, min_valid={min_valid_frac:.2f}")
    idxs, patches = gen_patches(cube_01, valid, patch, stride, min_valid_frac)

    # 7.1) Gerar e salvar máscaras por patch (para diagnósticos/treino)
    mask_dir = Path(cfg["paths"]["patches_pca_masks"] if proc.get("apply_pca", False)
                    else cfg["paths"]["patches_raw_masks"])
    mask_dir.mkdir(parents=True, exist_ok=True)

    mask_patches = []
    valid_fracs = []
    zero_fracs  = []

    for (y, x), p in zip(idxs, patches):
        m = valid[y:y+patch, x:x+patch]
        mask_patches.append(m.astype(np.uint8))
        valid_fracs.append(float(m.mean()))
        zero_fracs.append(float((p == 0.0).mean()))

    for i, m in enumerate(mask_patches):
        np.save(mask_dir / f"{stem}_{i:06d}.npy", m)

    tprint(f"Patches aprovados: {len(idxs)}")

    # 8) Salvar patches (.npy) e manifesto (.csv)
    if bool(proc.get("apply_pca", False)):
        img_dir = Path(cfg["paths"]["patches_pca_images"])
        run_tag = "pca"
    else:
        img_dir = Path(cfg["paths"]["patches_raw_images"])
        run_tag = "raw"
    img_dir.mkdir(parents=True, exist_ok=True)

    save_patches_npy(patches, img_dir, stem)

    # manifesto
    import csv
    man_path = out_root / f"{stem}_{run_tag}_manifest.csv"
    with man_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "scene","y","x","patch_size","stride","n_bands","kept_bands_nm",
            "valid_frac","zero_frac"
        ])
        kept_nm = ";".join([f"{v:.1f}" for v in wav_keep]) if wav_keep.size else ""
        for (y, x), vf, zf in zip(idxs, valid_fracs, zero_fracs):
            w.writerow([stem, y, x, patch, stride, patches.shape[-1], kept_nm, f"{vf:.3f}", f"{zf:.3f}"])

    # estatísticas da cena
    stats = {
        "scene": stem,
        "shape_hw_b": [int(cube.shape[0]), int(cube.shape[1]), int(cube.shape[2])],
        "n_patches": int(len(idxs)),
        "valid_frac_scene": float(np.mean(valid)),
        "kept_bands_nm": (wav_keep.tolist() if wav_keep.size else []),
        "swir_range_nm": swir_range,
        "ignore_val": (None if ignore_val is None else float(ignore_val)),
    }
    (out_root / f"{stem}_{run_tag}_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


def find_envi_headers(raw_dir: Path) -> List[Path]:
    hdrs = list(raw_dir.rglob("*.hdr"))
    good = []
    for h in hdrs:
        # pular LUTs/arquivos auxiliares
        if h.name.lower() in {"ch4.hdr"}:
            continue
        # garantir que exista bin correspondente
        for suf in (".img", ".dat", ".raw", ".bsq", ".bil", ".bip"):
            if (h.with_suffix(suf)).exists():
                good.append(h)
                break
    return sorted(good)


def main(raw_hdr: Optional[str], config_path: Optional[str]):
    cfg = load_config(config_path)
    out_root = Path(cfg["paths"]["output_dir"])  # para manifestos/estatísticas

    if raw_hdr is not None:
        scenes = [Path(raw_hdr)]
    else:
        scenes = find_envi_headers(Path(cfg["paths"]["raw_data_dir"]))
        tprint(f"Encontradas {len(scenes)} cenas .hdr em {cfg['paths']['raw_data_dir']}")

    it = tqdm(scenes, desc="Cenas") if _HAVE_TQDM else scenes
    for hdr in it:
        try:
            preprocess_scene(Path(hdr), cfg, out_root)
        except Exception as e:
            tprint(f"[WARN] Falhou {hdr}: {e}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--hdr", type=str, default=None, help="Opcional: processa apenas essa cena .hdr")
    args = ap.parse_args()
    main(raw_hdr=args.hdr, config_path=args.config)
