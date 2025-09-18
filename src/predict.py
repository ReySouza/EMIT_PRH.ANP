# src/predict.py
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import rasterio
from rasterio.transform import Affine
import matplotlib.pyplot as plt

# U-Net (arquitetura atual do projeto)
from unet import build_unet

# ============================ Utils & IO ====================================

def tprint(msg: str) -> None:
    print(f"[predict] {msg}")

def load_config(cfg_path: Path) -> Dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def find_stats_and_manifest(outputs_dir: Path, scene_hint: Optional[str]=None):
    """Retorna (stats_path, manifest_path, stats_dict) do PREPROCESS.
    Busca *_raw_stats.json ou *_pca_stats.json e o respectivo manifest.
    """
    def _pick(p: Path) -> bool:
        return p.name.endswith("_raw_stats.json") or p.name.endswith("_pca_stats.json")

    if scene_hint:
        for tag in ("raw", "pca"):
            sp = outputs_dir / f"{scene_hint}_{tag}_stats.json"
            mp = outputs_dir / f"{scene_hint}_{tag}_manifest.csv"
            if sp.exists() and mp.exists():
                with open(sp, "r", encoding="utf-8") as f: jd = json.load(f)
                return sp, mp, jd
        tprint(f"[WARN] stats/manifest para '{scene_hint}' não encontrados; usando o mais recente.")

    stats = sorted([p for p in outputs_dir.glob("*_stats.json") if _pick(p)], key=lambda p: p.stat().st_mtime)
    if not stats:
        raise FileNotFoundError(f"Nenhum *_raw_stats.json ou *_pca_stats.json em {outputs_dir}")
    sp = stats[-1]; stem = sp.name.replace("_stats.json", "")
    mp = outputs_dir / f"{stem}_manifest.csv"
    if not mp.exists():
        raise FileNotFoundError(f"Manifesto não encontrado para {sp.name}")
    with open(sp, "r", encoding="utf-8") as f: jd = json.load(f)
    return sp, mp, jd

def read_manifest_rows(manifest_csv: Path) -> List[dict]:
    rows = []
    with manifest_csv.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r: rows.append(row)
    if not rows: raise RuntimeError(f"Manifesto vazio: {manifest_csv}")
    return rows

def zscore_patch(arr: np.ndarray, eps: float=1e-6) -> np.ndarray:
    """Z-score canal-a-canal no patch (apenas nas bandas espectrais)."""
    mu = arr.reshape(-1, arr.shape[-1]).mean(axis=0)
    sd = arr.reshape(-1, arr.shape[-1]).std(axis=0)
    sd = np.maximum(sd, eps)
    return ((arr - mu) / sd).astype(np.float32)

def overlay_png(base: Optional[np.ndarray], prob: np.ndarray, mask: np.ndarray,
                thr: float, out_png: Path):
    from scipy import ndimage as ndi
    plt.figure(figsize=(14, 7))

    ax1 = plt.subplot(2, 2, 1)
    im1 = ax1.imshow(prob, vmin=0, vmax=1, cmap="viridis")
    ax1.set_title("U-Net — Probabilidade (0–1)")
    ax1.axis("off"); plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = plt.subplot(2, 2, 2)
    ax2.imshow(mask, cmap="gray")
    ax2.set_title(f"Máscara binária (thr={thr:.2f})")
    ax2.axis("off")

    ax3 = plt.subplot(2, 2, 3)
    if base is not None and np.isfinite(base).any():
        vv = base[np.isfinite(base)]
        lo, hi = np.percentile(vv, [1, 99]) if vv.size else (0.0, 1.0)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.nanmin(vv)), float(np.nanmax(vv))
        im3 = ax3.imshow(base, vmin=lo, vmax=hi, cmap="magma")
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        ax3.set_title("Base (MAG1C) + contorno máscara")
    else:
        im3 = ax3.imshow(prob, vmin=0, vmax=1, cmap="viridis")
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        ax3.set_title("Probabilidade + contorno máscara")
    edge = ndi.binary_dilation(mask.astype(bool), iterations=1) ^ mask.astype(bool)
    ax3.imshow(np.ma.masked_where(~edge, edge), cmap="Reds", alpha=0.5, interpolation="none")
    ax3.axis("off")

    ax4 = plt.subplot(2, 2, 4)
    vals = prob[np.isfinite(prob)].reshape(-1)
    ax4.hist(vals, bins=200, range=(0,1))
    ax4.axvline(thr, color="red", linestyle="--", label=f"thr={thr:.2f}")
    ax4.set_title("Histograma de probabilidade"); ax4.legend()
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()

def find_mag1c_any(mag1c_dir: Path, scene_stem: str):
    """Procura raster do MAG1C (preferência por GeoTIFF); retorna (arr, profile, path)."""
    subs = [mag1c_dir / scene_stem] if (mag1c_dir / scene_stem).exists() else []
    tifs = list(mag1c_dir.glob(f"{scene_stem}*.tif"))
    for sub in subs: tifs += list(sub.rglob("*.tif"))
    for p in sorted(set(tifs)):
        name = p.name.lower()
        if any(k in name for k in ("mag", "mf", "ch4")):
            try:
                with rasterio.open(p) as ds:
                    arr = ds.read(1).astype(np.float32)
                    prof = ds.profile.copy()
                return arr, prof, p
            except Exception:
                continue
    # Fallback: ENVI .hdr/.img
    try:
        import spectral.io.envi as sp_envi
        hdrs = list(mag1c_dir.glob(f"{scene_stem}*.hdr"))
        for p in hdrs:
            img = sp_envi.open(str(p))
            arr = np.asarray(img.open_memmap(interleave='bip')[:, :, 0], dtype=np.float32)
            return arr, None, p
    except Exception:
        pass
    return None, None, None

def write_geotiff(out_path: Path, arr: np.ndarray, ref_profile: Optional[Dict]=None,
                  dtype: str="float32", nodata: Optional[float]=None) -> None:
    H, W = arr.shape
    if ref_profile is not None:
        prof = ref_profile.copy()
        prof.update(count=1, dtype=dtype, compress="deflate")
        if nodata is not None:
            prof.update(nodata=nodata)
        with rasterio.open(out_path, "w", **prof) as ds:
            ds.write(arr.astype(dtype), 1)
    else:
        transform = Affine.translation(0, 0) * Affine.scale(1, 1)
        with rasterio.open(
            out_path, "w", driver="GTiff", height=H, width=W, count=1,
            dtype=dtype, crs=None, transform=transform, compress="deflate",
            nodata=nodata if nodata is not None else None
        ) as ds:
            ds.write(arr.astype(dtype), 1)

# ====================== Compat: Legacy U-Net loader ==========================

class _Block(nn.Module):
    """Bloco conv3x3 + BN + ReLU + conv3x3 + BN + ReLU com nome 'net' (p/ bater com checkpoint legado)."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),   # -> keys ...net.0.*
            nn.BatchNorm2d(out_ch),                                # -> keys ...net.1.*
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),   # -> keys ...net.3.*
            nn.BatchNorm2d(out_ch),                                # -> keys ...net.4.*
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class LegacyUNet(nn.Module):
    """
    Variante 4-down/3-up com nomes:
    - down1/down2/down3/down4 (cada um .net.*)
    - up3/up2/up1 (cada um .net.*)
    - outc (Conv1x1)
    Compatível com checkpoints com chaves 'down1.net.0.weight', etc.
    """
    def __init__(self, in_channels: int, out_channels: int = 1, base_channels: int = 64):
        super().__init__()
        b = base_channels
        self.pool = nn.MaxPool2d(2)
        # Encoder (4 níveis)
        self.down1 = _Block(in_channels, b)       # 128 -> 128
        self.down2 = _Block(b, 2*b)               # 64  -> 64
        self.down3 = _Block(2*b, 4*b)             # 32  -> 32
        self.down4 = _Block(4*b, 8*b)             # 16  -> 16 (bottleneck)
        # Decoder (3 níveis)
        self.up3 = _Block(8*b + 4*b, 4*b)         # 16->32 (cat com skip de 4b)
        self.up2 = _Block(4*b + 2*b, 2*b)         # 32->64
        self.up1 = _Block(2*b + 1*b, 1*b)         # 64->128
        self.outc = nn.Conv2d(b, out_channels, kernel_size=1)

    def forward(self, x):
        c1 = self.down1(x); x = self.pool(c1)     # 128->64
        c2 = self.down2(x); x = self.pool(c2)     # 64->32
        c3 = self.down3(x); x = self.pool(c3)     # 32->16
        x  = self.down4(x)                        # 16
        x  = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x  = torch.cat([x, c3], dim=1)
        x  = self.up3(x)
        x  = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x  = torch.cat([x, c2], dim=1)
        x  = self.up2(x)
        x  = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x  = torch.cat([x, c1], dim=1)
        x  = self.up1(x)
        return self.outc(x)

def is_legacy_state(state: Dict[str, torch.Tensor]) -> bool:
    for k in state.keys():
        if k.startswith("down1.net.0.weight") or k.startswith("down1.net.0.running_mean"):
            return True
    return "down1.net.0.weight" in state

def build_model_from_checkpoint(state: Dict[str, torch.Tensor],
                                in_channels_cfg: Optional[int],
                                B_from_stats: int,
                                out_channels_cfg: int,
                                base_channels_cfg: int,
                                dropout_cfg: float):
    """
    Decide qual arquitetura instanciar com base nas chaves do checkpoint.
    - Se legado: deduz in_channels/base_channels do próprio state_dict e cria LegacyUNet.
    - Caso contrário: devolve a U-Net atual (build_unet) usando config/stats.
    """
    if is_legacy_state(state):
        # Deduz in_ch do primeiro conv e b do último 1x1
        w0 = state.get("down1.net.0.weight", None)
        outc_w = state.get("outc.weight", None)
        if w0 is None or outc_w is None:
            raise RuntimeError("Checkpoint legado sem chaves esperadas ('down1.net.0.weight' / 'outc.weight').")

        in_ch_ckpt = int(w0.shape[1])
        b_ckpt     = int(outc_w.shape[1])  # canais que entram no 1x1 final

        if in_channels_cfg is not None and in_channels_cfg != in_ch_ckpt:
            tprint(f"[WARN] in_channels do YAML ({in_channels_cfg}) difere do checkpoint ({in_ch_ckpt}) — usando {in_ch_ckpt}.")
        if B_from_stats != in_ch_ckpt:
            tprint(f"[WARN] B do stats ({B_from_stats}) difere do checkpoint ({in_ch_ckpt}) — verifique preprocess/patches.")

        tprint(f"Detectado checkpoint LEGADO: in_ch={in_ch_ckpt}, base_channels={b_ckpt}")
        model = LegacyUNet(in_channels=in_ch_ckpt, out_channels=out_channels_cfg, base_channels=b_ckpt)
        arch = "legacy"
    else:
        # U-Net nova (enc*/dec*)
        in_ch = int(in_channels_cfg) if (in_channels_cfg is not None) else int(B_from_stats)
        model = build_unet(in_channels=in_ch,
                           out_channels=int(out_channels_cfg),
                           base_channels=int(base_channels_cfg),
                           dropout=float(dropout_cfg))
        arch = "current"
    return model, arch

# ========================== Inferência por mosaico ===========================

def infer_scene_mosaic(model: torch.nn.Module,
                       device: torch.device,
                       patch_dir: Path,
                       manifest_rows: List[dict],
                       scene_shape_hw: Tuple[int,int],
                       batch_size: int = 8,
                       scene_stem: Optional[str]=None,
                       include_mag1c: bool=False,
                       mag_patch_dir: Optional[Path]=None) -> Tuple[np.ndarray, np.ndarray]:
    H, W = scene_shape_hw
    model.eval()
    logits_sum = np.zeros((H, W), dtype=np.float32)
    counts     = np.zeros((H, W), dtype=np.float32)

    batch_patches, batch_coords = [], []

    def flush():
        nonlocal batch_patches, batch_coords, logits_sum, counts
        if not batch_patches: return
        x = np.stack(batch_patches, axis=0)                  # (B,H,W,C)
        x = np.moveaxis(x, -1, 1).astype(np.float32)         # (B,C,H,W)
        xt = torch.from_numpy(x).to(device)
        with torch.no_grad():
            ylogit = model(xt).squeeze(1).detach().cpu().numpy()  # (B,H,W)
        for (yy, xx), ylg in zip(batch_coords, ylogit):
            h, w = ylg.shape
            logits_sum[yy:yy+h, xx:xx+w] += ylg.astype(np.float32)
            counts[yy:yy+h, xx:xx+w]     += 1.0
        batch_patches.clear(); batch_coords.clear()

    for i, row in enumerate(manifest_rows):
        y, x = int(row["y"]), int(row["x"])
        # 1) resolver o caminho do patch de imagem
        if scene_stem is not None:
            patch_path = patch_dir / f"{scene_stem}_{i:06d}.npy"
            if not patch_path.exists():
                cands = list(patch_dir.glob(f"*_{i:06d}.npy"))
                if not cands:
                    raise FileNotFoundError(f"Patch não encontrado idx {i:06d} em {patch_dir}")
                patch_path = cands[0]
        else:
            cands = list(patch_dir.glob(f"*_{i:06d}.npy"))
            if not cands:
                raise FileNotFoundError(f"Patch não encontrado idx {i:06d} em {patch_dir}")
            patch_path = cands[0]
        # 2) carregar SEMPRE o patch depois de resolver patch_path (corrige UnboundLocalError)
        p = np.load(patch_path).astype(np.float32)  # (h,w,Cbands)
        # 3) z-score apenas nas bandas espectrais
        pz = zscore_patch(p)
        # 4) concatenar MAG1C como +1 canal, se habilitado
        if include_mag1c:
            m = None
            if mag_patch_dir is not None:
                cand = (mag_patch_dir / f"{scene_stem}_{i:06d}.npy") if scene_stem is not None else None
                if cand is not None and cand.exists():
                    m = np.load(cand).astype(np.float32)
                if m is None:
                    cands = list(mag_patch_dir.glob(f"*_{i:06d}.npy"))
                    if cands:
                        m = np.load(cands[0]).astype(np.float32)
            if m is None:
                m = np.zeros(p.shape[:2], dtype=np.float32)
            arr = np.concatenate([pz, m[..., None]], axis=-1)
        else:
            arr = pz
        batch_patches.append(arr); batch_coords.append((y, x))
        if len(batch_patches) >= batch_size:
            flush()
    flush()
    return logits_sum, counts

# ================================= Main =====================================

def main():
    ap = argparse.ArgumentParser(description="Predição U-Net (mosaico por patches)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--scene", default=None)
    ap.add_argument("--thr", type=float, default=None)
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--outdir", type=str, default=None,
                    help="Diretório de saída (se ausente, usa paths.prediction_dir ou paths.output_dir)")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    paths = cfg["paths"]; proc = cfg["processing"]; mdl = cfg["model"]; trn = cfg.get("training", {})

    # Diretório de saída (novo: --outdir tem prioridade)
    out_base = args.outdir or paths.get("prediction_dir") or paths.get("output_dir", ".")
    outputs_dir = Path(out_base); outputs_dir.mkdir(parents=True, exist_ok=True)

    # Stats/manifest da cena
    stats_path, manifest_path, stats = find_stats_and_manifest(Path(cfg["paths"]["output_dir"]), scene_hint=args.scene)
    scene_stem = stats["scene"]; H, W, B = stats["shape_hw_b"]
    tprint(f"Scene: {scene_stem} ({H}x{W}x{B})")

    # Device
    dev_str = str(trn.get("device", "cpu"))
    device = torch.device(dev_str if (torch.cuda.is_available() and "cuda" in dev_str) else "cpu")
    tprint(f"Device: {device}")

    # Checkpoint
    ckpt_path = Path(args.ckpt) if args.ckpt else None
    if ckpt_path is None:
        model_dir = Path(paths["model_dir"])
        cands = []
        best = model_dir / "unet_emit_best.pt"
        if best.exists(): cands.append(best)
        cands += sorted(model_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not cands:
            raise FileNotFoundError(f"Nenhum checkpoint .pt em {model_dir}")
        ckpt_path = cands[0]
    tprint(f"Checkpoint: {ckpt_path}")

    # Carrega state_dict (ou wrapper)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = None
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt: state = ckpt["state_dict"]
        elif "model" in ckpt:   state = ckpt["model"]
    if state is None: state = ckpt  # assume state_dict plano
    if not isinstance(state, dict):
        raise RuntimeError("Checkpoint inesperado: não é um state_dict nem contém 'model'/'state_dict'.")

    # Construir modelo (preferindo in_channels salvo no checkpoint)
    ckpt_in = ckpt.get("in_channels", None) if isinstance(ckpt, dict) else None
    val_in = ckpt_in if (ckpt_in is not None) else mdl.get("in_channels", None)

    out_ch  = int(mdl.get("out_channels", 1))
    base_c  = int(mdl.get("base_channels", 32))
    dropout = float(mdl.get("dropout", 0.0))

    model, arch = build_model_from_checkpoint(state, val_in, B, out_ch, base_c, dropout)
    model.to(device)

    try:
        model.load_state_dict(state, strict=True)
        tprint(f"Checkpoint carregado (arquitetura: {arch}, strict=True).")
    except Exception as e:
        tprint(f"[WARN] Falha strict=True ({e}); tentando carregamento por shape (strict=False).")
        model_dict = model.state_dict()
        filtered = {k: v for k, v in state.items() if (k in model_dict and v.shape == model_dict[k].shape)}
        missed = [k for k in model_dict.keys() if k not in filtered]
        model_dict.update(filtered)
        model.load_state_dict(model_dict, strict=False)
        tprint(f"Carregado com {len(filtered)}/{len(model_dict)} tensores compatíveis. Faltaram: {len(missed)}")

    model.eval()

    # Leitura de manifest e diretório de patches
    rows = read_manifest_rows(manifest_path)
    apply_pca = bool(proc.get("apply_pca", False))
    patch_dir = Path(paths["patches_pca_images"] if apply_pca else paths["patches_raw_images"])

    # Inferência (mosaico)
    logits_sum, counts = infer_scene_mosaic(
        model, device, patch_dir, rows, (H, W),
        batch_size=int(args.batch), scene_stem=scene_stem,
        include_mag1c=bool(proc.get("include_mag1c", False)),
        mag_patch_dir=Path(paths.get("patches_mag1c", "."))
    )
    safe_counts = np.where(counts > 0, counts, 1.0).astype(np.float32)
    avg_logit = logits_sum / safe_counts
    prob = 1.0 / (1.0 + np.exp(-avg_logit))

    # Limiar
    thr = args.thr
    if thr is None:
        # tenta achar em outputs oficiais do projeto (onde o evaluate salva)
        eval_metrics = Path(cfg["paths"]["output_dir"]) / "eval_val_metrics.json"
        if eval_metrics.exists():
            try:
                md = json.loads(eval_metrics.read_text(encoding="utf-8"))
                thr = float(md.get("best_thr_F1", md.get("best_threshold_f1", np.nan)))
            except Exception:
                thr = None
    if thr is None or not np.isfinite(thr): thr = 0.5

    binm = (prob >= float(thr)).astype(np.uint8)

    # Georreferência a partir do MAG1C (se houver)
    mag_arr, mag_prof, mag_src = find_mag1c_any(Path(paths["mag1c_dir"]), scene_stem)
    if mag_prof is None:
        tprint("[WARN] Sem georreferência do MAG1C; salvando GeoTIFFs sem CRS/transform.")

    # Saídas
    prob_tif = outputs_dir / f"{scene_stem}_unet_prob.tif"
    mask_tif = outputs_dir / f"{scene_stem}_unet_mask.tif"
    write_geotiff(prob_tif, prob.astype(np.float32), ref_profile=mag_prof, dtype="float32", nodata=np.nan)
    write_geotiff(mask_tif, binm.astype(np.uint8),   ref_profile=mag_prof, dtype="uint8",   nodata=0)
    tprint(f"prob: {prob_tif}")
    tprint(f"mask: {mask_tif}")

    overlay_png(mag_arr if mag_arr is not None else avg_logit, prob, binm, float(thr),
                out_png=outputs_dir / f"{scene_stem}_unet_overlay.png")
    tprint(f"overlay: {outputs_dir / f'{scene_stem}_unet_overlay.png'}")

if __name__ == "__main__":
    main()
