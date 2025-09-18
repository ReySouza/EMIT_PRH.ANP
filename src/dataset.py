from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset


def _list_npy(dir_: Path) -> List[Path]:
    return sorted([p for p in dir_.glob("*.npy") if p.is_file()])


def _bands_first(x: np.ndarray) -> np.ndarray:
    # (H,W,B) -> (B,H,W) ; (H,W) -> (1,H,W)
    if x.ndim == 2:
        x = x[..., None]
    return np.transpose(x, (2, 0, 1))


class EMITPatchDataset(Dataset):
    """
    Loads (image, mask) patch pairs stored as .npy arrays.
    - image: (H,W,B) or (B,H,W). Converted internally to (C,H,W) float32.
    - mask:  (H,W) binary/float; converted to (1,H,W) float32 in {0,1}.
    Optional: include MAG1C single-channel patch from `mag1c_dir` with same filename.
    """
    def __init__(self,
                 img_dir: Path,
                 mask_dir: Path,
                 include_mag1c: bool = False,
                 mag1c_dir: Optional[Path] = None,
                 normalization: bool = True,
                 min_valid_fraction: float = 0.6,
                 seed: int = 42):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.include_mag1c = include_mag1c
        self.mag1c_dir = Path(mag1c_dir) if include_mag1c and mag1c_dir is not None else None
        self.normalization = normalization
        self.min_valid_fraction = float(min_valid_fraction)
        self.seed = seed

        imgs = _list_npy(self.img_dir)
        masks = set(_list_npy(self.mask_dir))
        if self.include_mag1c and self.mag1c_dir is not None:
            mags = set(_list_npy(self.mag1c_dir))
        else:
            mags = None

        pairs: List[Tuple[Path, Path, Optional[Path]]] = []
        for ip in imgs:
            mp = self.mask_dir / ip.name
            if mp in masks:
                gp = (self.mag1c_dir / ip.name) if (mags is not None and (self.mag1c_dir / ip.name) in mags) else None
                if self.include_mag1c and mags is not None and gp is None:
                    # If MAG1C requested but missing for this patch, skip
                    continue
                # validity check (NaN ratio)
                img = np.load(ip, allow_pickle=False)
                if img.ndim == 3:
                    valid = np.isfinite(img).sum() / img.size
                else:
                    # seldom
                    valid = np.isfinite(img).sum() / float(img.size)
                if valid >= self.min_valid_fraction:
                    pairs.append((ip, mp, gp))

        if len(pairs) == 0:
            raise RuntimeError(f"No pairs in {self.img_dir} vs {self.mask_dir} (include_mag1c={self.include_mag1c})")

        self.pairs = pairs

        # Pre-compute positive fraction per patch for sampler
        self.pos_frac: List[float] = []
        self.pos_flags: List[bool] = []
        for _, mp, _ in self.pairs:
            m = np.load(mp, allow_pickle=False).astype(np.float32)
            if m.ndim == 3:
                if m.shape[0] == 1:
                    m = m[0]
                elif m.shape[-1] == 1:
                    m = m[...,0]
                else:
                    m = m[...,0]
            # binarize > 0
            mbin = (m > 0).astype(np.float32)
            frac = float(mbin.mean()) if mbin.size > 0 else 0.0
            self.pos_frac.append(frac)
            self.pos_flags.append(frac > 0.0)

        # Determine input channels for information purposes
        # Load one sample
        s_img = np.load(self.pairs[0][0], allow_pickle=False)
        if s_img.ndim == 3:
            b = s_img.shape[-1]  # (H,W,B)
        elif s_img.ndim == 2:
            b = 1
        else:
            # (B,H,W)
            b = s_img.shape[0]
        if self.include_mag1c:
            b += 1
        self.n_channels = int(b)

    def __len__(self):
        return len(self.pairs)

    def _normalize_bands_only(self, bands: np.ndarray) -> np.ndarray:
        # bands: (B,H,W) spectral bands
        B, H, W = bands.shape
        x = bands.reshape(B, -1)
        mean = np.nanmean(x, axis=1, keepdims=True)
        std = np.nanstd(x, axis=1, keepdims=True) + 1e-6
        xn = (x - mean) / std
        return xn.reshape(B, H, W)

    def __getitem__(self, idx: int):
        ip, mp, gp = self.pairs[idx]

        img = np.load(ip, allow_pickle=False)
        if img.ndim == 2:
            img = img[..., None]  # (H,W,1)
        if img.shape[0] != img.ndim - 1:  # assume (H,W,B), convert
            img = np.transpose(img, (2, 0, 1))  # -> (B,H,W)
        else:
            # already (B,H,W)
            pass
        img = img.astype(np.float32)

        if self.include_mag1c:
            g = np.load(gp, allow_pickle=False).astype(np.float32)
            if g.ndim == 3:
                # assume single-channel in last or first dim
                if g.shape[0] == 1:
                    g = g[0]
                elif g.shape[-1] == 1:
                    g = g[..., 0]
                else:
                    g = g[..., 0]
            # normalize spectral bands only if enabled
            if self.normalization:
                img = self._normalize_bands_only(img)
            g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
            g = g[None, ...]  # (1,H,W)
            x = np.concatenate([img, g], axis=0).astype(np.float32)
        else:
            if self.normalization:
                img = self._normalize_bands_only(img)
            x = img.astype(np.float32)

        m = np.load(mp, allow_pickle=False).astype(np.float32)
        if m.ndim == 3:
            if m.shape[0] == 1:
                m = m[0]
            elif m.shape[-1] == 1:
                m = m[..., 0]
            else:
                m = m[..., 0]
        m = (m > 0).astype(np.float32)  # binary
        m = m[None, ...]  # (1,H,W)

        # Replace NaNs/Infs
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        return torch.from_numpy(x), torch.from_numpy(m), str(ip.name)
