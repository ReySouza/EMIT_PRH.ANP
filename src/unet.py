# src/unet.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, p_drop: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.p_drop = float(p_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if self.p_drop > 0: x = F.dropout(x, p=self.p_drop, training=self.training)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        if self.p_drop > 0: x = F.dropout(x, p=self.p_drop, training=self.training)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 1, base_channels: int = 32, dropout: float = 0.0):
        super().__init__()
        b = int(base_channels)
        p = float(dropout)
        # encoder
        self.enc1 = DoubleConv(in_channels, b, p)
        self.enc2 = DoubleConv(b,   2*b, p)
        self.enc3 = DoubleConv(2*b, 4*b, p)
        self.pool = nn.MaxPool2d(2)
        # bottleneck
        self.bott = DoubleConv(4*b, 8*b, p)
        # decoder (bilinear upsample)
        self.dec3 = DoubleConv(8*b + 4*b, 4*b, p)
        self.dec2 = DoubleConv(4*b + 2*b, 2*b, p)
        self.dec1 = DoubleConv(2*b + 1*b, 1*b, p)
        self.outc = nn.Conv2d(b, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = self.enc1(x)
        x  = self.pool(c1)
        c2 = self.enc2(x)
        x  = self.pool(c2)
        c3 = self.enc3(x)
        x  = self.pool(c3)
        x  = self.bott(x)

        x  = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x  = torch.cat([x, c3], dim=1)
        x  = self.dec3(x)
        x  = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x  = torch.cat([x, c2], dim=1)
        x  = self.dec2(x)
        x  = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x  = torch.cat([x, c1], dim=1)
        x  = self.dec1(x)
        return self.outc(x)

# Fábrica compatível com train.py/evaluate.py/predict.py

def build_unet(in_channels: int, out_channels: int = 1, base_channels: int = 32, dropout: float = 0.0) -> UNet:
    """
    Constrói uma U-Net padrão com upsample bilinear.
    Parâmetros
    ---------
    in_channels : int
        Nº de bandas do patch de entrada (detectado pelo Dataset quando None no YAML)
    out_channels : int
        Nº de canais de saída (para segmentação binária, use 1)
    base_channels : int
        Nº de filtros do primeiro bloco (escala por 2 nas camadas mais profundas)
    dropout : float
        Dropout (0.0–0.5). Aplicado após cada ReLU dos blocos convolutivos.
    """
    return UNet(in_channels=in_channels, out_channels=out_channels, base_channels=base_channels, dropout=dropout)

__all__ = ["UNet", "build_unet"]
