# src/io.py

import os
import numpy as np
import spectral.io.envi as envi
from spectral import open_image
import xarray as xr
import rasterio
from rasterio.transform import from_origin

def load_envi_image(hdr_path, return_metadata=False):
    """
    Carrega imagem hiperespectral EMIT em formato ENVI (.hdr + .img)
    """
    img = envi.open(hdr_path)
    image = img.load().astype(np.float32)
    if return_metadata:
        return image, img.metadata
    return image

def load_netcdf_variable(nc_path, variable_name):
    """
    Carrega uma variável específica de um arquivo .nc (NetCDF)
    Ex: radiância, latitude, longitude
    """
    ds = xr.open_dataset(nc_path)
    return ds[variable_name].values

def save_npy(array, save_path):
    """
    Salva array numpy no disco
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, array)

def load_npy(load_path):
    """
    Carrega array numpy do disco
    """
    return np.load(load_path)

def save_patch_as_tif(patch, save_path, transform=None, crs="EPSG:4326"):
    """
    Salva patch (H x W x B) como GeoTIFF multibanda
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if patch.ndim == 3:
        patch = np.moveaxis(patch, -1, 0)  # (B, H, W)
    count, height, width = patch.shape
    with rasterio.open(
        save_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=count,
        dtype=patch.dtype,
        crs=crs,
        transform=transform or from_origin(0, 0, 1, 1),
    ) as dst:
        dst.write(patch)

def read_tif(tif_path):
    """
    Lê um arquivo GeoTIFF e retorna como array numpy (H x W x B)
    """
    with rasterio.open(tif_path) as src:
        array = src.read()  # (B, H, W)
        array = np.moveaxis(array, 0, -1)  # (H, W, B)
    return array
