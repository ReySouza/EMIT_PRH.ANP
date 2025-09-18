# src/paths.py
import os
import yaml

class PathManager:
    """
    Gerencia leitura do config.yaml e garante a existência das pastas.
    Aceita um caminho customizado no construtor; se não for passado, usa
    o caminho padrão do projeto.
    """
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = "C:/Users/251874/Desktop/TCC_EMIT_PRH/configs/config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        p = self.config["paths"]
        self.raw_data_dir       = p["raw_data_dir"]
        self.patches_raw_images = p["patches_raw_images"]
        self.patches_raw_masks  = p["patches_raw_masks"]
        self.patches_pca_images = p["patches_pca_images"]
        self.patches_pca_masks  = p["patches_pca_masks"]
        self.output_dir         = p["output_dir"]
        self.model_dir          = p["model_dir"]

        self.ensure_dirs()

    def ensure_dirs(self):
        for path in [
            self.patches_raw_images,
            self.patches_raw_masks,
            self.patches_pca_images,
            self.patches_pca_masks,
            self.output_dir,
            self.model_dir,
        ]:
            os.makedirs(path, exist_ok=True)

    def get_config(self):
        return self.config
