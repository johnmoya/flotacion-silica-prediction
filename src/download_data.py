"""Descarga el dataset de Kaggle a data/raw/.

Uso (desde la raíz del repo):
    python -m src.download_data
"""
import shutil
from pathlib import Path

import kagglehub

from src.config import DATA_RAW, KAGGLE_DATASET, RAW_CSV


def download() -> Path:
    if RAW_CSV.exists():
        print(f"Ya existe: {RAW_CSV}")
        return RAW_CSV

    cache_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    csvs = list(cache_dir.rglob("*.csv"))
    if len(csvs) != 1:
        raise RuntimeError(f"Se esperaba 1 CSV en {cache_dir}, se encontraron: {csvs}")

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    shutil.copy2(csvs[0], RAW_CSV)
    print(f"Copiado a: {RAW_CSV} ({RAW_CSV.stat().st_size / 1e6:.1f} MB)")
    return RAW_CSV


if __name__ == "__main__":
    download()
