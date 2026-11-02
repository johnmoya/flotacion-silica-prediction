"""Rutas y constantes compartidas por notebooks y módulos."""
from pathlib import Path

SEED = 42

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"

KAGGLE_DATASET = "edumagalhaes/quality-prediction-in-a-mining-process"
RAW_CSV = DATA_RAW / "MiningProcess_Flotation_Plant_Database.csv"

TARGET = "% Silica Concentrate"
LEAKY_COLUMNS = ["% Iron Concentrate"]  # mismo ensayo de laboratorio que el target

# Hold-out temporal: todo lo anterior entrena, desde aquí se evalúa (~22 % de las horas válidas)
TEST_START = "2017-08-01"
