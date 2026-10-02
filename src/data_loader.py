from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

def load_raw_data():
    train = pd.read_csv(RAW_DIR / "fraudTrain.csv")
    test = pd.read_csv(RAW_DIR / "fraudTest.csv")
    return train, test

def load_processed_data():
    train = pd.read_csv(PROCESSED_DIR / "train_processed.csv")
    test = pd.read_csv(PROCESSED_DIR / "test_processed.csv")
    return train, test