"""
data_utils.py – Funkcje pomocnicze do wczytywania i czyszczenia danych.
"""

import pandas as pd
from pathlib import Path

DATA_RAW = Path(__file__).parent.parent / "data" / "raw"
DATA_PROCESSED = Path(__file__).parent.parent / "data" / "processed"


def load_raw(filename: str) -> pd.DataFrame:
    """Wczytuje plik CSV z katalogu data/raw/."""
    path = DATA_RAW / filename
    return pd.read_csv(path)


def save_processed(df: pd.DataFrame, filename: str) -> None:
    """Zapisuje DataFrame do katalogu data/processed/."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PROCESSED / filename, index=False)
    print(f"Zapisano: {DATA_PROCESSED / filename}")


def basic_info(df: pd.DataFrame) -> None:
    """Wyświetla podstawowe informacje o DataFrame."""
    print(f"Kształt: {df.shape}")
    print(f"\nTypy danych:\n{df.dtypes}")
    print(f"\nBrakujące wartości:\n{df.isnull().sum()}")
    print(f"\nPierwsze wiersze:\n{df.head()}")

