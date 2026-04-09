"""
plot_utils.py – Funkcje pomocnicze do tworzenia wizualizacji.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

FIGURES_DIR = Path(__file__).parent.parent / "reports" / "figures"


def save_fig(filename: str, dpi: int = 150) -> None:
    """Zapisuje aktualny wykres do katalogu reports/figures/."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Zapisano wykres: {path}")


def plot_correlation_heatmap(df: pd.DataFrame, title: str = "Macierz korelacji") -> None:
    """Rysuje heatmapę korelacji dla zmiennych numerycznych."""
    corr = df.select_dtypes(include="number").corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title(title)
    plt.tight_layout()


def plot_distributions(df: pd.DataFrame, columns: list) -> None:
    """Rysuje histogramy dla wybranych kolumn."""
    n = len(columns)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, columns):
        df[col].dropna().plot(kind="hist", bins=30, ax=ax, edgecolor="black")
        ax.set_title(col)
        ax.set_xlabel(col)
    plt.tight_layout()

