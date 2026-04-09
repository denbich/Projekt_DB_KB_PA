"""
run_all.py – Skrypt uruchamiający całą analizę w kolejności.
Wymaga zainstalowanych zależności z requirements.txt oraz pobranych danych do data/raw/.
"""

import subprocess
import sys
from pathlib import Path

NOTEBOOKS = [
    "notebooks/01_EDA.ipynb",
    "notebooks/02_preprocessing.ipynb",
    "notebooks/03_modeling.ipynb",
    "notebooks/04_visualization.ipynb",
]

def run_notebook(path: str):
    print(f"\n Uruchamianie: {path}")
    result = subprocess.run(
        [
            sys.executable, "-m", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--inplace",
            path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  Błąd:\n{result.stderr}")
        sys.exit(result.returncode)
    print(f"  Zakończono: {path}")

if __name__ == "__main__":
    root = Path(__file__).parent.parent
    for nb in NOTEBOOKS:
        run_notebook(str(root / nb))
    print("\nCała analiza zakończona pomyślnie.")

