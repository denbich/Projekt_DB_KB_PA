# Instrukcja replikacji wyników

## Wymagania wstępne

- Python 3.10 lub nowszy
- Git

## Kroki replikacji

### 1. Sklonuj repozytorium

```bash
git clone https://github.com/<nazwa-organizacji>/Projekt_DB_KB_PA.git
cd Projekt_DB_KB_PA
```

### 2. Utwórz środowisko wirtualne i zainstaluj zależności

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Pobierz dane

Dane źródłowe należy pobrać ręcznie z linku podanego w pliku `data/raw/ZRODLO.md`
i umieścić je w katalogu `data/raw/`.

### 4. Uruchom pełną analizę

```bash
python replication/run_all.py
```

Lub uruchom kolejno notebooki Jupyter:

```bash
jupyter lab
```

Kolejność notebooków:
1. `notebooks/01_EDA.ipynb`
2. `notebooks/02_preprocessing.ipynb`
3. `notebooks/03_modeling.ipynb`
4. `notebooks/04_visualization.ipynb`

### 5. Wyniki

Wyniki (wykresy, tabele) zostaną zapisane w katalogu `reports/figures/`.

