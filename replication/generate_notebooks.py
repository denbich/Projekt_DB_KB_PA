"""
Skrypt generujący wszystkie 4 notebooki Jupyter jako poprawne pliki .ipynb
"""
import json, os

NB_DIR = os.path.join(os.path.dirname(__file__), '..', 'notebooks')
os.makedirs(NB_DIR, exist_ok=True)

def nb(cells):
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"}
        },
        "cells": cells
    }

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}

# Wspólny nagłówek ścieżek używany w każdym notebooku
# nbconvert uruchamia kernel z katalogu projektu (CWD = PROJECT_ROOT)
PATHS_SETUP = """from pathlib import Path
import os
PROJECT_ROOT = Path(os.getcwd())
# Jeśli CWD to notebooks/, cofnij się poziom wyżej
if not (PROJECT_ROOT / 'data').exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
DATA_PATH    = str(PROJECT_ROOT / 'data' / 'raw' / 'Car_Prices_Poland_Kaggle.csv')
PROC_PATH    = str(PROJECT_ROOT / 'data' / 'processed') + os.sep
FIGURES_PATH = str(PROJECT_ROOT / 'reports' / 'figures') + os.sep
os.makedirs(PROC_PATH, exist_ok=True)
os.makedirs(FIGURES_PATH, exist_ok=True)
print('PROJECT_ROOT:', PROJECT_ROOT)
print('FIGURES_PATH:', FIGURES_PATH)"""

# ─────────────────────────────────────────────
# 01_EDA.ipynb
# ─────────────────────────────────────────────
nb01 = nb([
    md("# 01 – Eksploracyjna Analiza Danych (EDA)\n**Dataset:** Car Prices Poland (Kaggle)  \n**Cel:** Zapoznanie się z danymi, analiza rozkładów, wykrywanie anomalii i wartości brakujących."),
    md("## 1. Import bibliotek i wczytanie danych"),
    code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.figsize'] = (10, 5)
""" + PATHS_SETUP),
    code("""df = pd.read_csv(DATA_PATH, index_col=0)
print(f'Kształt danych: {df.shape}')
df.head(10)"""),
    md("## 2. Podstawowe informacje o zbiorze danych"),
    code("df.info()"),
    code("df.describe(include='all').T"),
    md("## 3. Brakujące wartości"),
    code("""missing = df.isnull().sum().sort_values(ascending=False)
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Brakujące': missing, '% brakujących': missing_pct})
print(missing_df[missing_df['Brakujące'] > 0])"""),
    md("## 4. Rozkłady zmiennych numerycznych"),
    code("""num_cols = ['year', 'mileage', 'vol_engine', 'price']
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    axes[i].hist(df[col].dropna(), bins=50, edgecolor='black', color='steelblue', alpha=0.8)
    axes[i].set_title(f'Rozkład: {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Liczba ogłoszeń')
plt.suptitle('Rozkłady zmiennych numerycznych', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(FIGURES_PATH + '01_distributions_numeric.png', dpi=150, bbox_inches='tight')
plt.show()"""),
    md("## 5. Rozkład ceny – transformacja logarytmiczna"),
    code("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df['price'].dropna(), bins=80, color='steelblue', edgecolor='black', alpha=0.8)
axes[0].set_title('Rozkład ceny (PLN)')
axes[0].set_xlabel('Cena (PLN)')
axes[1].hist(np.log1p(df['price'].dropna()), bins=80, color='darkorange', edgecolor='black', alpha=0.8)
axes[1].set_title('Rozkład log(cena + 1)')
axes[1].set_xlabel('log(Cena + 1)')
plt.tight_layout()
plt.savefig(FIGURES_PATH + '01_price_distribution.png', dpi=150, bbox_inches='tight')
plt.show()"""),
    md("## 6. Zmienne kategoryczne"),
    code("""cat_cols = ['mark', 'fuel', 'province']
for col in cat_cols:
    print(f'\\n--- {col} --- ({df[col].nunique()} unikalnych wartości)')
    print(df[col].value_counts().head(15))"""),
    code("""top_marks = df['mark'].value_counts().head(15)
plt.figure(figsize=(12, 5))
sns.barplot(x=top_marks.values, y=top_marks.index, palette='Blues_d')
plt.title('Top 15 marek samochodów (liczba ogłoszeń)')
plt.xlabel('Liczba ogłoszeń')
plt.tight_layout()
plt.savefig(FIGURES_PATH + '01_top_marks.png', dpi=150, bbox_inches='tight')
plt.show()"""),
    code("""fuel_counts = df['fuel'].value_counts()
plt.figure(figsize=(8, 5))
sns.barplot(x=fuel_counts.index, y=fuel_counts.values, palette='Set2')
plt.title('Rozkład rodzaju paliwa')
plt.xlabel('Rodzaj paliwa')
plt.ylabel('Liczba ogłoszeń')
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(FIGURES_PATH + '01_fuel_distribution.png', dpi=150, bbox_inches='tight')
plt.show()"""),
    md("## 7. Cena według rodzaju paliwa"),
    code("""plt.figure(figsize=(12, 6))
order = df.groupby('fuel')['price'].median().sort_values(ascending=False).index
sns.boxplot(data=df, x='fuel', y='price', order=order, palette='Set3')
plt.title('Rozkład ceny według rodzaju paliwa')
plt.xlabel('Rodzaj paliwa')
plt.ylabel('Cena (PLN)')
plt.yscale('log')
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(FIGURES_PATH + '01_price_by_fuel.png', dpi=150, bbox_inches='tight')
plt.show()"""),
    md("## 8. Cena według roku produkcji"),
    code("""year_price = df.groupby('year')['price'].median().reset_index()
plt.figure(figsize=(14, 5))
sns.lineplot(data=year_price, x='year', y='price', marker='o', color='steelblue')
plt.title('Mediana ceny według roku produkcji')
plt.xlabel('Rok produkcji')
plt.ylabel('Mediana ceny (PLN)')
plt.tight_layout()
plt.savefig(FIGURES_PATH + '01_price_by_year.png', dpi=150, bbox_inches='tight')
plt.show()"""),
    md("## 9. Cena vs. przebieg"),
    code("""sample = df.dropna(subset=['mileage', 'price']).sample(min(5000, len(df)), random_state=42)
plt.figure(figsize=(10, 6))
plt.scatter(sample['mileage'], sample['price'], alpha=0.2, s=10, color='steelblue')
plt.title('Cena vs. Przebieg')
plt.xlabel('Przebieg (km)')
plt.ylabel('Cena (PLN)')
plt.yscale('log')
plt.tight_layout()
plt.savefig(FIGURES_PATH + '01_price_vs_mileage.png', dpi=150, bbox_inches='tight')
plt.show()"""),
    md("## 10. Heatmapa korelacji zmiennych numerycznych"),
    code("""corr = df[num_cols].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt='.3f', cmap='coolwarm', center=0,
            square=True, linewidths=0.5)
plt.title('Heatmapa korelacji (Pearson)')
plt.tight_layout()
plt.savefig(FIGURES_PATH + '01_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print('\\nKorelacje z ceną:')
print(corr['price'].sort_values(ascending=False))"""),
    md("## 11. Top 15 marek – mediana ceny"),
    code("""top15 = df['mark'].value_counts().head(15).index
mark_price = (df[df['mark'].isin(top15)]
              .groupby('mark')['price']
              .median()
              .sort_values(ascending=False))
plt.figure(figsize=(12, 5))
sns.barplot(x=mark_price.index, y=mark_price.values, palette='viridis')
plt.title('Mediana ceny dla Top 15 marek')
plt.xlabel('Marka')
plt.ylabel('Mediana ceny (PLN)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(FIGURES_PATH + '01_median_price_by_mark.png', dpi=150, bbox_inches='tight')
plt.show()"""),
    md("## 12. Cena według województwa"),
    code("""province_price = (df.groupby('province')['price']
                  .median()
                  .sort_values(ascending=False)
                  .reset_index())
plt.figure(figsize=(14, 5))
sns.barplot(data=province_price, x='province', y='price', palette='coolwarm')
plt.title('Mediana ceny według województwa')
plt.xlabel('Województwo')
plt.ylabel('Mediana ceny (PLN)')
plt.xticks(rotation=60, ha='right')
plt.tight_layout()
plt.savefig(FIGURES_PATH + '01_price_by_province.png', dpi=150, bbox_inches='tight')
plt.show()"""),
    md("""## 13. Wnioski z EDA

- Zbiór danych zawiera ogłoszenia sprzedaży samochodów z całej Polski (styczeń 2022).
- Cena jest **prawostronnie skośna** – transformacja `log` przybliża ją do rozkładu normalnego.
- **Rok produkcji** wykazuje silną dodatnią korelację z ceną; **przebieg** – ujemną.
- Pojemność silnika (`vol_engine`) ma umiarkowaną korelację z ceną.
- Samochody elektryczne i hybrydowe mają wyraźnie wyższe ceny mediany.
- Dane wymagają czyszczenia – patrz notebook `02_preprocessing.ipynb`."""),
])

# ─────────────────────────────────────────────
# 02_preprocessing.ipynb
# ─────────────────────────────────────────────
nb02 = nb([
    md("# 02 – Preprocessing (czyszczenie i inżynieria cech)\n**Cel:** Usunięcie wartości odstających, obsługa braków, kodowanie zmiennych kategorycznych, inżynieria cech, zapis przetworzonego datasetu do `data/processed/`."),
    md("## 1. Import i wczytanie danych"),
    code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid')
""" + PATHS_SETUP),
    code("""df = pd.read_csv(DATA_PATH, index_col=0)
print(f'Wczytano: {df.shape[0]} wierszy, {df.shape[1]} kolumn')
df.head()"""),
    md("## 2. Czyszczenie danych"),
    code("""key_cols = ['year', 'mileage', 'vol_engine', 'fuel', 'mark', 'price']
df_clean = df.dropna(subset=key_cols).copy()
print(f'Po usunięciu braków: {df_clean.shape[0]} wierszy')"""),
    code("""print('Przed filtrowaniem:')
print(df_clean[['year','mileage','vol_engine','price']].describe())
df_clean = df_clean[
    (df_clean['price'] > 500) & (df_clean['price'] < 1_500_000) &
    (df_clean['mileage'] >= 0) & (df_clean['mileage'] < 1_500_000) &
    (df_clean['year'] >= 1990) & (df_clean['year'] <= 2022) &
    (df_clean['vol_engine'] >= 0) & (df_clean['vol_engine'] < 10_000)
]
print(f'\\nPo filtrowaniu: {df_clean.shape[0]} wierszy')"""),
    code("""n_before = len(df_clean)
df_clean = df_clean.drop_duplicates()
print(f'Usunięto duplikatów: {n_before - len(df_clean)}')
print(f'Finalny rozmiar: {df_clean.shape}')"""),
    md("## 3. Normalizacja zmiennych kategorycznych"),
    code("""for col in ['mark', 'model', 'fuel', 'city', 'province', 'generation_name']:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].str.strip().str.lower()
print('Rodzaje paliwa po normalizacji:')
print(df_clean['fuel'].value_counts())"""),
    md("## 4. Inżynieria cech"),
    code("""CURRENT_YEAR = 2022
df_clean['car_age'] = CURRENT_YEAR - df_clean['year']
df_clean['mileage_per_year'] = df_clean['mileage'] / (df_clean['car_age'].clip(lower=1))
df_clean['log_price'] = np.log1p(df_clean['price'])
df_clean['is_electric_hybrid'] = df_clean['fuel'].isin(['electric', 'hybrid']).astype(int)
print(df_clean[['car_age', 'mileage_per_year', 'log_price', 'is_electric_hybrid']].describe())"""),
    md("## 5. Kodowanie zmiennych kategorycznych"),
    code("""mark_median = df_clean.groupby('mark')['price'].median()
df_clean['mark_median_price'] = df_clean['mark'].map(mark_median)
fuel_dummies = pd.get_dummies(df_clean['fuel'], prefix='fuel', drop_first=True)
df_clean = pd.concat([df_clean, fuel_dummies], axis=1)
print(f'Kolumny po kodowaniu: {df_clean.shape[1]}')
df_clean.head(3)"""),
    md("## 6. Wizualizacja po czyszczeniu"),
    code("""fig, axes = plt.subplots(1, 3, figsize=(16, 5))
axes[0].hist(df_clean['price'], bins=60, color='steelblue', edgecolor='black', alpha=0.8)
axes[0].set_title('Cena (po czyszczeniu)')
axes[0].set_xlabel('PLN')
axes[1].hist(df_clean['log_price'], bins=60, color='darkorange', edgecolor='black', alpha=0.8)
axes[1].set_title('log(Cena + 1)')
axes[1].set_xlabel('log PLN')
axes[2].hist(df_clean['car_age'], bins=30, color='seagreen', edgecolor='black', alpha=0.8)
axes[2].set_title('Wiek pojazdu (lata)')
axes[2].set_xlabel('Lata')
plt.tight_layout()
plt.savefig(FIGURES_PATH + '02_clean_distributions.png', dpi=150, bbox_inches='tight')
plt.show()"""),
    md("## 7. Zapis przetworzonego datasetu"),
    code("""df_clean.to_csv(PROC_PATH + 'cars_clean.csv', index=False)
print(f'Zapisano: {PROC_PATH}cars_clean.csv  ({df_clean.shape[0]} wierszy, {df_clean.shape[1]} kolumn)')"""),
])

# ─────────────────────────────────────────────
# 03_modeling.ipynb
# ─────────────────────────────────────────────
nb03 = nb([
    md("# 03 – Modelowanie i ewaluacja\n**Cel:** Testowanie hipotez statystycznych oraz budowa i porównanie modeli predykcyjnych ceny samochodu.\n\n**Pytania badawcze:**\n1. Jakie czynniki mają największy wpływ na cenę?\n2. Jak dobrze można przewidzieć cenę na podstawie parametrów pojazdu?\n3. Czy Random Forest / XGBoost bije regresję liniową?"),
    md("## 1. Import i wczytanie danych"),
    code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    from xgboost import XGBRegressor
    USE_XGB = True
except ImportError:
    USE_XGB = False
    print('XGBoost niedostępny.')
""" + PATHS_SETUP),
    code("""df = pd.read_csv(PROC_PATH + 'cars_clean.csv')
print(f'Wczytano: {df.shape}')
df.head(3)"""),
    md("## 2. Testy statystyczne\n### 2.1 ANOVA – czy rodzaj paliwa różnicuje cenę?"),
    code("""fuel_groups = [group['price'].values for _, group in df.groupby('fuel')]
f_stat, p_anova = stats.f_oneway(*fuel_groups)
print(f'ANOVA F-statystyka: {f_stat:.2f},  p-wartość: {p_anova:.4e}')
if p_anova < 0.05:
    print('→ Odrzucamy H0: istnieją statystycznie istotne różnice cen między grupami paliw (α=0.05)')
else:
    print('→ Brak podstaw do odrzucenia H0.')"""),
    md("### 2.2 Kruskal-Wallis"),
    code("""h_stat, p_kw = stats.kruskal(*fuel_groups)
print(f'Kruskal-Wallis H: {h_stat:.2f},  p-wartość: {p_kw:.4e}')
if p_kw < 0.05:
    print('→ Różnice cen między grupami paliw są statystycznie istotne.')"""),
    md("### 2.3 Test t – diesel vs. benzyna"),
    code("""diesel = df[df['fuel'] == 'diesel']['price']
petrol_keys = [k for k in df['fuel'].unique() if any(x in k for x in ['petrol','benzyna','gasoline','gas'])]
petrol_key = petrol_keys[0] if petrol_keys else None
if petrol_key:
    petrol = df[df['fuel'] == petrol_key]['price']
    t_stat, p_t = stats.ttest_ind(diesel, petrol, equal_var=False)
    print(f'Test t (diesel vs. {petrol_key}): t={t_stat:.3f}, p={p_t:.4e}')
    print(f'Mediana diesel: {diesel.median():,.0f} PLN')
    print(f'Mediana {petrol_key}: {petrol.median():,.0f} PLN')
else:
    print('Dostępne rodzaje paliwa:')
    print(df['fuel'].value_counts())"""),
    md("### 2.4 Korelacja Spearmana"),
    code("""num_features = [c for c in ['year','mileage','vol_engine','car_age','mileage_per_year','mark_median_price'] if c in df.columns]
rows = []
for col in num_features:
    mask = df[col].notna()
    rho, p = stats.spearmanr(df.loc[mask, col], df.loc[mask, 'price'])
    rows.append({'Cecha': col, 'rho': round(rho, 4), 'p-wartość': f'{p:.2e}'})
sp_df = pd.DataFrame(rows).sort_values('rho', key=abs, ascending=False)
print(sp_df.to_string(index=False))"""),
    md("## 3. Przygotowanie danych do modelowania"),
    code("""base_features = ['year','mileage','vol_engine','car_age','mileage_per_year','mark_median_price','is_electric_hybrid']
fuel_cols = [c for c in df.columns if c.startswith('fuel_')]
FEATURES = [c for c in base_features + fuel_cols if c in df.columns]
TARGET = 'log_price'
df_model = df[FEATURES + [TARGET, 'price']].dropna()
print(f'Cechy ({len(FEATURES)}): {FEATURES}')
print(f'Rozmiar zbioru: {df_model.shape}')
X = df_model[FEATURES]
y = df_model[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'Train: {X_train.shape[0]}, Test: {X_test.shape[0]}')"""),
    md("## 4. Trening modeli"),
    code("""def evaluate(name, model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    preds = np.expm1(model.predict(X_te))
    true  = np.expm1(y_te)
    rmse  = np.sqrt(mean_squared_error(true, preds))
    mae   = mean_absolute_error(true, preds)
    r2    = r2_score(true, preds)
    cv_r2 = cross_val_score(model, X_tr, y_tr, cv=5, scoring='r2').mean()
    print(f'{name:30s}  R2={r2:.4f}  RMSE={rmse:,.0f}  MAE={mae:,.0f}  CV-R2={cv_r2:.4f}')
    return {'Model': name, 'RMSE': round(rmse, 0), 'MAE': round(mae, 0),
            'R2': round(r2, 4), 'CV R2 (5-fold)': round(cv_r2, 4),
            '_model': model, '_preds': preds}

scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_train)
X_te_sc  = scaler.transform(X_test)

results = []
results.append(evaluate('Linear Regression', LinearRegression(), X_tr_sc, y_train, X_te_sc, y_test))
results.append(evaluate('Ridge (a=1)',        Ridge(alpha=1.0),   X_tr_sc, y_train, X_te_sc, y_test))
results.append(evaluate('Lasso (a=0.001)',    Lasso(alpha=0.001, max_iter=10000), X_tr_sc, y_train, X_te_sc, y_test))
results.append(evaluate('Random Forest',      RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1), X_train, y_train, X_test, y_test))
if USE_XGB:
    results.append(evaluate('XGBoost', XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42, verbosity=0), X_train, y_train, X_test, y_test))"""),
    code("""metrics_df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith('_')} for r in results])
metrics_df"""),
    md("## 5. Wizualizacja – porównanie modeli"),
    code("""fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (metric, higher_better) in zip(axes, [('R2', True), ('RMSE', False), ('MAE', False)]):
    sdf = metrics_df.sort_values(metric, ascending=not higher_better).reset_index(drop=True)
    colors = ['darkgreen' if i == 0 else 'steelblue' for i in range(len(sdf))]
    ax.barh(sdf['Model'], sdf[metric], color=colors)
    ax.set_title(metric.replace('R2', 'R\u00b2'))
    ax.set_xlabel(metric.replace('R2', 'R\u00b2'))
plt.suptitle('Porównanie modeli predykcyjnych', fontsize=13)
plt.tight_layout()
plt.savefig(FIGURES_PATH + '03_model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()"""),
    md("## 6. Najlepszy model – residua i ważność cech"),
    code("""best_idx   = metrics_df['R2'].idxmax()
best_res   = results[best_idx]
best_name  = best_res['Model']
best_preds = best_res['_preds']
print(f'Najlepszy model: {best_name}  (R2={best_res[\"R2\"]})')
true_vals = np.expm1(y_test.values)
residuals = true_vals - best_preds

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].scatter(best_preds, true_vals, alpha=0.2, s=8, color='steelblue')
lims = [min(best_preds.min(), true_vals.min()), max(best_preds.max(), true_vals.max())]
axes[0].plot(lims, lims, 'r--', linewidth=1.5, label='idealna predykcja')
axes[0].set_title(f'{best_name}\\nPrzewidywane vs. Rzeczywiste')
axes[0].set_xlabel('Przewidywana cena (PLN)')
axes[0].set_ylabel('Rzeczywista cena (PLN)')
axes[0].legend()
axes[1].scatter(best_preds, residuals, alpha=0.2, s=8, color='darkorange')
axes[1].axhline(0, color='red', linestyle='--', linewidth=1.5)
axes[1].set_title('Residua vs. Przewidywana cena')
axes[1].set_xlabel('Przewidywana cena (PLN)')
axes[1].set_ylabel('Residuum (PLN)')
plt.tight_layout()
plt.savefig(FIGURES_PATH + '03_residuals.png', dpi=150, bbox_inches='tight')
plt.show()"""),
    code("""best_model_obj = best_res['_model']
if hasattr(best_model_obj, 'feature_importances_'):
    imp = pd.Series(best_model_obj.feature_importances_, index=FEATURES).sort_values(ascending=False).head(15)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=imp.values, y=imp.index, palette='viridis')
    plt.title(f'Ważność cech – {best_name}')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(FIGURES_PATH + '03_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()
elif hasattr(best_model_obj, 'coef_'):
    coefs = pd.Series(best_model_obj.coef_, index=FEATURES).abs().sort_values(ascending=False).head(15)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=coefs.values, y=coefs.index, palette='viridis')
    plt.title(f'Współczynniki (abs.) – {best_name}')
    plt.xlabel('|Coefficient|')
    plt.tight_layout()
    plt.savefig(FIGURES_PATH + '03_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()"""),
    md("## 7. Zapis wyników"),
    code("""metrics_df.to_csv(PROC_PATH + 'model_results.csv', index=False)
print('Wyniki zapisane do:', PROC_PATH + 'model_results.csv')
metrics_df"""),
    md("## 8. Wnioski z modelowania\n\n| Hipoteza | Wynik |\n|---|---|\n| H1: Rok produkcji i przebieg są najsilniejszymi predyktorami | ✅/❌ (sprawdź feature importance) |\n| H2: Samochody elektryczne/hybrydowe są droższe | ✅/❌ (sprawdź test ANOVA) |\n| H3: RF/XGBoost bije regresję liniową | ✅/❌ (sprawdź tabelę R²) |\n\n> Uzupełnij tabelę po uruchomieniu notebooka."),
])

# ─────────────────────────────────────────────
# 04_visualization.ipynb
# ─────────────────────────────────────────────
nb04 = nb([
    md("# 04 – Wizualizacje do raportu\n**Cel:** Przygotowanie finalnych wykresów odpowiadających na wszystkie pytania badawcze. Wykresy zapisywane są do `reports/figures/`."),
    md("## 1. Import i wczytanie danych"),
    code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({'figure.dpi': 150, 'font.size': 11,
                     'axes.titlesize': 13, 'axes.labelsize': 11})
sns.set_theme(style='whitegrid', palette='muted')
""" + PATHS_SETUP + """

df = pd.read_csv(PROC_PATH + 'cars_clean.csv')
print(f'Wczytano: {df.shape}')
try:
    results = pd.read_csv(PROC_PATH + 'model_results.csv')
    print('Wczytano wyniki modeli.')
except FileNotFoundError:
    print('Brak model_results.csv – uruchom najpierw notebook 03.')
    results = None"""),
    md("## 2. PB1 – Czynniki wpływające na cenę (korelacje)"),
    code("""num_cols = [c for c in ['year','mileage','vol_engine','car_age','mileage_per_year','mark_median_price','price'] if c in df.columns]
corr = df[num_cols].corr()
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=axes[0])
axes[0].set_title('Macierz korelacji (Pearson)')
price_corr = corr['price'].drop('price').sort_values()
colors = ['tomato' if v < 0 else 'steelblue' for v in price_corr]
axes[1].barh(price_corr.index, price_corr.values, color=colors)
axes[1].axvline(0, color='black', linewidth=0.8)
axes[1].set_title('Korelacja cech z ceną (Pearson)')
axes[1].set_xlabel('Współczynnik korelacji')
plt.tight_layout()
plt.savefig(FIGURES_PATH + '04_pb1_correlations.png', bbox_inches='tight')
plt.show()"""),
    md("## 3. PB2 – Cena vs. rok produkcji i przebieg"),
    code("""sample = df.sample(min(8000, len(df)), random_state=42)
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sc1 = axes[0].scatter(sample['year'], sample['price'], alpha=0.15, s=8,
                       c=sample['mileage'], cmap='coolwarm', vmin=0, vmax=300000)
plt.colorbar(sc1, ax=axes[0], label='Przebieg (km)')
axes[0].set_title('Cena vs. rok produkcji\\n(kolor = przebieg)')
axes[0].set_xlabel('Rok produkcji')
axes[0].set_ylabel('Cena (PLN)')
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
sc2 = axes[1].scatter(sample['mileage'], sample['price'], alpha=0.15, s=8,
                       c=sample['year'], cmap='viridis', vmin=2000, vmax=2022)
plt.colorbar(sc2, ax=axes[1], label='Rok produkcji')
axes[1].set_title('Cena vs. przebieg\\n(kolor = rok produkcji)')
axes[1].set_xlabel('Przebieg (km)')
axes[1].set_ylabel('Cena (PLN)')
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

plt.tight_layout()
plt.savefig(FIGURES_PATH + '04_pb2_price_year_mileage.png', bbox_inches='tight')
plt.show()"""),
    md("## 4. PB3 – Cena według rodzaju paliwa"),
    code("""fuel_counts = df['fuel'].value_counts()
valid_fuels = fuel_counts[fuel_counts >= 100].index
df_fuel = df[df['fuel'].isin(valid_fuels)]
order = df_fuel.groupby('fuel')['price'].median().sort_values(ascending=False).index
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.boxplot(data=df_fuel, x='fuel', y='price', order=order,
            palette='Set2', showfliers=False, ax=axes[0])
axes[0].set_title('Rozkład ceny według paliwa\\n(bez wartości ekstremalnych)')
axes[0].set_xlabel('Rodzaj paliwa')
axes[0].set_ylabel('Cena (PLN)')
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
axes[0].tick_params(axis='x', rotation=30)
medians = df_fuel.groupby('fuel')['price'].median().reindex(order)
axes[1].barh(list(medians.index[::-1]), list(medians.values[::-1]),
             color=sns.color_palette('Set2', len(medians)))
axes[1].set_title('Mediana ceny według paliwa')
axes[1].set_xlabel('Mediana ceny (PLN)')
axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
for i, v in enumerate(medians.values[::-1]):
    axes[1].text(v + 200, i, f'{v:,.0f} PLN', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(FIGURES_PATH + '04_pb3_price_by_fuel.png', bbox_inches='tight')
plt.show()"""),
    md("## 5. PB4 – Marki i województwa z najwyższymi cenami"),
    code("""mark_stats = (df.groupby('mark')
               .agg(median_price=('price','median'), count=('price','count'))
               .query('count >= 50')
               .sort_values('median_price', ascending=False)
               .head(20))
prov_stats = df.groupby('province')['price'].median().sort_values(ascending=False)
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
sns.barplot(x=mark_stats['median_price'], y=mark_stats.index, palette='Blues_d', ax=axes[0])
axes[0].set_title('Top 20 marek – mediana ceny\\n(min. 50 ogłoszeń)')
axes[0].set_xlabel('Mediana ceny (PLN)')
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
sns.barplot(x=prov_stats.values, y=prov_stats.index, palette='Oranges_d', ax=axes[1])
axes[1].set_title('Mediana ceny według województwa')
axes[1].set_xlabel('Mediana ceny (PLN)')
axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

plt.tight_layout()
plt.savefig(FIGURES_PATH + '04_pb4_marks_provinces.png', bbox_inches='tight')
plt.show()"""),
    md("## 6. PB5 – Porównanie modeli ML"),
    code("""if results is not None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (metric, ascending) in zip(axes, [('R2', False), ('RMSE', True), ('MAE', True)]):
        sorted_r = results.sort_values(metric, ascending=ascending).reset_index(drop=True)
        palette = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(sorted_r))]
        sns.barplot(x=sorted_r[metric], y=sorted_r['Model'], palette=palette, ax=ax)
        ax.set_title(metric.replace('R2','R\u00b2'))
        ax.set_xlabel(metric.replace('R2','R\u00b2'))
        if metric == 'R2':
            ax.set_xlim(0, 1)
    plt.suptitle('Porównanie modeli predykcyjnych ceny samochodu', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_PATH + '04_pb5_model_comparison.png', bbox_inches='tight')
    plt.show()
else:
    print('Brak danych – uruchom notebook 03_modeling.ipynb')"""),
    md("## 7. Lista wygenerowanych wykresów"),
    code("""import os
figures = sorted(f for f in os.listdir(FIGURES_PATH) if f != '.gitkeep')
print(f'Wykresy w {FIGURES_PATH} ({len(figures)} plików):')
for f in figures:
    print(f'  • {f}')"""),
])

# ─────────────────────────────────────────────
# Zapis do plików
# ─────────────────────────────────────────────
notebooks = {
    '01_EDA': nb01,
    '02_preprocessing': nb02,
    '03_modeling': nb03,
    '04_visualization': nb04,
}

for name, content in notebooks.items():
    path = os.path.join(NB_DIR, f'{name}.ipynb')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(content, f, ensure_ascii=False, indent=1)
    cells = content['cells']
    code_cells = sum(1 for c in cells if c['cell_type'] == 'code')
    print(f'OK {name}.ipynb – {len(cells)} komórek ({code_cells} code)')

print('\nWszystkie notebooki wygenerowane.')


