===PROJEKT===
Numer grupy: 6
Nazwa projektu: Analiza i przewidywanie cen samochodów używanych w Polsce
Opis: Analiza czynników wpływających na ceny samochodów używanych w Polsce na podstawie danych z ogłoszeń z 2022 roku. Projekt obejmuje eksplorację danych, testowanie hipotez statystycznych oraz budowę modeli regresji (regresja liniowa, Random Forest, XGBoost) do przewidywania ceny pojazdu.

===GRUPA===
Lab grupa, ID, Nazwisko, Imie
Gr1, 71564, Bichler, Denis
Gr1, 71970, Borecki, Kasper
Gr1, 72530, Angielczyk, Pawel
Gr1, 66846, Wierzbicki, Jakub

===WKLAD===
ID, Nazwisko, Imie: Krótki opis wkładu każdego studenta do grupowego projektu na tym etapie.
71564, Bichler, Denis: Pozyskiwanie danych, wstępna obróbka i czyszczenie datasetu (data_utils.py, 02_preprocessing.ipynb).
71970, Borecki, Kasper: Analiza danych, modelowanie predykcyjne, ewaluacja modeli (03_modeling.ipynb).
72530, Angielczyk, Pawel: Wizualizacje, raport końcowy, notebook EDA (01_EDA.ipynb, 04_visualization.ipynb, plot_utils.py).
66846, Wierzbicki, Jakub: Wsparcie przy analizie i replikacji wyników (replication/).


===PYTANIA BADAWCZE===
1. Jakie czynniki mają największy wpływ na cenę samochodu używanego w Polsce (rok produkcji, przebieg, marka, pojemność silnika, rodzaj paliwa)?
2. Jak cena zmienia się w zależności od przebiegu i roku produkcji — czy zależność jest liniowa, czy bardziej złożona?
3. Czy rodzaj paliwa (benzyna, diesel, elektryczny, hybrydowy) istotnie różnicuje cenę pojazdu?
4. Które marki i województwa charakteryzują się najwyższymi cenami ofertowymi?
5. Jak dobrze można przewidzieć cenę samochodu na podstawie jego parametrów technicznych — porównanie modeli regresji liniowej, Random Forest i XGBoost?

===ZRODLA DANYCH===
=1=
Nazwa zrodla: Kaggle
Nazwa danych: Car Prices Poland
Dataset URL: https://www.kaggle.com/datasets/aleksandrglotov/car-prices-poland

===ZMIENNE===
mark, model, generation_name, year, mileage, vol_engine, fuel, city, province, price

===CECHY===
price (zmienna docelowa), year, mileage, vol_engine, fuel (encoded), mark (encoded), province (encoded)

===ANALIZA===
1. Eksploracyjna analiza danych (EDA) — rozkłady zmiennych, analiza brakujących wartości i outlierów
2. Analiza korelacji (Pearson) — heatmapa korelacji zmiennych numerycznych z ceną
3. Testy statystyczne — ANOVA/Kruskal-Wallis (porównanie cen między grupami paliw i marek), test t-Studenta
4. Regresja liniowa (baseline) — przewidywanie ceny
5. Random Forest Regressor — nieliniowe zależności, analiza ważności cech
6. XGBoost Regressor — gradient boosting, porównanie RMSE/MAE/R²

===SRODOWISKO===
Python version: Python 3.10+
Main libraries: pandas>=2.0.0, numpy>=1.24.0, matplotlib>=3.7.0, seaborn>=0.12.0, scikit-learn>=1.3.0, scipy>=1.11.0, statsmodels>=0.14.0, xgboost>=2.0.0, plotly>=5.15.0, jupyter>=1.0.0

===ZAWARTOSC===
Projekt_DB_KB_PA/
|
|--- README_FOR_PROFESSOR.md
|--- README.md
|--- requirements.txt
|--- data/
|    |--- raw/
|    |    |--- ZRODLO.md
|    |    |--- Car_Prices_Poland_Kaggle.csv   (pobierz ręcznie wg ZRODLO.md)
|    |--- processed/
|         |--- cars_clean.csv
|         |--- model_results.csv
|--- notebooks/
|    |--- 01_EDA.ipynb
|    |--- 02_preprocessing.ipynb
|    |--- 03_modeling.ipynb
|    |--- 04_visualization.ipynb
|--- src/
|    |--- data_utils.py
|    |--- plot_utils.py
|--- reports/
|    |--- figures/
|    |--- raport.pdf
|--- replication/
     |--- run_all.py
     |--- generate_notebooks.py
     |--- INSTRUKCJA.md

