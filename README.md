# Projekt_ZjPwDSiAI_Gr6
**Projekt: Denis B. · Kasper B. · Paweł A. · Jakub W.**

---

## 👥 Członkowie zespołu

| Imię i nazwisko        | Proponowana rola                                            |
|------------------------|-------------------------------------------------------------|
| Denis Bichler 71564    | Pozyskiwanie danych, wstępna obróbka i czyszczenie datasetu |
| Kasper Borecki 71970   | Analiza danych, modelowanie predykcyjne, ewaluacja modeli   |
| Paweł Angielczyk 72530 | Wizualizacje, raport końcowy, notebook EDA                  |
| Jakub Wierzbicki 66846 | Wsparcie przy analizie i replikacji wyników                 |

> Role mogą być modyfikowane w trakcie projektu według potrzeb.

---

## Temat i pytania badawcze

### Temat: Analiza i przewidywanie cen samochodów używanych w Polsce

**Dataset:** [Car Prices Poland – Kaggle](https://www.kaggle.com/datasets/aleksandrglotov/car-prices-poland)  
Dane zebrane w styczniu 2022 metodą web scrapingu z polskiego serwisu ogłoszeń samochodowych.

### Pytania badawcze

1. **Jakie czynniki mają największy wpływ na cenę samochodu używanego w Polsce?**  
   (rok produkcji, przebieg, marka, pojemność silnika, rodzaj paliwa)

2. **Jak cena zmienia się w zależności od przebiegu i roku produkcji?**  
   Czy zależność jest liniowa, czy bardziej złożona?

3. **Czy rodzaj paliwa (benzyna, diesel, elektryczny, hybrydowy) istotnie różnicuje cenę?**  
   Analiza statystyczna różnic między grupami.

4. **Które marki i województwa charakteryzują się najwyższymi cenami ofertowymi?**  
   Analiza rozkładu cen geograficznie i per marka.

5. **Jak dobrze można przewidzieć cenę samochodu na podstawie jego parametrów?**  
   Porównanie modeli regresji (liniowa, Random Forest, XGBoost).

### Hipotezy

- H1: Rok produkcji i przebieg są najsilniejszymi predyktorami ceny.
- H2: Samochody elektryczne i hybrydowe są istotnie droższe niż benzynowe/dieselowe przy porównywalnym roku produkcji.
- H3: Model Random Forest lub XGBoost osiągnie lepsze wyniki predykcji niż regresja liniowa.

---

## Struktura repozytorium

```
Projekt_DB_KB_PA/
│
├── data/
│   ├── raw/
│   │   ├── Car_Prices_Poland_Kaggle.csv   # oryginalne dane (nie w Git)
│   │   └── ZRODLO.md                      # opis źródła i instrukcja pobrania
│   └── processed/                         # dane po czyszczeniu
│
├── notebooks/
│   ├── 01_EDA.ipynb            # eksploracja i wizualizacja danych
│   ├── 02_preprocessing.ipynb  # czyszczenie i inżynieria cech
│   ├── 03_modeling.ipynb       # modelowanie i ewaluacja
│   └── 04_visualization.ipynb  # wykresy do raportu
│
├── src/
│   ├── data_utils.py   # funkcje do wczytywania i czyszczenia danych
│   └── plot_utils.py   # funkcje do wizualizacji
│
├── reports/
│   ├── figures/        # wyeksportowane wykresy
│   └── raport.pdf      # raport końcowy
│
├── replication/
│   ├── run_all.py      # skrypt replikujący całą analizę
│   └── INSTRUKCJA.md   # krok po kroku jak odtworzyć wyniki
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Metody analizy danych

### Eksploracja danych (EDA)
- Statystyki opisowe (średnia, mediana, odchylenie std, kwartyle)
- Rozkłady zmiennych numerycznych i kategorycznych
- Analiza brakujących wartości i wartości odstających

### Testowanie hipotez
- Test ANOVA / Kruskal-Wallis (porównanie cen między grupami paliw, marek)
- Test t-Studenta (np. diesel vs. benzyna)
- Test chi-kwadrat (zależności między zmiennymi kategorycznymi)

### Analiza korelacji
- Korelacja Pearsona: cena vs. rok, przebieg, pojemność silnika
- Heatmapa korelacji zmiennych numerycznych

### Modelowanie predykcyjne
| Model | Cel |
|---|---|
| Regresja liniowa (baseline) | Przewidywanie ceny |
| Ridge / Lasso | Regularyzacja, selekcja cech |
| Random Forest Regressor | Nieliniowe zależności, ważność cech |
| XGBoost Regressor | Gradient boosting, najlepsza dokładność |

### Ewaluacja modeli
- RMSE, MAE, R² (na zbiorze testowym)
- Walidacja krzyżowa (5-fold)
- Wykres residuów, wykres przewidywana vs. rzeczywista cena
- Wykres ważności cech (Feature Importance)

---

## Struktura raportu końcowego

1. **Streszczenie** – cel, główne wyniki, wnioski
2. **Wprowadzenie** – kontekst rynku samochodów używanych w Polsce, pytania badawcze
3. **Opis danych** – źródło, liczba rekordów, opis kolumn, statystyki opisowe
4. **Metodologia** – opis metod analizy i modelowania
5. **Wyniki**
   - EDA i wizualizacje
   - Wyniki testów statystycznych
   - Porównanie modeli ML
6. **Dyskusja** – interpretacja wyników, ograniczenia (dane z 2022, brak koloru/mocy/nadwozia)
7. **Wnioski** – odpowiedzi na pytania badawcze
8. **Bibliografia**

---

## Harmonogram

| Tydzień | Zadanie |
|---|---|
| 1–2 | Finalizacja tematu, podział ról, konfiguracja repozytorium |
| 3–4 | EDA: eksploracja danych, wizualizacje, statystyki opisowe |
| 5–6 | Preprocessing: czyszczenie, encoding, feature engineering |
| 7–8 | Modelowanie: regresja liniowa, Random Forest, XGBoost |
| 9–10 | Pisanie raportu końcowego |
| 11 | Finalizacja folderu replikacji, przegląd i oddanie projektu |

---

## Środowisko i technologie

- **Język:** Python 3.10+
- **Notebooki:** Jupyter Notebook / JupyterLab
- **Biblioteki:**
  - Dane: `pandas`, `numpy`
  - Wizualizacja: `matplotlib`, `seaborn`, `plotly`
  - Statystyki: `scipy`, `statsmodels`
  - ML: `scikit-learn`, `xgboost`

```bash
pip install -r requirements.txt
```

---

## 🔗 Źródło danych

- **Dataset:** [Car Prices Poland – Kaggle](https://www.kaggle.com/datasets/aleksandrglotov/car-prices-poland)
- **Autor:** Aleksandr Glotov
- **Instrukcja pobrania:** [data/raw/ZRODLO.md](data/raw/ZRODLO.md)
