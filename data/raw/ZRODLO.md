# Źródło danych

## Dataset: Car Prices Poland (Kaggle)

**Plik:** `Car_Prices_Poland_Kaggle.csv`  
**Źródło:** [https://www.kaggle.com/datasets/aleksandrglotov/car-prices-poland](https://www.kaggle.com/datasets/aleksandrglotov/car-prices-poland)  
**Autor:** Aleksandr Glotov  
**Data zebrania danych:** styczeń 2022  
**Metoda pozyskania:** web scraping (Selenium + requests, Python) z polskiego serwisu ogłoszeń samochodowych

## Opis datasetu

Dataset zawiera ogłoszenia sprzedaży samochodów używanych w Polsce. Kolumny:

| Kolumna | Opis |
|---|---|
| `mark` | Marka samochodu (np. BMW, Toyota) |
| `model` | Model samochodu |
| `generation_name` | Generacja modelu |
| `year` | Rok produkcji |
| `mileage` | Przebieg (km) |
| `vol_engine` | Pojemność silnika (cm³) |
| `fuel` | Rodzaj paliwa (benzyna, diesel, elektryczny, itd.) |
| `city` | Miasto, z którego pochodzi ogłoszenie |
| `province` | Województwo |
| `price` | Cena (PLN) – zmienna docelowa |

## Jak pobrać dane

1. Przejdź na stronę Kaggle: https://www.kaggle.com/datasets/aleksandrglotov/car-prices-poland
2. Pobierz plik `Car_Prices_Poland_Kaggle.csv`
3. Umieść go w tym folderze: `data/raw/Car_Prices_Poland_Kaggle.csv`

> Uwaga: pliki danych nie są śledzone przez Git (patrz `.gitignore`).  
> Każdy członek zespołu musi pobrać je ręcznie według powyższej instrukcji.

