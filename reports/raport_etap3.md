# Raport Etap 3 — Wybór Modelu Transformerowego
**Projekt:** Analiza i przewidywanie cen samochodów używanych w Polsce  
**Grupa:** 6 | Gr1 | 2025/2026  
**Członkowie:** Denis Bichler (71564), Kasper Borecki (71970), Paweł Angielczyk (72530), Jakub Wierzbicki (66846)  
**Data:** 8 maja 2026

---

## 1. Definicja zadania

### Czym jest zadanie?

Zadaniem jest **regresja ceny** — przewidywanie ceny ofertowej używanego samochodu w Polsce na podstawie jego parametrów technicznych i lokalizacji ogłoszenia.  
Projekt rozszerzamy o podejście oparte na modelu Transformerowym: zamiast podawać cechy tabelaryczne bezpośrednio (jak w XGBoost/Random Forest), serializujemy dane samochodu do postaci tekstowej i wykorzystujemy model językowy z dodatkową głowicą regresyjną.

### Wejście / Wyjście

| Element | Opis |
|---|---|
| **Wejście** | Tekstowy opis samochodu w języku angielskim, np.: `"Year: 2018, Brand: BMW, Model: 3 Series, Mileage: 52000 km, Engine: 1995 cc, Fuel: diesel, Province: mazowieckie"` |
| **Wyjście** | Przewidywana cena pojazdu w PLN (wartość ciągła, regresja) |

---

## 2. Wybrany model

| Parametr | Wartość |
|---|---|
| **Nazwa modelu** | `distilbert-base-uncased` |
| **Producent** | Hugging Face / Victor Sanh et al. (2019) |
| **Typ modelu** | Transformer encoder (BERT-style), destylowany |
| **Liczba parametrów** | ~66 mln |
| **Typ zadania po fine-tuningu** | Regresja (dodana głowica `nn.Linear` na wyjściu `[CLS]`) |
| **Link HuggingFace** | https://huggingface.co/distilbert-base-uncased |

---

## 3. Uzasadnienie wyboru

### Dlaczego ta architektura pasuje do zadania?

Architektura **BERT / DistilBERT** jest modelem z kodowaniem dwukierunkowym (encoder-only). Dzięki mechanizmowi **self-attention** model potrafi uchwycić wzajemne zależności pomiędzy wszystkimi tokenami wejściowymi jednocześnie — np. relację między marką pojazdu a pojemnością silnika czy rokiem produkcji. Podejście serializacji cech tabelarycznych do tekstu i fine-tuningu modelu językowego zostało opisane w literaturze (TURL, TAPAS, UnifiedSKG) i daje dobre wyniki w przypadkach, gdy cechy kategoryczne mają semantyczne znaczenie (np. nazwy marek samochodów: „BMW" jest semantycznie bliższe „Audi" niż „Daewoo").

W przeciwieństwie do drzew decyzyjnych, model Transformerowy może skorzystać z wiedzy zdobytej podczas pre-treningu na dużych korpusach tekstowych — nazwy marek, modeli, województw i rodzajów paliw są znane modelowi z precoders, co wspomaga generalizację.

### Dlaczego taki rozmiar (DistilBERT zamiast pełnego BERT)?

- DistilBERT jest **o 40% mniejszy** i **o 60% szybszy** od `bert-base-uncased` zachowując ~97% jego wydajności.
- Dane wejściowe są krótkie (ok. 30–50 tokenów na próbkę) — duży model nie jest konieczny.
- Projekt realizowany jest na sprzęcie bez dedykowanego GPU klasy enterprise — mniejszy model pozwala na realne przeprowadzenie fine-tuningu.

---

## 4. Planowane użycie

| Element | Opis |
|---|---|
| **Pipeline / zadanie HuggingFace** | `text-classification` z modyfikacją na regresję (`num_labels=1`, `problem_type="regression"`) |
| **Klasa modelu** | `AutoModelForSequenceClassification` z `num_labels=1` |
| **Tokenizer** | `AutoTokenizer.from_pretrained("distilbert-base-uncased")` |
| **Wejście do modelu** | Tekstowy opis samochodu (patrz punkt 1), max 128 tokenów |
| **Oczekiwane wyjście** | Pojedyncza wartość ciągła — przewidywana cena (PLN); stosujemy `log(price)` jako zmienną docelową aby znormalizować rozkład |

### Przykład użycia (Python)

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

text = "Year: 2018, Brand: BMW, Mileage: 52000 km, Engine: 1995 cc, Fuel: diesel, Province: mazowieckie"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

with torch.no_grad():
    output = model(**inputs)

import numpy as np
predicted_log_price = output.logits.item()
predicted_price = np.exp(predicted_log_price)
print(f"Przewidywana cena: {predicted_price:.0f} PLN")
```

---

## 5. Ograniczenia i ryzyka

| Ograniczenie / Ryzyko | Opis |
|---|---|
| **Utrata informacji numerycznej** | Serializacja liczb (np. `52000`) do tokenów tekstowych może być nieoptymalna — model może mieć trud z arytmetyką i porównywaniem wartości liczbowych. |
| **Format wejścia wrażliwy na kolejność** | Zmiana kolejności atrybutów może wpłynąć na wyniki (model może kodować pozycję tokenów). |
| **Brak danych polskojęzycznych** | `distilbert-base-uncased` jest wytrenowany na angielskim — nazwy województw wpisane po polsku (np. `"mazowieckie"`) mogą być dzielone na wiele tokenów (`"maz"`, `"ow"`, `"iec"`, `"kie"`). Alternatywnie należy je przetłumaczyć na angielski lub użyć polskiego modelu. |
| **Ryzyko overfittingu** | Dataset ma ~111 000 rekordów — przy fine-tuningu dużego modelu językowego na stosunkowo jednorodnych danych finansowych istnieje ryzyko przeuczenia. Konieczna regularyzacja (dropout, weight decay). |
| **Interpretowalność** | Model Transformerowy jest trudniejszy w interpretacji niż Random Forest/XGBoost (mniejsza transparentność predykcji). |
| **Czas treningu** | Fine-tuning na CPU może trwać wiele godzin — konieczna przynajmniej karta graficzna konsumencka (np. RTX 3060+) lub Google Colab. |

---

## 6. Plan ewaluacji

Model będzie oceniany na zbiorze testowym (20% danych) przy użyciu tych samych metryk co modele bazowe w Etapie 2, co umożliwi porównanie:

| Metryka | Opis |
|---|---|
| **RMSE** (Root Mean Squared Error) | Główna metryka — kara za duże błędy; jednostka: PLN |
| **MAE** (Mean Absolute Error) | Średni absolutny błąd — bardziej odporna na outliery |
| **R²** (Coefficient of Determination) | Udział wyjaśnionej wariancji ceny |
| **MAPE** (Mean Absolute Percentage Error) | Względny błąd procentowy — łatwiejszy do interpretacji biznesowej |

### Baseline do porównania

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Regresja liniowa | 38 707 PLN | 17 500 PLN | 0.784 |
| Random Forest | 28 204 PLN | 12 500 PLN | 0.885 |
| XGBoost | 27 622 PLN | 12 351 PLN | 0.890 |
| **DistilBERT (cel)** | **< 28 000 PLN** | — | **> 0.88** |

Celem jest osiągnięcie wyników porównywalnych lub lepszych niż Random Forest.

### Metodologia ewaluacji

1. Podział: 80% trening / 20% test (taki sam jak w Etapie 2 — te same indeksy).
2. Walidacja krzyżowa 5-fold na zbiorze treningowym podczas fine-tuningu.
3. Analiza residuów (histogram, wykres residua vs. wartości przewidywane).
4. Analiza błędów w podgrupach: wg marki, rodzaju paliwa, województwa.

---

## 7. Ograniczenia zasobowe

| Zasób | Sytuacja |
|---|---|
| **CPU vs GPU** | Fine-tuning zalecany na GPU; na CPU jest możliwy ale znacząco wolniejszy. Planujemy użycie Google Colab (GPU T4, darmowy tier) lub własnej karty graficznej. |
| **Pamięć RAM** | Dataset ~111 000 rekordów × krótki tekst — zbiór danych zmieści się w pamięci (ok. 500 MB). |
| **Pamięć GPU (VRAM)** | DistilBERT z batch_size=32 wymaga ok. 4–6 GB VRAM — mieści się w T4 (16 GB). |
| **Czas treningu** | Szacowany czas fine-tuningu: 3–5 epok × ~111k próbek — ok. 45–90 minut na GPU T4, ok. 8–15 godzin na CPU. |
| **Dysk** | Model checkpoint ~250 MB; pełny dataset po tokenizacji ~800 MB. |
| **Tokenizacja** | Tokenizacja całego datasetu jednorazowo (offline) — ok. 5–10 minut na CPU. |

---

## 8. Plan fine-tuningu

1. **Przygotowanie danych:** Serializacja cech tabelarycznych do tekstu; normalizacja ceny przez `log1p(price)`; podział 80/20; tokenizacja offline z `max_length=128`.
2. **Załadowanie modelu:** `AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=1, problem_type="regression")`.
3. **Konfiguracja treningu:**
   - Optimizer: AdamW, lr = 2e-5
   - Scheduler: liniowy warmup (10% kroków)
   - Batch size: 32 (GPU) / 8 (CPU)
   - Epoki: 3–5
   - Weight decay: 0.01
4. **Trening z Hugging Face Trainer API** (`Trainer` + `TrainingArguments`).
5. **Ewaluacja** po każdej epoce na zbiorze walidacyjnym (RMSE, R²).
6. **Zapisanie najlepszego checkpointa** (early stopping na RMSE).
7. **Analiza wyników** i porównanie z modelami z Etapu 2.

---

## 9. Modele alternatywne

| Model | Powód rozważenia | Powód odrzucenia |
|---|---|---|
| `bert-base-uncased` (pełny BERT) | Wyższe potencjalne możliwości | Dwa razy większy niż DistilBERT; wolniejszy trening; minimalna korzyść dla krótkich sekwencji |
| `allegro/herbert-base-cased` (HerBERT) | Polskojęzyczny model BERT; lepsze zrozumienie nazw polskich województw i marek | Dane są w językujęzykliku angielskim (nazwy marek, paliw); mniejsza społeczność i dokumentacja |
| `roberta-base` | Lepsze wyniki na benchmarkach NLP | Nie wnosi istotnej poprawy dla danych tabelarycznych; większy niż DistilBERT |
| FT-Transformer (`rtdl`) | Zaprojektowany specjalnie dla danych tabelarycznych | Brak pre-treningu na dużych zbiorach — trening od zera; trudniejszy w implementacji w ramach kursu |
| `TabPFN` | Pre-trenowany transformer dla danych tabelarycznych | Ograniczony do 1000 próbek i 100 cech — zbyt mały dla naszego datasetu |
| `google/tapas-base` | Transformer dla danych tabelarycznych | Zaprojektowany dla QA na tabelach, nie dla regresji ciągłej |

### Dlaczego DistilBERT jest najlepszym wyborem?

DistilBERT oferuje najlepszy kompromis między:
- **wydajnością obliczeniową** (możliwy do wytrenowania na darmowym GPU w Colab),
- **łatwością implementacji** (Hugging Face Trainer, AutoClasses),
- **możliwościami transferu wiedzy** (pre-trening na dużym korpusie — nazwy marek, modeli i parametrów są mu znane),
- **skalowalnością** na nasz dataset (~111k rekordów).

---

## Podsumowanie

Wybieramy model **`distilbert-base-uncased`** fine-tunowany na zadanie regresji ceny. Cechy tabelaryczne samochodu zostaną przekonwertowane na tekstowy opis, który następnie zostanie przetworzony przez model. Oczekujemy wyników porównywalnych z Random Forest (R² ≈ 0.88, RMSE ≈ 28 000 PLN). Fine-tuning zostanie przeprowadzony z użyciem Hugging Face Trainer API na środowisku Google Colab (GPU T4).

---

*Projekt_DB_KB_PA | Grupa 6 | Gr1 | 2025/2026*

