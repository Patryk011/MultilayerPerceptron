# Raport klasyfikacji raka piersi

## Metryki klasyfikacji

| Metryka | Wartoœæ |
|---------|--------|
| Accuracy | 0.9441 |
| Precision | 0.9592 |
| Recall | 0.8868 |
| F1-score | 0.9216 |
| AUC-ROC | 0.9929 |
| Average Precision | 0.9881 |

## Macierz pomy³ek

```
[[88  2]
 [ 6 47]]
```

### Interpretacja macierzy pomy³ek:

- True Negatives (TN): 88 (Poprawnie sklasyfikowane przypadki ³agodne)
- False Positives (FP): 2 (Przypadki ³agodne b³êdnie sklasyfikowane jako z³oœliwe)
- False Negatives (FN): 6 (Przypadki z³oœliwe b³êdnie sklasyfikowane jako ³agodne)
- True Positives (TP): 47 (Poprawnie sklasyfikowane przypadki z³oœliwe)

## Szczegó³owy raport klasyfikacji

```
              precision    recall  f1-score   support

 £agodny (0)       1.00      1.00      1.00         1
Z³oœliwy (1)       1.00      1.00      1.00         1

    accuracy                           1.00         2
   macro avg       1.00      1.00      1.00         2
weighted avg       1.00      1.00      1.00         2

```

## Wnioski

- Model osi¹ga **wysok¹ dok³adnoœæ** (>90%), co œwiadczy o jego dobrej jakoœci.
- Czu³oœæ modelu (recall <90%) mo¿e wymagaæ poprawy, aby minimalizowaæ liczbê fa³szywie negatywnych wyników.
- Model ma **bardzo wysok¹ precyzjê** (>95%), co oznacza niskie ryzyko fa³szywie pozytywnych wyników.
- Model nie rozpozna³ 6 przypadków z³oœliwych, co stanowi 11.3% wszystkich przypadków z³oœliwych. W kontekœcie medycznym nale¿y d¹¿yæ do minimalizacji tej wartoœci.

### Implikacje kliniczne:

- W kontekœcie diagnostyki raka piersi, wy¿sza czu³oœæ (recall) jest czêsto priorytetem, aby minimalizowaæ liczbê nierozpoznanych przypadków raka.
- B³êdy typu fa³szywie negatywne (FN) s¹ bardziej kosztowne ni¿ fa³szywie pozytywne (FP), poniewa¿ opóŸnienie w diagnozie raka mo¿e prowadziæ do powa¿niejszych konsekwencji zdrowotnych.

### Mo¿liwe kierunki poprawy modelu:

1. Eksperymentowanie z ró¿nymi architekturami sieci neuronowej (liczba warstw, neuronów).
2. Testowanie ró¿nych metod regulacji (alpha) i algorytmów optymalizacji.
3. Rozwa¿enie innych metod selekcji cech lub u¿ycie wiêkszej liczby cech.
4. Zwiêkszenie zbioru danych treningowych, jeœli to mo¿liwe.
