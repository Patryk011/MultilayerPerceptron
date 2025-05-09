# Raport klasyfikacji raka piersi

## Metryki klasyfikacji

| Metryka | Warto�� |
|---------|--------|
| Accuracy | 0.9441 |
| Precision | 0.9592 |
| Recall | 0.8868 |
| F1-score | 0.9216 |
| AUC-ROC | 0.9929 |
| Average Precision | 0.9881 |

## Macierz pomy�ek

```
[[88  2]
 [ 6 47]]
```

### Interpretacja macierzy pomy�ek:

- True Negatives (TN): 88 (Poprawnie sklasyfikowane przypadki �agodne)
- False Positives (FP): 2 (Przypadki �agodne b��dnie sklasyfikowane jako z�o�liwe)
- False Negatives (FN): 6 (Przypadki z�o�liwe b��dnie sklasyfikowane jako �agodne)
- True Positives (TP): 47 (Poprawnie sklasyfikowane przypadki z�o�liwe)

## Szczeg�owy raport klasyfikacji

```
              precision    recall  f1-score   support

 �agodny (0)       1.00      1.00      1.00         1
Z�o�liwy (1)       1.00      1.00      1.00         1

    accuracy                           1.00         2
   macro avg       1.00      1.00      1.00         2
weighted avg       1.00      1.00      1.00         2

```

## Wnioski

- Model osi�ga **wysok� dok�adno��** (>90%), co �wiadczy o jego dobrej jako�ci.
- Czu�o�� modelu (recall <90%) mo�e wymaga� poprawy, aby minimalizowa� liczb� fa�szywie negatywnych wynik�w.
- Model ma **bardzo wysok� precyzj�** (>95%), co oznacza niskie ryzyko fa�szywie pozytywnych wynik�w.
- Model nie rozpozna� 6 przypadk�w z�o�liwych, co stanowi 11.3% wszystkich przypadk�w z�o�liwych. W kontek�cie medycznym nale�y d��y� do minimalizacji tej warto�ci.

### Implikacje kliniczne:

- W kontek�cie diagnostyki raka piersi, wy�sza czu�o�� (recall) jest cz�sto priorytetem, aby minimalizowa� liczb� nierozpoznanych przypadk�w raka.
- B��dy typu fa�szywie negatywne (FN) s� bardziej kosztowne ni� fa�szywie pozytywne (FP), poniewa� op�nienie w diagnozie raka mo�e prowadzi� do powa�niejszych konsekwencji zdrowotnych.

### Mo�liwe kierunki poprawy modelu:

1. Eksperymentowanie z r�nymi architekturami sieci neuronowej (liczba warstw, neuron�w).
2. Testowanie r�nych metod regulacji (alpha) i algorytm�w optymalizacji.
3. Rozwa�enie innych metod selekcji cech lub u�ycie wi�kszej liczby cech.
4. Zwi�kszenie zbioru danych treningowych, je�li to mo�liwe.
