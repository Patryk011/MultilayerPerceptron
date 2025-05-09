
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)
import os

import config

def evaluate_model(model, X_test, y_test, threshold=None):
    """
    Ocenia model na zbiorze testowym.
    
    Args:
        model: Wytrenowany model
        X_test: Dane testowe
        y_test: Etykiety testowe
        threshold (float): Próg decyzyjny dla klasyfikacji
        
    Returns:
        dict: Wyniki ewaluacji
    """
    # Używamy parametrów z konfiguracji, jeśli nie podano
    if threshold is None:
        threshold = config.EVALUATION_CONFIG['threshold']
    
    print("\n--- EWALUACJA MODELU ---")
    
    # Predykcje modelu
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
    else:
        y_pred = model.predict(X_test)
        y_pred_proba = None
    
    # Metryki podstawowe
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Krzywa ROC i AUC (jeśli mamy prawdopodobieństwa)
    if y_pred_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
    else:
        fpr, tpr, roc_auc = None, None, None
    
    # Krzywa Precision-Recall (jeśli mamy prawdopodobieństwa)
    if y_pred_proba is not None:
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
    else:
        precision_curve, recall_curve, avg_precision = None, None, None
    
    # Macierz pomyłek
    cm = confusion_matrix(y_test, y_pred)
    
    # Raport klasyfikacji
    report = classification_report(y_test, y_pred, target_names=['Łagodny (0)', 'Złośliwy (1)'], output_dict=True)
    
    # Wyświetlenie wyników
    print(f"Wyniki ewaluacji modelu (próg: {threshold}):")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-score: {f1:.4f}")
    if roc_auc is not None:
        print(f"  AUC-ROC: {roc_auc:.4f}")
    
    print("\nMacierz pomyłek:")
    print(cm)
    
    # Przygotowanie wyników
    evaluation_results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'avg_precision': avg_precision,
        'confusion_matrix': cm,
        'classification_report': report,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'fpr': fpr,
        'tpr': tpr,
        'precision_curve': precision_curve,
        'recall_curve': recall_curve,
        'threshold': threshold
    }
    
    # Wizualizacja wyników
    visualize_confusion_matrix(cm)
    if y_pred_proba is not None:
        visualize_roc_curve(fpr, tpr, roc_auc)
        visualize_precision_recall_curve(recall_curve, precision_curve, avg_precision)
    visualize_metrics(evaluation_results)
    
    return evaluation_results

def visualize_confusion_matrix(cm, filename="confusion_matrix.png"):
    """
    Wizualizuje macierz pomyłek.
    
    Args:
        cm: Macierz pomyłek
        filename (str): Nazwa pliku do zapisu wykresu
    """
    plt.figure(figsize=config.VISUALIZATION_CONFIG['figsize_medium'])
    
    # Normalizacja macierzy pomyłek (do procentów)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Tworzymy dwa wykresy obok siebie
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Macierz pomyłek z wartościami liczbowymi
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=['Łagodny (0)', 'Złośliwy (1)'],
        yticklabels=['Łagodny (0)', 'Złośliwy (1)'],
        ax=axes[0]
    )
    axes[0].set_xlabel('Przewidywane')
    axes[0].set_ylabel('Rzeczywiste')
    axes[0].set_title('Macierz pomyłek (liczby)')
    
    # Macierz pomyłek z wartościami procentowymi
    sns.heatmap(
        cm_normalized, 
        annot=True, 
        fmt='.1f', 
        cmap='Blues', 
        xticklabels=['Łagodny (0)', 'Złośliwy (1)'],
        yticklabels=['Łagodny (0)', 'Złośliwy (1)'],
        ax=axes[1]
    )
    axes[1].set_xlabel('Przewidywane')
    axes[1].set_ylabel('Rzeczywiste')
    axes[1].set_title('Macierz pomyłek (%)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, "plots", "model", filename), 
                dpi=config.VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    print(f"Zapisano wizualizację macierzy pomyłek: {filename}")

def visualize_roc_curve(fpr, tpr, roc_auc, filename="roc_curve.png"):
    """
    Wizualizuje krzywą ROC.
    
    Args:
        fpr: False Positive Rate
        tpr: True Positive Rate
        roc_auc: Pole pod krzywą ROC
        filename (str): Nazwa pliku do zapisu wykresu
    """
    plt.figure(figsize=config.VISUALIZATION_CONFIG['figsize_medium'])
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    plt.savefig(os.path.join(config.OUTPUT_DIR, "plots", "model", filename), 
                dpi=config.VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    print(f"Zapisano wizualizację krzywej ROC: {filename}")

def visualize_precision_recall_curve(recall_curve, precision_curve, avg_precision, filename="precision_recall_curve.png"):
    """
    Wizualizuje krzywą Precision-Recall.
    
    Args:
        recall_curve: Wartości Recall
        precision_curve: Wartości Precision
        avg_precision: Średnia precyzja
        filename (str): Nazwa pliku do zapisu wykresu
    """
    plt.figure(figsize=config.VISUALIZATION_CONFIG['figsize_medium'])
    
    plt.plot(recall_curve, precision_curve, color='blue', lw=2, 
             label=f'Precision-Recall curve (AP = {avg_precision:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True)
    
    plt.savefig(os.path.join(config.OUTPUT_DIR, "plots", "model", filename), 
                dpi=config.VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    print(f"Zapisano wizualizację krzywej Precision-Recall: {filename}")

def visualize_metrics(evaluation_results, filename="metrics.png"):
    """
    Wizualizuje metryki klasyfikacji.
    
    Args:
        evaluation_results (dict): Wyniki ewaluacji
        filename (str): Nazwa pliku do zapisu wykresu
    """
    plt.figure(figsize=config.VISUALIZATION_CONFIG['figsize_medium'])
    
    # Wybieramy metryki do wizualizacji
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    if evaluation_results['roc_auc'] is not None:
        metrics.append('roc_auc')
    
    values = [evaluation_results[metric] for metric in metrics]
    
    # Tworzymy wykres słupkowy
    bars = plt.bar(metrics, values)
    
    # Dodajemy wartości na szczycie słupków
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{value:.4f}',
            ha='center',
            va='bottom'
        )
    
    plt.title('Metryki klasyfikacji')
    plt.ylabel('Wartość')
    plt.ylim(0, 1.1)  # Ustawiamy zakres osi Y od 0 do 1.1
    plt.grid(True, axis='y')
    plt.tight_layout()
    
    plt.savefig(os.path.join(config.OUTPUT_DIR, "plots", "model", filename), 
                dpi=config.VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    print(f"Zapisano wizualizację metryk klasyfikacji: {filename}")

def generate_classification_report(evaluation_results, filename="classification_report.md"):
    """
    Generuje raport klasyfikacji w formacie Markdown.
    
    Args:
        evaluation_results (dict): Wyniki ewaluacji
        filename (str): Nazwa pliku do zapisu raportu
    """
    filepath = os.path.join(config.OUTPUT_DIR, "reports", filename)
    
    with open(filepath, 'w') as f:
        # Nagłówek
        f.write("# Raport klasyfikacji raka piersi\n\n")
        
        # Podstawowe metryki
        f.write("## Metryki klasyfikacji\n\n")
        f.write("| Metryka | Wartość |\n")
        f.write("|---------|--------|\n")
        f.write(f"| Accuracy | {evaluation_results['accuracy']:.4f} |\n")
        f.write(f"| Precision | {evaluation_results['precision']:.4f} |\n")
        f.write(f"| Recall | {evaluation_results['recall']:.4f} |\n")
        f.write(f"| F1-score | {evaluation_results['f1']:.4f} |\n")
        if evaluation_results['roc_auc'] is not None:
            f.write(f"| AUC-ROC | {evaluation_results['roc_auc']:.4f} |\n")
        if evaluation_results['avg_precision'] is not None:
            f.write(f"| Average Precision | {evaluation_results['avg_precision']:.4f} |\n")
        f.write("\n")
        
        # Macierz pomyłek
        f.write("## Macierz pomyłek\n\n")
        f.write("```\n")
        f.write(str(evaluation_results['confusion_matrix']))
        f.write("\n```\n\n")
        
        # Interpretacja macierzy pomyłek
        cm = evaluation_results['confusion_matrix']
        tn, fp, fn, tp = cm.ravel()
        
        f.write("### Interpretacja macierzy pomyłek:\n\n")
        f.write(f"- True Negatives (TN): {tn} (Poprawnie sklasyfikowane przypadki łagodne)\n")
        f.write(f"- False Positives (FP): {fp} (Przypadki łagodne błędnie sklasyfikowane jako złośliwe)\n")
        f.write(f"- False Negatives (FN): {fn} (Przypadki złośliwe błędnie sklasyfikowane jako łagodne)\n")
        f.write(f"- True Positives (TP): {tp} (Poprawnie sklasyfikowane przypadki złośliwe)\n\n")
        
        # Szczegółowy raport klasyfikacji
        f.write("## Szczegółowy raport klasyfikacji\n\n")
        f.write("```\n")
        report_text = classification_report(
            y_true=(np.array([0, 1])[np.argmax(cm, axis=1)]),
            y_pred=(np.array([0, 1])[np.argmax(cm, axis=0)]),
            target_names=['Łagodny (0)', 'Złośliwy (1)']
        )
        f.write(f"{report_text}\n")
        f.write("```\n\n")
        
        # Wnioski
        f.write("## Wnioski\n\n")
        
        if evaluation_results['accuracy'] > 0.95:
            f.write("- Model osiąga **bardzo wysoką dokładność** (>95%), co świadczy o jego doskonałej jakości.\n")
        elif evaluation_results['accuracy'] > 0.9:
            f.write("- Model osiąga **wysoką dokładność** (>90%), co świadczy o jego dobrej jakości.\n")
        elif evaluation_results['accuracy'] > 0.8:
            f.write("- Model osiąga **dobrą dokładność** (>80%), ale jest miejsce na poprawę.\n")
        else:
            f.write("- Dokładność modelu jest **umiarkowana** (<80%), co wskazuje na potrzebę dalszej optymalizacji.\n")
        
        if evaluation_results['recall'] > 0.95:
            f.write("- Model ma **bardzo wysoką czułość** (recall >95%), co jest kluczowe w kontekście diagnostyki medycznej.\n")
        elif evaluation_results['recall'] > 0.9:
            f.write("- Model ma **wysoką czułość** (recall >90%), co jest istotne w kontekście diagnostyki medycznej.\n")
        else:
            f.write("- Czułość modelu (recall <90%) może wymagać poprawy, aby minimalizować liczbę fałszywie negatywnych wyników.\n")
        
        if evaluation_results['precision'] > 0.95:
            f.write("- Model ma **bardzo wysoką precyzję** (>95%), co oznacza niskie ryzyko fałszywie pozytywnych wyników.\n")
        elif evaluation_results['precision'] > 0.9:
            f.write("- Model ma **wysoką precyzję** (>90%), co oznacza względnie niskie ryzyko fałszywie pozytywnych wyników.\n")
        
        if fn > 0:
            f.write(f"- Model nie rozpoznał {fn} przypadków złośliwych, co stanowi {fn/(fn+tp)*100:.1f}% wszystkich przypadków złośliwych. ")
            f.write("W kontekście medycznym należy dążyć do minimalizacji tej wartości.\n")
        
        f.write("\n### Implikacje kliniczne:\n\n")
        f.write("- W kontekście diagnostyki raka piersi, wyższa czułość (recall) jest często priorytetem, ")
        f.write("aby minimalizować liczbę nierozpoznanych przypadków raka.\n")
        f.write("- Błędy typu fałszywie negatywne (FN) są bardziej kosztowne niż fałszywie pozytywne (FP), ")
        f.write("ponieważ opóźnienie w diagnozie raka może prowadzić do poważniejszych konsekwencji zdrowotnych.\n")
        
        f.write("\n### Możliwe kierunki poprawy modelu:\n\n")
        f.write("1. Eksperymentowanie z różnymi architekturami sieci neuronowej (liczba warstw, neuronów).\n")
        f.write("2. Testowanie różnych metod regulacji (alpha) i algorytmów optymalizacji.\n")
        f.write("3. Rozważenie innych metod selekcji cech lub użycie większej liczby cech.\n")
        f.write("4. Zwiększenie zbioru danych treningowych, jeśli to możliwe.\n")
    
    print(f"Raport klasyfikacji zapisany do: {filepath}")

def evaluate_and_report(data_dict):
    """
    Główna funkcja ewaluująca model i generująca raport.
    
    Args:
        data_dict (dict): Słownik z przetworzonymi danymi i wytrenowanym modelem
        
    Returns:
        dict: Zaktualizowany słownik z wynikami ewaluacji
    """
    print("\n--- EWALUACJA I RAPORTOWANIE ---")
    
    # Ewaluacja modelu
    evaluation_results = evaluate_model(
        data_dict['model'],
        data_dict['X_test_selected'],
        data_dict['y_test']
    )
    
    # Generowanie raportu
    generate_classification_report(evaluation_results)
    
    # Aktualizacja słownika danych
    data_dict.update({
        'evaluation_results': evaluation_results
    })
    
    return data_dict

# Gdy moduł jest uruchamiany bezpośrednio
if __name__ == "__main__":
    from data_processing import process_data
    from feature_selection import perform_feature_selection
    from training import train_and_validate_model
    
    # Przetwarzanie danych
    data_dict = process_data()
    
    # Selekcja cech
    data_dict = perform_feature_selection(data_dict)
    
    # Trenowanie i walidacja modelu
    data_dict = train_and_validate_model(data_dict)
    
    # Ewaluacja i raportowanie
    data_dict = evaluate_and_report(data_dict)