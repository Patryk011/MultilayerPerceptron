import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import os


import config

def evaluate_model(model, X_test, y_test, threshold=None):
 
    # Używamy parametrów z konfiguracji, jeśli nie podano
    if threshold is None:
        threshold = config.EVALUATION_CONFIG['threshold']
    
    print("\n--- EWALUACJA MODELU ---")
    
    # Predykcje modelu
    y_pred = model.predict(X_test)
    
    # Metryki podstawowe
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    # Macierz pomyłek
    cm = confusion_matrix(y_test, y_pred)
    
    # Wyświetlenie wyników
    print(f"Wyniki ewaluacji modelu:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    
    print("\nMacierz pomylek:")
    print("Rzeczywiste \\ Przewidywane")
    print(f"Lagodny (0)    | {cm[0,0]:3d} | {cm[0,1]:3d} |")
    print(f"Zlosliwy (1)   | {cm[1,0]:3d} | {cm[1,1]:3d} |")
    print(f"               |  0  |  1  |")
    
    # Przygotowanie wyników
    evaluation_results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'threshold': threshold
    }
    
    # Wizualizacja macierzy pomyłek
    visualize_confusion_matrix(cm)
    
    return evaluation_results

def visualize_confusion_matrix(cm, filename="confusion_matrix.png"):
 
    plt.figure(figsize=config.VISUALIZATION_CONFIG['figsize_medium'])
    
    # Tworzymy mapę ciepła macierzy pomyłek
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=['Lagodny (0)', 'Zlosliwy (1)'],
        yticklabels=['Lagodny (0)', 'Zlosliwy (1)'],
        cbar_kws={'label': 'Liczba probek'}
    )
    
    plt.xlabel('Przewidywane')
    plt.ylabel('Rzeczywiste')
    plt.title('Macierz pomylek')
    
    # Dodajemy wartości procentowe jako tekst
    total = np.sum(cm)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            percentage = cm[i, j] / total * 100
            plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                    ha='center', va='center', fontsize=10, color='red')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, "plots", "model", filename), 
                dpi=config.VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    print(f"Zapisano wizualizacje macierzy pomylek: {filename}")

def generate_text_report(evaluation_results, cv_results=None, filename="classification_report.txt"):
 
    filepath = os.path.join(config.OUTPUT_DIR, "reports", filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        # Nagłówek
        f.write("=" * 60 + "\n")
        f.write("RAPORT KLASYFIKACJI RAKA PIERSI\n")
        f.write("=" * 60 + "\n\n")
        
        # Podstawowe metryki
        f.write("METRYKI KLASYFIKACJI\n")
        f.write("-" * 30 + "\n")
        f.write(f"Accuracy:  {evaluation_results['accuracy']:.4f}\n")
        f.write(f"Precision: {evaluation_results['precision']:.4f}\n")
        f.write(f"Recall:    {evaluation_results['recall']:.4f}\n\n")
        
        # Macierz pomyłek
        f.write("MACIERZ POMYLEK\n")
        f.write("-" * 30 + "\n")
        cm = evaluation_results['confusion_matrix']
        f.write("Rzeczywiste \\ Przewidywane\n")
        f.write(f"Lagodny (0)    | {cm[0,0]:3d} | {cm[0,1]:3d} |\n")
        f.write(f"Zlosliwy (1)   | {cm[1,0]:3d} | {cm[1,1]:3d} |\n")
        f.write(f"               |  0  |  1  |\n\n")
        
        # Interpretacja macierzy pomyłek
        tn, fp, fn, tp = cm.ravel()
        total = tn + fp + fn + tp
        
        f.write("INTERPRETACJA MACIERZY POMYLEK\n")
        f.write("-" * 30 + "\n")
        f.write(f"True Negatives (TN):  {tn:3d} ({tn/total*100:.1f}%) - Poprawnie sklasyfikowane przypadki lagodne\n")
        f.write(f"False Positives (FP): {fp:3d} ({fp/total*100:.1f}%) - Przypadki lagodne blednie sklasyfikowane jako zlosliwe\n")
        f.write(f"False Negatives (FN): {fn:3d} ({fn/total*100:.1f}%) - Przypadki zlosliwe blednie sklasyfikowane jako lagodne\n")
        f.write(f"True Positives (TP):  {tp:3d} ({tp/total*100:.1f}%) - Poprawnie sklasyfikowane przypadki zlosliwe\n\n")
        
        # Walidacja krzyżowa (jeśli dostępna)
        if cv_results:
            f.write("WYNIKI WALIDACJI KRZYZOWEJ\n")
            f.write("-" * 30 + "\n")
            f.write(f"Accuracy:  {cv_results['accuracy']['mean']:.4f} ± {cv_results['accuracy']['std']:.4f}\n")
            f.write(f"Precision: {cv_results['precision']['mean']:.4f} ± {cv_results['precision']['std']:.4f}\n")
            f.write(f"Recall:    {cv_results['recall']['mean']:.4f} ± {cv_results['recall']['std']:.4f}\n")
            f.write(f"Loss:      {cv_results['loss']['mean']:.4f} ± {cv_results['loss']['std']:.4f}\n\n")
            
            f.write("SZCZEGOLOWE WYNIKI DLA KAZDEGO FOLDA\n")
            f.write("-" * 30 + "\n")
            f.write("Fold | Accuracy | Precision | Recall   | Loss\n")
            f.write("-----|----------|-----------|----------|----------\n")
            for i in range(len(cv_results['accuracy']['values'])):
                acc = cv_results['accuracy']['values'][i]
                prec = cv_results['precision']['values'][i]
                rec = cv_results['recall']['values'][i]
                loss = cv_results['loss']['values'][i]
                f.write(f"{i+1:4d} | {acc:8.4f} | {prec:9.4f} | {rec:8.4f} | {loss:8.4f}\n")
            f.write("\n")
        
        # Wnioski
        f.write("WNIOSKI\n")
        f.write("-" * 30 + "\n")
        
        if evaluation_results['accuracy'] > 0.95:
            f.write("- Model osiaga bardzo wysoka dokladnosc (>95%), co swiadczy o jego doskonalej jakosci.\n")
        elif evaluation_results['accuracy'] > 0.9:
            f.write("- Model osiaga wysoka dokladnosc (>90%), co swiadczy o jego dobrej jakosci.\n")
        elif evaluation_results['accuracy'] > 0.8:
            f.write("- Model osiaga dobra dokladnosc (>80%), ale jest miejsce na poprawe.\n")
        else:
            f.write("- Dokladnosc modelu jest umiarkowana (<80%), co wskazuje na potrzebe dalszej optymalizacji.\n")
        
        if evaluation_results['recall'] > 0.95:
            f.write("- Model ma bardzo wysoka czulosc (recall >95%), co jest kluczowe w kontekscie diagnostyki medycznej.\n")
        elif evaluation_results['recall'] > 0.9:
            f.write("- Model ma wysoka czulosc (recall >90%), co jest istotne w kontekscie diagnostyki medycznej.\n")
        else:
            f.write("- Czulosc modelu (recall <90%) moze wymagac poprawy, aby minimalizowac liczbe falszywie negatywnych wynikow.\n")
        
        if evaluation_results['precision'] > 0.95:
            f.write("- Model ma bardzo wysoka precyzje (>95%), co oznacza niskie ryzyko falszywie pozytywnych wynikow.\n")
        elif evaluation_results['precision'] > 0.9:
            f.write("- Model ma wysoka precyzje (>90%), co oznacza wzglednie niskie ryzyko falszywie pozytywnych wynikow.\n")
        
        if fn > 0:
            f.write(f"- Model nie rozpoznal {fn} przypadkow zlosliwych, co stanowi {fn/(fn+tp)*100:.1f}% wszystkich przypadkow zlosliwych.\n")
            f.write("  W kontekscie medycznym nalezy dazyc do minimalizacji tej wartosci.\n")
        
    
    
    print(f"Raport klasyfikacji zapisany do: {filepath}")

def evaluate_and_report(data_dict):
 
    print("\n--- EWALUACJA I RAPORTOWANIE ---")
    
    # Ewaluacja modelu
    evaluation_results = evaluate_model(
        data_dict['model'],
        data_dict['X_test_selected'],
        data_dict['y_test']
    )
    
    # Generowanie raportu
    generate_text_report(
        evaluation_results, 
        cv_results=data_dict.get('cv_results', None)
    )
    
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