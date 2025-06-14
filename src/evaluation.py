import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, classification_report
import os


import config

def evaluate_model(model, X_test, y_test, threshold=None):
 
    
    if threshold is None:
        threshold = config.EVALUATION_CONFIG['threshold']
    
    print("\n--- EWALUACJA MODELU ---")
    
    
    y_pred = model.predict(X_test)
    
    
    accuracy = accuracy_score(y_test, y_pred)

    recall = recall_score(y_test, y_pred)
    
    
    cm = confusion_matrix(y_test, y_pred)
    
    
    print(f"Wyniki ewaluacji modelu:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Recall: {recall:.4f}")
    
    print("\nMacierz pomylek:")
    print("Rzeczywiste \\ Przewidywane")
    print(f"Lagodny (0)    | {cm[0,0]:3d} | {cm[0,1]:3d} |")
    print(f"Zlosliwy (1)   | {cm[1,0]:3d} | {cm[1,1]:3d} |")
    print(f"               |  0  |  1  |")
    
    
    evaluation_results = {
        'accuracy': accuracy,
        'recall': recall,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'threshold': threshold
    }
    
    
    visualize_confusion_matrix(cm)
    
    return evaluation_results

def visualize_confusion_matrix(cm, filename="confusion_matrix.png"):
 
    plt.figure(figsize=config.VISUALIZATION_CONFIG['figsize_medium'])
    
    
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

def generate_text_report(evaluation_results, cv_results=None, grid_search_results=None, model_comparison=None, filename="classification_report.txt"):
    """
    Generuje szczegółowy raport tekstowy z wyników klasyfikacji.
    
    Args:
        evaluation_results (dict): Wyniki ewaluacji modelu
        cv_results (dict): Wyniki walidacji krzyżowej
        grid_search_results (dict): Wyniki grid search (opcjonalne)
        model_comparison (dict): Porównanie modelu bazowego vs zoptymalizowanego (opcjonalne)
        filename (str): Nazwa pliku raportu
    """
    filepath = os.path.join(config.OUTPUT_DIR, "reports", filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        
        f.write("=" * 60 + "\n")
        f.write("RAPORT KLASYFIKACJI RAKA PIERSI\n")
        f.write("=" * 60 + "\n\n")
        
        
        if model_comparison:
            f.write("PORÓWNANIE: MODEL BAZOWY vs ZOPTYMALIZOWANY\n")
            f.write("-" * 30 + "\n")
            f.write(f"MODEL BAZOWY (MODEL_CONFIG):\n")
            f.write(f"  Accuracy: {model_comparison['base_accuracy']:.4f}\n")
            f.write(f"  Recall:   {model_comparison['base_recall']:.4f}\n\n")
            
            f.write(f"MODEL ZOPTYMALIZOWANY (Grid Search):\n")
            f.write(f"  Accuracy: {model_comparison['optimized_accuracy']:.4f}\n")
            f.write(f"  Recall:   {model_comparison['optimized_recall']:.4f}\n\n")
            
            f.write(f"POPRAWA PO OPTYMALIZACJI:\n")
            acc_perc = model_comparison['accuracy_improvement']/model_comparison['base_accuracy']*100
            rec_perc = model_comparison['recall_improvement']/model_comparison['base_recall']*100
            f.write(f"  Accuracy: {model_comparison['accuracy_improvement']:+.4f} ({acc_perc:+.2f}%)\n")
            f.write(f"  Recall:   {model_comparison['recall_improvement']:+.4f} ({rec_perc:+.2f}%)\n\n")
        
        
        if grid_search_results:
            f.write("OPTYMALIZACJA HIPERPARAMETRÓW (GRID SEARCH)\n")
            f.write("-" * 30 + "\n")
            f.write(f"Przetestowano kombinacji parametrów: {grid_search_results['n_combinations']}\n")
            f.write(f"Czas grid search: {grid_search_results['search_time']:.2f} sekund\n")
            f.write(f"Najlepszy recall CV: {grid_search_results['best_score']:.4f}\n\n")
            
            f.write("NAJLEPSZE PARAMETRY:\n")
            for param, value in grid_search_results['best_params'].items():
                f.write(f"  {param}: {value}\n")
            f.write("\n")
        
        
        f.write("METRYKI KLASYFIKACJI (ZBIÓR TESTOWY)\n")
        f.write("-" * 30 + "\n")
        f.write(f"Accuracy:  {evaluation_results['accuracy']:.4f}\n")
        f.write(f"Recall:    {evaluation_results['recall']:.4f}\n\n")
        
        
        f.write("MACIERZ POMYLEK\n")
        f.write("-" * 30 + "\n")
        cm = evaluation_results['confusion_matrix']
        f.write("Rzeczywiste \\ Przewidywane\n")
        f.write(f"Lagodny (0)    | {cm[0,0]:3d} | {cm[0,1]:3d} |\n")
        f.write(f"Zlosliwy (1)   | {cm[1,0]:3d} | {cm[1,1]:3d} |\n")
        f.write(f"               |  0  |  1  |\n\n")
        
        
        tn, fp, fn, tp = cm.ravel()
        total = tn + fp + fn + tp
        
        f.write("INTERPRETACJA MACIERZY POMYLEK\n")
        f.write("-" * 30 + "\n")
        f.write(f"True Negatives (TN):  {tn:3d} ({tn/total*100:.1f}%) - Poprawnie sklasyfikowane przypadki lagodne\n")
        f.write(f"False Positives (FP): {fp:3d} ({fp/total*100:.1f}%) - Przypadki lagodne blednie sklasyfikowane jako zlosliwe\n")
        f.write(f"False Negatives (FN): {fn:3d} ({fn/total*100:.1f}%) - Przypadki zlosliwe blednie sklasyfikowane jako lagodne\n")
        f.write(f"True Positives (TP):  {tp:3d} ({tp/total*100:.1f}%) - Poprawnie sklasyfikowane przypadki zlosliwe\n\n")
        
        
        if cv_results:
            f.write("WYNIKI WALIDACJI KRZYZOWEJ\n")
            f.write("-" * 30 + "\n")
            f.write(f"Accuracy:  {cv_results['accuracy']['mean']:.4f} ± {cv_results['accuracy']['std']:.4f}\n")
            f.write(f"Recall:    {cv_results['recall']['mean']:.4f} ± {cv_results['recall']['std']:.4f}\n")
            f.write(f"Loss:      {cv_results['loss']['mean']:.4f} ± {cv_results['loss']['std']:.4f}\n\n")
            
            f.write("SZCZEGOLOWE WYNIKI DLA KAZDEGO FOLDA\n")
            f.write("-" * 30 + "\n")
            f.write("Fold | Accuracy |  Recall   | Loss\n")
            f.write("-----|----------|----------|----------\n")
            for i in range(len(cv_results['accuracy']['values'])):
                acc = cv_results['accuracy']['values'][i]
                rec = cv_results['recall']['values'][i]
                loss = cv_results['loss']['values'][i]
                f.write(f"{i+1:4d} | {acc:8.4f} | {rec:8.4f} | {loss:8.4f}\n")
            f.write("\n")
        
        
        f.write("WNIOSKI\n")
        f.write("-" * 30 + "\n")
        
        if model_comparison:
            acc_improved = model_comparison['accuracy_improvement'] > 0
            rec_improved = model_comparison['recall_improvement'] > 0
            if acc_improved or rec_improved:
                f.write("- Grid search skutecznie zoptymalizował model - osiągnięto poprawę względem konfiguracji bazowej.\n")
            else:
                f.write("- Model bazowy z MODEL_CONFIG był już dobrze skonfigurowany - grid search nie przyniósł znaczącej poprawy.\n")
        
        if grid_search_results:
            f.write("- Zastosowano optymalizację hiperparametrów za pomocą grid search, co pozwoliło na\n")
            f.write(f"  znalezienie najlepszej konfiguracji spośród {grid_search_results['n_combinations']} przetestowanych.\n")
        
        if evaluation_results['accuracy'] > 0.95:
            f.write("- Model osiąga bardzo wysoką dokładność (>95%), co świadczy o jego doskonałej jakości.\n")
        elif evaluation_results['accuracy'] > 0.9:
            f.write("- Model osiąga wysoką dokładność (>90%), co świadczy o jego dobrej jakości.\n")
        elif evaluation_results['accuracy'] > 0.8:
            f.write("- Model osiąga dobrą dokładność (>80%), ale jest miejsce na poprawę.\n")
        else:
            f.write("- Dokładność modelu jest umiarkowana (<80%), co wskazuje na potrzebę dalszej optymalizacji.\n")
        
        if evaluation_results['recall'] > 0.95:
            f.write("- Model ma bardzo wysoką czułość (recall >95%), co jest kluczowe w kontekście diagnostyki medycznej.\n")
        elif evaluation_results['recall'] > 0.9:
            f.write("- Model ma wysoką czułość (recall >90%), co jest istotne w kontekście diagnostyki medycznej.\n")
        else:
            f.write("- Czułość modelu (recall <90%) może wymagać poprawy, aby minimalizować liczbę fałszywie negatywnych wyników.\n")
        
        if fn > 0:
            f.write(f"- Model nie rozpoznał {fn} przypadków złośliwych, co stanowi {fn/(fn+tp)*100:.1f}% wszystkich przypadków złośliwych.\n")
            f.write("  W kontekście medycznym należy dążyć do minimalizacji tej wartości.\n")
    
    print(f"Raport klasyfikacji zapisany do: {filepath}")

def evaluate_and_report(data_dict):
    """
    Przeprowadza ewaluację modelu i generuje raport.
    
    Args:
        data_dict (dict): Słownik z danymi i wytrenowanym modelem
        
    Returns:
        dict: Zaktualizowany słownik z wynikami ewaluacji
    """
    print("\n--- EWALUACJA I RAPORTOWANIE ---")
    
    
    evaluation_results = evaluate_model(
        data_dict['model'],
        data_dict['X_test_selected'],
        data_dict['y_test']
    )
    
    
    generate_text_report(
        evaluation_results, 
        cv_results=data_dict.get('cv_results', None),
        grid_search_results=data_dict.get('grid_search_results', None),
        model_comparison=data_dict.get('model_comparison', None)
    )
    
    
    data_dict.update({
        'evaluation_results': evaluation_results
    })
    
    return data_dict
    """
    Przeprowadza ewaluację modelu i generuje raport.
    
    Args:
        data_dict (dict): Słownik z danymi i wytrenowanym modelem
        
    Returns:
        dict: Zaktualizowany słownik z wynikami ewaluacji
    """
    print("\n--- EWALUACJA I RAPORTOWANIE ---")
    
    
    evaluation_results = evaluate_model(
        data_dict['model'],
        data_dict['X_test_selected'],
        data_dict['y_test']
    )
    
    
    generate_text_report(
        evaluation_results, 
        cv_results=data_dict.get('cv_results', None),
        grid_search_results=data_dict.get('grid_search_results', None)
    )
    
    
    data_dict.update({
        'evaluation_results': evaluation_results
    })
    
    return data_dict
 
    print("\n--- EWALUACJA I RAPORTOWANIE ---")
    
    
    evaluation_results = evaluate_model(
        data_dict['model'],
        data_dict['X_test_selected'],
        data_dict['y_test']
    )
    
    
    generate_text_report(
        evaluation_results, 
        cv_results=data_dict.get('cv_results', None)
    )
    
    
    data_dict.update({
        'evaluation_results': evaluation_results
    })
    
    return data_dict


if __name__ == "__main__":
    from data_processing import process_data
    from feature_selection import perform_feature_selection
    from training import train_and_validate_model
    
    
    data_dict = process_data()
    
    
    data_dict = perform_feature_selection(data_dict)
    
    
    data_dict = train_and_validate_model(data_dict)
    
    
    data_dict = evaluate_and_report(data_dict)