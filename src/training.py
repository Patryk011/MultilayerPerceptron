import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score
import time
import os

import config
from src.model import create_mlp_classifier, save_model

def train_model(model, X_train, y_train):

    print("\n--- TRENOWANIE MODELU ---")
    
    start_time = time.time()
    
    # Trenowanie modelu
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    print(f"Trening zakończony po {model.n_iter_} iteracjach ({training_time:.2f} sekund)")
    
    return model

def perform_cross_validation(model, X, y, cv=None):
    """
    Przeprowadza walidację krzyżową modelu.
    
    Args:
        model: Model do walidacji
        X: Dane wejściowe
        y: Etykiety
        cv (int): Liczba foldów
        
    Returns:
        dict: Wyniki walidacji krzyżowej
    """
    # Używamy parametrów z konfiguracji, jeśli nie podano
    if cv is None:
        cv = config.TRAINING_CONFIG['cv_folds']
    
    print(f"\nPrzeprowadzanie {cv}-krotnej walidacji krzyżowej...")
    
    # Inicjalizacja stratyfikowanej walidacji krzyżowej
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=config.RANDOM_STATE)
    
    # Definicja metryk do obliczenia
    scoring = ['accuracy', 'recall', 'neg_log_loss']
    
    # Przeprowadzenie walidacji krzyżowej
    start_time = time.time()
    cv_results = cross_validate(
        model, X, y, 
        cv=cv_strategy, 
        scoring=scoring,
        return_train_score=False
    )
    cv_time = time.time() - start_time
    
    # Wyświetlenie szczegółowych wyników dla każdego folda
    print(f"\nSzczegółowe wyniki dla każdego folda:")
    print("-" * 60)
    for i in range(cv):
        accuracy = cv_results['test_accuracy'][i]
        recall = cv_results['test_recall'][i]
        loss = -cv_results['test_neg_log_loss'][i]  # Negujemy bo sklearn zwraca neg_log_loss
        
        print(f"Fold {i+1}: Accuracy={accuracy:.4f},  "
              f"Recall={recall:.4f}, Loss={loss:.4f}")
    
    # Obliczenie średnich i odchyleń standardowych
    results_summary = {
        'accuracy': {
            'mean': np.mean(cv_results['test_accuracy']),
            'std': np.std(cv_results['test_accuracy']),
            'values': cv_results['test_accuracy']
        },
        
        'recall': {
            'mean': np.mean(cv_results['test_recall']),
            'std': np.std(cv_results['test_recall']),
            'values': cv_results['test_recall']
        },
        'loss': {
            'mean': np.mean(-cv_results['test_neg_log_loss']),
            'std': np.std(-cv_results['test_neg_log_loss']),
            'values': -cv_results['test_neg_log_loss']
        },
        'time': cv_time
    }
    
    # Wyświetlenie podsumowania
    print("-" * 60)
    print(f"Podsumowanie walidacji krzyzowej ({cv_time:.2f} sekund):")
    print(f"  Accuracy: {results_summary['accuracy']['mean']:.4f} ± {results_summary['accuracy']['std']:.4f}")
    print(f"  Recall: {results_summary['recall']['mean']:.4f} ± {results_summary['recall']['std']:.4f}")
    print(f"  Loss: {results_summary['loss']['mean']:.4f} ± {results_summary['loss']['std']:.4f}")
    
    # Wizualizacja wyników
    visualize_cross_validation_results(results_summary)
    
    return results_summary

def visualize_cross_validation_results(cv_results, filename="cv_results.png"):
    """
    Wizualizuje wyniki walidacji krzyżowej - tylko podstawowe metryki.
    
    Args:
        cv_results (dict): Wyniki walidacji krzyżowej
        filename (str): Nazwa pliku do zapisu wykresu
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Wykres 1: Średnie wartości metryk z błędami
    ax1 = axes[0]
    metrics = ['accuracy', 'recall', 'loss']
    mean_values = [cv_results[metric]['mean'] for metric in metrics]
    std_values = [cv_results[metric]['std'] for metric in metrics]
    
    bars = ax1.bar(metrics, mean_values, yerr=std_values, capsize=10, color=['skyblue', 'lightcoral', 'lightgreen'])
    
    # Dodajemy wartości na szczycie słupków
    for bar, mean, std in zip(bars, mean_values, std_values):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.,
            height + std,
            f'{mean:.3f}±{std:.3f}',
            ha='center',
            va='bottom',
            fontsize=9
        )
    
    ax1.set_title('Srednie wyniki walidacji krzyzowej')
    ax1.set_ylabel('Wartosc')
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Wykres 2: Rozkład wartości dla accuracy i recall
    ax2 = axes[1]
    fold_numbers = list(range(1, len(cv_results['accuracy']['values']) + 1))
    
    ax2.plot(fold_numbers, cv_results['accuracy']['values'], 'o-', label='Accuracy', linewidth=2, markersize=6)
    ax2.plot(fold_numbers, cv_results['recall']['values'], 's-', label='Recall', linewidth=2, markersize=6)
    
    ax2.set_title('Wyniki dla poszczegolnych foldow')
    ax2.set_xlabel('Numer folda')
    ax2.set_ylabel('Wartosc')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(fold_numbers)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, "plots", "model", filename), 
                dpi=config.VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    print(f"Zapisano wizualizacje wynikow walidacji krzyzowej: {filename}")

def train_and_validate_model(data_dict):
    """
    Główna funkcja trenująca i walidująca model.
    
    Args:
        data_dict (dict): Słownik z przetworzonymi danymi
        
    Returns:
        dict: Zaktualizowany słownik z wytrenowanym modelem
    """
    print("\n--- TRENOWANIE I WALIDACJA MODELU ---")
    
    # Tworzenie modelu
    model = create_mlp_classifier()
    
    # Przeprowadzenie walidacji krzyżowej
    cv_results = perform_cross_validation(
        model, 
        data_dict['X_train_selected'], 
        data_dict['y_train']
    )
    
    # Trenowanie modelu na całym zbiorze treningowym
    trained_model = train_model(
        model, 
        data_dict['X_train_selected'], 
        data_dict['y_train']
    )
    
    # Zapisanie modelu
    save_model(trained_model)
    
    # Aktualizacja słownika danych
    data_dict.update({
        'model': trained_model,
        'cv_results': cv_results
    })
    
    return data_dict

# Gdy moduł jest uruchamiany bezpośrednio
if __name__ == "__main__":
    from data_processing import process_data
    from feature_selection import perform_feature_selection
    
    # Przetwarzanie danych
    data_dict = process_data()
    
    # Selekcja cech
    data_dict = perform_feature_selection(data_dict)
    
    # Trenowanie i walidacja modelu
    data_dict = train_and_validate_model(data_dict)