import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score
import time
import os

import config
from src.model import create_mlp_classifier, save_model

def perform_grid_search(X_train, y_train, cv=None):
    """
    Przeprowadza grid search dla optymalizacji hiperparametrów modelu MLP.
    
    Args:
        X_train: Dane treningowe
        y_train: Etykiety treningowe
        cv (int): Liczba foldów dla walidacji krzyżowej
        
    Returns:
        dict: Wyniki grid search z najlepszym modelem
    """
    if cv is None:
        cv = config.TRAINING_CONFIG['cv_folds']
    
    print(f"\n--- GRID SEARCH ---")
    print(f"Przeszukiwanie najlepszych hiperparametrów za pomocą {cv}-krotnej walidacji krzyżowej...")
    
    # Definicja przestrzeni hiperparametrów do przeszukania
    param_grid = {
        'hidden_layer_sizes': [
            (32,), (64,), (128,),           # Jedna warstwa ukryta
            (64, 32), (128, 64), (100, 50), # Dwie warstwy ukryte
            (128, 64, 32)                   # Trzy warstwy ukryte
        ],
        'alpha': [0.0001, 0.001, 0.01, 0.1],  # Regularyzacja L2
        'learning_rate_init': [0.001, 0.01, 0.1],  # Współczynnik uczenia
        'activation': ['relu', 'tanh'],      # Funkcja aktywacji
        'solver': ['adam', 'lbfgs']          # Algorytm optymalizacji
    }
    
    # Tworzenie bazowego modelu (parametry będą nadpisane przez grid search)
    base_model = create_mlp_classifier()
    
    # Inicjalizacja Grid Search
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=config.RANDOM_STATE),
        scoring=['accuracy', 'recall'],  # Główne metryki
        refit='recall',  # Wybieramy model z najlepszym recall (ważne w medycynie)
        n_jobs=-1,  # Wykorzystanie wszystkich dostępnych rdzeni
        verbose=1,  # Pokazuj postęp
        return_train_score=False
    )
    
    # Przeprowadzenie grid search
    start_time = time.time()
    print("Rozpoczynanie grid search... To może chwilę potrwać.")
    
    grid_search.fit(X_train, y_train)
    
    search_time = time.time() - start_time
    
    # Wyświetlenie wyników
    print(f"\nGrid search zakończony po {search_time:.2f} sekundach")
    print(f"Przetestowano {len(grid_search.cv_results_['params'])} kombinacji parametrów")
    
    print(f"\nNajlepsze parametry:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    
    print(f"\nNajlepsze wyniki walidacji krzyżowej:")
    print(f"  Recall: {grid_search.cv_results_['mean_test_recall'][grid_search.best_index_]:.4f}")
    print(f"  Accuracy: {grid_search.cv_results_['mean_test_accuracy'][grid_search.best_index_]:.4f}")
    
    # Top 5 najlepszych kombinacji
    print(f"\nTop 5 najlepszych kombinacji parametrów (według recall):")
    results_df = pd.DataFrame(grid_search.cv_results_)
    top_results = results_df.nlargest(5, 'mean_test_recall')[
        ['params', 'mean_test_recall', 'mean_test_accuracy', 'std_test_recall', 'std_test_accuracy']
    ]
    
    for i, (idx, row) in enumerate(top_results.iterrows(), 1):
        print(f"\n  {i}. Recall: {row['mean_test_recall']:.4f} ± {row['std_test_recall']:.4f}, "
              f"Accuracy: {row['mean_test_accuracy']:.4f} ± {row['std_test_accuracy']:.4f}")
        print(f"     Parametry: {row['params']}")
    
    # Wizualizacja wyników grid search
    visualize_grid_search_results(grid_search)
    
    # Przygotowanie wyników
    grid_search_results = {
        'best_model': grid_search.best_estimator_,
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'cv_results': grid_search.cv_results_,
        'search_time': search_time,
        'n_combinations': len(grid_search.cv_results_['params'])
    }
    
    return grid_search_results

def visualize_grid_search_results(grid_search, filename="grid_search_results.png"):
    """
    Wizualizuje wyniki grid search.
    
    Args:
        grid_search: Obiekt GridSearchCV po wykonaniu fit()
        filename (str): Nazwa pliku do zapisu
    """
    # Przygotowanie danych
    results_df = pd.DataFrame(grid_search.cv_results_)
    
    # Tworzenie wykresów
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Wykres 1: Top 20 wyników
    ax1 = axes[0, 0]
    top_20 = results_df.nlargest(20, 'mean_test_recall')
    x_pos = np.arange(len(top_20))
    
    bars = ax1.bar(x_pos, top_20['mean_test_recall'], 
                   yerr=top_20['std_test_recall'], capsize=3, alpha=0.7)
    
    # Oznaczenie najlepszego wyniku
    bars[0].set_color('red')
    
    ax1.set_title('Top 20 wyników Grid Search (Recall)')
    ax1.set_xlabel('Ranking kombinacji')
    ax1.set_ylabel('Recall')
    ax1.grid(True, alpha=0.3)
    
    # Wykres 2: Rozkład wyników recall
    ax2 = axes[0, 1]
    ax2.hist(results_df['mean_test_recall'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(grid_search.best_score_, color='red', linestyle='--', 
                label=f'Najlepszy: {grid_search.best_score_:.4f}')
    ax2.set_title('Rozkład wyników Recall')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Częstość')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Wykres 3: Recall vs Accuracy
    ax3 = axes[1, 0]
    scatter = ax3.scatter(results_df['mean_test_accuracy'], results_df['mean_test_recall'], 
                         alpha=0.6, c=results_df['mean_test_recall'], cmap='viridis')
    
    # Oznaczenie najlepszego wyniku
    best_idx = grid_search.best_index_
    ax3.scatter(results_df.iloc[best_idx]['mean_test_accuracy'], 
               results_df.iloc[best_idx]['mean_test_recall'],
               color='red', s=100, marker='*', label='Najlepszy model')
    
    ax3.set_title('Recall vs Accuracy')
    ax3.set_xlabel('Accuracy')
    ax3.set_ylabel('Recall')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='Recall')
    
    # Wykres 4: Wpływ liczby neuronów w pierwszej warstwie
    ax4 = axes[1, 1]
    
    # Wyciągnięcie pierwszej wartości z hidden_layer_sizes dla każdej kombinacji
    first_layer_sizes = []
    recalls = []
    
    for params, recall in zip(results_df['params'], results_df['mean_test_recall']):
        hidden_layers = params['hidden_layer_sizes']
        if isinstance(hidden_layers, tuple) and len(hidden_layers) > 0:
            first_layer_sizes.append(hidden_layers[0])
            recalls.append(recall)
    
    # Grupowanie i obliczanie średnich
    size_recall_dict = {}
    for size, recall in zip(first_layer_sizes, recalls):
        if size not in size_recall_dict:
            size_recall_dict[size] = []
        size_recall_dict[size].append(recall)
    
    sizes = sorted(size_recall_dict.keys())
    mean_recalls = [np.mean(size_recall_dict[size]) for size in sizes]
    std_recalls = [np.std(size_recall_dict[size]) for size in sizes]
    
    ax4.errorbar(sizes, mean_recalls, yerr=std_recalls, marker='o', capsize=5)
    ax4.set_title('Wpływ wielkości pierwszej warstwy na Recall')
    ax4.set_xlabel('Liczba neuronów w pierwszej warstwie')
    ax4.set_ylabel('Średni Recall')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, "plots", "model", filename), 
                dpi=config.VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    print(f"Zapisano wizualizację wyników grid search: {filename}")

def train_model(model, X_train, y_train):
    """
    Trenuje model na danych treningowych.
    
    Args:
        model: Model do wytrenowania
        X_train: Dane treningowe
        y_train: Etykiety treningowe
        
    Returns:
        model: Wytrenowany model
    """
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

def train_and_validate_model(data_dict, use_grid_search=True):
    """
    Główna funkcja trenująca i walidująca model.
    
    Args:
        data_dict (dict): Słownik z przetworzonymi danymi
        use_grid_search (bool): Czy użyć grid search do optymalizacji hiperparametrów
        
    Returns:
        dict: Zaktualizowany słownik z wytrenowanym modelem
    """
    print("\n--- TRENOWANIE I WALIDACJA MODELU ---")
    
    if use_grid_search:
        # Przeprowadzenie grid search
        grid_search_results = perform_grid_search(
            data_dict['X_train_selected'], 
            data_dict['y_train']
        )
        
        # Używamy najlepszego modelu z grid search
        best_model = grid_search_results['best_model']
        print(f"\nUżywam najlepszego modelu z grid search.")
        
        # Dodatkowa walidacja krzyżowa dla najlepszego modelu
        cv_results = perform_cross_validation(
            best_model, 
            data_dict['X_train_selected'], 
            data_dict['y_train']
        )
        
        # Aktualizacja słownika danych
        data_dict.update({
            'model': best_model,
            'cv_results': cv_results,
            'grid_search_results': grid_search_results
        })
        
    else:
        # Standardowa procedura bez grid search
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
        
        # Aktualizacja słownika danych
        data_dict.update({
            'model': trained_model,
            'cv_results': cv_results
        })
    
    # Zapisanie modelu
    save_model(data_dict['model'])
    
    return data_dict

# Gdy moduł jest uruchamiany bezpośrednio
if __name__ == "__main__":
    from data_processing import process_data
    from feature_selection import perform_feature_selection
    
    # Przetwarzanie danych
    data_dict = process_data()
    
    # Selekcja cech
    data_dict = perform_feature_selection(data_dict)
    
    # Trenowanie i walidacja modelu z grid search
    data_dict = train_and_validate_model(data_dict, use_grid_search=True)