"""
Moduł odpowiedzialny za eksperymenty z różnymi algorytmami uczenia i parametrami modelu.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import time
import os
import itertools


import config

def compare_learning_algorithms(X_train, y_train, X_test, y_test):
    """
    Porównuje różne algorytmy uczenia (solvers) dla MLPClassifier.
    
    Args:
        X_train: Dane treningowe
        y_train: Etykiety treningowe
        X_test: Dane testowe
        y_test: Etykiety testowe
        
    Returns:
        dict: Wyniki porównania algorytmów
    """
    print("\n--- PORÓWNANIE ALGORYTMÓW UCZENIA ---")
    
    solvers = config.ALGORITHM_EXPERIMENTS['solvers']
    results = []
    
    for solver in solvers:
        print(f"\nTestowanie algorytmu: {solver}")
        
        # Dostosowanie parametrów dla każdego solvera
        if solver == 'lbfgs':
            # LBFGS nie obsługuje mini-batch, więc używamy domyślnych parametrów
            model = MLPClassifier(
                hidden_layer_sizes=config.MODEL_CONFIG['hidden_layer_sizes'],
                activation=config.MODEL_CONFIG['activation'],
                solver=solver,
                alpha=config.MODEL_CONFIG['alpha'],
                max_iter=1000,
                random_state=config.RANDOM_STATE
            )
        else:
            model = MLPClassifier(
                hidden_layer_sizes=config.MODEL_CONFIG['hidden_layer_sizes'],
                activation=config.MODEL_CONFIG['activation'],
                solver=solver,
                alpha=config.MODEL_CONFIG['alpha'],
                batch_size=config.MODEL_CONFIG['batch_size'],
                learning_rate=config.MODEL_CONFIG['learning_rate'],
                learning_rate_init=config.MODEL_CONFIG['learning_rate_init'],
                max_iter=config.MODEL_CONFIG['max_iter'],
                early_stopping=True,
                validation_fraction=0.2,
                n_iter_no_change=20,
                random_state=config.RANDOM_STATE
            )
        
        # Trenowanie modelu
        start_time = time.time()
        try:
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Predykcje
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Metryki
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc_score = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
            
            # Zapisanie wyników
            result = {
                'solver': solver,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc_score,
                'training_time': training_time,
                'n_iter': model.n_iter_,
                'converged': model.n_iter_ < model.max_iter
            }
            
            results.append(result)
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-score: {f1:.4f}")
            print(f"  AUC: {auc_score:.4f}" if auc_score else "  AUC: N/A")
            print(f"  Czas treningu: {training_time:.2f}s")
            print(f"  Liczba iteracji: {model.n_iter_}")
            
        except Exception as e:
            print(f"  Błąd podczas treningu z {solver}: {e}")
            continue
    
    # Konwersja do DataFrame
    results_df = pd.DataFrame(results)
    
    # Wizualizacja wyników
    if not results_df.empty:
        visualize_algorithm_comparison(results_df)
    
    return results_df

def visualize_algorithm_comparison(results_df, filename="algorithm_comparison.png"):
    """
    Wizualizuje porównanie algorytmów uczenia.
    
    Args:
        results_df (DataFrame): Wyniki porównania algorytmów
        filename (str): Nazwa pliku do zapisu wykresu
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Wykres 1: Porównanie metryk
    ax1 = axes[0, 0]
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    x = np.arange(len(results_df))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        ax1.bar(x + i * width, results_df[metric], width, label=metric)
    
    ax1.set_xlabel('Algorytm')
    ax1.set_ylabel('Wartość')
    ax1.set_title('Porównanie metryk dla różnych algorytmów')
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(results_df['solver'])
    ax1.legend()
    ax1.grid(True, axis='y')
    
    # Wykres 2: Czas treningu
    ax2 = axes[0, 1]
    ax2.bar(results_df['solver'], results_df['training_time'], color='orange')
    ax2.set_xlabel('Algorytm')
    ax2.set_ylabel('Czas treningu (sekundy)')
    ax2.set_title('Czas treningu dla różnych algorytmów')
    ax2.grid(True, axis='y')
    
    # Wykres 3: Liczba iteracji
    ax3 = axes[1, 0]
    ax3.bar(results_df['solver'], results_df['n_iter'], color='green')
    ax3.set_xlabel('Algorytm')
    ax3.set_ylabel('Liczba iteracji')
    ax3.set_title('Liczba iteracji do konwergencji')
    ax3.grid(True, axis='y')
    
    # Wykres 4: AUC (jeśli dostępne)
    ax4 = axes[1, 1]
    if 'auc' in results_df.columns and not results_df['auc'].isna().all():
        ax4.bar(results_df['solver'], results_df['auc'], color='purple')
        ax4.set_xlabel('Algorytm')
        ax4.set_ylabel('AUC-ROC')
        ax4.set_title('Pole pod krzywą ROC')
        ax4.grid(True, axis='y')
    else:
        ax4.text(0.5, 0.5, 'AUC niedostępne', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('AUC-ROC')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, "plots", "model", filename), 
                dpi=config.VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    print(f"Zapisano porównanie algorytmów: {filename}")

def analyze_parameter_importance(X_train, y_train, cv=5):
    """
    Analizuje wpływ różnych parametrów modelu przy użyciu przeszukiwania siatki.
    
    Args:
        X_train: Dane treningowe
        y_train: Etykiety treningowe
        cv (int): Liczba foldów dla walidacji krzyżowej
        
    Returns:
        dict: Wyniki analizy parametrów
    """
    print("\n--- ANALIZA WAŻNOŚCI PARAMETRÓW MODELU ---")
    
    # Definiujemy uproszczoną siatkę parametrów (dla szybszego działania)
    param_grid = {
        'hidden_layer_sizes': [(8,), (16,), (16, 8), (32, 16)],
        'alpha': [0.001, 0.01, 0.1],
        'learning_rate_init': [0.01, 0.001, 0.0001]
    }
    
    # Tworzenie modelu bazowego
    base_model = MLPClassifier(
        activation=config.MODEL_CONFIG['activation'],
        solver=config.MODEL_CONFIG['solver'],
        max_iter=config.MODEL_CONFIG['max_iter'],
        early_stopping=True,
        validation_fraction=0.2,
        random_state=config.RANDOM_STATE
    )
    
    # Przeszukiwanie siatki
    print("Przeprowadzanie przeszukiwania siatki parametrów...")
    grid_search = GridSearchCV(
        base_model, 
        param_grid, 
        cv=cv, 
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    search_time = time.time() - start_time
    
    print(f"Przeszukiwanie zakończone w {search_time:.2f} sekund")
    print(f"Najlepsze parametry: {grid_search.best_params_}")
    print(f"Najlepsza dokładność: {grid_search.best_score_:.4f}")
    
    # Analiza wyników
    results_df = pd.DataFrame(grid_search.cv_results_)
    
    # Wizualizacja wpływu parametrów
    visualize_parameter_importance(results_df, grid_search.best_params_)
    
    return {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'best_estimator': grid_search.best_estimator_,
        'cv_results': results_df,
        'search_time': search_time
    }

def visualize_parameter_importance(results_df, best_params, filename="parameter_importance.png"):
    """
    Wizualizuje wpływ różnych parametrów na wydajność modelu.
    
    Args:
        results_df (DataFrame): Wyniki przeszukiwania siatki
        best_params (dict): Najlepsze parametry
        filename (str): Nazwa pliku do zapisu wykresu
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Wykres 1: Top 10 konfiguracji
    ax1 = axes[0, 0]
    top_results = results_df.nlargest(10, 'mean_test_score')
    y_pos = np.arange(len(top_results))
    
    ax1.barh(y_pos, top_results['mean_test_score'], 
             xerr=top_results['std_test_score'], capsize=5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([f"Config {i+1}" for i in range(len(top_results))])
    ax1.set_xlabel('Średnia dokładność')
    ax1.set_title('Top 10 konfiguracji parametrów')
    ax1.grid(True, axis='x')
    
    # Wykres 2: Wpływ hidden_layer_sizes
    ax2 = axes[0, 1]
    hidden_layer_data = []
    for _, row in results_df.iterrows():
        hidden_layers = str(row['param_hidden_layer_sizes'])
        score = row['mean_test_score']
        hidden_layer_data.append({'hidden_layers': hidden_layers, 'score': score})
    
    hidden_df = pd.DataFrame(hidden_layer_data)
    hidden_grouped = hidden_df.groupby('hidden_layers')['score'].agg(['mean', 'std']).reset_index()
    
    ax2.bar(range(len(hidden_grouped)), hidden_grouped['mean'], 
            yerr=hidden_grouped['std'], capsize=5)
    ax2.set_xticks(range(len(hidden_grouped)))
    ax2.set_xticklabels(hidden_grouped['hidden_layers'], rotation=45)
    ax2.set_xlabel('Architektura warstw ukrytych')
    ax2.set_ylabel('Średnia dokładność')
    ax2.set_title('Wpływ architektury sieci')
    ax2.grid(True, axis='y')
    
    # Wykres 3: Wpływ alpha (regularyzacja)
    ax3 = axes[1, 0]
    alpha_data = []
    for _, row in results_df.iterrows():
        alpha = row['param_alpha']
        score = row['mean_test_score']
        alpha_data.append({'alpha': alpha, 'score': score})
    
    alpha_df = pd.DataFrame(alpha_data)
    alpha_grouped = alpha_df.groupby('alpha')['score'].agg(['mean', 'std']).reset_index()
    
    ax3.bar(range(len(alpha_grouped)), alpha_grouped['mean'], 
            yerr=alpha_grouped['std'], capsize=5)
    ax3.set_xticks(range(len(alpha_grouped)))
    ax3.set_xticklabels(alpha_grouped['alpha'])
    ax3.set_xlabel('Parametr regularyzacji (alpha)')
    ax3.set_ylabel('Średnia dokładność')
    ax3.set_title('Wpływ regularyzacji')
    ax3.grid(True, axis='y')
    
    # Wykres 4: Wpływ learning_rate_init
    ax4 = axes[1, 1]
    lr_data = []
    for _, row in results_df.iterrows():
        lr = row['param_learning_rate_init']
        score = row['mean_test_score']
        lr_data.append({'learning_rate': lr, 'score': score})
    
    lr_df = pd.DataFrame(lr_data)
    lr_grouped = lr_df.groupby('learning_rate')['score'].agg(['mean', 'std']).reset_index()
    
    ax4.bar(range(len(lr_grouped)), lr_grouped['mean'], 
            yerr=lr_grouped['std'], capsize=5)
    ax4.set_xticks(range(len(lr_grouped)))
    ax4.set_xticklabels(lr_grouped['learning_rate'])
    ax4.set_xlabel('Początkowy współczynnik uczenia')
    ax4.set_ylabel('Średnia dokładność')
    ax4.set_title('Wpływ współczynnika uczenia')
    ax4.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, "plots", "model", filename), 
                dpi=config.VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    print(f"Zapisano analizę ważności parametrów: {filename}")

def comprehensive_model_testing(X_train, y_train, X_test, y_test):
    """
    Przeprowadza kompleksowe testowanie modelu z różnymi konfiguracjami.
    
    Args:
        X_train: Dane treningowe
        y_train: Etykiety treningowe
        X_test: Dane testowe
        y_test: Etykiety testowe
        
    Returns:
        dict: Wyniki kompleksowego testowania
    """
    print("\n--- KOMPLEKSOWE TESTOWANIE MODELU ---")
    
    # Testowanie różnych architektur
    architectures = [
        (8,), (16,), (32,),
        (8, 4), (16, 8), (32, 16),
        (16, 8, 4), (32, 16, 8)
    ]
    
    results = []
    
    for arch in architectures:
        print(f"\nTestowanie architektury: {arch}")
        
        # Tworzenie modelu
        model = MLPClassifier(
            hidden_layer_sizes=arch,
            activation=config.MODEL_CONFIG['activation'],
            solver=config.MODEL_CONFIG['solver'],
            alpha=config.MODEL_CONFIG['alpha'],
            learning_rate_init=config.MODEL_CONFIG['learning_rate_init'],
            max_iter=config.MODEL_CONFIG['max_iter'],
            early_stopping=True,
            validation_fraction=0.2,
            random_state=config.RANDOM_STATE
        )
        
        # Trenowanie
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Ewaluacja
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metryki
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Zapisanie wyników
        result = {
            'architecture': str(arch),
            'n_parameters': sum([arch[i] * (arch[i-1] if i > 0 else X_train.shape[1]) for i in range(len(arch))]) + sum(arch),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc_score,
            'training_time': training_time,
            'n_iter': model.n_iter_
        }
        
        results.append(result)
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-score: {f1:.4f}")
        print(f"  Czas treningu: {training_time:.2f}s")
    
    # Konwersja do DataFrame
    results_df = pd.DataFrame(results)
    
    # Wizualizacja wyników
    visualize_comprehensive_testing(results_df)
    
    return results_df

def visualize_comprehensive_testing(results_df, filename="comprehensive_testing.png"):
    """
    Wizualizuje wyniki kompleksowego testowania.
    
    Args:
        results_df (DataFrame): Wyniki testowania
        filename (str): Nazwa pliku do zapisu wykresu
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Wykres 1: Dokładność vs liczba parametrów
    ax1 = axes[0, 0]
    ax1.scatter(results_df['n_parameters'], results_df['accuracy'], alpha=0.7)
    ax1.set_xlabel('Liczba parametrów modelu')
    ax1.set_ylabel('Dokładność')
    ax1.set_title('Dokładność vs złożoność modelu')
    ax1.grid(True)
    
    # Dodanie etykiet dla każdego punktu
    for i, row in results_df.iterrows():
        ax1.annotate(row['architecture'], 
                    (row['n_parameters'], row['accuracy']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.7)
    
    # Wykres 2: Porównanie metryk
    ax2 = axes[0, 1]
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    x = np.arange(len(results_df))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        ax2.bar(x + i * width, results_df[metric], width, label=metric, alpha=0.7)
    
    ax2.set_xlabel('Architektura')
    ax2.set_ylabel('Wartość')
    ax2.set_title('Porównanie metryk dla różnych architektur')
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels([f"Arch {i+1}" for i in range(len(results_df))], rotation=45)
    ax2.legend()
    ax2.grid(True, axis='y')
    
    # Wykres 3: Czas treningu vs dokładność
    ax3 = axes[1, 0]
    ax3.scatter(results_df['training_time'], results_df['accuracy'], alpha=0.7)
    ax3.set_xlabel('Czas treningu (sekundy)')
    ax3.set_ylabel('Dokładność')
    ax3.set_title('Kompromis czas-dokładność')
    ax3.grid(True)
    
    # Wykres 4: Ranking architektur
    ax4 = axes[1, 1]
    # Sortujemy według F1-score
    sorted_results = results_df.sort_values('f1', ascending=True)
    y_pos = np.arange(len(sorted_results))
    
    ax4.barh(y_pos, sorted_results['f1'], alpha=0.7)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(sorted_results['architecture'])
    ax4.set_xlabel('F1-score')
    ax4.set_title('Ranking architektur według F1-score')
    ax4.grid(True, axis='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, "plots", "model", filename), 
                dpi=config.VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    print(f"Zapisano wyniki kompleksowego testowania: {filename}")

def run_all_experiments(data_dict):
    """
    Uruchamia wszystkie eksperymenty.
    
    Args:
        data_dict (dict): Słownik z przetworzonymi danymi
        
    Returns:
        dict: Zaktualizowany słownik z wynikami eksperymentów
    """
    print("\n--- URUCHAMIANIE WSZYSTKICH EKSPERYMENTÓW ---")
    
    X_train = data_dict['X_train_selected']
    y_train = data_dict['y_train']
    X_test = data_dict['X_test_selected']
    y_test = data_dict['y_test']
    
    # Eksperyment 1: Porównanie algorytmów uczenia
    algorithm_results = compare_learning_algorithms(X_train, y_train, X_test, y_test)
    
    # Eksperyment 2: Analiza ważności parametrów
    parameter_results = analyze_parameter_importance(X_train, y_train)
    
    # Eksperyment 3: Kompleksowe testowanie architektur
    comprehensive_results = comprehensive_model_testing(X_train, y_train, X_test, y_test)
    
    # Aktualizacja słownika danych
    data_dict.update({
        'algorithm_comparison': algorithm_results,
        'parameter_analysis': parameter_results,
        'comprehensive_testing': comprehensive_results
    })
    
    # Zapisanie wyników do plików CSV
    algorithm_results.to_csv(f"{config.OUTPUT_DIR}/reports/algorithm_comparison.csv", index=False)
    parameter_results['cv_results'].to_csv(f"{config.OUTPUT_DIR}/reports/parameter_analysis.csv", index=False)
    comprehensive_results.to_csv(f"{config.OUTPUT_DIR}/reports/comprehensive_testing.csv", index=False)
    
    print(f"\nWszystkie wyniki eksperymentów zapisane w {config.OUTPUT_DIR}/reports/")
    
    return data_dict

# Gdy moduł jest uruchamiany bezpośrednio
if __name__ == "__main__":
    from data_processing import process_data
    from feature_selection import perform_feature_selection
    
    # Przetwarzanie danych
    data_dict = process_data()
    
    # Selekcja cech
    data_dict = perform_feature_selection(data_dict)
    
    # Uruchomienie eksperymentów
    data_dict = run_all_experiments(data_dict)