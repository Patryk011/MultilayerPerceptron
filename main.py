import os
import time
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from src.data_processing import process_data
from src.feature_selection import perform_feature_selection
from src.training import train_and_validate_model
from src.evaluation import evaluate_and_report

def main(use_grid_search=True):
    """
    Główna funkcja uruchamiająca cały pipeline projektu.
    
    Args:
        use_grid_search (bool): Czy używać grid search do optymalizacji hiperparametrów
    """
    print("=" * 80)
    print("PROJEKT: KLASYFIKACJA DANYCH RAKA PIERSI Z UŻYCIEM PERCEPTRONU WIELOWARSTWOWEGO")
    print("=" * 80)
    
    
    if use_grid_search:
        print("🔍 TRYB: Grid Search - optymalizacja hiperparametrów (może potrwać dłużej)")
    else:
        print("⚡ TRYB: Standardowy - domyślne parametry (szybszy)")
    
    start_time = time.time()
    
    print("\n[KROK 1] PRZETWARZANIE DANYCH")
    data_dict = process_data()
    
    print("\n[KROK 2] SELEKCJA CECH")
    data_dict = perform_feature_selection(data_dict)
    
    print("\n[KROK 3] TRENOWANIE I WALIDACJA MODELU")
    
    data_dict = train_and_validate_model(data_dict, use_grid_search=use_grid_search)
    
    print("\n[KROK 4] EWALUACJA I RAPORTOWANIE")
    data_dict = evaluate_and_report(data_dict)
 
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"PROJEKT ZAKOŃCZONY POMYŚLNIE! Całkowity czas wykonania: {total_time:.2f} sekund")
    print("=" * 80)
    
    print(f"\nPODSUMOWANIE WYNIKÓW:")

    if 'evaluation_results' in data_dict:
        eval_res = data_dict['evaluation_results']
        print(f"  Dokładność finalnego modelu: {eval_res['accuracy']:.4f}")
        print(f"  Recall finalnego modelu: {eval_res['recall']:.4f}")
    
    if 'cv_results' in data_dict:
        cv_res = data_dict['cv_results']
        print(f"  Średnia dokładność CV: {cv_res['accuracy']['mean']:.4f} ± {cv_res['accuracy']['std']:.4f}")
        print(f"  Średni recall CV: {cv_res['recall']['mean']:.4f} ± {cv_res['recall']['std']:.4f}")
    
    
    if use_grid_search and 'grid_search_results' in data_dict:
        gs_res = data_dict['grid_search_results']
        print(f"\nWYNIKI GRID SEARCH:")
        print(f"  Przetestowano {gs_res['n_combinations']} kombinacji parametrów")
        print(f"  Czas grid search: {gs_res['search_time']:.2f} sekund")
        print(f"  Najlepszy recall CV: {gs_res['best_score']:.4f}")
        print(f"\n  Najlepsze parametry:")
        for param, value in gs_res['best_params'].items():
            print(f"    {param}: {value}")
    
    print(f"\nWszystkie wyniki zostały zapisane w katalogu: {config.OUTPUT_DIR}")
    print("Wygenerowane pliki:")
    print(f"  - Model: {config.OUTPUT_DIR}/models/mlp_model.pkl")
    print(f"  - Wykresy: {config.OUTPUT_DIR}/plots/")
    print(f"  - Raport: {config.OUTPUT_DIR}/reports/classification_report.txt")
    
    if use_grid_search:
        print(f"  - Wyniki Grid Search: {config.OUTPUT_DIR}/plots/model/grid_search_results.png")

def run_with_grid_search():
    """Uruchamia projekt z grid search - najlepsze wyniki, ale wolniejsze."""
    main(use_grid_search=True)

def run_fast():
    """Uruchamia projekt bez grid search - szybciej, domyślne parametry."""
    main(use_grid_search=False)

if __name__ == "__main__":
  
    main(use_grid_search=True)
    
  