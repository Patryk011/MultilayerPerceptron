import os
import time
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from src.data_processing import process_data
from src.feature_selection import perform_feature_selection
from src.training import train_and_validate_model
from src.evaluation import evaluate_and_report

def main():
    print("=" * 80)
    print("PROJEKT: KLASYFIKACJA DANYCH RAKA PIERSI Z UŻYCIEM PERCEPTRONU WIELOWARSTWOWEGO")
    print("=" * 80)
    
    start_time = time.time()
    
    print("\n[KROK 1] PRZETWARZANIE DANYCH")
    data_dict = process_data()
    
    print("\n[KROK 2] SELEKCJA CECH")
    data_dict = perform_feature_selection(data_dict)
    
    print("\n[KROK 3] TRENOWANIE I WALIDACJA MODELU")
    data_dict = train_and_validate_model(data_dict)
    
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
    
    if 'cv_results' in data_dict:
        cv_res = data_dict['cv_results']
        print(f"  Średnia dokładność CV: {cv_res['accuracy']['mean']:.4f} ± {cv_res['accuracy']['std']:.4f}")
    
    print(f"\nWszystkie wyniki zostały zapisane w katalogu: {config.OUTPUT_DIR}")
    print("Wygenerowane pliki:")
    print(f"  - Model: {config.OUTPUT_DIR}/models/mlp_model.pkl")
    print(f"  - Wykresy: {config.OUTPUT_DIR}/plots/")
    print(f"  - Raport: {config.OUTPUT_DIR}/reports/classification_report.txt")


if __name__ == "__main__":
    main()