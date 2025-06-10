
import os
import time
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from src.data_processing import process_data
from src.feature_selection import perform_feature_selection
from src.training import train_and_validate_model

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
    
 
    
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"PROJEKT ZAKOŃCZONY POMYŚLNIE! Całkowity czas wykonania: {total_time:.2f} sekund")
    print("=" * 80)
    
    print(f"\nWszystkie wyniki zostały zapisane w katalogu: {config.OUTPUT_DIR}")
    print("Wygenerowane pliki:")
    print(f"  - Wykresy: {config.OUTPUT_DIR}/plots/")
    print(f"  - Model: {config.OUTPUT_DIR}/models/mlp_model.pkl")
    print(f"  - Raport: {config.OUTPUT_DIR}/reports/classification_report.md")

if __name__ == "__main__":
    main()