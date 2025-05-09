"""
Główny skrypt projektu klasyfikacji danych raka piersi.
"""

import os
import time
import sys

# Dodajemy katalog src do ścieżki wyszukiwania modułów
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from src.data_processing import process_data
from src.feature_selection import perform_feature_selection
from src.training import train_and_validate_model
from src.evaluation import evaluate_and_report

def main():
    """
    Główna funkcja projektu.
    """
    print("=" * 80)
    print("PROJEKT: KLASYFIKACJA DANYCH RAKA PIERSI Z UŻYCIEM PERCEPTRONU WIELOWARSTWOWEGO")
    print("=" * 80)
    
    start_time = time.time()
    
    # Krok 1: Przetwarzanie danych
    print("\n[KROK 1] PRZETWARZANIE DANYCH")
    data_dict = process_data()
    
    # Krok 2: Selekcja cech
    print("\n[KROK 2] SELEKCJA CECH")
    data_dict = perform_feature_selection(data_dict)
    
    # Krok 3: Trenowanie i walidacja modelu
    print("\n[KROK 3] TRENOWANIE I WALIDACJA MODELU")
    data_dict = train_and_validate_model(data_dict)
    
    # Krok 4: Ewaluacja i raportowanie
    print("\n[KROK 4] EWALUACJA I RAPORTOWANIE")
    data_dict = evaluate_and_report(data_dict)
    
    # Zakończenie
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"PROJEKT ZAKOŃCZONY POMYŚLNIE! Całkowity czas wykonania: {total_time:.2f} sekund")
    print("=" * 80)
    
    # Informacja o wynikach
    print(f"\nWszystkie wyniki zostały zapisane w katalogu: {config.OUTPUT_DIR}")
    print("Wygenerowane pliki:")
    print(f"  - Wykresy: {config.OUTPUT_DIR}/plots/")
    print(f"  - Model: {config.OUTPUT_DIR}/models/mlp_model.pkl")
    print(f"  - Raport: {config.OUTPUT_DIR}/reports/classification_report.md")

if __name__ == "__main__":
    main()