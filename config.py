RANDOM_STATE = 42
OUTPUT_DIR = "results"

# Parametry przetwarzania danych
DATA_CONFIG = {
    'test_size': 0.20,        # Rozmiar zbioru testowego (25% danych)
    'random_state': RANDOM_STATE,
    'dataset_id': 17,         # ID zbioru danych Breast Cancer Wisconsin w ucimlrepo
}

#Parametry selekcji cech
FEATURE_SELECTION_CONFIG = {
   'method': 'anova',        # Metoda selekcji cech (ANOVA F-value)
   'k': 10,                  # Liczba cech do wybrania
}


# Parametry wizualizacji
VISUALIZATION_CONFIG = {
    'figsize_medium': (8, 6),     # Standardowy rozmiar wykresów
    'figsize_large': (10, 8),     # Duży rozmiar wykresów  
    'dpi': 300,                   # Rozdzielczość wykresów
    'style': 'seaborn-v0_8',      # Styl wykresów
}

# Parametry modelu scikit-learn MLP
MODEL_CONFIG = {
    'hidden_layer_sizes': (64,32),   # Architektura sieci: dwie warstwy ukryte (16 i 8 neuronów)
    'activation': 'relu',            # Funkcja aktywacji: ReLU
    'solver': 'adam',                   # Algorytm optymalizacji: Adam
    'alpha': 0.001,                   # Parametr regularyzacji L2
    'batch_size':64,                 # Rozmiar batcha
    'learning_rate':'constant',       # Typ współczynnika uczenia
    'learning_rate_init': 0.001,      # Początkowy współczynnik uczenia
    'max_iter': 1000,                 # Maksymalna liczba iteracji (epok)
    'early_stopping': True,           # Wczesne zatrzymanie
    'validation_fraction': 0.2,       # Frakcja danych treningowych używana do walidacji
    'n_iter_no_change': 20,           # Liczba iteracji bez poprawy do wczesnego zatrzymania
    'random_state': RANDOM_STATE,     # Ziarno losowości
}

# Parametry treningu
TRAINING_CONFIG = {
    'cv_folds': 5,                    # Liczba foldów dla walidacji krzyżowej
}

# Parametry ewaluacji modelu
EVALUATION_CONFIG = {
    'threshold': 0.5,                 # Próg decyzyjny dla klasyfikacji binarnej
}