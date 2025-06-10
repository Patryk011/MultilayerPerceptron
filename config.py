
# Parametry ogólne
RANDOM_STATE = 42
OUTPUT_DIR = "results"

# Parametry przetwarzania danych
DATA_CONFIG = {
    'test_size': 0.25,       
    'random_state': RANDOM_STATE,
    'dataset_id': 17,        
}

# Parametry selekcji cech
FEATURE_SELECTION_CONFIG = {
    'method': 'anova',       
    'k': 10,                 
}

# Parametry wizualizacji
VISUALIZATION_CONFIG = {
    'figsize_medium': (10, 6),   
    'figsize_large': (14, 10),   
    'dpi': 300,                  
    'cmap': 'viridis',           
    'style': 'seaborn-v0_8',     
}

# Parametry modelu scikit-learn MLP
MODEL_CONFIG = {
    'hidden_layer_sizes': (16, 8),   
    'activation': 'relu',            
    'solver': 'adam',                
    'alpha': 0.001,                  
    'batch_size': 32,                
    'learning_rate': 'adaptive',     
    'learning_rate_init': 0.001,     
    'max_iter': 1000,                
    'early_stopping': True,          
    'validation_fraction': 0.2,      
    'n_iter_no_change': 20,          
    'random_state': RANDOM_STATE,    
}

# Parametry treningu
TRAINING_CONFIG = {
    'cv_folds': 5,                   
}
