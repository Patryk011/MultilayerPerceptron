
from sklearn.neural_network import MLPClassifier
import pickle
import os


import config

def create_mlp_classifier(hidden_layer_sizes=None, activation=None, solver=None, alpha=None,
                         batch_size=None, learning_rate=None, learning_rate_init=None,
                         max_iter=None, early_stopping=None, validation_fraction=None,
                         n_iter_no_change=None, random_state=None):
    """
    Tworzy model perceptronu wielowarstwowego (MLP) z scikit-learn.
    
    Args:
        hidden_layer_sizes (tuple): Rozmiary warstw ukrytych
        activation (str): Funkcja aktywacji ('relu', 'tanh', 'logistic')
        solver (str): Algorytm optymalizacji ('adam', 'sgd', 'lbfgs')
        alpha (float): Parametr regularyzacji L2
        batch_size (int): Rozmiar batcha (dla 'adam' i 'sgd')
        learning_rate (str): Strategia uczenia ('constant', 'adaptive', 'invscaling')
        learning_rate_init (float): Początkowy współczynnik uczenia
        max_iter (int): Maksymalna liczba iteracji (epok)
        early_stopping (bool): Czy używać wczesnego zatrzymania
        validation_fraction (float): Frakcja danych treningowych do walidacji
        n_iter_no_change (int): Liczba iteracji bez poprawy do wczesnego zatrzymania
        random_state (int): Ziarno losowości
        
    Returns:
        model: Model MLPClassifier
    """
    
    if hidden_layer_sizes is None:
        hidden_layer_sizes = config.MODEL_CONFIG['hidden_layer_sizes']
    if activation is None:
        activation = config.MODEL_CONFIG['activation']
    if solver is None:
        solver = config.MODEL_CONFIG['solver']
    if alpha is None:
        alpha = config.MODEL_CONFIG['alpha']
    if batch_size is None:
        batch_size = config.MODEL_CONFIG['batch_size']
    if learning_rate is None:
        learning_rate = config.MODEL_CONFIG['learning_rate']
    if learning_rate_init is None:
        learning_rate_init = config.MODEL_CONFIG['learning_rate_init']
    if max_iter is None:
        max_iter = config.MODEL_CONFIG['max_iter']
    if early_stopping is None:
        early_stopping = config.MODEL_CONFIG['early_stopping']
    if validation_fraction is None:
        validation_fraction = config.MODEL_CONFIG['validation_fraction']
    if n_iter_no_change is None:
        n_iter_no_change = config.MODEL_CONFIG['n_iter_no_change']
    if random_state is None:
        random_state = config.MODEL_CONFIG['random_state']
    
    
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        alpha=alpha,
        batch_size=batch_size,
        learning_rate=learning_rate,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        early_stopping=early_stopping,
        validation_fraction=validation_fraction,
        n_iter_no_change=n_iter_no_change,
        random_state=random_state,
        verbose=True  
    )
    
    print(f"Utworzono model MLP z {len(hidden_layer_sizes)} warstwami ukrytymi: {hidden_layer_sizes}")
    
    return model

def save_model(model, filename="mlp_model.pkl"):
    """
    Zapisuje model do pliku.
    
    Args:
        model: Model do zapisania
        filename (str): Nazwa pliku
    """
    filepath = os.path.join(config.OUTPUT_DIR, "models", filename)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model zapisany do pliku: {filepath}")

def load_model(filename="mlp_model.pkl"):
    """
    Wczytuje model z pliku.
    
    Args:
        filename (str): Nazwa pliku
        
    Returns:
        model: Wczytany model
    """
    filepath = os.path.join(config.OUTPUT_DIR, "models", filename)
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    print(f"Model wczytany z pliku: {filepath}")
    
    return model


if __name__ == "__main__":
    
    model = create_mlp_classifier()
    
    
    print("\nParametry modelu:")
    for param, value in model.get_params().items():
        print(f"  {param}: {value}")