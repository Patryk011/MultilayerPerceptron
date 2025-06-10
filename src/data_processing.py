
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo
import os

import config

def load_breast_cancer_data():

    print("Wczytywanie zbioru danych Breast Cancer Wisconsin Diagnostic...")
    
   
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=config.DATA_CONFIG['dataset_id'])
    
   
    X = breast_cancer_wisconsin_diagnostic.data.features
    y = breast_cancer_wisconsin_diagnostic.data.targets
    
   
    metadata = breast_cancer_wisconsin_diagnostic.metadata
    variables = breast_cancer_wisconsin_diagnostic.variables
    
   
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0] 
    
    print(f"Wczytano dane: {X.shape[0]} próbek, {X.shape[1]} cech")
    print(f"Rozkład klas: {y.value_counts().to_dict()}")
    
    return X, y, metadata, variables

def convert_target_to_numeric(y):
 
   
    if y.dtype == 'int64' or y.dtype == 'float64':
        print("Zmienna celu jest już numeryczna")
        return y
    
   
    y_numeric = y.map({'B': 0, 'M': 1})
    print(f"Przekształcono etykiety do formatu numerycznego (0: łagodny, 1: złośliwy)")
    
    return y_numeric

def split_data(X, y, test_size=None, random_state=None):
   
   
    if test_size is None:
        test_size = config.DATA_CONFIG['test_size']
    if random_state is None:
        random_state = config.DATA_CONFIG['random_state']
    
   
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Podział danych: treningowe={X_train.shape[0]}, testowe={X_test.shape[0]}")
    print(f"Rozkład klas w zbiorze treningowym: {np.bincount(y_train)}")
    print(f"Rozkład klas w zbiorze testowym: {np.bincount(y_test)}")
    
    return X_train, X_test, y_train, y_test

def standardize_data(X_train, X_test):
   
   
    scaler = StandardScaler()
    
   
    X_train_scaled = scaler.fit_transform(X_train)
    
   
    X_test_scaled = scaler.transform(X_test)
    
   
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    print("Dane zostały znormalizowane")
    
    return X_train_scaled, X_test_scaled, scaler

def create_output_directories():
  
   
    os.makedirs(f"{config.OUTPUT_DIR}/plots/features", exist_ok=True)
    os.makedirs(f"{config.OUTPUT_DIR}/plots/model", exist_ok=True)
    
   
    os.makedirs(f"{config.OUTPUT_DIR}/models", exist_ok=True)
    
   
    os.makedirs(f"{config.OUTPUT_DIR}/reports", exist_ok=True)
    
    print(f"Utworzono strukturę katalogów w {config.OUTPUT_DIR}")

def visualize_class_distribution(y, filename="class_distribution.png"):
   
    plt.figure(figsize=config.VISUALIZATION_CONFIG['figsize_medium'])
    sns.countplot(x=y)
    plt.title('Rozkład klas w zbiorze danych')
    plt.xlabel('Klasa (0: łagodny, 1: złośliwy)')
    plt.ylabel('Liczba próbek')
    plt.tight_layout()
    plt.savefig(f"{config.OUTPUT_DIR}/plots/features/{filename}", 
                dpi=config.VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    print(f"Zapisano wizualizację rozkładu klas: {filename}")

def visualize_feature_distributions(X, y, feature_names, n_features=4, filename="feature_distributions.png"):
   
    if n_features > len(feature_names):
        n_features = len(feature_names)
    
   
    features_to_plot = feature_names[:n_features]
    
   
    n_cols = 2
    n_rows = (n_features + 1) // 2
    
   
    fig, axes = plt.subplots(n_rows, n_cols, figsize=config.VISUALIZATION_CONFIG['figsize_large'])
    axes = axes.flatten() 
    
   
    for i, feature in enumerate(features_to_plot):
        for target, color in zip([0, 1], ['skyblue', 'salmon']):
           
            sns.kdeplot(
                X[feature][y == target], 
                ax=axes[i], 
                fill=True, 
                label=f'Klasa {target}', 
                color=color,
                alpha=0.6
            )
        
        axes[i].set_title(f'Rozkład cechy {feature}')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Gęstość')
        axes[i].legend(['Łagodny (0)', 'Złośliwy (1)'])
    
   
    for i in range(n_features, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig(f"{config.OUTPUT_DIR}/plots/features/{filename}", 
                dpi=config.VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    print(f"Zapisano wizualizację rozkładów cech: {filename}")

def visualize_correlation_matrix(X, filename="correlation_matrix.png"):
    
    plt.figure(figsize=config.VISUALIZATION_CONFIG['figsize_large'])
    
   
    correlation_matrix = X.corr()
    
   
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
   
    sns.heatmap(
        correlation_matrix, 
        mask=mask, 
        annot=False, 
        cmap='coolwarm', 
        linewidths=0.5, 
        vmin=-1, 
        vmax=1,
        center=0
    )
    
    plt.title('Macierz korelacji cech')
    plt.tight_layout()
    plt.savefig(f"{config.OUTPUT_DIR}/plots/features/{filename}", 
                dpi=config.VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    print(f"Zapisano wizualizację macierzy korelacji: {filename}")

def process_data():
  
   
    create_output_directories()
    
   
    X, y, metadata, variables = load_breast_cancer_data()
    
   
    y_numeric = convert_target_to_numeric(y)
    
   
    visualize_class_distribution(y_numeric)
    
   
    visualize_feature_distributions(X, y_numeric, X.columns[:4])
    
   
    visualize_correlation_matrix(X)
    
   
    X_train, X_test, y_train, y_test = split_data(X, y_numeric)
    
   
    X_train_scaled, X_test_scaled, scaler = standardize_data(X_train, X_test)
    
   
    data_dict = {
        'X': X,
        'y': y,
        'y_numeric': y_numeric,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'scaler': scaler,
        'metadata': metadata,
        'variables': variables
    }
    
    return data_dict

if __name__ == "__main__":
    process_data()