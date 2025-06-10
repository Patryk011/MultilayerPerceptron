
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedKFold
import time
import os

import config
from src.model import create_mlp_classifier, save_model

def train_model(model, X_train, y_train):

    print("\n--- TRENOWANIE MODELU ---")
    
    start_time = time.time()
    
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    print(f"Trening zakończony po {model.n_iter_} iteracjach ({training_time:.2f} sekund)")
    
    return model

def perform_cross_validation(model, X, y, cv=None):

    if cv is None:
        cv = config.TRAINING_CONFIG['cv_folds']
    
    print(f"\nPrzeprowadzanie {cv}-krotnej walidacji krzyżowej...")
    
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=config.RANDOM_STATE)
    
    start_time = time.time()
    cv_accuracy = cross_val_score(model, X, y, cv=cv_strategy, scoring='accuracy')
    cv_precision = cross_val_score(model, X, y, cv=cv_strategy, scoring='precision')
    cv_recall = cross_val_score(model, X, y, cv=cv_strategy, scoring='recall')
    cv_f1 = cross_val_score(model, X, y, cv=cv_strategy, scoring='f1')
    cv_roc_auc = cross_val_score(model, X, y, cv=cv_strategy, scoring='roc_auc')
    cv_time = time.time() - start_time
    
    cv_results = {
        'accuracy': {
            'mean': np.mean(cv_accuracy),
            'std': np.std(cv_accuracy),
            'values': cv_accuracy
        },
        'precision': {
            'mean': np.mean(cv_precision),
            'std': np.std(cv_precision),
            'values': cv_precision
        },
        'recall': {
            'mean': np.mean(cv_recall),
            'std': np.std(cv_recall),
            'values': cv_recall
        },
        'f1': {
            'mean': np.mean(cv_f1),
            'std': np.std(cv_f1),
            'values': cv_f1
        },
        'roc_auc': {
            'mean': np.mean(cv_roc_auc),
            'std': np.std(cv_roc_auc),
            'values': cv_roc_auc
        },
        'time': cv_time
    }
    
    print(f"Wyniki walidacji krzyżowej ({cv_time:.2f} sekund):")
    for metric, values in cv_results.items():
        if metric != 'time':
            print(f"  {metric}: {values['mean']:.4f} ± {values['std']:.4f}")
    
    visualize_cross_validation_results(cv_results)
    
    return cv_results

def visualize_cross_validation_results(cv_results, filename="cv_results.png"):
  
    plt.figure(figsize=config.VISUALIZATION_CONFIG['figsize_medium'])
    
    metrics = [metric for metric in cv_results.keys() if metric != 'time']
    mean_values = [cv_results[metric]['mean'] for metric in metrics]
    std_values = [cv_results[metric]['std'] for metric in metrics]
    
    bars = plt.bar(metrics, mean_values, yerr=std_values, capsize=10)
    
    for bar, mean, std in zip(bars, mean_values, std_values):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.,
            height + std,
            f'{mean:.4f}±{std:.4f}',
            ha='center',
            va='bottom',
            fontsize=8
        )
    
    plt.title('Wyniki walidacji krzyżowej')
    plt.ylabel('Wartość')
    plt.ylim(0, 1.1) 
    plt.grid(True, axis='y')
    plt.tight_layout()
    
    plt.savefig(os.path.join(config.OUTPUT_DIR, "plots", "model", filename), 
                dpi=config.VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    print(f"Zapisano wizualizację wyników walidacji krzyżowej: {filename}")

def train_and_validate_model(data_dict):
   
    print("\n--- TRENOWANIE I WALIDACJA MODELU ---")
    
    model = create_mlp_classifier()
    
    cv_results = perform_cross_validation(
        model, 
        data_dict['X_train_selected'], 
        data_dict['y_train']
    )
    
    trained_model = train_model(
        model, 
        data_dict['X_train_selected'], 
        data_dict['y_train']
    )
    
    save_model(trained_model)
    
    data_dict.update({
        'model': trained_model,
        'cv_results': cv_results
    })
    
    return data_dict

if __name__ == "__main__":
    from data_processing import process_data
    from feature_selection import perform_feature_selection
    
    data_dict = process_data()
    
    data_dict = perform_feature_selection(data_dict)
    
    data_dict = train_and_validate_model(data_dict)