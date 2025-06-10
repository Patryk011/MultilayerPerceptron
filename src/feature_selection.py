
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

import config

def select_features_anova(X_train, y_train, X_test, k=None):

   
    if k is None:
        k = config.FEATURE_SELECTION_CONFIG['k']
    
    print(f"Wybieranie {k} najważniejszych cech używając testu ANOVA F-value...")
    
   
    selector = SelectKBest(f_classif, k=k)
    
   
    X_train_selected = selector.fit_transform(X_train, y_train)
    
   
    X_test_selected = selector.transform(X_test)
    
   
    feature_indices = selector.get_support(indices=True)
    selected_feature_names = X_train.columns[feature_indices]
    
   
    X_train_selected_df = pd.DataFrame(
        X_train_selected, 
        columns=selected_feature_names,
        index=X_train.index
    )
    
    X_test_selected_df = pd.DataFrame(
        X_test_selected, 
        columns=selected_feature_names,
        index=X_test.index
    )
    
   
    feature_scores = pd.DataFrame({
        'Feature': X_train.columns,
        'F_score': selector.scores_,
        'p_value': selector.pvalues_
    }).sort_values('F_score', ascending=False)
    
    print("Najważniejsze 5 cech:")
    print(feature_scores.head(5))
    
   
    feature_selection_dict = {
        'X_train_selected': X_train_selected_df,
        'X_test_selected': X_test_selected_df,
        'selector': selector,
        'feature_scores': feature_scores,
        'selected_feature_names': selected_feature_names,
        'feature_indices': feature_indices
    }
    
    return feature_selection_dict

def visualize_feature_importance(feature_scores, top_n=15, filename="feature_importance.png"):

    plt.figure(figsize=config.VISUALIZATION_CONFIG['figsize_large'])
    
   
    top_features = feature_scores.head(top_n)
    
   
    sns.barplot(x='F_score', y='Feature', data=top_features)
    plt.title(f'{top_n} najważniejszych cech')
    plt.xlabel('Wartość F')
    plt.ylabel('Cecha')
    plt.tight_layout()
    
   
    plt.savefig(f"{config.OUTPUT_DIR}/plots/features/{filename}", 
                dpi=config.VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    print(f"Zapisano wizualizację ważności cech: {filename}")

def visualize_selected_feature_distributions(X, y, selected_features, filename="selected_feature_distributions.png"):

   
    n_features = min(6, len(selected_features))
    
   
    n_cols = 2
    n_rows = (n_features + 1) // 2
    
   
    fig, axes = plt.subplots(n_rows, n_cols, figsize=config.VISUALIZATION_CONFIG['figsize_large'])
    axes = axes.flatten() 
    
   
    for i, feature in enumerate(selected_features[:n_features]):
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
    
    print(f"Zapisano wizualizację rozkładów wybranych cech: {filename}")

def visualize_selected_feature_correlations(X, selected_features, filename="selected_feature_correlations.png"):

    plt.figure(figsize=config.VISUALIZATION_CONFIG['figsize_large'])
    
   
    correlation_matrix = X[selected_features].corr()
    
   
    sns.heatmap(
        correlation_matrix, 
        annot=True, 
        cmap='coolwarm', 
        linewidths=0.5, 
        vmin=-1, 
        vmax=1,
        center=0,
        fmt='.2f'
    )
    
    plt.title('Macierz korelacji wybranych cech')
    plt.tight_layout()
    plt.savefig(f"{config.OUTPUT_DIR}/plots/features/{filename}", 
                dpi=config.VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    print(f"Zapisano wizualizację korelacji wybranych cech: {filename}")

def perform_feature_selection(data_dict):
 
    print("\n--- SELEKCJA CECH ---")
    
   
    if config.FEATURE_SELECTION_CONFIG['method'] == 'anova':
        feature_selection_dict = select_features_anova(
            data_dict['X_train_scaled'], 
            data_dict['y_train'],
            data_dict['X_test_scaled'], 
            k=config.FEATURE_SELECTION_CONFIG['k']
        )
    else:
        raise ValueError(f"Nieznana metoda selekcji cech: {config.FEATURE_SELECTION_CONFIG['method']}")
    
   
    visualize_feature_importance(feature_selection_dict['feature_scores'])
    
   
    visualize_selected_feature_distributions(
        data_dict['X'], 
        data_dict['y_numeric'],
        feature_selection_dict['selected_feature_names']
    )
    
   
    visualize_selected_feature_correlations(
        data_dict['X'],
        feature_selection_dict['selected_feature_names']
    )
    
   
    data_dict.update({
        'feature_selection': feature_selection_dict,
        'X_train_selected': feature_selection_dict['X_train_selected'],
        'X_test_selected': feature_selection_dict['X_test_selected'],
        'selected_feature_names': feature_selection_dict['selected_feature_names']
    })
    
    return data_dict

if __name__ == "__main__":
    from data_processing import process_data
    
   
    data_dict = process_data()
    
   
    data_dict = perform_feature_selection(data_dict)