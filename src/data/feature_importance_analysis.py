import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


def feature_importance_analysis(X, y_heartattack, name_plot, disease):
    # Define and fit the model
    rf = RandomForestRegressor(random_state=42)

    # Fit the heart attack model
    rf.fit(X, np.ravel(y_heartattack))

    # Extract the features 
    features = X.columns

    # Extract the importances to plot them
    importances = rf.feature_importances_

    # Sort the features by importances
    indices = np.argsort(importances)

    # Plot the feature importances heartattack
    plt.figure(figsize=(10, 8))
    plt.title(f"Análisis de Importancia de Características {disease}")
    plt.barh(range(len(indices)), importances[indices], color = "darkorange", align = "center")
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel("Importancia relativa")
    plt.tight_layout()
    plt.savefig(name_plot)

    return importances


def remove_features(X, importances, threshold):
    # Removing the features with the lowest importance (0.01)
    X_filtered = X.drop(X.columns[np.where(importances < threshold)], axis = 1)
    return X_filtered


# Load unbalanced data
X_unbalanced = pd.read_csv("/home/esther/Desktop/master-thesis/data/processed/X_data_preprocessed.csv")
y_heartattack_unbalanced = pd.read_csv("/home/esther/Desktop/master-thesis/data/processed/y_heartattack_preprocessed.csv")
y_angina_unbalanced = pd.read_csv("/home/esther/Desktop/master-thesis/data/processed/y_angina_preprocessed.csv")

# Load balanced data
X_heartattack_balanced = pd.read_csv("/home/esther/Desktop/master-thesis/data/processed/X_heartattack_balanced.csv")
X_angina_balanced = pd.read_csv("/home/esther/Desktop/master-thesis/data/processed/X_angina_balanced.csv")
y_heartattack_balanced = pd.read_csv("/home/esther/Desktop/master-thesis/data/processed/y_heartattack_balanced.csv")
y_angina_balanced = pd.read_csv("/home/esther/Desktop/master-thesis/data/processed/y_angina_balanced.csv")

# # Feature importance analysis unbalanced data
importances_heartattack = feature_importance_analysis(X_unbalanced, y_heartattack_unbalanced, "feature_importances_heartattack.png", "Infarto de Miocardio")
importances_angina = feature_importance_analysis(X_unbalanced, y_angina_unbalanced, "feature_importances_angina.png", "Angina de Pecho")

# # Remove the features with the lowest importance
X_heartattack_unbalanced = remove_features(X_unbalanced, importances_heartattack, 0.01)
X_angina_unbalanced = remove_features(X_unbalanced, importances_angina, 0.01)

# Feature importance analysis balanced data in unbalanced data
importances_heartattack_balanced = feature_importance_analysis(X_heartattack_balanced, y_heartattack_balanced, "feature_importances_heartattack_balanced.png", "Infarto de Miocardio")
importances_angina_balanced = feature_importance_analysis(X_angina_balanced, y_angina_balanced, "feature_importances_angina_balanced.png", "Angina de Pecho")

# Remove the features with the lowest importance in balanced data
X_heartattack_balanced = remove_features(X_heartattack_balanced, importances_heartattack_balanced, 0.002)
X_angina_balanced = remove_features(X_angina_balanced, importances_angina_balanced, 0.002)

# Keep the features
X_heartattack_unbalanced.to_csv("/home/esther/Desktop/master-thesis/data/processed/X_heartattack_importance_analysis.csv", index = False)
X_angina_unbalanced.to_csv("/home/esther/Desktop/master-thesis/data/processed/X_angina_importance_analysis.csv", index = False)

X_heartattack_balanced.to_csv("/home/esther/Desktop/master-thesis/data/processed/X_heartattack_balanced_importance_analysis_0.002.csv", index = False)
X_angina_balanced.to_csv("/home/esther/Desktop/master-thesis/data/processed/X_angina_balanced_importance_analysis_0.002.csv", index = False)
