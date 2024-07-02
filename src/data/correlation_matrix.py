import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def create_correlation_matrix(X, plot_name, disease):
    # Calculate the correlation matrix
    corr_matrix_heartattack = X.corr()

    # Plot the Heatmap
    plt.figure(figsize=(15, 13))
    sns.heatmap(corr_matrix_heartattack, annot = False, cmap = 'coolwarm', linewidths = .5)
    plt.title(f"Matriz de Correlación {disease}")
    plt.tight_layout()
    plt.savefig(plot_name)


# Load the unbalanced data
X_heartattack_unbalanced = pd.read_csv("/home/esther/Desktop/master-thesis/data/processed/X_heartattack_importance_analysis.csv")
X_angina_unbalanced = pd.read_csv("/home/esther/Desktop/master-thesis/data/processed/X_angina_importance_analysis.csv")

# Load the unbalanced data
X_heartattack_balanced = pd.read_csv("/home/esther/Desktop/master-thesis/data/processed/X_heartattack_balanced_importance_analysis_0.002.csv")
X_angina_balanced = pd.read_csv("/home/esther/Desktop/master-thesis/data/processed/X_angina_balanced_importance_analysis_0.002.csv")

# # Create the correlation matrix for unbalanced data
create_correlation_matrix(X_heartattack_unbalanced, "correlation_matrix_heartattack_unbalanced.png", "Infarto de Miocardio")
create_correlation_matrix(X_angina_unbalanced, "correlation_matrix_angina_unbalanced.png", "Angina de Pecho")

# Create the correlation matrix for balanced data
create_correlation_matrix(X_heartattack_balanced, "correlation_matrix_heartattack_balanced.png", "Infarto de Miocardio")
create_correlation_matrix(X_angina_balanced, "correlation_matrix_angina_balanced.png", "Angina de Pecho")
