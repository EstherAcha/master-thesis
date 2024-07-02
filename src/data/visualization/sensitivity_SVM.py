import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib
from sklearn.metrics import confusion_matrix
from calculation_metrics import calculate_accuracy, calculate_precision, calculate_sensitivity, calculate_specificity


def plot_heatmap(data, title, name_image):
    plt.figure(figsize=(10, 6))
    sns.heatmap(data, annot = False, fmt = ".2f", cmap = "coolwarm", linewidths = .5)
    plt.title(title, fontsize = 16)
    plt.xlabel("Gamma", fontsize = 14)
    plt.ylabel("C", fontsize = 14)
    plt.savefig(f"{name_image}.png")

# Load data
data_hyperparameters = pd.read_csv("/home/esther/Desktop/master-thesis/data/processed/sensitivity_C_gamma.csv", index_col = 0)

# Pivot the data
pivot_senstivity = data_hyperparameters.pivot(index = "C", columns = "Gamma", values = "Sensitivity")
pivot_accuracy = data_hyperparameters.pivot(index = "C", columns = "Gamma", values = "Accuracy")
pivot_specificity = data_hyperparameters.pivot(index = "C", columns = "Gamma", values = "Specificity")

# Plot the heatmaps
plot_heatmap(pivot_senstivity, "Sensibilidad Gamma vs C", "sensitivity_SVM")
plot_heatmap(pivot_accuracy, "Excactitud Gamma vs C", "accuracy_SVM")
plot_heatmap(pivot_specificity, "Especificidad Gamma vs C", "specificity_SVM")


