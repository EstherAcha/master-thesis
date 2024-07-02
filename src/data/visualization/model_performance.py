import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from calculation_metrics import calculate_accuracy, calculate_precision, calculate_sensitivity, calculate_specificity

# Set the font globally to Arial
matplotlib.rcParams['font.sans-serif'] = "Arial"

def plot_model_performance(VP_m, FP_m, FN_m, VN_m, VP_s, FP_s, FN_s, VN_s, title, name_image):
    # Calculate accuracy
    accuracy_m = calculate_accuracy(VP_m, FP_m, FN_m, VN_m)
    accuracy_s = calculate_accuracy(VP_s, FP_s, FN_s, VN_s)

    # Calculate precision
    precision_m = calculate_precision(VP_m, FP_m)
    precision_s = calculate_precision(VP_s, FP_s)

    # Calculate Sensitivity
    sensitivity_m = calculate_sensitivity(VP_m, FN_m)
    sensitivity_s = calculate_sensitivity(VP_s, FN_s)

    # Calculate Specificity
    specificity_m = calculate_specificity(FP_m, VN_m)
    specificity_s = calculate_specificity(FP_s, VN_s)

    # Plot the metrics
    metrics = {
        "Metric": ["Exactitud", "Precisión", "Sensibilidad", "Especificidad"],
        "RLM": [accuracy_m, precision_m, sensitivity_m, specificity_m],
        "SVM": [accuracy_s, precision_s, sensitivity_s, specificity_s]
    }


    metrics_df = pd.DataFrame(metrics).melt(id_vars="Metric", var_name="Data Type", value_name="Value")

    sns.set_theme(style = "whitegrid")
    plt.figure(figsize=(10, 6))
    sns.barplot(x = "Metric", y = "Value", hue = "Data Type", data = metrics_df, palette = ["coral", "orangered"])
    plt.xticks(fontsize=12)  
    plt.yticks(fontsize=12) 
    plt.xlabel("Métrica", fontsize = 14) 
    plt.ylabel("Valor", fontsize = 14)  
    plt.title(title, fontsize = 16)
    plt.legend(title = "Tipo de modelo", bbox_to_anchor = (1.05, 1), loc = "upper left")
    plt.tight_layout(rect=[0, 0, 0.99, 1])
    plt.savefig(name_image)


# Plot the metrics for multiple linear regression vs SVM Hearattack
plot_model_performance(42670, 18478, 3780, 28107, 198, 52, 7, 143, "Infarto: Regresión linear múltiple vs SVM", "RLM_vs_SVM_Heartattack.png")

# Plot the metrics for multiple linear regression vs SVM Angina
plot_model_performance(42916, 17476, 3291, 28745, 199, 44, 23, 134, "Angina de pecho: Regresión linear múltiple vs SVM", "RLM_vs_SVM_Angina.png")