import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib
from sklearn.metrics import confusion_matrix
from calculation_metrics import calculate_accuracy, calculate_precision, calculate_sensitivity, calculate_specificity

# Set the font globally to Arial
matplotlib.rcParams['font.sans-serif'] = "Arial"

def get_sensitivity_values(metrics_df):
    list_sensitivity_values = []
    for idx in range(len(metrics_df)):
        VP = metrics_df.loc[idx, "VP"]
        FN = metrics_df.loc[idx, "FN"]
        sensitivity = calculate_sensitivity(VP, FN)
        list_sensitivity_values.append(sensitivity)
    return list_sensitivity_values

def get_specificity_values(metrics_df):
    list_specificity_values = []
    for idx in range(len(metrics_df)):
        FP = metrics_df.loc[idx, "FP"]
        VN = metrics_df.loc[idx, "VN"]
        specificity = calculate_specificity(FP, VN)
        list_specificity_values.append(specificity)
    return list_specificity_values

# Get the sensitivity values
metrics_heartattack = pd.read_csv("/home/esther/Desktop/master-thesis/data/processed/metrics_values_heartattack.csv", index_col = 0)
list_sensitivity_values_MLR = get_sensitivity_values(metrics_heartattack)
list_specificity_values_MLR = get_specificity_values(metrics_heartattack)

list_thresholds = []
for idx in np.arange(0, 1, 0.0001):
    idx = round(idx, 4)
    list_thresholds.append(idx)


metrics_sen = {
    "threshold": list_thresholds,
    "sensitivity": list_sensitivity_values_MLR
}

metrics_spe = {
    "threshold": list_thresholds,
    "specificity": list_specificity_values_MLR
}
metrics_df_sen = pd.DataFrame(metrics_sen).melt(id_vars = "threshold", var_name = "Data Type", value_name = "Value")
metrics_df_sp = pd.DataFrame(metrics_spe).melt(id_vars = "threshold", var_name = "Data Type", value_name = "Value")


# Plot sensitivity values
sns.set_theme(style = "whitegrid")
plt.figure(figsize=(10, 6))
sns.lineplot(x = "threshold", y = "Value", data = metrics_df_sen, color = "darkorange", label = "Sensibilidad")
sns.lineplot(x = "threshold", y = "Value", data = metrics_df_sp, color = "orangered", label = "Especificidad", linestyle = "dashed")

# Calculate positions for 10 evenly spaced ticks
tick_positions = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
tick_labels = [f"{x:.2f}" for x in tick_positions]  
plt.xticks(tick_positions, tick_labels, fontsize=12)  
plt.yticks(fontsize=12) 
plt.xlabel("Valor Umbral", fontsize = 14) 
plt.ylabel("")
plt.title("Sensibilidad vs Especificidad RLM", fontsize = 16)
plt.legend(title = "Tipo de métrica", bbox_to_anchor = (1.05, 1), loc = "upper left")
plt.tight_layout(rect=[0, 0, 0.99, 1])

plt.savefig("Senstivity_MLR.png")