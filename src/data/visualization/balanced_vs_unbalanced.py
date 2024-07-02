import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from matplotlib.lines import Line2D
from calculation_metrics import calculate_accuracy, calculate_precision, calculate_sensitivity, calculate_specificity

# Set the font globally to Arial
matplotlib.rcParams['font.sans-serif'] = "Arial"

def calculate_metrics(VP, FP, FN, VN):
    list_metrics = []

    # Calculate accuracy
    list_metrics.append(calculate_accuracy(VP, FP, FN, VN))

    # Calculate precision
    list_metrics.append(calculate_precision(VP, FP))

    # Calculate Sensitivity
    list_metrics.append(calculate_sensitivity(VP, FN))

    # Calculate Specificity
    list_metrics.append(calculate_specificity(FP, VN))

    return list_metrics

    

def make_bar_plot(VP_u, FP_u, FN_u, VN_u, VP_b, FP_b, FN_b, VN_b, list_d_u, list_d_b, title, name_image):
    # Calculate the lists of the metrics 
    list_unbalanced_metrics = calculate_metrics(VP_u, FP_u, FN_u, VN_u)
    list_balanced_metrics = calculate_metrics(VP_b, FP_b, FN_b, VN_b)
    list_dummy_metrics = list_d_u + list_d_b

    # Plot the metrics
    metrics = {
        "Metric": ["Exactitud", "Precisión", "Sensibilidad", "Especificidad"],
        "No balanceados":list_unbalanced_metrics,
        "Balancedos": list_balanced_metrics
    }


    metrics_df = pd.DataFrame(metrics).melt(id_vars="Metric", var_name="Data Type", value_name="Value")

    sns.set_theme(style = "whitegrid")
    plt.figure(figsize=(10, 6))
    sns.barplot(x = "Metric", y = "Value", hue = "Data Type", data = metrics_df, palette = ["coral", "orangered"])
    plt.yticks(fontsize=12) 
    plt.xlabel("Métrica", fontsize = 14) 
    plt.ylabel("Valor", fontsize = 14)  
    plt.title(title, fontsize = 16)

    bar_containers = plt.gca().containers
    for container_index, container in enumerate(bar_containers):
        for bar_index, bar in enumerate(container):
            # Calculate the index in list_dummy_metrics for the current bar
            metric_index = container_index * len(container) + bar_index
            # Calculate the x positions for the start and end of the horizontal line
            x_start = bar.get_x()
            x_end = bar.get_x() + bar.get_width()
            # Draw the horizontal line at the specified value for this bar
            plt.hlines(y=list_dummy_metrics[metric_index], xmin = x_start, xmax = x_end, color = "black", linestyles= "dashed")

    # Create custom legend
    legend_elements = [Line2D([0], [0], color='coral', lw = 4, label='No balanceados'),
                    Line2D([0], [0], color = 'orangered', lw = 4, label='Balanceados'),
                    Line2D([0], [0], color = "black", lw = 2, linestyle = "dashed", label = "Métrica Dummy")]
    plt.legend(handles=legend_elements, title="Tipo de datos", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout(rect=[0, 0, 0.99, 1])
    plt.savefig(name_image)


# Calculate dummy metrics Heartattack
list_heartattack_dummy_unbalanced = calculate_metrics(758, 12669, 12677, 219918)
list_heartattack_dummy_balanced = calculate_metrics(116322, 116380, 116265, 116207)

# Calculate dummy metrics Angina
list_angina_dummy_unbalanced = calculate_metrics(966, 13983, 13987, 217086)
list_angina_dummy_balanced = calculate_metrics(115682, 115556, 115387, 115513)

# Plot the metrics for multiple linear regression Heartaccack
make_bar_plot(9, 6, 2623, 46567, 42670, 18478, 3780, 28107, list_heartattack_dummy_unbalanced, list_heartattack_dummy_balanced, "RLM Infarto: Datos sin balancear vs balanceados", "RLM_Heartattack_U_vs_B.png")

# Plot the metrics for multiple linear regression Angina
make_bar_plot(38, 26, 2889, 46252, 42916, 17476, 3291, 28745, list_angina_dummy_unbalanced, list_angina_dummy_balanced, "RLM Angina de pecho: Datos sin balancear vs balanceados", "RLM_Angina_U_vs_B.png")

# Plot the metrics for SVM Heartattack
make_bar_plot(3, 57, 25, 315, 198, 52, 7, 143, list_heartattack_dummy_unbalanced, list_heartattack_dummy_balanced, "SVM Infarto: Datos sin balancear vs balanceados", "SVM_Heartattack_U_vs_B.png")

# Plot th metrics for SVM Angina
make_bar_plot(5, 19, 15, 361, 199, 44, 23, 134, list_angina_dummy_unbalanced, list_angina_dummy_balanced, "SVM Angina de pecho: Datos sin balancear vs balanceados", "SVM_Angina_U_vs_B.png")


