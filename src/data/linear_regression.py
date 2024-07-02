import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, recall_score
from sklearn import metrics
from visualization.calculation_metrics import calculate_sensitivity, calculate_specificity, calculate_accuracy


def transform_probabilities(predictions_list, test_list, disease):
    best_predictions_list = []
    best_sensitivity = 0
    best_threshold = 0
    metrics_values = pd.DataFrame(columns = ["Threshold", "VP", "FN", "FP", "VN"])
    for threshold in np.arange(0, 1, 0.0001):
        threshold = round(threshold, 4)
        predictions_list_temp = []
        metrics_list = []
        for idx in range(len(predictions_list)):
            if predictions_list[idx] > threshold:
                predictions_list_temp.append(1)
            else:
                predictions_list_temp.append(0)
        
        conf_matrix = confusion_matrix(test_list, predictions_list_temp, labels = [1, 0])
        metrics_list.append(threshold)
        for i in range(2):
            for j in range(2):
                metrics_list.append(conf_matrix[i][j])
        metrics_values.loc[len(metrics_values)] = metrics_list
        metrics_values.to_csv(f"/home/esther/Desktop/master-thesis/data/processed/metrics_values_{disease}.csv")

        accuracy = calculate_accuracy(conf_matrix[0][0], conf_matrix[0][1], conf_matrix[1][0], conf_matrix[1][1])
        if accuracy > best_sensitivity:
            best_sensitivity = accuracy
            best_predictions_list = predictions_list_temp
            best_threshold = threshold
        
    return best_predictions_list, best_threshold


def calculate_multiple_linear_regression(X, y, name_model):
    # Split the data heartattack
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    # Model heartattack
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)

    # Predictions 
    predictions = model_lr.predict(X_test)

    # Evaluate the model 
    print(f"\nEvaluation of the {name_model} model:\n")
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictions))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, predictions))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

    # Transform the predictions to a list
    predictions_list = predictions.ravel().tolist()
    test_list = y_test.values.ravel().tolist()

    # Transform the probabilities to 0 and 1
    predictions_list, threshold=  transform_probabilities(predictions_list, test_list, "heartattack")
    print(f"Best threshold: {threshold}")

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, predictions_list, labels = [1, 0])
    print("Confusion matrix:\n", conf_matrix)

    # Save y_test
    y_test_df = pd.DataFrame(test_list)
    y_test_df.to_csv("/home/esther/Desktop/master-thesis/data/processed/y_test_MLR.csv")


# Load the unbalanced data
X_heartattack = pd.read_csv("/home/esther/Desktop/master-thesis/data/processed/X_heartattack_importance_analysis.csv")
X_angina = pd.read_csv("/home/esther/Desktop/master-thesis/data/processed/X_angina_importance_analysis.csv")
y_heartattack = pd.read_csv("/home/esther/Desktop/master-thesis/data/processed/y_heartattack_preprocessed.csv")
y_angina = pd.read_csv("/home/esther/Desktop/master-thesis/data/processed/y_angina_preprocessed.csv")

# Load balanced data
X_heartattack_balanced = pd.read_csv("/home/esther/Desktop/master-thesis/data/processed/X_heartattack_balanced_importance_analysis.csv")
X_angina_balanced = pd.read_csv("/home/esther/Desktop/master-thesis/data/processed/X_angina_balanced_importance_analysis.csv")
y_heartattack_balanced = pd.read_csv("/home/esther/Desktop/master-thesis/data/processed/y_heartattack_balanced.csv")
y_angina_balanced = pd.read_csv("/home/esther/Desktop/master-thesis/data/processed/y_angina_balanced.csv")

# Calculate the multiple linear regression for heartattack
# calculate_multiple_linear_regression(X_heartattack, y_heartattack, "unbalanced heartattack")

# Calculate the multiple linear regression for angina
# calculate_multiple_linear_regression(X_angina, y_angina, "unbalanced angina")

# Calculate the multiple linear regression for balanced heartattack
calculate_multiple_linear_regression(X_heartattack_balanced, y_heartattack_balanced, "balanced heartattack")

# Calculate the multiple linear regression for balanced angina
# calculate_multiple_linear_regression(X_angina_balanced, y_angina_balanced, "balanced angina")


