import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer, recall_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import expon, reciprocal


def calculate_svm(X, y, name_model):
    # Split the data heartattack
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    X_subset = X_train.iloc[:5000]
    y_subset = y_train.iloc[:5000]
    X_test_subset = X_test[:400]
    y_test_subset = y_test[:400]

    # # Model heartattack
    # model_svm = SVC(kernel = "sigmoid", random_state = 42)

    custom_weight1 = {0: 1, 1: 20}
    custom_weight2 = {0: 1, 1: 3}

    # Parameter grid
    param_grid = {
        'C': [1],
        'gamma': [0.01],
        'kernel': ['rbf'],
        'class_weight': [custom_weight1],
    }

    # Grid search
    grid = GridSearchCV(SVC(), param_grid, verbose=2, scoring = "accuracy", cv = 10)
    grid.fit(X_subset, y_subset.values.ravel()) 

    # Best parameters and model
    print("Best parameters found: ", grid.best_params_)
    best_model = grid.best_estimator_

    # Make predictions with the best model
    predictions = best_model.predict(X_test_subset)


    # Evaluate the model 
    print(f"\nEvaluation of the {name_model} model:\n")
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_subset, predictions))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test_subset, predictions))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_subset, predictions)))

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test_subset, predictions, labels = [1, 0])
    print("Confusion matrix:\n", conf_matrix)

    # Transform the predictions to a list
    predictions_list = predictions.ravel().tolist()
    test_list = y_test_subset.values.ravel().tolist()

    correct_predictions = 0
    correct_disease = 0
    incorrect_disease = 0
    incorrect_healthy = 0
    for i in range(len(predictions_list)):
        # Normalize data types for comparison
        predicted = int(predictions_list[i])
        actual = int(test_list[i])
        
        if predicted == actual:
            correct_predictions += 1
            if predicted == 1:
                correct_disease += 1
        else:
            if predicted == 1:
                incorrect_disease += 1
            elif predicted == 0:  
                incorrect_healthy += 1

    print("Total of predicitions:", len(predictions_list))
    print("Correct predictions heartattack:", correct_predictions)
    print("Total of correcct predicitions that suffer from heartattack:", correct_disease)
    print("Total of wrong predicitions that suffer form heart attack:", incorrect_disease)
    print("Total of wrong predictions that are healthy:", incorrect_healthy, "\n")


# Load unbalanced data
X_heartattack = pd.read_csv("/home/esther/Desktop/master-thesis/data/processed/X_heartattack_importance_analysis.csv")
X_angina = pd.read_csv("/home/esther/Desktop/master-thesis/data/processed/X_angina_importance_analysis.csv")
y_heartattack = pd.read_csv("/home/esther/Desktop/master-thesis/data/processed/y_heartattack_preprocessed.csv")
y_angina = pd.read_csv("/home/esther/Desktop/master-thesis/data/processed/y_angina_preprocessed.csv")

numerical_cols = ["PhysicalHealthDays", "MentalHealthDays", "SleepHours", "HeightInMeters", "WeightInKilograms", "BMI"] 
scaler = StandardScaler()

X_heartattack_scaled = scaler.fit_transform(X_heartattack[numerical_cols])
X_angina_scaled = scaler.fit_transform(X_angina[numerical_cols])

# Load balanced data
X_heartattack_balanced = pd.read_csv("/home/esther/Desktop/master-thesis/data/processed/X_heartattack_balanced_importance_analysis_0.002.csv")
X_angina_balanced = pd.read_csv("/home/esther/Desktop/master-thesis/data/processed/X_angina_balanced_importance_analysis_0.002.csv")
y_heartattack_balanced = pd.read_csv("/home/esther/Desktop/master-thesis/data/processed/y_heartattack_balanced.csv")
y_angina_balanced = pd.read_csv("/home/esther/Desktop/master-thesis/data/processed/y_angina_balanced.csv")

X_heartattack_balanced_scaled = scaler.fit_transform(X_heartattack_balanced[numerical_cols])
X_angina_balanced_scaled = scaler.fit_transform(X_angina_balanced[numerical_cols])


# Calculate SVM for heartattack
# calculate_svm(X_heartattack, y_heartattack, "unbalanced heartattack")

# Calculate SVM for angina
# calculate_svm(X_angina, y_angina, "unbalanced angina")

# Calculate SVM for balanced heartattack
calculate_svm(X_heartattack_balanced, y_heartattack_balanced, "balanced heartattack")

# Calculate SVM for balanced angina
# calculate_svm(X_angina_balanced, y_angina_balanced, "balanced angina")
