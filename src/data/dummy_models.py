import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix

def calculate_dummy_model(X, y, type_strategy, name_model):
    dummy = DummyClassifier(strategy = type_strategy, random_state = 42)
    dummy.fit(X, y)
    predictions = dummy.predict(X)
    confusion = confusion_matrix(y, predictions, labels = [1, 0])
    print("Confusion matrix for", name_model, "data:\n")
    print(confusion, "\n")




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

# Calculate the dummy model for unbalanced data
calculate_dummy_model(X_heartattack, y_heartattack, "stratified", "unbalanced heartattack")
calculate_dummy_model(X_angina, y_angina, "stratified", "unbalanced angina")

# Calculate the dummy model for balanced data
calculate_dummy_model(X_heartattack_balanced, y_heartattack_balanced, "uniform", "balanced heartattack")
calculate_dummy_model(X_angina_balanced, y_angina_balanced, "uniform", "balanced angina")

