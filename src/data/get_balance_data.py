import pandas as pd
import numpy as np
from matplotlib import colormaps
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import math

X = pd.read_csv("/home/esther/Desktop/master-thesis/data/processed/X_data_preprocessed.csv", sep = ",", na_values = ["NaN"])
y_heartattack = pd.read_csv("/home/esther/Desktop/master-thesis/data/processed/y_heartattack_preprocessed.csv", sep = ",", na_values = ["NaN"])
y_angina = pd.read_csv("/home/esther/Desktop/master-thesis/data/processed/y_angina_preprocessed.csv", sep = ",", na_values = ["NaN"])

# Apply SMOTE to balance the data
oversample = SMOTE()

X_heartattack_balanced, y_heartattack_balanced = oversample.fit_resample(X, y_heartattack)
X_angina_balanced, y_angina_balanced = oversample.fit_resample(X, y_angina)

print(y_heartattack["HeartAttack"].value_counts())
print(y_heartattack_balanced["HeartAttack"].value_counts())


# # Save the balanced data
# X_heartattack_balanced.to_csv("/home/esther/Desktop/master-thesis/data/processed/X_heartattack_balanced.csv", index = False)
# y_heartattack_balanced.to_csv("/home/esther/Desktop/master-thesis/data/processed/y_heartattack_balanced.csv", index = False)
# X_angina_balanced.to_csv("/home/esther/Desktop/master-thesis/data/processed/X_angina_balanced.csv", index = False)
# y_angina_balanced.to_csv("/home/esther/Desktop/master-thesis/data/processed/y_angina_balanced.csv", index = False)


