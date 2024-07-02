import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
import math

# Import raw data
heart_csv = pd.read_csv("/home/esther/Desktop/master-thesis/data/raw/heart_2022_with_nans.csv", sep = ",", na_values = ["NaN"])

# Eliminate rows with missing values
heart_csv.dropna(axis = 0, inplace = True)

# Separate features and target
X = heart_csv.drop('HadHeartAttack', axis=1)
X = X.drop("HadAngina", axis=1)

y_heartattack = heart_csv["HadHeartAttack"]  
y_angina = heart_csv["HadAngina"]

# Extract the names of the columns
columns = X.columns
# Categorical columns
categorical_column_list = ["State", "Sex", "GeneralHealth", "PhysicalHealthDays", "MentalHealthDays", "LastCheckupTime", "PhysicalActivities",
                           "RemovedTeeth", "HadStroke", "HadAsthma", "HadSkinCancer", "HadCOPD", "HadDepressiveDisorder",
                           "HadKidneyDisease", "HadArthritis", "HadDiabetes", "DeafOrHardOfHearing",
                           "BlindOrVisionDifficulty", "DifficultyConcentrating", "DifficultyWalking", "DifficultyDressingBathing",
                           "DifficultyErrands", "SmokerStatus", "ECigaretteUsage", "ChestScan", "RaceEthnicityCategory",
                           "AgeCategory", "AlcoholDrinkers", "HIVTesting", "FluVaxLast12", "PneumoVaxEver", 
                           "TetanusLast10Tdap", "HighRiskLastYear", "CovidPos"]

# Encode the categorical variables
encoder = OrdinalEncoder()
X[categorical_column_list] = encoder.fit_transform(X[categorical_column_list])

# Use LabelEncoder for the target variables
le = LabelEncoder()
y_heartattack = le.fit_transform(y_heartattack)
y_angina = le.fit_transform(y_angina)

# Transform to a DataFrame
X = pd.DataFrame(X, columns=columns)
y_heartattack = pd.DataFrame(y_heartattack, columns=["HeartAttack"])
y_angina = pd.DataFrame(y_angina, columns=["Angina"])
print(y_heartattack)

# # Export DataFrame to CSV
X.to_csv('/home/esther/Desktop/master-thesis/data/processed/X_data_preprocessed.csv', index=False)
y_heartattack.to_csv('/home/esther/Desktop/master-thesis/data/processed/y_heartattack_preprocessed.csv', index=False)
y_angina.to_csv('/home/esther/Desktop/master-thesis/data/processed/y_angina_preprocessed.csv', index=False)
