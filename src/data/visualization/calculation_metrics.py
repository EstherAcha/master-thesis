# This file contains the functions to calculate the metrics of the confusion matrix

def calculate_accuracy(VP, FP, FN, VN):
    accuracy = (VP + VN) / (VP + FP + FN + VN)
    return accuracy

def calculate_precision(VP, FP):
    precision = VP / (VP + FP)
    return precision

def calculate_sensitivity(VP, FN):
    sensitivity = VP / (VP + FN)
    return sensitivity

def calculate_specificity(FP, VN):  
    specificity = VN / (FP + VN)
    return specificity