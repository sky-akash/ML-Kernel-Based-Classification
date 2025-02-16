# Import Libraries

import sys
sys.path.append(r"C:\Users\Akash Mittal\Documents\GitHub\ML-Project\sources")
import pandas as pd
import numpy as np

from data_process_scale import StandardScaling, TrainTestSplit
import data_process_scale
from perceptron_class import Perceptron
import SVM_pegasos
from Reg_log_class_2 import LogisticRegressionPegasos
from polynomial_features import PolynomialFeatureExpansion
from Kernel_Perceptron_F import KernelizedPerceptron
from Kernel_Pegasos_SVM_F1 import KernelizedPegasosSVM
from K_fold_CV import MyKFold # Custom Kfold CV
from GridSearch import cross_validate
from sklearn.metrics import accuracy_score


# Loading the dataset
file_path = r"C:\Users\Akash Mittal\Documents\GitHub\ML-Project\Test Data\test.xlsx"
df = pd.read_excel(file_path, engine='openpyxl')

# Splitting the data

# instance of class
splitter = TrainTestSplit(test_size=0.25, random_state=42, stratify=True)
X_train, y_train, X_test, y_test = splitter.split(df)


# Standardize features (fitting only on training data to avoid data leakage)
scaler = StandardScaling()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Converting to Arrays
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

# Setting Seed for reproducability
np.random.seed(42)


""" Simple Perceptron
"""
# Perceptron Parameter Grid
perceptron_param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'epochs': [500, 1000],
    'activation_func': ['unit_step', 'sigmoid']
}

print("\nGrid Search for Perceptron...")
best_params_perceptron, _ = cross_validate(
    model_class=Perceptron,
    param_grid=perceptron_param_grid,
    X=X_train,
    y=y_train,
    k=5,
    shuffle=True,
    random_state=42,
    verbose=True
)
perceptron_best = Perceptron(**best_params_perceptron)
weights, bias = perceptron_best.fit(X_train, y_train)
y_pred_best = perceptron_best.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f"Test Accuracy for Perceptron with Best Hyperparameters: {accuracy_best:.4f}")

"""Pegasos SVM
"""

# SVM Pegasos Grid
svm_pegasos_param_grid = {
    'lambda_param': [0.001, 0.01, 0.1],
    'epochs': [500, 1000]
}

# Grid Search for Pegasos SVM
print("\nGrid Search for Pegasos SVM...")
best_params_svm_pegasos, _ = cross_validate(
    model_class=SVM_pegasos.SVM_Pegasos,
    param_grid=svm_pegasos_param_grid,
    X=X_train,
    y=y_train,
    k=5,
    shuffle=True,
    random_state=42,
    verbose=True
)
svm_pegasos_best = SVM_pegasos.SVM_Pegasos(**best_params_svm_pegasos)
svm_pegasos_best.fit(X_train, y_train)
y_pred_svm_pegasos = svm_pegasos_best.predict(X_test)
accuracy_svm_pegasos = accuracy_score(y_test, y_pred_svm_pegasos)
print(f"Test Accuracy for Pegasos SVM with Best Hyperparameters: {accuracy_svm_pegasos:.4f}")


""" Regularized logistic classification (i.e., the Pegasos objective function with logistic loss instead of hinge loss)
"""
# Logistic Parameter Grid
logreg_pegasos_param_grid = {
    'lambda_param': [0.001, 0.01, 0.1],
    'epochs': [500, 1000]
}

# Grid Search for Logistic Regression Pegasos
print("\nGrid Search for Logistic Regression Pegasos...")
best_params_logreg_pegasos, _ = cross_validate(
    model_class=LogisticRegressionPegasos,
    param_grid=logreg_pegasos_param_grid,
    X=X_train,
    y=y_train,
    k=5,
    shuffle=True,
    random_state=42,
    verbose=True
)
logreg_pegasos_best = LogisticRegressionPegasos(**best_params_logreg_pegasos)
logreg_pegasos_best.fit(X_train, y_train)
y_pred_logreg_pegasos = logreg_pegasos_best.predict(X_test)
accuracy_logreg_pegasos = accuracy_score(y_test, y_pred_logreg_pegasos)
print(f"Test Accuracy for Logistic Regression Pegasos with Best Hyperparameters: {accuracy_logreg_pegasos:.4f}")

#### attempt to improve the performance of the previous models by using polynomial feature expansion of degree 2. Include and compare the linear weights corresponding to the various numerical features you found after the training phase.

""" Feature Expansion
"""

# Polynomial Feature Expansion
poly_expander = PolynomialFeatureExpansion(degree=2)
X_train_poly = poly_expander.fit_transform(X_train)
X_test_poly = poly_expander.fit_transform(X_test)

""" Perceptron with Polynomial Features
"""
# Grid Search with Polynomial Features for Perceptron
print("\nGrid Search for Perceptron with Polynomial Features...")
best_params_perceptron_poly, _ = cross_validate(
    model_class=Perceptron,
    param_grid=perceptron_param_grid,
    X=X_train_poly,
    y=y_train,
    k=5,
    shuffle=True,
    random_state=42,
    verbose=True
)
perceptron_poly_best = Perceptron(**best_params_perceptron_poly)
perceptron_poly_best.fit(X_train_poly, y_train)
y_pred_perceptron_poly = perceptron_poly_best.predict(X_test_poly)
accuracy_perceptron_poly = accuracy_score(y_test, y_pred_perceptron_poly)
print(f"Test Accuracy for Perceptron with Polynomial Features: {accuracy_perceptron_poly:.4f}")

""" Pegasos SVM with Polynomial Features
"""
# Grid Search with Polynomial Features for Pegasos SVM
print("\nGrid Search for Pegasos SVM with Polynomial Features...")
best_params_svm_pegasos_poly, _ = cross_validate(
    model_class=SVM_pegasos.SVM_Pegasos,
    param_grid=svm_pegasos_param_grid,
    X=X_train_poly,
    y=y_train,
    k=5,
    shuffle=True,
    random_state=42,
    verbose=True
)
svm_pegasos_poly_best = SVM_pegasos.SVM_Pegasos(**best_params_svm_pegasos_poly)
svm_pegasos_poly_best.fit(X_train_poly, y_train)
y_pred_svm_pegasos_poly = svm_pegasos_poly_best.predict(X_test_poly)
accuracy_svm_pegasos_poly = accuracy_score(y_test, y_pred_svm_pegasos_poly)
print(f"Test Accuracy for Pegasos SVM with Polynomial Features: {accuracy_svm_pegasos_poly:.4f}")

""" Logistic Regression Pegasos with Polynomial Features
"""
# Grid Search with Polynomial Features for Logistic Regression Pegasos
print("\nGrid Search for Logistic Regression Pegasos with Polynomial Features...")
best_params_logreg_pegasos_poly, _ = cross_validate(
    model_class=LogisticRegressionPegasos,
    param_grid=logreg_pegasos_param_grid,
    X=X_train_poly,
    y=y_train,
    k=5,
    shuffle=True,
    random_state=42,
    verbose=True
)
logreg_pegasos_poly_best = LogisticRegressionPegasos(**best_params_logreg_pegasos_poly)
logreg_pegasos_poly_best.fit(X_train_poly, y_train)
y_pred_logreg_pegasos_poly = logreg_pegasos_poly_best.predict(X_test_poly)
accuracy_logreg_pegasos_poly = accuracy_score(y_test, y_pred_logreg_pegasos_poly)
print(f"Test Accuracy for Logistic Regression Pegasos with Polynomial Features: {accuracy_logreg_pegasos_poly:.4f}")

""" Weight Comparisons
"""
# Model Weight Comparison
# Function to compare weights of different models
def compare_weights(models, model_names):
    """
    Compares weights of different models.

    Parameters:
    models (list): List of trained models.
    model_names (list): List of corresponding model names.
    """
    for model, name in zip(models, model_names):
        print(f"\nWeights for {name} Model:")
        if hasattr(model, 'weights'):  # Check if model has weights attribute
            print(model.weights)
        elif hasattr(model, 'coef_'):  # For models like LogisticRegression or SVM
            print(model.coef_)
        elif hasattr(model, 'support_vectors_'):  # For SVM models
            print(model.support_vectors_)
        else:
            print(f"No direct weight attribute found for {name} model.")

# Training models
perceptron_best.fit(X_train, y_train)
svm_pegasos_best.fit(X_train, y_train)
logreg_pegasos_best.fit(X_train, y_train)

# For models with Polynomial Features:
perceptron_poly_best.fit(X_train_poly, y_train)
svm_pegasos_poly_best.fit(X_train_poly, y_train)
logreg_pegasos_poly_best.fit(X_train_poly, y_train)

# Store models and their names for comparison
models = [
    perceptron_best,
    svm_pegasos_best,
    logreg_pegasos_best,
    perceptron_poly_best,
    svm_pegasos_poly_best,
    logreg_pegasos_poly_best
]

model_names = [
    "Perceptron",
    "Pegasos SVM",
    "Logistic Regression Pegasos",
    "Perceptron with Polynomial Features",
    "Pegasos SVM with Polynomial Features",
    "Logistic Regression Pegasos with Polynomial Features"
]

# Compare weights for all models
compare_weights(models, model_names)


##### Kernelized Perceptron - Custom Grid Search

# Define the parameter grids for RBF and Polynomial kernels
param_grid_rbf = {
    'gamma': [0.1, 1.0, 10],  # Range of gamma values for RBF kernel
}

param_grid_poly = {
    'degree': [2, 3, 4],  # Degree values for Polynomial kernel
}

# RBF Kernel
print("Performing Custom Cross-Validation for RBF Kernel...")
model_rbf = KernelizedPerceptron(kernel="rbf") # Instantiation
best_params_rbf, best_accuracy_rbf = cross_validate(
    model_class=KernelizedPerceptron, # passing the class
    param_grid=param_grid_rbf,
    X=X_train,
    y=y_train
)

print(f"Best Parameters for RBF Kernel: {best_params_rbf}")
print(f"Best Accuracy for RBF Kernel: {best_accuracy_rbf * 100:.2f}%")

# Polynomial Kernel
print("Performing Custom Cross-Validation for Polynomial Kernel...")
model_poly = KernelizedPerceptron(kernel="poly")  # Instantiation
best_params_poly, best_accuracy_poly = cross_validate(
    model_class=KernelizedPerceptron,  # passing the class
    param_grid=param_grid_poly,
    X=X_train,
    y=y_train
)

print(f"Best Parameters for Polynomial Kernel: {best_params_poly}")
print(f"Best Accuracy for Polynomial Kernel: {best_accuracy_poly * 100:.2f}%")
