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

""" Simple Perceptron
"""
# Model Initialization
perceptron_model = Perceptron(learning_rate=0.01, epochs=1000, activation_func='unit_step')
# Fitting Model
weights, bias = perceptron_model.fit(X_train, y_train)
# Predictions
y_pred = perceptron_model.predict(X_test)
# Evaluate Accuracy
accuracy_perceptron = perceptron_model.accuracy(X_test, y_test)
print(f"Accuracy for Simple Perceptron: {accuracy_perceptron:.4f}")


"""Pegasos SVM
"""
# Random Seed
np.random.seed(42)
# Model Initialization
svm_pegasos_model = SVM_pegasos.SVM_Pegasos(lambda_param=0.01, epochs=1000)
# Fitting Model
svm_pegasos_model.fit(X_train, y_train)
# Make predictions
y_pred = svm_pegasos_model.predict(X_test)
# Evaluate accuracy
accuracy = svm_pegasos_model.score(X_test, y_test)
print(f"Accuracy for Pegasos SVM: {accuracy * 100:.2f}%")


""" Regularized logistic classification (i.e., the Pegasos objective function with logistic loss instead of hinge loss)
"""

# Random Seed
np.random.seed(42)

# Model Instantialization and training
logreg_pegasos = LogisticRegressionPegasos(lambda_param=0.01, epochs=1000)
logreg_pegasos.fit(X_train, y_train)
# Make predictions
y_pred = logreg_pegasos.predict(X_train)
# Evaluate accuracy
accuracy_reglog = logreg_pegasos.score(X_train, y_train)
print(f"Accuracy: {accuracy_reglog * 100:.2f}%")

# attempt to improve the performance of the previous models by using polynomial feature expansion of degree 2. Include and compare the linear weights corresponding to the various numerical features you found after the training phase.

# Polynomial Expansion
poly_expander = PolynomialFeatureExpansion(degree=2)

X_train_poly = poly_expander.fit_transform(X_train)
X_test_poly = poly_expander.fit_transform(X_test)

# Train models with polynomial features
np.random.seed(42)
perceptron_model.fit(X_train_poly, y_train)
svm_pegasos_model.fit(X_train_poly, y_train)
logreg_pegasos.fit(X_train_poly, y_train)

# Compare weights
print("Weights for Perceptron:\n", perceptron_model.weights)
print("Weights for SVM Pegasos:\n", svm_pegasos_model.weights)
print("Weights for Logistic Regression Pegasos:\n", logreg_pegasos.weights)

# Evaluate performance
accuracy_perceptron = perceptron_model.accuracy(X_test_poly, y_test)
accuracy_svm_pegasos = svm_pegasos_model.score(X_test_poly, y_test)
accuracy_logreg_pegasos = logreg_pegasos.score(X_test_poly, y_test)

# Print accuracy scores
print(f"Perceptron: {accuracy_perceptron * 100:.2f}%")
print(f"SVM Pegasos Accuracy: {accuracy_svm_pegasos * 100:.2f}%")
print(f"Logistic Regression Pegasos Accuracy: {accuracy_logreg_pegasos * 100:.2f}%")


""" Kernelized Perceptron (Gaussian and Polynomial Kernels)
"""
# RBF (Gaussian) Kernel
print("Training with RBF Kernel...")
# Model Initialization
model_rbf = KernelizedPerceptron(kernel="rbf", gamma=1.0)
# Fitting Model
model_rbf.fit(X_train, y_train, epochs=10)  # Specify number of epochs
# Predictions
y_pred_rbf = model_rbf.predict(X_test)
# Evaluation
print(f"RBF Kernel Accuracy: {accuracy_score(y_test, y_pred_rbf):.4f}")

# Polynomial Kernel
print("Training with Polynomial Kernel...")
# Model Initialization
model_poly = KernelizedPerceptron(kernel="poly", degree=3)
# Fitting Model
model_poly.fit(X_train, y_train)
# Predictions
y_pred_poly = model_poly.predict(X_test)
# Evaluation
print(f"Polynomial Kernel Accuracy: {accuracy_score(y_test, y_pred_poly):.4f}")


""" Kernelized Pegasos SVM (Gaussian and Polynomial Kernels)
"""

# Train and test RBF Kernelized Pegasos SVM
# Model Initialization
model_rbf_ksvm = KernelizedPegasosSVM(kernel='rbf', gamma=0.5)
# Fitting Model
model_rbf_ksvm.train(X_train, y_train)
# Predictions
predictions_rbf_ksvm = model_rbf_ksvm.predict(X_test)
# Evaluation
print(f"RBF_SVM Kernel Accuracy: {accuracy_score(y_test, predictions_rbf_ksvm):.2f}")

# Train and test Polynomial Kernelized Pegasos SVM
# Model Initialization
model_poly_ksvm = KernelizedPegasosSVM(kernel='poly', degree=3)
# Fitting Model
model_poly_ksvm.train(X_train, y_train)
# Predictions
predictions_poly_ksvm = model_poly_ksvm.predict(X_test)
# Evaluation
print(f"Polynomial Kernel Accuracy: {accuracy_score(y_test, predictions_poly_ksvm):.2f}")







