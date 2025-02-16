"""
Classes to implement the kernels 
1st part ->
- Kernelized Perceptron
- Kernelized Perceptron with Gaussian
- Kernelized Perceptron with Polynomials
2nd Part ->
Kernelized Pegasos
-> with Gaussian & with Polynomials for SVM using the Given algo, after correcting the update mistake in the algo - 
"""

import numpy as np

class KernelPerceptron:
    def __init__(self, kernel_type='linear', degree=3, gamma=1.0, epochs=100):
        self.kernel_type = kernel_type
        self.degree = degree  # For polynomial kernel
        self.gamma = gamma  # For Gaussian kernel
        self.epochs = epochs

    def kernel(self, x1, x2):
        """Calculate the kernel between two samples"""
        if self.kernel_type == 'linear':
            return np.dot(x1, x2)
        elif self.kernel_type == 'polynomial':
            return (np.dot(x1, x2) + 1) ** self.degree
        elif self.kernel_type == 'gaussian':
            return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)

    def fit(self, X, y):
        """Train the kernel perceptron"""
        self.alpha = np.zeros(len(X))  # Initialize alpha values
        self.support_vectors = X  # Store the support vectors
        self.support_labels = y  # Store the corresponding labels
        
        for epoch in range(self.epochs):
            for i in range(len(X)):
                summation = sum(self.alpha[j] * y[j] * self.kernel(X[i], X[j]) for j in range(len(X)))
                if y[i] * summation <= 0:
                    self.alpha[i] += 1  # Update alpha

    def predict(self, X):
        """Predict new data points"""
        y_pred = []
        for x in X:
            summation = sum(self.alpha[j] * self.support_labels[j] * self.kernel(x, self.support_vectors[j]) for j in range(len(self.support_vectors)))
            y_pred.append(np.sign(summation))
        return np.array(y_pred)


class KernelPegasosSVM:
    def __init__(self, kernel_type='linear', lambda_param=0.01, epochs=1000, degree=3, gamma=1.0):
        self.kernel_type = kernel_type
        self.lambda_param = lambda_param
        self.epochs = epochs
        self.degree = degree  # For polynomial kernel
        self.gamma = gamma  # For Gaussian kernel

    def kernel(self, x1, x2):
        """Calculate the kernel between two samples"""
        if self.kernel_type == 'linear':
            return np.dot(x1, x2)
        elif self.kernel_type == 'polynomial':
            return (np.dot(x1, x2) + 1) ** self.degree
        elif self.kernel_type == 'gaussian':
            return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)

    def fit(self, X, y):
        """Train the kernel Pegasos SVM"""
        n_samples = len(X)
        self.alpha = np.zeros(n_samples)  # Initialize alpha values
        self.support_vectors = X
        self.support_labels = y

        for t in range(1, self.epochs + 1):
            # Randomly select an index it
            it = np.random.randint(0, n_samples)
            eta = 1 / (self.lambda_param * t)

            # Summation over all j
            summation = sum(self.alpha[j] * y[j] * self.kernel(X[it], X[j]) for j in range(n_samples))

            # Update rule for Pegasos
            if y[it] * summation < 1:
                self.alpha[it] += eta
            else:
                self.alpha[it] *= (1 - eta * self.lambda_param)

    def predict(self, X):
        """Predict using the kernel Pegasos SVM"""
        y_pred = []
        for x in X:
            summation = sum(self.alpha[j] * self.support_labels[j] * self.kernel(x, self.support_vectors[j]) for j in range(len(self.support_vectors)))
            y_pred.append(np.sign(summation))
        return np.array(y_pred)


class CustomGridSearchCV:
    def __init__(self, model_class, param_grid, scoring='accuracy', cv=None):
        """
        Custom implementation of GridSearchCV.
        
        model_class: Class of the model (e.g., KernelPerceptron, KernelPegasosSVM).
        param_grid: Dictionary of hyperparameters to search over.
        scoring: Metric to optimize (e.g., 'accuracy').
        cv: Number of cross-validation splits (optional).
        """
        self.model_class = model_class
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv = cv
        self.best_params_ = None
        self.best_score_ = None
    
    def fit(self, X_train, Y_train, X_test, Y_test):
        """
        Perform grid search over the parameter grid.
        
        X_train, Y_train: Training data.
        X_test, Y_test: Validation/Test data.
        """
        best_score = 0
        best_params = None
        
        # Iterate over all combinations of hyperparameters
        for params in self.generate_param_combinations(self.param_grid):
            model = self.model_class(**params)  # Initialize the model with the current parameters
            model.fit(X_train, Y_train)  # Train the model
            
            Y_pred = model.predict(X_test)  # Predict on test/validation set
            accuracy = self.compute_accuracy(Y_test, Y_pred)  # Compute accuracy
            
            if accuracy > best_score:
                best_score = accuracy
                best_params = params
        
        self.best_params_ = best_params
        self.best_score_ = best_score
    
    def generate_param_combinations(self, param_grid):
        """
        Generate all combinations of hyperparameters.
        """
        from itertools import product
        keys, values = zip(*param_grid.items())
        for combo in product(*values):
            yield dict(zip(keys, combo))
    
    def compute_accuracy(self, Y_true, Y_pred):
        """
        Compute accuracy score.
        """
        return np.mean(Y_true == Y_pred)

    def accuracy_score(Y_true, Y_pred):
        """
        Custom accuracy score function.
        """
        return np.mean(Y_true == Y_pred)
    
    def classification_report(Y_true, Y_pred):
        """
        Custom classification report function.
        """
        from collections import Counter
        
        # Unique classes
        classes = np.unique(Y_true)
        report = {}
        
        for cls in classes:
            # True Positives, False Positives, False Negatives
            TP = np.sum((Y_true == cls) & (Y_pred == cls))
            FP = np.sum((Y_true != cls) & (Y_pred == cls))
            FN = np.sum((Y_true == cls) & (Y_pred != cls))
            
            # Precision, Recall, F1-score
            precision = TP / (TP + FP) if (TP + FP) != 0 else 0
            recall = TP / (TP + FN) if (TP + FN) != 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
            
            # Store the metrics for this class
            report[cls] = {
                'precision': precision,
                'recall': recall,
                'f1-score': f1_score
            }
        
        # Print the report
        print("Classification Report:")
        for cls, metrics in report.items():
            print(f"Class {cls}:")
            print(f"  Precision: {metrics['precision']:.2f}")
            print(f"  Recall: {metrics['recall']:.2f}")
            print(f"  F1-Score: {metrics['f1-score']:.2f}")
        
        return report
