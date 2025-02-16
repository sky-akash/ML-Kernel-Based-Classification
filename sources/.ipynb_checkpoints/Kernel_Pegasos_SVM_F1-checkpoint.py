import numpy as np

class KernelizedPegasosSVM:
    def __init__(self, lambda_param=0.01, T=1000, kernel='rbf', gamma=1, degree=3):
        self.lambda_param = lambda_param  # Regularization parameter
        self.T = T  # Number of iterations
        self.kernel_type = kernel  # Kernel type: 'rbf' or 'poly'
        self.gamma = gamma  # Parameter for RBF kernel
        self.degree = degree  # Degree for polynomial kernel
        self.alpha = None  # Lagrange multipliers
        self.X_train = None  # Store training data
        self.y_train = None  # Store training labels
        self.K = None  # Kernel matrix for training data
    
    def kernel(self, x1, x2):
        """Compute the kernel function."""
        if self.kernel_type == 'rbf':
            return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)
        elif self.kernel_type == 'poly':
            return (1 + np.dot(x1, x2)) ** self.degree
        else:
            raise ValueError("Unsupported kernel type")

    def train(self, X, y):
        """Train the SVM using the Kernelized Pegasos algorithm."""
        m, n = X.shape
        self.alpha = np.zeros(m)
        self.X_train = X
        self.y_train = y
        
        # Precompute the kernel matrix for training data
        self.K = np.array([[self.kernel(x_train, x_test) for x_test in X] for x_train in X])
        
        for t in range(1, self.T + 1):
            i = np.random.randint(m)  # Pick a random sample
            eta = 1 / (self.lambda_param * t)  # Learning rate
            
            # Compute the decision function for the selected sample
            sum_kernel = np.sum(self.alpha * self.y_train * self.K[:, i])
            
            # Update alpha[i] if the sample is misclassified
            if y[i] * sum_kernel < 1:
                self.alpha[i] += eta

    def predict(self, X):
        """Make predictions using the trained model."""
        # Compute the kernel matrix between training data and input data
        K_matrix = np.array([[self.kernel(x_train, x_test) for x_test in X] for x_train in self.X_train])
        
        # Transpose K_matrix to (n_test, n_train) and compute decision values
        decision_values = np.dot(K_matrix.T, self.alpha * self.y_train)
        
        # Return the predicted labels
        return np.sign(decision_values)


""""
# Example usage:
if __name__ == "__main__":
    # Generate synthetic data
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=100, n_features=2, n_classes=2, n_redundant=0, n_clusters_per_class=1)
    y = np.where(y == 0, -1, 1)  # Convert labels to {-1, 1}
    
    # Train and test the model
    model = KernelizedPegasosSVM(kernel='rbf', gamma=0.5)
    model.train(X, y)
    predictions = model.predict(X)
    
    # Evaluate accuracy
    accuracy = np.mean(predictions == y)
    print(f"Accuracy: {accuracy:.2f}")
"""