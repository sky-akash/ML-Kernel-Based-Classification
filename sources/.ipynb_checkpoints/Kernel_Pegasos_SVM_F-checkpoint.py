


import numpy as np

class KernelizedPegasosSVM:
    def __init__(self, kernel="rbf", degree=3, gamma=1.0, lambda_param=0.01, epochs=1000):
        """
        Kernelized Pegasos SVM supporting RBF and Polynomial kernels.

        Parameters:
        kernel : str, "rbf" or "poly"
        degree : int, Degree of polynomial kernel (used if kernel="poly")
        gamma : float, Kernel coefficient for RBF kernel
        lambda_param : float, Regularization parameter
        epochs : int, Number of training iterations
        """
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.lambda_param = lambda_param
        self.epochs = epochs
        self.alpha = None  # Dual coefficients
        self.support_vectors = None
        self.support_labels = None

    def _rbf_kernel(self, X1, X2):
        """Gaussian RBF kernel function"""
        pairwise_sq_dist = np.sum(X1**2, axis=1, keepdims=True) - 2 * np.dot(X1, X2.T) + np.sum(X2**2, axis=1)
        return np.exp(-self.gamma * pairwise_sq_dist)

    def _poly_kernel(self, X1, X2):
        """Polynomial kernel function"""
        return (1 + np.dot(X1, X2.T)) ** self.degree

    def _compute_kernel(self, X1, X2):
        """Compute kernel matrix based on selected kernel"""
        if self.kernel == "rbf":
            return self._rbf_kernel(X1, X2)
        elif self.kernel == "poly":
            return self._poly_kernel(X1, X2)
        else:
            raise ValueError("Unsupported kernel. Use 'rbf' or 'poly'.")

    def fit(self, X, y):
        """Train Kernelized Pegasos SVM"""
        n_samples = X.shape[0]
        self.alpha = np.zeros(n_samples)  # Dual coefficients
        self.support_vectors = X
        self.support_labels = y
        t = 0  # Time step

        for epoch in range(self.epochs):
            for i in range(n_samples):
                t += 1  # Increment step
                eta = 1 / (self.lambda_param * t)  # Learning rate
                kernel_sum = np.sum(self.alpha * y * self._compute_kernel(X, X[i:i+1]))
                if y[i] * kernel_sum < 1:  # Support vector condition
                    self.alpha[i] += eta  # Update alpha
                self.alpha *= (1 - eta * self.lambda_param)  # Regularization

    def predict(self, X):
        """Predict class labels for input data"""
        kernel_matrix = self._compute_kernel(self.support_vectors, X)
        prediction_scores = np.sum(self.alpha * self.support_labels[:, None] * kernel_matrix, axis=0)
        return np.sign(prediction_scores)  # Return -1 or 1
