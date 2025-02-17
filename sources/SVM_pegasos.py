import numpy as np

class SVM_Pegasos:
    def __init__(self, lambda_param=0.01, epochs=1000, include_bias=True):
        self.lambda_param = lambda_param  # Regularization parameter
        self.epochs = epochs
        self.weights = None
        self.include_bias = include_bias
        self.weights_sum = None  # Initialize weights sum for averaging
        self.t = 0  # Total number of iterations for averaging

    def _add_bias(self, X):
        return np.hstack([X, np.ones((X.shape[0], 1))])

    def _project_weights(self):
        """Project weights onto L2 ball with radius 1/sqrt(lambda)"""
        norm_squared = np.dot(self.weights, self.weights)
        max_norm_squared = 1.0 / self.lambda_param
        if norm_squared > max_norm_squared:
            scale_factor = np.sqrt(norm_squared) / np.sqrt(max_norm_squared)
            self.weights /= scale_factor  # Scale the weights

    def fit(self, X, y):
        # Bias term
        if self.include_bias:
            X = self._add_bias(X)
        
        n_samples, n_features = X.shape
        # Initialize weights
        self.weights = np.zeros(n_features)
        self.weights_sum = np.zeros_like(self.weights)  # Initialize weights sum for averaging

        # Pegasos Algorithm
        for epoch in range(self.epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(n_samples):
                self.t += 1
                xi = X_shuffled[i]
                yi = y_shuffled[i]

                # Learning Rate for this step
                eta_t = 1.0 / (self.lambda_param * self.t)
                
                # Compute the margin
                margin = yi * np.dot(xi, self.weights)

                # Update weights according to Pegasos rules
                if margin < 1:
                    self.weights = (1 - eta_t * self.lambda_param) * self.weights + eta_t * yi * xi
                else:
                    self.weights = (1 - eta_t * self.lambda_param) * self.weights

                # Project Weights
                self._project_weights()
                
                # Accumulating weights for averaging
                self.weights_sum += self.weights

        # Averaging weights after all epochs
        self.weights = self.weights_sum / self.t  # Averaging over total iterations

    def predict(self, X):
        if self.include_bias and X.shape[1] == self.weights.shape[0] - 1:
            X = self._add_bias(X)
        linear_output = np.dot(X, self.weights)
        return np.sign(linear_output)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

###########################################################################