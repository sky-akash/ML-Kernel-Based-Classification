import numpy as np

class LogisticRegressionPegasos:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param  # Regularization parameter
        self.epochs = epochs
        self.weights = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Initialize weights
        self.weights = np.zeros(n_features)

        # Pegasos algorithm with logistic loss
        for epoch in range(self.epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(n_samples):
                xi = X_shuffled[i]
                yi = y_shuffled[i]

                # Compute the margin
                margin = yi * np.dot(xi, self.weights)

                # Update weights according to Pegasos logistic loss
                gradient = (yi * xi) / (1 + np.exp(yi * margin))
                self.weights = (1 - self.learning_rate * self.lambda_param) * self.weights + self.learning_rate * gradient

    def predict_proba(self, X):
        # Compute the decision function (logistic sigmoid)
        linear_output = np.dot(X, self.weights)
        return 1 / (1 + np.exp(-linear_output))

    def predict(self, X):
        # Return class label: 1 if sigmoid output >= 0.5, -1 if less
        proba = self.predict_proba(X)
        return np.where(proba >= 0.5, 1, -1)

    def score(self, X, y):
        # Predict labels for the input data
        y_pred = self.predict(X)
        # Calculate accuracy as the mean of correct predictions
        accuracy = np.mean(y_pred == y)
        return accuracy