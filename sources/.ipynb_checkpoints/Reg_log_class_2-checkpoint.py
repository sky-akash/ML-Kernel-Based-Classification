import numpy as np

class LogisticRegressionPegasos:
    def __init__(self, lambda_param=0.01, epochs=1000, batch_size=32):
        # self.learning_rate = learning_rate
        self.lambda_param = lambda_param  # Regularization parameter
        self.epochs = epochs
        self.batch_size = batch_size  # Batch size for mini-batch updates
        self.weights = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -250, 250))) # clipping limits for very large z

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Initialize weights
        self.weights = np.zeros(n_features)
        t = 0 # Time step for learning Rate 

        # Pegasos algorithm with logistic loss
        for epoch in range(self.epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, n_samples, self.batch_size):
                t += 1 # Increment
                eta = 1 / (self.lambda_param * t) # Pegasos-style learning rate

                # Get the current mini-batch
                batch_indices = indices[i:i + self.batch_size]
                X_batch = X_shuffled[batch_indices]
                y_batch = y_shuffled[batch_indices]
                
                # Compute the margin for the mini-batch
                margin = np.dot(X_batch, self.weights)
                margin = np.clip(margin, -500, 500)  # Clip the margin for stability

                # Compute the gradient for the logistic loss and regularization
                exp_term = np.exp(-y_batch * margin)  # Logistic loss part
                gradient = np.dot(X_batch.T, (-y_batch * exp_term) / (1 + exp_term)) / self.batch_size
                gradient += self.lambda_param * self.weights  # Add regularization gradient

                # Update weights
                self.weights -= eta * gradient

   # def predict_proba(self, X):
    #    # Compute the decision function (logistic sigmoid)
     #   linear_output = np.dot(X, self.weights)
      #  return 1 / (1 + np.exp(-linear_output))
    

    def predict_proba(self, X):
        linear_output = np.dot(X, self.weights)
        return np.vectorize(self._sigmoid)(linear_output)  # Apply the sigmoid to the output
    
    """     # Use a numerically stable sigmoid function
        def stable_sigmoid(x):
        # If x is very large, return 1
            if x >= 0:
                return 1 / (1 + np.exp(-x))
        # If x is very small, use an approximation to avoid overflow
            else:
                return np.exp(x) / (1 + np.exp(x))
    
    # Apply the stable sigmoid function element-wise
        return np.vectorize(stable_sigmoid)(linear_output)
    """

    def predict(self, X):
        # Return class label: 1 if sigmoid output >= 0.5, -1 if less
        proba = self.predict_proba(X)
        return np.where(proba >= 0.5, 1, -1)  # Convert probabilities to class labels

    def score(self, X, y):
        # Predict labels for the input data
        y_pred = self.predict(X)
        # Calculate accuracy as the mean of correct predictions
        return np.mean(y_pred == y)
        
#########################################################################################