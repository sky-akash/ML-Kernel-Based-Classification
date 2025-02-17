import numpy as np

class KernelizedPegasosSVM:
    def __init__(self, lambda_param=0.01, T=1000, kernel='rbf', gamma=1, degree=3, batch_size=32):
        self.lambda_param = lambda_param  # Regularization parameter
        self.T = T  # Number of iterations
        self.kernel_type = kernel  # Kernel type: 'rbf' or 'poly'
        self.gamma = gamma  # Parameter for RBF kernel
        self.degree = degree  # Degree for polynomial kernel
        self.batch_size = batch_size  # Mini-batch size
        self.alpha = None  # Lagrange multipliers
        self.X_train = None  # Store training data
        self.y_train = None  # Store training labels
    
    def kernel(self, x1, x2):
        """Compute the kernel function."""
        if self.kernel_type == 'rbf':
            return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)
        elif self.kernel_type == 'poly':
            return (1 + np.dot(x1, x2)) ** self.degree
        else:
            raise ValueError("Unsupported kernel type")

    def fit(self, X, y):
        """Train the SVM using the Kernelized Pegasos algorithm."""
        m, n = X.shape
        self.alpha = np.zeros(m)  # Initialize the Lagrange multipliers
        self.X_train = X
        self.y_train = y
    
        for t in range(1, self.T + 1):
            eta = 1 / (self.lambda_param * t)  # Learning rate
            batch_indices = np.random.choice(m, self.batch_size, replace=False)  # Select random batch
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            
            # Compute the kernel matrix for the mini-batch
            K_batch = np.array([[self.kernel(x_i, x_j) for x_j in X_batch] for x_i in X_batch])
    
            for i in range(self.batch_size):
                # Select the kernel vector corresponding to the current sample in the batch
                K_i = K_batch[i]
                
                # Compute the decision function using only the alpha and y values for the selected batch
                decision_value = np.sum(self.alpha[batch_indices] * self.y_train[batch_indices] * K_i)
                
                # Update alpha[batch_indices[i]] if the sample is misclassified
                if y_batch[i] * decision_value < 1:
                    self.alpha[batch_indices[i]] += eta

    def predict(self, X):
        """Make predictions using the trained model."""
        # Compute the kernel matrix between training data and input data
        K_matrix = np.array([[self.kernel(x_train, x_test) for x_test in X] for x_train in self.X_train])
        
        # Compute the decision values for predictions
        decision_values = np.dot(K_matrix.T, self.alpha * self.y_train)
        
        # Return the predicted labels based on the sign of the decision values
        return np.sign(decision_values)

#################################