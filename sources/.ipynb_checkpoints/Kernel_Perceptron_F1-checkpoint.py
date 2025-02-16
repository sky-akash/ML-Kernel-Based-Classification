import numpy as np

class KernelizedPerceptron:
    """
    Kernelized Perceptron Classifier supporting RBF (Gaussian) and Polynomial kernels.
    
    This class implements a kernelized version of the Perceptron algorithm. The kernel trick allows for
    non-linear classification by transforming the data into higher-dimensional space via a kernel function.
    The available kernels are Radial Basis Function (RBF) and Polynomial kernels.

    Parameters:
    kernel : str, optional (default="rbf")
        The kernel to use. Can be "rbf" for Gaussian Radial Basis Function kernel, or "poly" for Polynomial kernel.
    
    degree : int, optional (default=2)
        The degree of the polynomial kernel (only used when kernel="poly").
    
    gamma : float, optional (default=1.0)
        The gamma parameter for the RBF kernel. It influences the width of the Gaussian function.
    """

    def __init__(self, kernel="rbf", degree=2, gamma=1.0):
        """
        Initializes the Kernelized Perceptron with specified kernel and parameters.

        Parameters:
        kernel : str, "rbf" or "poly" (default="rbf")
        degree : int, Degree of polynomial kernel (used if kernel="poly", default=2)
        gamma : float, Kernel coefficient for RBF kernel (default=1.0)
        """
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.alpha = None  # Dual coefficients, which represent the importance of each sample
        self.support_vectors = None  # Support vectors after training (used in predictions)
        self.support_labels = None  # Corresponding labels for support vectors

    def _rbf_kernel(self, X1, X2):
        """
        Computes the Radial Basis Function (RBF) kernel between two sets of vectors.

        Parameters:
        X1 : ndarray, shape (n_samples_1, n_features)
            First input matrix of samples.

        X2 : ndarray, shape (n_samples_2, n_features)
            Second input matrix of samples.

        Returns:
        ndarray, shape (n_samples_1, n_samples_2)
            The RBF kernel matrix between the two input sets of samples.
        """
        pairwise_sq_dist = np.sum(X1**2, axis=1, keepdims=True) - 2 * np.dot(X1, X2.T) + np.sum(X2**2, axis=1) # just the distance formula, x1^2-2.X1.X2+X2^2
        return np.exp(-pairwise_sq_dist / (2 * self.gamma))  # Gaussian function

    def _poly_kernel(self, X1, X2):
        """
        Computes the Polynomial kernel between two sets of vectors.

        Parameters:
        X1 : ndarray, shape (n_samples_1, n_features)
            First input matrix of samples.

        X2 : ndarray, shape (n_samples_2, n_features)
            Second input matrix of samples.

        Returns:
        ndarray, shape (n_samples_1, n_samples_2)
            The Polynomial kernel matrix between the two input sets of samples.
        """
        return (1 + np.dot(X1, X2.T)) ** self.degree  # Polynomial function

    def _compute_kernel(self, X1, X2):
        """
        Selects and computes the kernel matrix based on the specified kernel type.

        Parameters:
        X1 : ndarray, shape (n_samples_1, n_features)
            First input matrix of samples.

        X2 : ndarray, shape (n_samples_2, n_features)
            Second input matrix of samples.

        Returns:
        ndarray, shape (n_samples_1, n_samples_2)
            The computed kernel matrix based on the selected kernel type.
        """
        if self.kernel == "rbf":
            return self._rbf_kernel(X1, X2)
        elif self.kernel == "poly":
            return self._poly_kernel(X1, X2)
        else:
            raise ValueError("Unsupported kernel. Use 'rbf' or 'poly'.")

    def fit(self, X, y, epochs=10, tolerance=1e-5):
        """
        Trains the Kernelized Perceptron classifier on the provided data.

        Parameters:
        X : ndarray, shape (n_samples, n_features)
            Input data matrix where each row represents a sample.

        y : ndarray, shape (n_samples,)
            Labels corresponding to the input data. Should be -1 or 1.

        epochs : int, optional (default=10)
            The number of training iterations (epochs) to perform.
        """
        n_samples = X.shape[0]
        self.alpha = np.zeros(n_samples)  # Dual variables (alphas), initialized to zero
        self.support_vectors = X  # All training samples are considered support vectors initially
        self.support_labels = y  # The labels for the support vectors

        # Training loop for a fixed number of epochs
        for epoch in range(epochs):
            alpha_prev = self.alpha.copy() # Storing previous alpha values for convergence check
            for i in range(n_samples):
                # Compute the kernel 
                kernel_vals = self._compute_kernel(X, X[i:i+1])
                # Compute the kernel sum Prediction score for the current sample
                kernel_sum = np.sum(self.alpha * y * kernel_vals.flatten())
                # Update alpha if the current sample is misclassified
                if np.sign(kernel_sum) != y[i]:
                    self.alpha[i] += 1  # Increase alpha if sample is misclassified
            # Check for convergence (if alpha doesn't change significantly)
            if np.all(np.abs(self.alpha - alpha_prev) < tolerance):
                print(f"Converged after {epoch + 1} epochs.")
                break

    def predict(self, X):
        """
        Predicts the class labels for a given set of input data.

        Parameters:
        X : ndarray, shape (n_samples, n_features)
            Input data to classify.

        Returns:
        ndarray, shape (n_samples,)
            Predicted class labels, either -1 or 1.
        """
        # Compute the kernel matrix between the support vectors and the input samples
        kernel_matrix = self._compute_kernel(self.support_vectors, X)
        # Compute the prediction scores for each input sample
        prediction_scores = np.sum((self.alpha[:, None] * self.support_labels[:, None]) * kernel_matrix, axis=0)
        # Return class labels based on the sign of the prediction scores
        return np.sign(prediction_scores)  # Return either -1 or 1
#################################################################################################################