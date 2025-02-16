import numpy as np

class KernelizedPerceptron:
    def __init__(self, kernel='gaussian', degree=2, gamma=None, epochs=100):
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma if gamma is not None else 1.0
        self.epochs = epochs
        self.alpha = None
        self.support_vectors = []
        self.support_labels = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        self.alpha = np.zeros(n_samples)

        # Kernelized perceptron training
        for epoch in range(self.epochs):
            for i in range(n_samples):
                pred = self.predict_single(X, i)
                if y[i] * pred <= 0:
                    self.alpha[i] += 1  # Update alpha
                    # Store support vectors and their corresponding labels
                    self.support_vectors.append(X[i])
                    self.support_labels.append(y[i])

        # Convert lists to numpy arrays after training
        self.support_vectors = np.array(self.support_vectors)
        self.support_labels = np.array(self.support_labels)

    def predict(self, X):
        predictions = []
        for i in range(X.shape[0]):
            pred = self.predict_single(X, i)
            predictions.append(pred)
        return np.array(predictions)

    def predict_single(self, X, i):
        result = 0.0
        for j in range(len(self.support_vectors)):
            if self.alpha[j] > 0:
                # Ensure we're indexing into the correct vectors
                if j < len(self.support_labels):  # Debugging check
                    result += self.alpha[j] * self.support_labels[j] * self.kernel_function(X[i], self.support_vectors[j])
                else:
                    print(f"Index out of bounds for j: {j}, len(support_labels): {len(self.support_labels)}")
        return np.sign(result)

    def kernel_function(self, x1, x2):
        if self.kernel == 'gaussian':
            return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)
        elif self.kernel == 'polynomial':
            return (np.dot(x1, x2) + 1) ** self.degree
        else:
            raise ValueError("Unsupported kernel type.")

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)