# Class to generate Polynomial Features:

import numpy as np

class PolynomialFeatureExpansion:
    def __init__(self, degree=2):
        """
        Class for manually generating polynomial features up to a given degree.
        Default degree = 2 (quadratic expansion with interaction terms).
        """
        self.degree = degree

    def fit_transform(self, X):
        """
        Expands the input feature matrix X with polynomial features up to the given degree.

        Parameters:
        X : np.array of shape (n_samples, n_features)
            Input dataset with numerical features.

        Returns:
        np.array : Expanded feature matrix with polynomial terms.
        """
        n_samples, n_features = X.shape
        poly_features = [X]  # Original features , creates the list of items in X
        
        for d in range(2, self.degree + 1):  # Generating terms up to the specified degree
            # Squared and higher-order terms
            for i in range(n_features):
                poly_features.append(X[:, i:i+1] ** d)

            # Interaction terms for each degree level
            for i in range(n_features):
                for j in range(i+1, n_features):  # Avoiding duplicate pairs
                    interaction_term = (X[:, i:i+1] * X[:, j:j+1]) ** (d - 1)
                    poly_features.append(interaction_term)

        # Concatenate all features to form the expanded matrix along with the initial features :)
        return np.hstack(poly_features)

####################################################################################################