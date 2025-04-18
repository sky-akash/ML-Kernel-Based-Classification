# This is a class for perceptron from scratch without using sklean library.

# For creating the perceptron class, we need 
# default __init__ function with inputs/ initialization parameters for the class
# activation function, to keep consistency with the chosen activation function
# fit funtion to fit the model on training data including the update of weights and returning the final weights and bias
# predict function to make predictions on the test data using the finalized weights and bias from the fit function. 

import numpy as np 

# Defining the Perceptron Class

class Perceptron:
    def __init__(self, epochs = 100, activation_func = 'sign'): # Default epochs = 1000
        #self.learning_rate = learning_rate
        self.epochs = epochs
        self.activation_func = activation_func
        self.weights = None
        self.bias = None
    
    # Defining an activation method for selection of a proper activation function as specified in class instantiation
    def activation(self, z):
        if self.activation_func == 'sign':
            return np.where(z >= 0, 1, -1)
        else:
            raise ValueError("Invalid activation function. Select 'sign' as activation function.")

    def fit(self, X, y): # Defining the Fit function for the perceptron
        n_samples, n_features = X.shape

        # Initializing Parameters (weights and bias)
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.array(y) #(just a copy of my data)
        self.boundary_lines = []

        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")

            # Shuffleing the data for random check on misclassification (with replacement or not, up to you)
            random_indices = np.random.choice(X.shape[0], size=X.shape[0], replace=True)

            for idx in random_indices:
                x_i = X[idx]
                y_true = y_[idx]

                z = np.dot(x_i, self.weights) + self.bias
                y_pred = self.activation(z)

                if y_pred != y_true:
                    # Classic perceptron update ( I removed the learning rate here )
                    self.weights += y_true * x_i
                    self.bias += y_true

        return self.weights, self.bias

    def predict(self,X):
        """Making predictions using the trained Perceptron model."""
        z = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation(z)
        return y_predicted

    def accuracy(self, X, y):
        """Return accuracy on the given dataset."""
        return np.mean(self.predict(X) == y)
#####################################################################################