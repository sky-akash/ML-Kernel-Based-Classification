# This is a class for perceptron from scratch without using sklean library.

# For creating the perceptron class, we need 
# default __init__ function with inputs/ initialization parameters for the class
# activation function, to keep consistency with the chosen activation function
# fit funtion to fit the model on training data including the update of weights and returning the final weights and bias
# predict function to make predictions on the test data using the finalized weights and bias from the fit function. 

import numpy as np

# Defining the Perceptron Class

class Perceptron:
    def __init__(self, learning_rate = 0.01, epochs = 1000, activation_func = 'sign'):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.activation_func = activation_func
        self.weights = None
        self.bias = None
    
    # Defining an activation method for selection of a proper activation function as specified in class instantiation
    def activation(self, z):
        if self.activation_func == 'sign':
            return np.sign(z)
        else:
            raise ValueError("Invalid activation function. Select 'sign' as activation function.")

    def fit(self, X, y): # Defining the Fit function for the perceptron
        n_samples, n_features = X.shape
        
        # Initialize Parameters i.e. weights and bias
        self.weights = np.zeros(n_features) # Weights is a vector for all the inputs
        self.bias = 0 # A scalar as it is same for one Perceptron unit
        
        y_ = np.array(y) # just a copy of my data (in case we require a modification later on)

        # Learning the Weights -> run a loop and iterate for number of epochs
        for i in range(self.epochs):
            # Run a loop to traverse the whole training set
            for idx, x_i in enumerate(X):
                
                # Linear Output, z
                z = np.dot(x_i, self.weights) + self.bias 
                
                # Prediction using activation function
                y_predicted = self.activation(z)

                # Checking Misclassifications
                if y_predicted != y_[idx]:
                    # Perceptron Update for learning rate and weights and bias
                    update = self.learning_rate * (y_[idx] - y_predicted)
                    self.weights += update * x_i
                    self.bias += update
                
        return self.weights, self.bias # Returning the final weights and bias

    def predict(self,X):
        """Making predictions using the trained Perceptron model."""
        z = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation(z)
        return y_predicted

    def accuracy(self, X, y):
        """Return accuracy on the given dataset."""
        return np.mean(self.predict(X) == y)
#####################################################################################