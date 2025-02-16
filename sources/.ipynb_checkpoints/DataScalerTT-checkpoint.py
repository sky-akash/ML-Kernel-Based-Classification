# This is a class for standardization and normalization on input data frame.
# Further, we handle the leakage of data between training and test dataset. 

import pandas as pd
import numpy as np

class DataScaler:
    def __init__(self, train_df, test_df):
        """
        Initializes the DataScaler with training and test DataFrames.
        Parameters:
        - train_df (pd.DataFrame): The training DataFrame.
        - test_df (pd.DataFrame): The test DataFrame.
        """
        self.train_df = train_df
        self.test_df = test_df
        self.scaled_train_df = None
        self.scaled_test_df = None
        self.standardized_train_df = None
        self.standardized_test_df = None

    def min_max_scaling(self):
        """
        Applies Min-Max Scaling to the training and test DataFrames.
        The scaling parameters are calculated based on the training data.
        """
        min_value = self.train_df.min()
        max_value = self.train_df.max()
        
        # Scale the training data
        self.scaled_train_df = (self.train_df - min_value) / (max_value - min_value)
        
        # Scale the test data using the training parameters
        self.scaled_test_df = (self.test_df - min_value) / (max_value - min_value)
        
        return self.scaled_train_df, self.scaled_test_df

    def standardization(self):
        """
        Applies Z-score Standardization to the training and test DataFrames.
        The standardization parameters are calculated based on the training data.
        """
        mean = self.train_df.mean()
        stdev = self.train_df.std()
        
        # Standardize the training data
        self.standardized_train_df = (self.train_df - mean) / stdev
        
        # Standardize the test data using the training parameters
        self.standardized_test_df = (self.test_df - mean) / stdev
        
        return self.standardized_train_df, self.standardized_test_df

    def get_scaled_data(self):
        """
        Returns the Min-Max Scaled training and test DataFrames.
        """
        if self.scaled_train_df is None or self.scaled_test_df is None:
            raise ValueError("Min-Max Scaling has not been applied yet.")
        return self.scaled_train_df, self.scaled_test_df

    def get_standardized_data(self):
        """
        Returns the Standardized training and test DataFrames.
        """
        if self.standardized_train_df is None or self.standardized_test_df is None:
            raise ValueError("Standardization has not been applied yet.")
        return self.standardized_train_df, self.standardized_test_df