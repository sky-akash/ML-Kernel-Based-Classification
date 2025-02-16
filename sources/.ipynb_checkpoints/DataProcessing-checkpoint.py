# the class performs data splitting as per given ratio for training set.

import pandas as pd
import numpy as np

class DataSplitter:
    # Initializing the Class
    def __init__(self, df, train_ratio=0.8):
        """
        Initializes the DataSplitter with a DataFrame and a train-test split ratio.
        Parameters:
        - df (pd.DataFrame): The DataFrame to be split.
        - train_ratio (float): The proportion of the data to be used for training (default is selected as 0.8).
        """
        if not (0 < train_ratio < 1):
            raise ValueError("train_ratio must be between 0 and 1")
        self.df = df
        self.train_ratio = train_ratio
        self.train_df = None
        self.test_df = None

    # Function for splitting data into training and test data
    def split_data(self):
        """
        Splits the DataFrame into training and test sets based on the specified train_ratio.
        """
        # Shuffling the DataFrame for Randomization
        df_shuffled = self.df.sample(frac=1).reset_index(drop=True)

        # Calculate the index for splitting
        train_size = int(len(df_shuffled) * self.train_ratio)

        # Split the DataFrame
        self.train_df = df_shuffled[:train_size]
        self.test_df = df_shuffled[train_size:]

    # Function for returning the training and test data
    def get_train_test_data(self):
        """
        Returns the training and test sets as separate DataFrames.
        Returns:
        - train_df (pd.DataFrame): The training set.
        - test_df (pd.DataFrame): The test set.
        """
        return self.train_df, self.test_df