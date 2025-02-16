# A class to perform the train test split of the dataset

import pandas as pd
import numpy as np


class TrainTestSplit:
    def __init__(self, test_size=0.3, random_state=None):
        """
        Initializing the TrainTestSplit object.
        Parameters:
        test_size (float): Proportion of the dataset to include in the test split (a default is considered to be 0.3).
        random_state (int, optional): A Seed for random number generator (default is None). # Including to use different train test models that helps to check the robustness of estimates
        """
        self.test_size = test_size
        self.random_state = random_state

    def split(self, df):
        # Parameter -> dataframe
        # Output -> a tuple with train and test dataframes

        # Set random seed for reproducibility if specified
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Shuffling the DataFrame
        shuffled_indices = np.random.permutation(df.index)
        num_test_samples = int(len(df) * self.test_size)

        # Split the indices
        test_indices = shuffled_indices[:num_test_samples]
        train_indices = shuffled_indices[num_test_samples:]

        # Create train and test DataFrames
        train_df = df.loc[train_indices].reset_index(drop=True)
        test_df = df.loc[test_indices].reset_index(drop=True)

        # Split into features (X) and target (Y) | Assuming Target Variable is last one in the input dataframe
        X_train = train_df.iloc[:, :-1]  # All rows, all columns except the target variable
        Y_train = train_df.iloc[:, -1]    # All rows, for the target variable
        X_test = test_df.iloc[:, :-1]     # All rows, all columns except the target variable
        Y_test = test_df.iloc[:, -1]       # All rows, for the target variable

        return X_train, Y_train, X_test, Y_test














