# A class to perform the standard scaling of the input dataframe & 
# to perform the train test split of the dataset

# Class Update to Use alpha-Winsorized Mean for Robustness to Outliers

import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize

class StandardScaling:
    def __init__(self):
        self.means = None # Mean of Columns
        self.stds = None # Std deviations of columns
        self.columns = None # Names of columns

    def fit(self, X):
        # Here I first calculate mean of each column, std deviation of each column.
        # Note: I took the input to be a dataframe for convenience.
        self.means = X.mean()
        self.stds = X.std()
        self.columns = X.columns # Storing the column Names

    def transform(self, X):
        # Transform the new data using the previously computed means and stds
        if self.means is None or self.stds is None:
            raise ValueError("Fit method must be called before transform.")
        return pd.DataFrame((X - self.means) / self.stds, columns=self.columns)

    def fit_transform(self, X):
        # Now I transform the dataframe
        # But what if I encounter division by zero ! so handle the exception
        self.fit(X)
        return self.transform(X)

class TrainTestSplit:
    def __init__(self, test_size=0.3, random_state=None, target_column=None, stratify=False):
        """
        TrainTestSplit: Splits dataset into train-test sets with optional stratification.
        
        Parameters:
        - test_size (float): Proportion of dataset for testing (default = 0.3).
        - random_state (int, optional): Seed for reproducibility.
        - target_column (str, optional): Target column name (default: last column).
        - stratify (bool): If True, maintains class proportions in train-test split.
        """
        self.test_size = test_size
        self.random_state = random_state
        self.target_column = target_column
        self.stratify = stratify

    def split(self, df):
        """
        Splits the dataset into train-test sets while maintaining class distribution if needed.

        Parameters:
        - df (pd.DataFrame): Dataset containing features and target variable.

        Returns:
        - X_train, y_train, X_test, y_test: Train-test split data.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Determine target column
        if self.target_column is None:
            self.target_column = df.columns[-1]  # Default: last column

        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in the dataset.")

        X = df.drop(columns=[self.target_column])  # Features
        y = df[self.target_column]  # Target variable

        if self.stratify:
            # Stratified Sampling: Maintain class proportions
            train_indices = []
            test_indices = []

            for _, group in df.groupby(self.target_column):
                num_test_samples = int(len(group) * self.test_size)
                shuffled_indices = group.index.to_numpy()
                np.random.shuffle(shuffled_indices)

                test_indices.extend(shuffled_indices[:num_test_samples])
                train_indices.extend(shuffled_indices[num_test_samples:])
        else:
            # Random Split without Stratification
            shuffled_indices = np.random.permutation(df.index)
            num_test_samples = int(len(df) * self.test_size)
            test_indices = shuffled_indices[:num_test_samples]
            train_indices = shuffled_indices[num_test_samples:]

        # Create train and test DataFrames
        train_df = df.loc[train_indices].reset_index(drop=True)
        test_df = df.loc[test_indices].reset_index(drop=True)

        # Split into features (X) and target (y)
        X_train, y_train = train_df.drop(columns=[self.target_column]), train_df[self.target_column]
        X_test, y_test = test_df.drop(columns=[self.target_column]), test_df[self.target_column]

        return X_train, y_train, X_test, y_test

# also implementing the 0-1 loss function here
class ZeroOneLoss:
    def __init__(self):
        # class initialization
        pass
    
    def compute_loss(self, y_true, y_pred):
        """Return Loss on the given dataset."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        loss = (np.sum(y_true != y_pred)) / len(y_true)

        return loss

     
# Below code was for without any stratification        
"""        # Parameter -> dataframe
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

        # Determine target column
        if self.target_column is None:
            self.target_column = df.columns[-1]  # Default: last column

        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in the dataset.")

        # Split into features (X) and target (Y) 
        
        X_train = train_df.drop(columns=[self.target_column])  # All rows, all columns except the target variable
        Y_train = train_df[self.target_column]    # All rows, for the target variable
        X_test = test_df.drop(columns=[self.target_column])     # All rows, all columns except the target variable
        Y_test = test_df[self.target_column]       # All rows, for the target variable

        return X_train, Y_train, X_test, Y_test

"""







