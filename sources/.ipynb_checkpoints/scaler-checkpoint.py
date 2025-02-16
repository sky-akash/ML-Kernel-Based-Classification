# A class to perform the standard scaling of the input dataframe

import pandas as pd
import numpy as np



class StandardScaling:
    def __init__(self):
        self.means = None # Mean of Columns
        self.stds = None # Std deviations of columns
        self.columns = None # Names of columns


    def fit_transform(self, df):
        # Here we first calculate mean of each column, std deviation of each column.
        # Note: we took the input to be a dataframe for convenience.
        self.means = df.mean()
        self.stds = df.std()
        self.columns = df.columns # Storing the column Names

        # Now we transform the dataframe
        # But what if we encounter division by zero ! so handle the exception
        with pd.option_context('mode.errors', 'raise'):
            try:
                scaled_df = (df - self.means) / self.stds
            except ZeroDivisionError:
                print("Error: Division by zero encountered. Check for columns with zero standard deviation.")
                raise
        return pd.DataFrame(scaled_df, columns=self.columns)