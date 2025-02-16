# This is a class for standardization and normalization on input data frame.
# This class doesn't handle train test split. 

class DataStanNorm:
    # Class Initialization
    def __init__(self, df):
        self.df = df
        self.scaled_df = None
        self.standardized_df = None

    def min_max_scaling(self):
        min_value = self.df.min()
        max_value = self.df.max()
        self.scaled_df = (self.df - min_value) / (max_value - min_value)
        return self.scaled_df
    
    def standardization(self):
        mean = self.df.mean()
        stdev = self.df.std()
        self.standardized_df = (self.df - mean) / stdev
        return self.standardized_df
    
    def get_scaled_data(self):
        """
        Returns the Min-Max Scaled DataFrame.
        """
        if self.scaled_df is None:
            raise ValueError("Min-Max Scaling has not been applied yet.")
        return self.scaled_df
    
    def get_standardized_data(self):
        """
        Returns the Standardized DataFrame.
        """
        if self.standardized_df is None:
            raise ValueError("Standardization has not been applied yet.")
        return self.standardized_df











 # Function for Min-Max Scaling
    def min_max_scaling(self, df, min_value=None, max_value=None):
        if min_value is None or max_value is None:
            min_value = df.min()
            max_value = df.max()
        df_scaled = (df - min_value) / (max_value - min_value)
        return df_scaled

    # Function for Standardization
    def standardization(self, df, mean=None, stdev=None):
        if mean is None or stdev is None:
            mean = df.mean()
            stdev = df.std()
        df_standardized = (df - mean) / stdev
        return df_standardized
    
    # Applying the scaling and normalization
    def scale_normalize(self):
        """
        Applies scaling and normalization to both training and test datasets.
        """
        # The required scaling parameters are as follows:
        min_value = self.train_df.min()
        max_value = self.train_df.max()
        mean = self.train_df.mean()
        stdev = self.train_df.std()

        # Scaling and normalizing training data
        self.train_df = self.min_max_scaling(self.train_df, min_value, max_value)
        self.train_df = self.standardization(self.train_df, mean, stdev)

        # Scaling and normalizing test data (using the training parameters)
        self.test_df = self.min_max_scaling(self.test_df, min_value, max_value)
        self.test_df = self.standardization(self.test_df, mean, stdev)

