import numpy as np

class MyKFold:
    """
    Custom implementation of k-fold cross-validation splitter.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    shuffle : bool, default=False
        Whether to shuffle the dataset before splitting.
    random_state : int or None, default=None
        Seed used by the random number generator for shuffling.

    Notes
    -----
    X is used only to infer the number of samples (n_samples) and does not affect
    feature-based splitting. The splitter divides the data indices into n_splits
    parts, ensuring that each fold is used as a test set once.
    """
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        # Validating the number of splits.
        if n_splits < 2:
            raise ValueError("n_splits must be ≥ 2.")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X):
        """
        Generate indices to split data into training and test sets.

        Parameters
        ----------
        X : array-like, shape (n_samples, ...)
            Input data. Only the number of samples (len(X)) is used for splitting.

        Yields
        ------
        train_indices : ndarray
            The training set indices for that split.
        test_indices : ndarray
            The testing set indices for that split.

        Raises
        ------
        ValueError
            If n_splits is greater than the number of samples.
        """
        n_samples = len(X)
        if self.n_splits > n_samples:
            raise ValueError("n_splits cannot exceed n_samples.")

        # Create an array of sample indices
        indices = np.arange(n_samples)
        
        # Shuffle the indices if required.
        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(indices)
        
        # Determine the size of each fold
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int) # numpy.full(shape, fill_value, dtype=None, order='C', *, device=None, like=None)
        fold_sizes[: n_samples % self.n_splits] += 1
        
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_idx = indices[start:stop]
            train_idx = np.concatenate([indices[:start], indices[stop:]])
            yield train_idx, test_idx
            current = stop

############################################################################################







