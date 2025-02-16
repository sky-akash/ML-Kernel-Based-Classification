from itertools import product
from K_fold_CV import MyKFold # Custom Kfold CV
import numpy as np


def cross_validate(
    model_class,
    param_grid,
    X,
    y,
    k=5,
    shuffle=True,
    random_state=42,
    scoring=None,
    verbose=True,
):
    """
    Perform k-fold cross-validation for hyperparameter tuning.

    Args:
        model_class: Class of the model to train.
        param_grid: Dictionary of hyperparameters to tune (values must be iterables).
        X, y: Dataset (arrays or DataFrames).
        k: Number of folds (default=5).
        shuffle: Whether to shuffle data before splitting.
        random_state: Seed for reproducibility.
        scoring: Scoring function (default: accuracy for classification, negative MSE for regression).
        verbose: To print progress (keeping it default=True, so I can check the parameters and also use for debug).

    Returns:
        best_params: Dictionary containing the hyperparameters with the best mean cross-validation score.
    """
    # Input validation
    X = np.asarray(X) # futher input X is also np array, just for precautions
    y = np.asarray(y) # futher input X is also np array, just for precautions
    n_samples = len(X) 
    
    # Checks on parameter grid values and value of K.
    if len(X) != len(y): 
        raise ValueError("X and y must have the same length.")
    if k > n_samples:
        raise ValueError(f"k={k} cannot exceed the number of samples ({n_samples}).")
    if k < 2:
        raise ValueError("k must be ≥ 2.")
    for key, value in param_grid.items():
        if not isinstance(value, (list, np.ndarray)):
            raise ValueError(f"param_grid value for '{key}' must be iterable.")

    # Default scoring based on task our focus is on classification
    scoring = lambda y_true, y_pred: np.mean(y_true == y_pred)
    
    """
    if scoring is None:
        # If target type is integer, assume classification; otherwise, regression.
        if np.issubdtype(y.dtype, np.integer):
            scoring = lambda y_true, y_pred: np.mean(y_true == y_pred)
        else:
            scoring = lambda y_true, y_pred: -np.mean((y_true - y_pred) ** 2)
    """
    
    # Initialize best score tracking
    best_params = None
    best_score = -np.inf  # Assumes higher score is better so initialize with -infinity
    best_cv = np.inf  # Highest Coefficient of Variation
    history = [] # History of hyperparameters, so we can keep a track and plot the variables later

    # Generate parameter combinations from the parameter grid.
    param_combinations = list(product(*param_grid.values()))
    
    # Creating instance of KFold splitter
    kf = MyKFold(n_splits=k, shuffle=shuffle, random_state=random_state)

    for params in param_combinations:
        param_dict = dict(zip(param_grid.keys(), params))
        if verbose:
            print(f"Testing Hyperparameters: {param_dict}")

        scores = []
        try:
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # Edge Case: Skip empty validation sets.
                if len(y_val) == 0:
                    raise ValueError("Validation set is empty.")

                # Create and train the model.
                model = model_class(**param_dict) # **param_dict to unpack the dictionary of grid search parameters for the models
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                scores.append(scoring(y_val, y_pred))
            
            # Mean Scores
            mean_score = np.mean(scores)

            # Standard Deviation of Scores
            std_dev = np.std(scores) # calculating the standard deviation for the scores

            # Coefficient of Varaition
            cv = std_dev / mean_score if mean_score != 0 else np.inf  # Avoid division by zero

            if verbose:
                print(f"Mean Score: {mean_score:.4f}, Std Dev: {std_dev:.4f}, CV: {cv:.4f}")

            # Store the result in the history list
            history.append({
                'params': param_dict,
                'scores': scores,
                'mean_score': mean_score,
                'std_dev' : std_dev,
                'cv' : cv
            })
            
            if mean_score > best_score or (mean_score == best_score and cv < best_cv):
                best_score = mean_score
                best_cv = cv
                best_params = param_dict

        except Exception as e:
            if verbose:
                print(f"Error with {param_dict}: {str(e)}")
            continue

    print(f"\nBest Parameters: {best_params} with Score: {best_score:.4f}")
    return best_params, best_score, best_cv, history