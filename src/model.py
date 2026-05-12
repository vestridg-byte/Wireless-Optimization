import numpy as np

def add_bias(X):
    """Add a bias column of 1s to a feature matrix."""
    return np.c_[np.ones((X.shape[0], 1)), X]

def fit_linear_regression(X, Y):
    """
    Fit linear regression using pseudo-inverse.
    X: (n,d), Y: (n,2)
    Returns W: (d+1,2)
    """
    Xb = add_bias(X)
    return np.linalg.pinv(Xb) @ Y

def predict_linear_regression(X, W):
    """Predict Y from X using learned weights W."""
    Xb = add_bias(X)
    return Xb @ W

def mean_position_error(Y_true, Y_pred):
    """Mean Euclidean distance error."""
    d = np.sqrt(np.sum((Y_true - Y_pred) ** 2, axis=1))
    return float(np.mean(d))