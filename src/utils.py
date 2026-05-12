import numpy as np

def split_indices(n, test_size=0.2, seed=695):
    """Return train_idx, test_idx."""
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(n * test_size)
    return idx[n_test:], idx[:n_test]

def standardize_fit(X):
    """Fit mean/std on training data only."""
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0
    return mu, sigma

def standardize_apply(X, mu, sigma):
    """Apply standardization using training mu/sigma."""
    return (X - mu) / sigma