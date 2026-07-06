import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    n_samples, n_features = X.shape
    
    w = np.zeros(n_features)
    b=0.0
    for step in range (0, steps):
        z = np.dot(X,w) + b
        p = _sigmoid(z)
        grad_w= np.dot(X.T, (p-y)) / n_samples
        grad_b = np.mean(p-y)

        w = w - lr*grad_w
        b = b - lr*grad_b
    return w, b
    pass