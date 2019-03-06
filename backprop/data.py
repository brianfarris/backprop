import numpy as np
from sklearn.model_selection import train_test_split


def get_train_test():
    N = int(1e5)
    x_in = np.random.normal(0, 1, (N//2, 2))
    r = np.random.normal(6, 0.5, (N//2, 1))
    phi = np.random.uniform(0, 2. * np.pi, (N//2, 1))
    x_out = np.hstack([r * np.cos(phi), r * np.sin(phi)])
    x_full = np.vstack([x_in, x_out])
    y_in = np.vstack([np.zeros((N//2, 1)), np.ones((N//2, 1))])
    y_out = np.vstack([np.ones((N//2, 1)), np.zeros((N//2, 1))])
    y_full = np.hstack([y_in, y_out])
    (x_train, x_test,
     y_train, y_test) = train_test_split(x_full, y_full,
                                         test_size=0.1)
    return x_train, x_test, y_train, y_test


def chunks(x, y, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, x.shape[0], n):
        yield x[i:i + n], y[i:i + n]
