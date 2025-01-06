import math

import numpy as np


def white_wishart_matrix(n: int, p: int) -> np.ndarray:
    """Return a white Wishart matrix of size p."""
    x = np.random.randn(p, n)
    xt = np.transpose(x)
    return np.matmul(x, xt) / n


def marchenko_pastur(n: int, p: int, x: np.ndarray) -> np.ndarray:
    """Distribution of the eigen values of a white Wishart matrix of size p.

    The singularity at 0 for n < p is ignored.
    """
    c = float(p) / float(n)
    lower, upper = marchenko_pastur_bounds(n, p)
    norm = 2 * math.pi * c
    return np.divide(
        np.divide(np.sqrt(np.maximum((upper - x) * (x - lower), 0.0)), x), norm
    )


def marchenko_pastur_bounds(n: int, p: int) -> tuple[float, float]:
    """Lower and upper bounds of the Marchenko-Pastur distribution."""
    c = float(p) / float(n)
    return ((1 - math.sqrt(c)) ** 2, (1 + math.sqrt(c)) ** 2)
