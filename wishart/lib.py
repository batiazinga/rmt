import math

import numpy as np


def white_wishart_matrix(n: int, p: int) -> np.ndarray:
    """Return a white Wishart matrix of size p.

    Or Sample Covariance Matrix of the variables."""
    x = np.random.randn(p, n)
    xt = np.transpose(x)
    return np.matmul(x, xt) / n


def marchenko_pastur(c: float, x: np.ndarray) -> np.ndarray:
    """Distribution of the eigen values of a white Wishart matrix of size p.

    The singularity at 0 for n < p is ignored.
    """
    lower, upper = marchenko_pastur_bounds(c)
    norm = 2 * math.pi * c
    return np.divide(
        np.divide(np.sqrt(np.maximum((upper - x) * (x - lower), 0.0)), x), norm
    )


def marchenko_pastur_bounds(c: float) -> tuple[float, float]:
    """Lower and upper bounds of the Marchenko-Pastur distribution."""
    return ((1 - math.sqrt(c)) ** 2, (1 + math.sqrt(c)) ** 2)


def limit_stieltjes(c: float, z: np.ndarray) -> np.ndarray:
    lower, upper = ((1 - math.sqrt(c)) ** 2, (1 + math.sqrt(c)) ** 2)
    num = 1 - c - z + np.emath.sqrt(z - lower) * np.emath.sqrt(z - upper)
    denom = 2 * c * z
    return np.divide(num, denom)


def gamma(c: float, z: np.ndarray) -> np.ndarray:
    """Related to the Stieltjes of the Sample Covariance Matrix of the observations by:
    gamma = - 1 / obs_stieljes
    """
    lower, upper = ((1 - math.sqrt(c)) ** 2, (1 + math.sqrt(c)) ** 2)
    num = 2 * z
    denom = 1 - c + z - np.emath.sqrt(z - lower) * np.emath.sqrt(z - upper)
    return np.divide(num, denom)
