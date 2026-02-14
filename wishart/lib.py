import math
from typing import Any

import numpy as np
import scipy


def scm(data: np.ndarray) -> np.ndarray:
    """Sample Covariance Matrix."""
    return data @ np.linalg.matrix_transpose(data) / data.shape[-1]


def marchenko_pastur(c: float, x: np.ndarray) -> np.ndarray:
    """Distribution of the eigen values of a white Wishart matrix of size p.

    The singularity at 0 for n < p is ignored.
    """
    lower, upper = marchenko_pastur_bounds(c)
    norm = 2 * math.pi * c
    return np.sqrt(np.maximum((upper - x) * (x - lower), 0.0)) / x / norm


def marchenko_pastur_bounds(c: float) -> tuple[float, float]:
    """Lower and upper bounds of the Marchenko-Pastur distribution."""
    return ((1 - math.sqrt(c)) ** 2, (1 + math.sqrt(c)) ** 2)


def limit_stieltjes(c: float, z: np.ndarray) -> np.ndarray:
    lower, upper = ((1 - math.sqrt(c)) ** 2, (1 + math.sqrt(c)) ** 2)
    num = 1 - c - z + np.emath.sqrt(z - lower) * np.emath.sqrt(z - upper)
    denom = 2 * c * z
    return num / denom


def inverse_limit_co_stieltjes(c: float, z: np.ndarray) -> np.ndarray:
    """Functional inverse of the Stieltjes transform of the Gram Matrix."""
    return -1.0 / z + c / (1.0 + z)


def inverse_limit_stieltjes(c: float, z: np.ndarray) -> np.ndarray:
    """Functional inverse of the Stieltjes transform of the Sample Covariance Matrix
    (~blue transform).
    """
    return -1.0 / z + 1.0 / (1.0 + c * z)


def gamma(c: float, z: np.ndarray) -> np.ndarray:
    """Related to the Stieltjes of the Gram Matrix by gamma = - 1 / stieljes"""
    lower, upper = ((1 - math.sqrt(c)) ** 2, (1 + math.sqrt(c)) ** 2)
    num = 2 * z
    denom = 1 - c + z - np.emath.sqrt(z - lower) * np.emath.sqrt(z - upper)
    return num / denom


def tracy_widom1(x: Any) -> Any:
    """Probability distribution of the Tracy Widom distribution (beta=1)."""
    # From https://www.mathworks.com/matlabcentral/fileexchange/44711-approximation-for-the-tracy-widom-laws
    k = 46.44604884387787
    theta = 0.18605402228279347
    alpha = 9.848007781128567
    return _pdf_gamma(np.array(x) + alpha, theta, k)


def _pdf_gamma(x: Any, theta: float, k: float) -> Any:
    return np.where(
        x > 0,
        1 / (scipy.special.gamma(k) * theta**k) * x ** (k - 1) * np.exp(-x / theta),
        0.0,
    )
