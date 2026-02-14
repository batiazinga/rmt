from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import scipy

if TYPE_CHECKING:
    import numpy.typing as npt


def scm(data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Sample Covariance Matrix."""
    return data @ np.linalg.matrix_transpose(data) / data.shape[-1]


def marchenko_pastur(c: float, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Distribution of the eigen values of a white Wishart matrix of size p.

    The singularity at 0 for n < p is ignored.
    """
    lower, upper = marchenko_pastur_bounds(c)
    norm = 2 * math.pi * c
    return np.sqrt(np.maximum((upper - x) * (x - lower), 0.0)) / x / norm


def marchenko_pastur_bounds(c: float) -> tuple[float, float]:
    """Lower and upper bounds of the Marchenko-Pastur distribution."""
    return ((1 - math.sqrt(c)) ** 2, (1 + math.sqrt(c)) ** 2)


def limit_stieltjes(c: float, z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    lower, upper = ((1 - math.sqrt(c)) ** 2, (1 + math.sqrt(c)) ** 2)
    num = 1 - c - z + np.emath.sqrt(z - lower) * np.emath.sqrt(z - upper)
    denom = 2 * c * z
    return num / denom


def inverse_limit_co_stieltjes(
    c: float, z: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Functional inverse of the Stieltjes transform of the Gram Matrix."""
    return -1.0 / z + c / (1.0 + z)


def inverse_limit_stieltjes(
    c: float, z: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Functional inverse of the Stieltjes transform of the Sample Covariance Matrix
    (~blue transform).
    """
    return -1.0 / z + 1.0 / (1.0 + c * z)


def gamma(c: float, z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Related to the Stieltjes of the Gram Matrix by gamma = - 1 / stieljes"""
    lower, upper = ((1 - math.sqrt(c)) ** 2, (1 + math.sqrt(c)) ** 2)
    num = 2 * z
    denom = 1 - c + z - np.emath.sqrt(z - lower) * np.emath.sqrt(z - upper)
    return num / denom


def tracy_widom1(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Probability distribution of the Tracy Widom distribution (beta=1)."""
    # From https://www.mathworks.com/matlabcentral/fileexchange/44711-approximation-for-the-tracy-widom-laws
    k = 46.44604884387787
    theta = 0.18605402228279347
    alpha = 9.848007781128567
    return _pdf_gamma(np.array(x) + alpha, theta, k)


def _pdf_gamma(
    x: npt.NDArray[np.float64], theta: float, k: float
) -> npt.NDArray[np.float64]:
    return np.where(
        x > 0,
        1 / (scipy.special.gamma(k) * theta**k) * x ** (k - 1) * np.exp(-x / theta),
        0.0,
    )
