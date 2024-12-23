from collections.abc import Iterable
from pathlib import Path

import numpy as np

np.random.seed(42)


def wishart_eigen_values(n: int, p: int) -> np.ndarray:
    """Return the eigen values of a white Wishart matrix in ascending order."""
    x = np.random.randn(p, n)
    xt = np.transpose(x)
    wishart = np.matmul(x, xt) / n
    return np.linalg.eigvalsh(wishart)


def dump_results(n: int, p: int, eigen_values: Iterable[float]) -> None:
    results_folder = Path("results")
    results_folder.mkdir(exist_ok=True)
    results_file = results_folder / "wishart_n_p_eigs.txt"
    with open(results_file, mode="w", encoding="utf8") as f:
        f.write(f"{n}\n")
        f.write(f"{p}\n")
        for eig_value in eigen_values:
            f.write(f"{eig_value}\n")


# parameters
p: int = 500  # dimension
n: int = 50000  # number of samples


eigen_values = wishart_eigen_values(n, p)
dump_results(n, p, eigen_values)
