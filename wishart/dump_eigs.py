from collections.abc import Iterable
from pathlib import Path

import numpy as np

from wishart.lib import white_wishart_matrix

np.random.seed(42)


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


eigen_values = np.linalg.eigvalsh(white_wishart_matrix(n, p))
dump_results(n, p, eigen_values)
