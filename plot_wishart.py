import math
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


def read_eigen_values(filename: str) -> tuple[int, int, tuple[float, ...]]:
    eigen_values: list[float] = []
    with open(Path("results") / filename, mode="r", encoding="utf8") as f:
        n = int(f.readline())
        p = int(f.readline())
        for str_value in f.readlines():
            eigen_values.append(float(str_value))
    return n, p, tuple(eigen_values)


def marchenko_pastur(n: int, p: int, x: np.ndarray) -> np.ndarray:
    c = float(p) / float(n)
    lambda_p = (1 + math.sqrt(c)) ** 2
    lambda_m = (1 - math.sqrt(c)) ** 2
    norm = 2 * math.pi * c
    return np.divide(
        np.divide(np.sqrt(np.maximum((lambda_p - x) * (x - lambda_m), 0.0)), x), norm
    )


fig = go.Figure()

n, p, eigen_values = read_eigen_values("wishart_n_p_eigs.txt")
empirical_density = np.histogram(eigen_values, bins=30)
empirical_delta = empirical_density[1][1] - empirical_density[1][0]
fig.add_trace(
    go.Bar(x=empirical_density[1], y=empirical_density[0] / float(p) / empirical_delta)
)

x = np.arange(0.01, 1.6, 0.01)
limit_density = marchenko_pastur(n, p, x)
fig.add_trace(
    go.Scatter(
        {
            "x": x,
            "y": limit_density,
        },
    )
)

fig.show()
