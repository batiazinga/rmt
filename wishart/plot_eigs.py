from pathlib import Path

import numpy as np
import plotly.graph_objects as go  # type: ignore

from wishart.lib import marchenko_pastur


def read_eigen_values(filename: str) -> tuple[int, int, tuple[float, ...]]:
    eigen_values: list[float] = []
    with open(Path("results") / filename, mode="r", encoding="utf8") as f:
        n = int(f.readline())
        p = int(f.readline())
        for str_value in f.readlines():
            eigen_values.append(float(str_value))
    return n, p, tuple(eigen_values)


fig = go.Figure()

n, p, eigen_values = read_eigen_values("wishart_n_p_eigs.txt")
empirical_density = np.histogram(eigen_values, bins=30)
empirical_delta = empirical_density[1][1] - empirical_density[1][0]
fig.add_trace(
    go.Bar(x=empirical_density[1], y=empirical_density[0] / float(p) / empirical_delta)
)

c = float(p) / float(n)
x = np.arange(0.005, 1.6, 0.005)
limit_density = marchenko_pastur(c, x)
fig.add_trace(
    go.Scatter(
        {
            "x": x,
            "y": limit_density,
        },
    )
)

fig.show()
