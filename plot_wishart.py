from pathlib import Path

import plotly.express as px


def read_eigen_values(filename: str) -> tuple[int, int, tuple[float, ...]]:
    eigen_values: list[float] = []
    with open(Path("results") / filename, mode="r", encoding="utf8") as f:
        n = int(f.readline())
        p = int(f.readline())
        for str_value in f.readlines():
            eigen_values.append(float(str_value))
    return n, p, tuple(eigen_values)


n, p, eigen_values = read_eigen_values("wishart_n_p_eigs.txt")
fig = px.histogram(eigen_values)
fig.show()
