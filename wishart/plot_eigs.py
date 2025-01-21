import numpy as np
import plotly.graph_objects as go  # type: ignore

from wishart.lib import marchenko_pastur, scm_matrix

np.random.seed(42)


def white_wishart_matrix_eigenvalues(n: int, p: int) -> np.ndarray:
    x = np.random.randn(p, n)
    return np.linalg.eigvalsh(scm_matrix(x))


# parameters
p = 500  # dimension
n = 50000  # sample size


fig = go.Figure()

eigen_values = white_wishart_matrix_eigenvalues(n, p)
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
