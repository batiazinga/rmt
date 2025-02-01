import numpy as np
import plotly.graph_objects as go  # type: ignore

from wishart.lib import marchenko_pastur, scm

np.random.seed(42)


def white_wishart_matrix_eigenvalues(n: int, p: int) -> np.ndarray:
    x = np.random.randn(p, n)
    return np.linalg.eigvalsh(scm(x))


# parameters
p = 500  # dimension
n = 50000  # sample size


fig = go.Figure(
    layout=go.Layout(
        title=go.layout.Title(
            text=f"Eigenvalues of a white Wishart matrix (c = {p/n})",
            subtitle=go.layout.title.Subtitle(
                text=(
                    "Histogram of the empirical eigenvalues "
                    "versus limit distribution"
                )
            ),
        )
    )
)

eigen_values = white_wishart_matrix_eigenvalues(n, p)
empirical_density = np.histogram(eigen_values, bins=30)
empirical_delta = empirical_density[1][1] - empirical_density[1][0]
fig.add_trace(
    go.Bar(
        x=empirical_density[1],
        y=empirical_density[0] / float(p) / empirical_delta,
        name="Empirical distribution",
    )
)

c = float(p) / float(n)
x = np.arange(0.005, 1.6, 0.005)
limit_density = marchenko_pastur(c, x)
fig.add_trace(
    go.Scatter(
        x=x,
        y=limit_density,
        line=go.scatter.Line(color="black", dash="dash"),
        name="Limit distribution",
    )
)

fig.show()
