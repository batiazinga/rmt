import numpy as np
import plotly.graph_objects as go  # type: ignore
from plotly.subplots import make_subplots

from wishart.lib import scm

np.random.seed(42)


def co_stieltjes(
    x: np.ndarray, eigen_values: np.ndarray, dim: int, sample_size: int
) -> np.ndarray:
    y = np.zeros((len(x)))
    y -= (sample_size - dim) * np.divide(1.0, x)
    for eig in eigen_values:
        y += np.divide(1.0, eig - x)
    y /= sample_size
    return y


def co_stieltjes_zeros(eigen_values: np.ndarray, sample_size: int) -> np.ndarray:
    sqrt_eigen_vec = np.array([np.sqrt(eigen_values)])
    l = np.diag(eigen_values)
    stieltjes_zeros_matrix = l - sqrt_eigen_vec.T.dot(sqrt_eigen_vec) / n
    return np.linalg.eigvalsh(stieltjes_zeros_matrix)


def stieltjes(x: np.ndarray, eigen_values: np.ndarray, dim: int) -> np.ndarray:
    y = np.zeros((len(x)))
    for eig in eigen_values:
        y += np.divide(1.0, eig - x)
    y /= dim
    return y


def stieltjes_zeros(eigen_values: np.ndarray, dim: int) -> np.ndarray:
    sqrt_eigen_vec = np.array([np.sqrt(eigen_values)])
    l = np.diag(eigen_values)
    stieltjes_zeros_matrix = l - sqrt_eigen_vec.T.dot(sqrt_eigen_vec) / dim
    return np.linalg.eigvalsh(stieltjes_zeros_matrix)


# parameters
p = 5  # dimension
n = 50  # sample size

# generate iid gaussian data
data = np.random.randn(p, n)

# eigen values of the Wishart matrix
eigen_values = np.linalg.eigvalsh(scm(data))

# zeros of the empirical Stieltjes transform of the co Wishart matrix
zeros = stieltjes_zeros(eigen_values, p)
co_zeros = co_stieltjes_zeros(eigen_values, n)


xs = [
    np.arange(0.8 * eigen_values[0], eigen_values[0] - 0.005, 0.0005),
    np.arange(eigen_values[0] + 0.005, eigen_values[1] - 0.005, 0.0005),
    np.arange(eigen_values[1] + 0.005, eigen_values[2] - 0.005, 0.0005),
    np.arange(eigen_values[2] + 0.005, eigen_values[3] - 0.005, 0.0005),
    np.arange(eigen_values[3] + 0.005, eigen_values[4] - 0.005, 0.0005),
    np.arange(eigen_values[4] + 0.005, 1.2 * eigen_values[4], 0.0005),
]

fig = make_subplots(
    rows=2,
    cols=1,
)
for x in xs:
    fig.add_trace(
        go.Scatter(
            x=x,
            y=stieltjes(x, eigen_values, p),
            line=go.scatter.Line(color="black"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=co_stieltjes(x, eigen_values, p, n),
            line=go.scatter.Line(color="red"),
        ),
        row=2,
        col=1,
    )

for i, z in enumerate(zeros):
    if i == 0:
        continue
    fig.add_trace(
        go.Scatter(
            x=[z],
            y=[0.0],
            mode="markers",
            marker_symbol="circle",
            marker_size=8,
            line=go.scatter.Line(color="black"),
        ),
        row=1,
        col=1,
    )
for z in co_zeros:
    fig.add_trace(
        go.Scatter(
            x=[z],
            y=[0.0],
            mode="markers",
            marker_symbol="circle",
            marker_size=8,
            line=go.scatter.Line(color="red"),
        ),
        row=2,
        col=1,
    )


fig.show()
