import numpy as np
import plotly.graph_objects as go  # type: ignore

from wishart.lib import scm

np.random.seed(42)

p = 600  # dimension
n = 60000  # sample size
c = p / n

# generate gaussian data
# the correlation matrix has 3 eigenvalues with high multiplicity
data1 = np.random.randn(p // 3, n)
data2 = np.sqrt(2.0) * np.random.randn(p // 3, n)
data3 = 2.0 * np.random.randn(p // 3, n)
data = np.concatenate([data1, data2, data3], axis=0)

# eigen values of the Wishart matrix
eigen_values = np.linalg.eigvalsh(scm(data))

# zeros of the empirical Stieltjes transform of the co Wishart matrix
sqrt_eigen_vec = np.array([np.sqrt(eigen_values)])
l = np.diag(eigen_values)
stieltjes_zeros_matrix = l - sqrt_eigen_vec.T.dot(sqrt_eigen_vec) / n
stieltjes_zeros = np.linalg.eigvalsh(stieltjes_zeros_matrix)

# infer the eigen values of the true correlation matrix
estimate1: float = np.sum(eigen_values[: p // 3] - stieltjes_zeros[: p // 3]) / c * 3.0
estimate2: float = (
    np.sum(eigen_values[p // 3 : 2 * p // 3] - stieltjes_zeros[p // 3 : 2 * p // 3])
    / c
    * 3.0
)
estimate3: float = (
    np.sum(eigen_values[2 * p // 3 :] - stieltjes_zeros[2 * p // 3 :]) / c * 3.0
)
print("estimate of the eigen values of the true correlation matrix:")
print(f"  {estimate1}")
print(f"  {estimate2}")
print(f"  {estimate3}")

fig = go.Figure(
    layout=go.Layout(
        title=go.layout.Title(
            text=f"Eigenvalues distribution (c = {c})",
        )
    )
)

empirical_density = np.histogram(eigen_values, bins=150)
empirical_delta = empirical_density[1][1] - empirical_density[1][0]
fig.add_trace(
    go.Bar(
        x=empirical_density[1],
        y=empirical_density[0] / float(p) / empirical_delta,
        name="Empirical distribution",
    )
)

fig.add_trace(
    go.Scatter(
        x=[estimate1],
        y=[0.0],
        mode="markers",
        marker_symbol="circle",
        marker_size=10,
        name="Estimate of the first true eigen value",
        line=go.scatter.Line(color="black"),
    )
)
fig.add_trace(
    go.Scatter(
        x=[estimate2],
        y=[0.0],
        mode="markers",
        marker_symbol="circle",
        marker_size=10,
        name="Estimate of the second true eigen value",
        line=go.scatter.Line(color="red"),
    )
)
fig.add_trace(
    go.Scatter(
        x=[estimate3],
        y=[0.0],
        mode="markers",
        marker_symbol="circle",
        marker_size=10,
        name="Estimate of the third true eigen value",
        line=go.scatter.Line(color="green"),
    )
)

fig.show()
