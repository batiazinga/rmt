import numpy as np
import plotly.graph_objects as go

from wishart.lib import marchenko_pastur, scm

np.random.seed(42)


# parameters
p = 500  # dimension
n = 50000  # sample size
c = p / n

# generate iid gaussian data
data = np.random.randn(p, n)

# eigen values of the Wishart matrix
eigen_values = np.linalg.eigvalsh(scm(data))

# zeros of the empirical Stieltjes transform of the co Wishart matrix
# p-1 zeros (one between each pair of eigenvalue) + a 0.0
sqrt_eigen_vec = np.array([np.sqrt(eigen_values)])
l = np.diag(eigen_values)
stieltjes_zeros_matrix = l - sqrt_eigen_vec.T.dot(sqrt_eigen_vec) / n
stieltjes_zeros = np.linalg.eigvalsh(stieltjes_zeros_matrix)

# infer the value of the (unique) eigen values of the true correlation matrix
# the true correlation matrix is the identity, so we expect to find 1.0
estimate: float = np.sum(eigen_values - stieltjes_zeros) / c
print(f"Estimate of the true eigen value: {estimate} (expecting 1.0)")

fig = go.Figure(
    layout=go.Layout(
        title=go.layout.Title(
            text=f"Eigenvalues of a white Wishart matrix (c = {c})",
            subtitle=go.layout.title.Subtitle(
                text=(
                    "Histogram of the empirical eigenvalues "
                    "versus limit distribution"
                )
            ),
        )
    )
)

empirical_density = np.histogram(eigen_values, bins=30)
empirical_delta = empirical_density[1][1] - empirical_density[1][0]
fig.add_trace(
    go.Bar(
        x=empirical_density[1],
        y=empirical_density[0] / float(p) / empirical_delta,
        name="Empirical distribution",
    )
)

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

fig.add_trace(
    go.Scatter(
        x=[estimate],
        y=[0.0],
        mode="markers",
        marker_symbol="circle",
        marker_size=10,
        name="Estimate of the true eigen value",
        line=go.scatter.Line(color="black"),
    )
)

fig.show()
