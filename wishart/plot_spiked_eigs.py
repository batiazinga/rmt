import numpy as np
import plotly.graph_objects as go  # type: ignore

from wishart.lib import scm

np.random.seed(42)

p = 600  # dimension
n = 60000  # sample size
c = p / n

# generate gaussian data
# the correlation matrix is the identity + low rank perturbation
data1 = np.sqrt(2.0) * np.random.randn(1, n)
data2 = np.sqrt(3.0) * np.random.randn(1, n)
data3 = 2.0 * np.random.randn(1, n)
data4 = np.sqrt(5.0) * np.random.randn(1, n)
data5 = np.random.randn(p - 4, n)
data = np.concatenate([data1, data2, data3, data4, data5], axis=0)

# eigen values of the Wishart matrix
eigen_values = np.linalg.eigvalsh(scm(data))


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

fig.show()
