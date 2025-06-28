import math

import numpy as np
import plotly.graph_objects as go  # type: ignore

from wishart.lib import marchenko_pastur, scm

np.random.seed(42)

p = 600  # dimension
n = 60000  # sample size
c = p / n

# generate gaussian data
# the correlation matrix is the identity + low rank perturbation
sqrt_c = math.sqrt(c)
eig_perturbation = np.array(
    [
        sqrt_c - 0.04,
        sqrt_c + 0.04,
        3 * sqrt_c,
        5 * sqrt_c,
    ]
)
data1 = np.sqrt(1.0 + eig_perturbation[0]) * np.random.randn(1, n)
data2 = np.sqrt(1.0 + eig_perturbation[1]) * np.random.randn(1, n)
data4 = np.sqrt(1.0 + eig_perturbation[2]) * np.random.randn(1, n)
data3 = np.sqrt(1.0 + eig_perturbation[3]) * np.random.randn(1, n)
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

empirical_density = np.histogram(eigen_values, bins=100)
empirical_delta = empirical_density[1][1] - empirical_density[1][0]
fig.add_trace(
    go.Bar(
        x=empirical_density[1],
        y=empirical_density[0] / float(p) / empirical_delta,
        name="Empirical distribution",
    )
)

x = np.arange(0.005, 2.0, 0.005)
limit_density = marchenko_pastur(c, x)
fig.add_trace(
    go.Scatter(
        x=x,
        y=limit_density,
        line=go.scatter.Line(color="black", dash="dash"),
        name="Limit distribution without perturbation",
    )
)

spiked = eig_perturbation[1:]
x_spikes = 1.0 + spiked + c * (1.0 + spiked) / spiked
fig.add_trace(
    go.Scatter(
        x=x_spikes,
        y=[0.0] * len(spiked),
        mode="markers",
        marker_symbol="circle",
        marker_size=10,
        name="Limit spike locations",
        line=go.scatter.Line(color="black"),
    )
)

fig.add_trace(
    go.Scatter(
        x=1.0 + eig_perturbation,
        y=[0.0] * len(eig_perturbation),
        mode="markers",
        marker_symbol="cross",
        marker_size=10,
        name="True spikes",
        line=go.scatter.Line(color="red"),
    )
)

fig.show()
