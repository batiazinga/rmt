import logging
import time

import numpy as np
import plotly.graph_objects as go

from wishart.lib import marchenko_pastur_bounds, scm, tracy_widom1

logging.basicConfig(level=logging.DEBUG)
np.random.seed(42)


# parameters
p = 256  # dimension
n = 2048  # sample size
sample_size = 4000
batch_size = 100
c = p / n

logging.info("Generating data...")
logging.debug(
    f"Generating {sample_size//batch_size} batches of size {batch_size} "
    f"of Wishart matrices (c={c})..."
)
start = time.time()
largest_eigen_values = np.zeros(sample_size)
for i in range(sample_size // batch_size):
    logging.debug(f"  batch {i+1}/{sample_size//batch_size}")
    data = np.random.randn(batch_size, p, n)
    largest_eigen_values[i * batch_size : (i + 1) * batch_size] = np.linalg.eigvalsh(
        scm(data)
    )[:, -1]
logging.info(f"Generated data in {time.time()-start:.2f}s")

logging.info("Rescaling largest eigen values...")
_, upper_bound = marchenko_pastur_bounds(c)
scaled = (
    n ** (2.0 / 3.0)
    * (largest_eigen_values - upper_bound)
    / upper_bound ** (2.0 / 3.0)
    / c ** (-1.0 / 6.0)
)

fig = go.Figure(
    layout=go.Layout(
        title=go.layout.Title(
            text=(
                "Distribution of the largest eigen value "
                f"of a white Wishart matrix (c = {c})"
            ),
            subtitle=go.layout.title.Subtitle(
                text=(
                    "Histogram of the empirical distribution "
                    "versus limit distribution"
                )
            ),
        )
    )
)


empirical_density = np.histogram(scaled, bins=30)
empirical_delta = empirical_density[1][1] - empirical_density[1][0]
fig.add_trace(
    go.Bar(
        x=empirical_density[1],
        y=empirical_density[0] / float(sample_size) / empirical_delta,
        name="Scaled empirical distribution",
    )
)

x = np.arange(-6.0, 4.0, 0.05)
density = tracy_widom1(x)
fig.add_trace(
    go.Scatter(
        x=x,
        y=density,
        line=go.scatter.Line(color="black", dash="dash"),
        name="Scaled theoretical distribution",
    )
)

fig.show()
