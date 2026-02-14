import numpy as np
import plotly.graph_objects as go

from wishart.lib import limit_stieltjes

# parameters
c = 0.01

fig = go.Figure(
    layout=go.Layout(
        title=go.layout.Title(
            text=(
                "Real and imaginary parts of the limit Stieltjes transform "
                f"(c = {c})"
            ),
        )
    )
)

x = np.arange(0.5, 1.5, 0.005)
limit_stielt = limit_stieltjes(c, x)
fig.add_trace(
    go.Scatter(
        x=x,
        y=np.real(limit_stielt),
        name="Real part",
    )
)
fig.add_trace(
    go.Scatter(
        x=x,
        y=np.imag(limit_stielt),
        name="Imaginary part",
    )
)

fig.show()
