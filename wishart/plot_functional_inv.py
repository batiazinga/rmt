import numpy as np
import plotly.graph_objects as go

from wishart.lib import (
    inverse_limit_co_stieltjes,
    inverse_limit_stieltjes,
    marchenko_pastur_bounds,
)

c = 0.01


fig = go.Figure(
    layout=go.Layout(
        title=go.layout.Title(
            text=f"Functional inverse of the limit Stieltjes transform (c = {c})",
        )
    )
)

x_min = -10.0
x_max = 12.0

x = np.arange(x_min, -1.002, 0.001)
fig.add_trace(
    go.Scatter(
        x=x,
        y=inverse_limit_co_stieltjes(c, x),
        line=go.scatter.Line(color="black"),
    )
)
x = np.arange(-0.998, -0.2, 0.001)
fig.add_trace(
    go.Scatter(
        x=x,
        y=inverse_limit_co_stieltjes(c, x),
        line=go.scatter.Line(color="black"),
    )
)
x = np.arange(0.2, x_max, 0.001)
fig.add_trace(
    go.Scatter(
        x=x,
        y=inverse_limit_co_stieltjes(c, x),
        line=go.scatter.Line(color="black"),
    )
)

x = np.arange(x_min, -0.2, 0.001)
fig.add_trace(
    go.Scatter(
        x=x,
        y=inverse_limit_stieltjes(c, x),
        line=go.scatter.Line(color="blue"),
    )
)
x = np.arange(0.15, x_max, 0.001)
fig.add_trace(
    go.Scatter(
        x=x,
        y=inverse_limit_stieltjes(c, x),
        line=go.scatter.Line(color="blue"),
    )
)

lower, upper = marchenko_pastur_bounds(c)
fig.add_trace(
    go.Scatter(
        x=(x_min, x_max),
        y=(lower, lower),
        line=go.scatter.Line(color="green", dash="dot", width=1),
    )
)
fig.add_trace(
    go.Scatter(
        x=(x_min, x_max),
        y=(upper, upper),
        line=go.scatter.Line(color="green", dash="dot", width=1),
    )
)

fig.show()
