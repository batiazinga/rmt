import numpy as np
import plotly.graph_objects as go

from wishart.lib import gamma, marchenko_pastur, marchenko_pastur_bounds

# parameters
c = 0.05

lower, upper = marchenko_pastur_bounds(c)
border = 0.015
step = 0.001
horizontal = np.arange(lower - border, upper + border, step)
z_top = horizontal + border * 1j
z_down = np.flipud(horizontal) - border * 1j
vertical = np.arange(-border, border, step) * 1j
z_left = vertical + lower - border
z_right = np.flipud(vertical) + upper + border
z = np.concatenate((z_top, z_right, z_down, z_left))
g = gamma(c, z)

fig = go.Figure(
    layout=go.Layout(
        title=go.layout.Title(
            text=(
                "Image of a contour containing the support of the limit distribution "
                f"(c = {c}) by the gamma function"
            ),
        )
    )
)

x = np.real(z)
y = np.imag(z)
fig.add_trace(
    go.Scatter(
        x=x,
        y=y,
        name="Contour containing the support of the limit distribution",
    )
)

gamma_x = np.real(g)
gamma_y = np.imag(g)
fig.add_trace(
    go.Scatter(
        x=gamma_x,
        y=gamma_y,
        name="Image of the contour by the gamma function",
    )
)

x = np.arange(step, upper + 2 * border, step)
limit_density = marchenko_pastur(c, x)
fig.add_trace(
    go.Scatter(
        x=x,
        y=limit_density,
        line=go.scatter.Line(color="black", dash="dot"),
    )
)

fig.show()
