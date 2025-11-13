# logistic_growth_gif.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# -----------------------------
# Colors
# -----------------------------
colors = ["#76508A", "#65A9A2"]  # [line_color, point_color]

# -----------------------------
# Model parameters
# -----------------------------
V0 = 8.4619      # initial cyst volume (cm³)
K = 50.0         # carrying capacity (cm³)
r = 0.4          # growth rate (per year)

years = 10
n_points = 300

t = np.linspace(0, years, n_points)

if V0 <= 0:
    raise ValueError("Initial volume V0 must be > 0 for this logistic model.")

Vt = K / (1.0 + ((K - V0) / V0) * np.exp(-r * t))

# -----------------------------
# Figure setup
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 5))

ax.set_title("Simulated Cyst Growth Over Time", fontsize=14)
ax.set_xlabel("Time (years)", fontsize=12)
ax.set_ylabel("Cyst volume (cm³)", fontsize=12)

ax.set_xlim(0, years)
ax.set_ylim(0, 1.1 * K)

# Optional: light grid
ax.grid(True, alpha=0.25)

# Line that will grow over time
(line,) = ax.plot(
    [], [],
    linewidth=2.5,
    label="Cyst volume",
    color=colors[0],
)

# Moving marker
(point,) = ax.plot(
    [],
    [],
    "o",
    markersize=8,
    color=colors[1],          # edge color
    markerfacecolor=colors[1] # fill color
)

# Text box for time and volume
time_text = ax.text(
    0.02,
    0.95,
    "",
    transform=ax.transAxes,
    fontsize=11,
    verticalalignment="top",
)

ax.legend(fontsize=10, loc="lower right")


# -----------------------------
# Animation functions
# -----------------------------
def init():
    """Initialize empty frame."""
    line.set_data([], [])
    point.set_data([], [])
    time_text.set_text("")
    return line, point, time_text


def update(frame):
    """
    frame: integer from 0 to n_points-1.
    We show data up to this frame.
    """
    t_curr = t[: frame + 1]
    V_curr = Vt[: frame + 1]

    # Line: full history
    line.set_data(t_curr, V_curr)

    # Point: last point (must be sequences)
    point.set_data([t_curr[-1]], [V_curr[-1]])

    time_text.set_text(
        f"t = {t_curr[-1]:.2f} years\nV = {V_curr[-1]:.2f} cm³"
    )

    return line, point, time_text


# -----------------------------
# Create animation
# -----------------------------
anim = FuncAnimation(
    fig,
    update,
    frames=n_points,
    init_func=init,
    blit=True,
    interval=40,   # ms between frames
)

# -----------------------------
# Save as GIF
# -----------------------------
gif_filename = "cyst_growth.gif"
writer = PillowWriter(fps=25)
anim.save(gif_filename, writer=writer)

print(f"Saved animation to {gif_filename}")
