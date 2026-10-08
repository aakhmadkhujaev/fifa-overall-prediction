"""Minimal shared matplotlib styling for the visualization layer.

Figures are created with :class:`matplotlib.figure.Figure` directly so no
pyplot global state is touched, and styling is applied only inside
``plot_style()`` via ``rc_context`` so global rcParams are never modified.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any

import matplotlib as mpl
import numpy as np
from matplotlib.figure import Figure

# Okabe-Ito colorblind-safe pair.
POPULATION_COLORS = {"goalkeeper": "#0072B2", "outfield": "#E69F00"}
NEUTRAL_COLOR = "#333333"
DEFAULT_FIGSIZE = (7.0, 5.0)

_STYLE: dict[Any, Any] = {
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "legend.frameon": False,
    "figure.dpi": 100,
}


def plot_style() -> AbstractContextManager[None]:
    """Context manager applying the shared style without changing global rcParams."""
    return mpl.rc_context(_STYLE)


def population_color(population: str) -> str:
    """Return the palette color for a population, or a neutral fallback."""
    return POPULATION_COLORS.get(population, NEUTRAL_COLOR)


def new_figure(
    nrows: int = 1,
    ncols: int = 1,
    *,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
) -> tuple[Figure, np.ndarray]:
    """Create a standalone Figure and a 2-D array of axes (never registered with pyplot)."""
    figure = Figure(figsize=figsize, layout="constrained")
    axes = figure.subplots(nrows, ncols, squeeze=False)
    return figure, axes
