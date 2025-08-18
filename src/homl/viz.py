from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from typing import Iterable, Optional, Tuple, Union

# Type alias for readability
Number = Union[int, float]


def plot_regression_fit(
    X: Iterable[Number],
    y: Iterable[Number],
    X_line: Iterable[Number],
    y_pred: Iterable[Number],
    *,
    x_label: str = "X",
    y_label: str = "Y",
    title: str = "Scatter with linear fit",
    y_lim: Optional[Tuple[Number, Number]] = None,
    format_x_thousands: bool = False,
    show: bool = True,
):
    """
    Scatter points (X, y) with a regression line (X_line, y_pred).

    Parameters
    ----------
    X, y : data samples
    X_line, y_pred : line coordinates
    x_label, y_label, title : axis labels and figure title
    y_lim : optional (ymin, ymax) range
    format_x_thousands : format x-axis with thousands separators if True
    show : display plot interactively (disable if saving to file)

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Create figure + axis
    fig, ax = plt.subplots(figsize=(16, 10))

    # Scatter points and regression line
    ax.scatter(X, y, s=60, alpha=0.85, label="Data")
    ax.plot(X_line, y_pred, linewidth=2, label="Linear fit")

    # Axis labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    # Optional y-limits and x-axis formatting
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    if format_x_thousands:
        ax.xaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))

    # Add grid (major + minor ticks)
    ax.minorticks_on()
    ax.grid(True, which="major")
    ax.grid(True, which="minor", linestyle=":")

    # Legend and layout
    ax.legend()
    fig.tight_layout()

    # Show interactively if requested
    if show:
        plt.show()

    return fig
