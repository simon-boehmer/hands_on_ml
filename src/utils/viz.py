from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.figure import Figure
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
) -> Figure:
    # Create figure + axis
    fig, ax = plt.subplots(figsize=(16, 10))

    # Scatter data points and regression line
    ax.scatter(X, y, s=60, alpha=0.85, label="Data")
    ax.plot(
        X_line, y_pred, color="red", linestyle="--", linewidth=2, label="Linear fit"
    )

    # Axis labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    # Optional y-limits and thousands formatting
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    if format_x_thousands:
        ax.xaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))

    # Grid + minor ticks
    ax.minorticks_on()
    ax.grid(True, which="major")
    ax.grid(True, which="minor", linestyle=":")

    # Legend + layout
    ax.legend()
    fig.tight_layout()

    # Show interactively if requested
    if show:
        plt.show()

    return fig


def plot_scatter(
    X: Iterable[Number],
    y: Iterable[Number],
    *,
    x_label: str = "X",
    y_label: str = "Y",
    title: str = "Scatter plot",
    alpha: float = 0.85,
    format_x_thousands: bool = False,
    show: bool = True,
) -> Figure:
    # Create figure + axis
    fig, ax = plt.subplots(figsize=(16, 10))

    # Scatter points only
    ax.scatter(X, y, s=60, alpha=alpha, label="Data")

    # Axis labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    # Thousands formatting (only if values are large)
    if format_x_thousands and max(abs(float(v)) for v in X) >= 1000:
        ax.xaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))

    # Grid + minor ticks
    ax.minorticks_on()
    ax.grid(True, which="major")
    ax.grid(True, which="minor", linestyle=":")

    # Legend + layout
    ax.legend()
    fig.tight_layout()

    # Show interactively if requested
    if show:
        plt.show()

    return fig


def plot_housing_map(
    df: pd.DataFrame,
    *,
    alpha: float = 0.4,
    size_scale: float = 0.02,  # scaling factor for population marker size
    cmap: str = "jet",
    show: bool = True,
) -> Figure:
    # Create figure + axis
    fig, ax = plt.subplots(figsize=(12, 8))

    # Scatter plot: longitude vs latitude
    scatter = ax.scatter(
        df["longitude"],
        df["latitude"],
        s=df["population"] * size_scale,  # marker size ~ population
        c=df["median_house_value"],  # marker color ~ house value
        cmap=cmap,
        alpha=alpha,
        label="Population-weighted housing values",
    )

    # Axis labels and title
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("California Housing Prices")

    # Grid + minor ticks
    ax.minorticks_on()
    ax.grid(True, which="major")
    ax.grid(True, which="minor", linestyle=":")

    # Colorbar for house values
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Median house value")

    # Legend + layout
    ax.legend()
    fig.tight_layout()

    # Show interactively if requested
    if show:
        plt.show()

    return fig
