"""Plotting utilities for regimes and performance."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

plt.style.use("seaborn-v0_8")


def plot_states(
    price: pd.Series,
    states: pd.Series,
    *,
    title: str = "Price with HMM States",
    output: Optional[Path] = None,
) -> plt.Figure:
    """Plot price series overlaid with discrete states."""

    price = price.sort_index()
    states = states.reindex(price.index).ffill()

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(price.index, price.values, color="black", label="Price")
    ax1.set_ylabel("Price")
    ax1.set_title(title)

    ax2 = ax1.twinx()
    ax2.fill_between(price.index, 0, states.values, color="tab:blue", alpha=0.1)
    ax2.set_yticks(sorted(states.dropna().unique()))
    ax2.set_ylabel("State")

    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines, labels, loc="upper left")
    fig.tight_layout()

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150)
    return fig


def plot_equity(
    equity: pd.Series,
    benchmark: pd.Series,
    *,
    title: str = "Equity Curve",
    output: Optional[Path] = None,
) -> plt.Figure:
    """Plot strategy vs benchmark equity curves."""

    equity = equity.sort_index()
    benchmark = benchmark.reindex(equity.index).ffill()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(equity.index, equity.values, label="Strategy", color="tab:blue")
    ax.plot(benchmark.index, benchmark.values, label="Benchmark", color="tab:orange")
    ax.set_title(title)
    ax.set_ylabel("Equity")
    ax.legend()
    fig.tight_layout()

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150)
    return fig
