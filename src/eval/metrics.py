"""Performance metrics for regime strategies."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def compute_metrics(
    equity_curve: pd.Series,
    returns: pd.Series,
    *,
    periods_per_year: int = 8760,
    risk_free_rate: float = 0.0,
) -> Dict[str, float]:
    """Compute key performance metrics."""

    if equity_curve.empty or returns.empty:
        raise ValueError("Cannot compute metrics on empty series.")

    equity = equity_curve.dropna()
    rets = returns.dropna()

    total_periods = len(rets)
    ending_value = equity.iloc[-1]
    if ending_value <= 0:
        cagr = float("nan")
    else:
        years = total_periods / periods_per_year
        cagr = ending_value ** (1 / years) - 1 if years > 0 else float("nan")

    excess_returns = rets - risk_free_rate / periods_per_year
    vol = excess_returns.std(ddof=0)
    sharpe = (np.sqrt(periods_per_year) * excess_returns.mean() / vol) if vol > 0 else float("nan")

    rolling_max = equity.cummax()
    drawdowns = equity / rolling_max - 1
    max_drawdown = drawdowns.min()

    hit_rate = (rets > 0).sum() / total_periods if total_periods > 0 else float("nan")

    return {
        "CAGR": float(cagr),
        "Sharpe": float(sharpe),
        "MaxDrawdown": float(max_drawdown),
        "HitRate": float(hit_rate),
    }
