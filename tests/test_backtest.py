"""Tests for backtesting utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.strategies import run_strategy


def test_spot_regime_strategy_returns_are_finite() -> None:
    index = pd.date_range("2023-01-01", periods=100, freq="h")
    price = pd.Series(np.linspace(20000, 21000, len(index)), index=index)
    returns = pd.Series(np.random.default_rng(0).normal(0, 0.001, size=len(index)), index=index)
    weights = pd.Series(np.random.default_rng(1).integers(0, 3, size=len(index)), index=index)

    result = run_strategy(
        "spot_regime",
        price=price,
        returns=returns,
        states=weights,
    )

    assert not result.returns.isna().any()
    assert not result.equity_curve.isna().any()
