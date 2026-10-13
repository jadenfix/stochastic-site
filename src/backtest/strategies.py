"""Backtesting utilities for regime-driven strategies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class StrategyResult:
    """Container for backtest results."""

    name: str
    returns: pd.Series
    equity_curve: pd.Series
    weights: pd.Series
    benchmark: pd.Series
    metadata: Dict[str, object]


def _classify_states(states: pd.Series, returns: pd.Series) -> Dict[int, str]:
    stats = pd.DataFrame(
        {
            "mean_return": returns.groupby(states).mean(),
            "volatility": returns.groupby(states).std(ddof=0),
        }
    ).fillna(0.0)
    if stats.empty:
        raise ValueError("Unable to classify states: stats table empty.")

    bullish = stats["mean_return"].idxmax()
    bearish = stats["mean_return"].idxmin()
    labels: Dict[int, str] = {int(idx): "neutral" for idx in stats.index}
    if bullish == bearish:
        return labels
    labels[int(bullish)] = "calm_up"
    labels[int(bearish)] = "bearish"
    remaining = stats.drop(index=[bullish, bearish])
    if not remaining.empty:
        labels[int(remaining["volatility"].idxmax())] = "high_vol"
    return labels


def _apply_weights(
    returns: pd.Series,
    weights: pd.Series,
    fee_bps: float,
) -> tuple[pd.Series, pd.Series]:
    aligned_returns = returns.reindex(weights.index).fillna(0.0)
    if weights.empty:
        raise ValueError("Cannot backtest empty weights")
    if not np.isfinite(fee_bps) or fee_bps < 0:
        raise ValueError("fee_bps must be finite and non-negative")
    shifted_weights = weights.shift().fillna(0.0)
    gross = shifted_weights * np.expm1(aligned_returns)
    turnover = shifted_weights.diff().abs().fillna(shifted_weights.abs())
    fees = turnover * (fee_bps / 10_000)
    strategy_returns = gross - fees
    equity_curve = (1 + strategy_returns).cumprod()
    equity_curve.name = "equity"
    strategy_returns.name = "strategy_return"
    return strategy_returns, equity_curve


def _spot_regime_strategy(
    price: pd.Series,
    returns: pd.Series,
    states: pd.Series,
    fee_bps: float,
    state_labels: Optional[Dict[int, str]] = None,
) -> StrategyResult:
    labels = state_labels if state_labels is not None else _classify_states(states, returns)
    weight_map = {state: 0.5 for state in labels}
    calm_states = [state for state, label in labels.items() if label == "calm_up"]
    high_vol_states = [state for state, label in labels.items() if label == "high_vol"]
    bearish_states = [state for state, label in labels.items() if label == "bearish"]

    for state in calm_states:
        weight_map[state] = 1.0
    for state in set(high_vol_states + bearish_states):
        weight_map[state] = 0.0

    weights = states.map(weight_map).fillna(0.5).astype(float)
    strategy_returns, equity_curve = _apply_weights(returns, weights, fee_bps)

    benchmark = np.exp(returns.cumsum())
    benchmark.name = "buy_hold"

    metadata = {
        "state_labels": labels,
        "fee_bps": fee_bps,
        "evaluation": "training_labels" if state_labels is not None else "retrospective_exploration",
    }

    return StrategyResult(
        name="spot_regime",
        returns=strategy_returns,
        equity_curve=equity_curve,
        weights=weights,
        benchmark=benchmark,
        metadata=metadata,
    )


def _options_overlay_stub(
    dvol: Optional[pd.Series],
    states: pd.Series,
    labels: Dict[int, str],
) -> pd.Series:
    if dvol is None or dvol.empty:
        return pd.Series(dtype=float)
    dvol = dvol.sort_index().reindex(states.index, method="ffill")
    dvol_change = dvol.diff().fillna(0.0)
    overlay = pd.Series(0.0, index=states.index, name="options_overlay")
    for state, label in labels.items():
        if label == "high_vol":
            overlay.loc[states == state] = np.where(dvol_change.loc[states == state] > 0, 1.0, 0.0)
    return overlay


def run_strategy(
    name: str,
    *,
    price: pd.Series,
    returns: pd.Series,
    states: pd.Series,
    fee_bps: float = 2.0,
    dvol: Optional[pd.Series] = None,
    state_labels: Optional[Dict[int, str]] = None,
) -> StrategyResult:
    """Run a named strategy and return performance data."""

    price = price.sort_index()
    returns = returns.reindex(price.index).fillna(0.0)
    states = states.reindex(price.index).ffill().dropna()

    if name == "spot_regime":
        base_result = _spot_regime_strategy(price, returns, states, fee_bps, state_labels)
        overlay = _options_overlay_stub(dvol, states, base_result.metadata["state_labels"])
        if not overlay.empty:
            base_result.metadata["options_overlay_signal"] = overlay
        return base_result

    raise ValueError(f"Unsupported strategy: {name}")
