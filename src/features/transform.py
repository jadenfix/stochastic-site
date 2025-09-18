"""Feature engineering helpers for regime models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "log_return",
    "rolling_vol_24h",
    "rsi_24",
    "volume_zscore",
]


@dataclass
class FeatureSet:
    """Container for engineered features and associated scaler."""

    features: pd.DataFrame
    raw: pd.DataFrame
    scaler: StandardScaler


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """Compute log returns for a price series."""

    prices = pd.to_numeric(prices, errors="coerce")
    log_prices = np.log(prices.replace(0, np.nan))
    returns = log_prices.diff().fillna(0.0)
    return returns.replace([np.inf, -np.inf], 0.0)


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Compute Wilder's RSI for a price series."""

    prices = pd.to_numeric(prices, errors="coerce")
    delta = prices.diff().fillna(0.0)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def compute_volume_zscore(volume: pd.Series, window: int = 24) -> pd.Series:
    """Compute rolling z-score of volume."""

    vol = pd.to_numeric(volume, errors="coerce")
    rolling_mean = vol.rolling(window=window, min_periods=max(3, window // 2)).mean()
    rolling_std = vol.rolling(window=window, min_periods=max(3, window // 2)).std(ddof=0)
    zscore = (vol - rolling_mean) / rolling_std.replace(0, np.nan)
    return zscore.fillna(0.0)


def build_feature_matrix(
    ohlcv: pd.DataFrame,
    *,
    price_col: str = "close",
    volume_col: str = "volume",
    return_window: int = 24,
    rsi_period: int = 24,
    scaler: Optional[StandardScaler] = None,
    dropna: bool = True,
) -> FeatureSet:
    """Engineer standardized features for regime models.

    Parameters
    ----------
    ohlcv:
        DataFrame containing at least close and volume columns, indexed by time.
    price_col, volume_col:
        Column names for prices and volume.
    return_window:
        Window (in bars) for rolling volatility.
    rsi_period:
        RSI lookback period.
    scaler:
        Optionally reuse an existing ``StandardScaler``.
    dropna:
        Drop rows containing NaNs before scaling.
    """

    if price_col not in ohlcv or volume_col not in ohlcv:
        missing: set[str] = {c for c in (price_col, volume_col) if c not in ohlcv}
        msg = f"Input DataFrame missing required columns: {missing}"
        raise ValueError(msg)

    df = ohlcv.copy()
    df = df.sort_index()

    raw = pd.DataFrame(index=df.index)
    raw["log_return"] = compute_log_returns(df[price_col])
    raw["rolling_vol_24h"] = (
        raw["log_return"].rolling(window=return_window, min_periods=max(3, return_window // 2)).std(ddof=0)
    )
    raw["rsi_24"] = compute_rsi(df[price_col], period=rsi_period)
    raw["volume_zscore"] = compute_volume_zscore(df[volume_col], window=return_window)

    if dropna:
        raw = raw.dropna()
    else:
        raw = raw.ffill().bfill()

    if raw.empty:
        raise ValueError("Feature matrix is empty after preprocessing; check data window.")

    scaler_to_use = scaler or StandardScaler()
    if scaler is None:
        scaled_values = scaler_to_use.fit_transform(raw.values)
    else:
        scaled_values = scaler_to_use.transform(raw.values)

    features = pd.DataFrame(scaled_values, index=raw.index, columns=FEATURE_COLUMNS)
    return FeatureSet(features=features, raw=raw, scaler=scaler_to_use)


def ensure_feature_columns(columns: Iterable[str]) -> None:
    """Validate that provided columns cover the default feature set."""

    missing = set(FEATURE_COLUMNS) - set(columns)
    if missing:
        msg = f"Missing required feature columns: {missing}"
        raise ValueError(msg)
