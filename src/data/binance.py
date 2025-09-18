"""Binance market data utilities with paging, retries, and caching."""
from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import requests

from src.config import settings

logger = logging.getLogger(__name__)

KLINE_COLUMNS: tuple[str, ...] = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
)
_NUMERIC_COLUMNS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
)

_BASE_ENDPOINT = "v3/klines"
_MAX_LIMIT = 1000
_DEFAULT_RETRY_DELAY = 0.6
_DEFAULT_TIMEOUT = 30
_SESSION_HEADERS = {"User-Agent": "hmm-regimes-crypto/0.1"}

Interval = str


def interval_to_milliseconds(interval: Interval) -> int:
    """Convert Binance interval string (e.g. 1h) to milliseconds."""

    unit = interval[-1]
    value = int(interval[:-1])
    multipliers = {
        "m": 60_000,
        "h": 3_600_000,
        "d": 86_400_000,
        "w": 604_800_000,
        "M": 2_592_000_000,  # approximate 30 days
    }
    if unit not in multipliers:
        msg = f"Unsupported interval unit: {interval}"
        raise ValueError(msg)
    return value * multipliers[unit]


def _ensure_datetime(value: datetime | str | pd.Timestamp) -> datetime:
    if isinstance(value, datetime):
        return value
    ts = pd.to_datetime(value, utc=True)
    return ts.tz_localize(None) if ts.tzinfo else ts.to_pydatetime()


def _build_cache_path(
    symbol: str,
    interval: Interval,
    start: datetime,
    end: datetime,
    cache_dir: Optional[Path] = None,
) -> Path:
    cache_dir = cache_dir or (settings.cache_dir / "binance")
    cache_dir.mkdir(parents=True, exist_ok=True)
    slug = f"{symbol}|{interval}|{start.isoformat()}|{end.isoformat()}"
    digest = hashlib.sha256(slug.encode()).hexdigest()[:16]
    filename = f"binance_{symbol.lower()}_{interval}_{digest}.parquet"
    return cache_dir / filename


def fetch_klines(
    symbol: str,
    interval: Interval = "1h",
    start: datetime | str | pd.Timestamp = datetime.utcnow(),
    end: datetime | str | pd.Timestamp | None = None,
    *,
    limit: int = _MAX_LIMIT,
    max_retries: int = 5,
    retry_delay: float = _DEFAULT_RETRY_DELAY,
    session: Optional[requests.Session] = None,
    use_cache: bool = True,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Fetch OHLCV klines for the given symbol and interval.

    Parameters
    ----------
    symbol:
        Trading pair symbol, e.g. ``BTCUSDT``.
    interval:
        Kline interval string accepted by Binance (default ``1h``).
    start:
        Start datetime (inclusive). Strings are parsed via ``pandas.to_datetime``.
    end:
        End datetime (exclusive). Defaults to current UTC time.
    limit:
        Page size per request (max 1000 per Binance API rules).
    max_retries:
        Number of retries for failed requests.
    retry_delay:
        Base delay between retries in seconds.
    session:
        Optional ``requests.Session`` to reuse connections.
    use_cache:
        If True, read/write parquet cache under ``data/cache``.
    cache_dir:
        Override cache directory.

    Returns
    -------
    pandas.DataFrame
        Indexed by open time with numeric columns parsed.
    """

    if limit > _MAX_LIMIT:
        msg = f"Binance limit cannot exceed {_MAX_LIMIT}. Got {limit}."
        raise ValueError(msg)

    symbol = symbol.upper()
    start_dt = _ensure_datetime(start)
    end_dt = _ensure_datetime(end) if end else datetime.utcnow()

    cache_path: Path | None = None
    if use_cache:
        cache_path = _build_cache_path(symbol, interval, start_dt, end_dt, cache_dir)
        if cache_path.exists():
            logger.debug("Loading Binance klines from cache: %s", cache_path)
            return pd.read_parquet(cache_path)

    request_session = session or requests.Session()
    owns_session = session is None
    if owns_session:
        request_session.headers.update(_SESSION_HEADERS)

    endpoint = f"{settings.binance_base_url}/{_BASE_ENDPOINT}"

    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    interval_ms = interval_to_milliseconds(interval)

    rows: list[list[str | float | int]] = []
    current = start_ms

    try:
        while current < end_ms:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": current,
                "endTime": end_ms,
                "limit": limit,
            }
            response_json: list[list[str | float | int]] | None = None
            for attempt in range(1, max_retries + 1):
                try:
                    response = request_session.get(
                        endpoint, params=params, timeout=_DEFAULT_TIMEOUT
                    )
                    if response.status_code == 429:
                        sleep_for = retry_delay * attempt
                        logger.warning(
                            "Binance rate limit (HTTP 429). Sleeping %.2fs", sleep_for
                        )
                        time.sleep(sleep_for)
                        continue
                    response.raise_for_status()
                    response_json = response.json()
                    break
                except requests.RequestException as exc:
                    if attempt == max_retries:
                        raise RuntimeError("Failed to fetch Binance klines") from exc
                    sleep_for = retry_delay * attempt
                    logger.warning(
                        "Binance request error: %s. Retry in %.2fs", exc, sleep_for
                    )
                    time.sleep(sleep_for)
            if not response_json:
                logger.debug("No further data returned from Binance; stopping pagination.")
                break

            rows.extend(response_json)
            last_open_time = int(response_json[-1][0])
            current = last_open_time + interval_ms
            if len(response_json) < limit:
                break
    finally:
        if owns_session:
            request_session.close()

    if not rows:
        logger.warning("No Binance klines retrieved for %s %s", symbol, interval)
        return pd.DataFrame(columns=KLINE_COLUMNS)

    frame = pd.DataFrame(rows, columns=KLINE_COLUMNS)
    frame["open_time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    frame["close_time"] = pd.to_datetime(frame["close_time"], unit="ms", utc=True)

    for col in _NUMERIC_COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame = frame.set_index("open_time").sort_index()

    if cache_path:
        frame.to_parquet(cache_path)
        logger.debug("Cached Binance klines to %s", cache_path)

    return frame


def ensure_columns(data: pd.DataFrame, required: Iterable[str]) -> pd.DataFrame:
    """Ensure the DataFrame includes required columns, raising if missing."""

    missing = set(required) - set(data.columns)
    if missing:
        msg = f"DataFrame missing expected Binance columns: {missing}"
        raise ValueError(msg)
    return data
