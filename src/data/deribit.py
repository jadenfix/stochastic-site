"""Deribit public data utilities with caching and retries."""
from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from src.config import settings

logger = logging.getLogger(__name__)

_TRADINGVIEW_ENDPOINT = "public/get_tradingview_chart_data"
_DVOL_ENDPOINT = "public/get_volatility_index_data"
_DEFAULT_TIMEOUT = 30
_DEFAULT_RETRY_DELAY = 0.6
_SESSION_HEADERS = {"User-Agent": "hmm-regimes-crypto/0.1"}


def _ensure_datetime(value: datetime | str | pd.Timestamp | None, fallback: datetime) -> datetime:
    if value is None:
        return fallback
    if isinstance(value, datetime):
        return value
    ts = pd.to_datetime(value, utc=True)
    return ts.tz_localize(None) if ts.tzinfo else ts.to_pydatetime()


def _build_cache_path(prefix: str, slug: str, cache_dir: Optional[Path]) -> Path:
    cache_dir = cache_dir or (settings.cache_dir / "deribit")
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(slug.encode()).hexdigest()[:16]
    return cache_dir / f"deribit_{prefix}_{digest}.parquet"


def fetch_candles(
    instrument_name: str = "BTC-PERPETUAL",
    resolution: str = "60",
    start: datetime | str | pd.Timestamp | None = None,
    end: datetime | str | pd.Timestamp | None = None,
    *,
    session: Optional[requests.Session] = None,
    days: int = 30,
    max_retries: int = 5,
    retry_delay: float = _DEFAULT_RETRY_DELAY,
    use_cache: bool = True,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Fetch Deribit TradingView-style candles for the given instrument."""

    now = datetime.utcnow()
    start_dt = _ensure_datetime(start, now - timedelta(days=days))
    end_dt = _ensure_datetime(end, now)

    cache_path: Path | None = None
    if use_cache:
        slug = f"candles|{instrument_name}|{resolution}|{start_dt.isoformat()}|{end_dt.isoformat()}"
        cache_path = _build_cache_path("candles", slug, cache_dir)
        if cache_path.exists():
            logger.debug("Loading Deribit candles from cache: %s", cache_path)
            return pd.read_parquet(cache_path)

    request_session = session or requests.Session()
    owns_session = session is None
    if owns_session:
        request_session.headers.update(_SESSION_HEADERS)

    params = {
        "instrument_name": instrument_name,
        "resolution": resolution,
        "start_timestamp": int(start_dt.timestamp() * 1000),
        "end_timestamp": int(end_dt.timestamp() * 1000),
    }
    url = f"{settings.deribit_base_url}/{_TRADINGVIEW_ENDPOINT}"

    try:
        response_json: dict | None = None
        for attempt in range(1, max_retries + 1):
            try:
                response = request_session.get(url, params=params, timeout=_DEFAULT_TIMEOUT)
                response.raise_for_status()
                response_json = response.json()
                break
            except requests.RequestException as exc:
                if attempt == max_retries:
                    raise RuntimeError("Failed to fetch Deribit candles") from exc
                sleep_for = retry_delay * attempt
                logger.warning("Deribit candle request error: %s. Retry in %.2fs", exc, sleep_for)
                time.sleep(sleep_for)
        if not response_json or "result" not in response_json:
            logger.warning("Deribit candle response empty for %s", instrument_name)
            return pd.DataFrame()

        result = response_json["result"]
        required_keys = {"ticks", "open", "high", "low", "close", "volume"}
        if not required_keys.issubset(result):
            missing = required_keys - set(result)
            msg = f"Missing expected keys from Deribit response: {missing}"
            raise ValueError(msg)

        frame = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(result["ticks"], unit="ms", utc=True),
                "open": result["open"],
                "high": result["high"],
                "low": result["low"],
                "close": result["close"],
                "volume": result["volume"],
            }
        )
        frame = frame.set_index("timestamp").sort_index()

        for col in ("open", "high", "low", "close", "volume"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

        if cache_path is not None:
            frame.to_parquet(cache_path)
            logger.debug("Cached Deribit candles to %s", cache_path)
        return frame
    finally:
        if owns_session:
            request_session.close()


def fetch_dvol(
    currency: str = "BTC",
    *,
    index_name: str = "DVOL",
    resolution: str = "1D",
    start: datetime | str | pd.Timestamp | None = None,
    end: datetime | str | pd.Timestamp | None = None,
    session: Optional[requests.Session] = None,
    days: int = 90,
    max_retries: int = 5,
    retry_delay: float = _DEFAULT_RETRY_DELAY,
    use_cache: bool = True,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Fetch Deribit DVOL (implied volatility index) time series."""

    now = datetime.utcnow()
    start_dt = _ensure_datetime(start, now - timedelta(days=days))
    end_dt = _ensure_datetime(end, now)

    cache_path: Path | None = None
    if use_cache:
        slug = f"dvol|{currency.upper()}|{index_name}|{resolution}|{start_dt.isoformat()}|{end_dt.isoformat()}"
        cache_path = _build_cache_path("dvol", slug, cache_dir)
        if cache_path.exists():
            logger.debug("Loading Deribit DVOL from cache: %s", cache_path)
            return pd.read_parquet(cache_path)

    request_session = session or requests.Session()
    owns_session = session is None
    if owns_session:
        request_session.headers.update(_SESSION_HEADERS)

    params = {
        "currency": currency.upper(),
        "index_name": index_name,
        "resolution": resolution,
        "start_timestamp": int(start_dt.timestamp() * 1000),
        "end_timestamp": int(end_dt.timestamp() * 1000),
    }
    url = f"{settings.deribit_base_url}/{_DVOL_ENDPOINT}"

    try:
        response_json: dict | None = None
        for attempt in range(1, max_retries + 1):
            try:
                response = request_session.get(url, params=params, timeout=_DEFAULT_TIMEOUT)
                response.raise_for_status()
                response_json = response.json()
                break
            except requests.RequestException as exc:
                if attempt == max_retries:
                    raise RuntimeError("Failed to fetch Deribit DVOL") from exc
                sleep_for = retry_delay * attempt
                logger.warning("Deribit DVOL request error: %s. Retry in %.2fs", exc, sleep_for)
                time.sleep(sleep_for)
        if not response_json or "result" not in response_json:
            logger.warning("Deribit DVOL response empty for %s", currency)
            return pd.DataFrame()

        result = response_json["result"]
        data = result.get("data")
        if not data:
            logger.warning("Deribit DVOL returned no data entries for %s", currency)
            return pd.DataFrame()

        if isinstance(data[0], (list, tuple)):
            columns = ["timestamp", "open", "high", "low", "close"][: len(data[0])]
            frame = pd.DataFrame(data, columns=columns)
        else:
            frame = pd.DataFrame(data)

        if "timestamp" not in frame:
            msg = "Deribit DVOL response missing timestamp field"
            raise ValueError(msg)

        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
        frame = frame.set_index("timestamp").sort_index()

        numeric_cols = [col for col in ("open", "high", "low", "close", "value") if col in frame]
        for col in numeric_cols:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

        if "value" not in frame and "close" in frame:
            frame["value"] = frame["close"]

        if cache_path is not None:
            frame.to_parquet(cache_path)
            logger.debug("Cached Deribit DVOL to %s", cache_path)
        return frame
    finally:
        if owns_session:
            request_session.close()
