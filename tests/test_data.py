"""Tests for data fetching utilities using mocked HTTP sessions."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List

import pandas as pd
import pytest
import requests

from src.data.binance import fetch_klines
from src.data.deribit import fetch_candles, fetch_dvol


class MockResponse:
    def __init__(self, payload: Any, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code

    def json(self) -> Any:
        return self.payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class MockSession:
    def __init__(self, responses: List[Any]) -> None:
        self._responses = responses
        self._counter = 0
        self.headers: Dict[str, str] = {}

    def get(self, *_args: Any, **_kwargs: Any) -> MockResponse:
        payload = self._responses[self._counter]
        self._counter = min(self._counter + 1, len(self._responses) - 1)
        return MockResponse(payload)

    def close(self) -> None:
        return None


@pytest.fixture
def monkey_session(monkeypatch: pytest.MonkeyPatch):
    def factory(responses: List[Any]):
        session = MockSession(responses)

        def session_ctor() -> MockSession:
            return session

        monkeypatch.setattr(requests, "Session", session_ctor)
        return session

    return factory


def test_fetch_klines_shape(monkey_session) -> None:
    start = datetime(2023, 1, 1)
    end = start + timedelta(hours=3)
    ms = int(60 * 60 * 1000)
    payload = [
        [int(start.timestamp() * 1000), "1", "2", "0", "3", "10", int(start.timestamp() * 1000) + ms - 1, "0", "1", "0", "0", "0"],
        [int((start + timedelta(hours=1)).timestamp() * 1000), "2", "3", "1", "4", "12", int((start + timedelta(hours=1)).timestamp() * 1000) + ms - 1, "0", "1", "0", "0", "0"],
    ]
    monkey_session([payload])

    df = fetch_klines("BTCUSDT", "1h", start=start, end=end, use_cache=False)
    assert not df.empty
    assert list(df.columns)[:4] == ["open", "high", "low", "close"]
    assert df.index.is_monotonic_increasing


def test_fetch_deribit_candles(monkey_session) -> None:
    start = datetime(2023, 1, 1)
    end = start + timedelta(hours=2)
    ticks = [int(start.timestamp() * 1000), int((start + timedelta(hours=1)).timestamp() * 1000)]
    payload = {
        "result": {
            "ticks": ticks,
            "open": [1.0, 2.0],
            "high": [1.5, 2.5],
            "low": [0.5, 1.5],
            "close": [1.2, 2.2],
            "volume": [100, 110],
        }
    }
    monkey_session([payload])

    df = fetch_candles(
        instrument_name="BTC-PERPETUAL",
        resolution="60",
        start=start,
        end=end,
        use_cache=False,
    )
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 2


def test_fetch_dvol(monkey_session) -> None:
    start = datetime(2023, 1, 1)
    end = start + timedelta(days=1)
    payload = {
        "result": {
            "data": [
                {"timestamp": int(start.timestamp() * 1000), "value": 60.0, "index_name": "DVOL"},
                {"timestamp": int(end.timestamp() * 1000), "value": 62.0, "index_name": "DVOL"},
            ]
        }
    }
    monkey_session([payload])

    df = fetch_dvol(currency="BTC", start=start, end=end, use_cache=False)
    assert "value" in df.columns
    assert not df["value"].isna().any()
