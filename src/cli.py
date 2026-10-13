"""Typer CLI entrypoints for the project."""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd
import typer

from src.backtest.strategies import run_strategy
from src.config import settings
from src.data.binance import fetch_klines
from src.data.deribit import fetch_candles, fetch_dvol
from src.eval.metrics import compute_metrics
from src.features.transform import build_feature_matrix
from src.models.hmm import fit_hmm_model, load_hmm_model, save_hmm, select_hmm
from src.viz.plots import plot_equity, plot_states

app = typer.Typer(help="Regime detection research CLI")


def _default_binance_path(symbol: str, interval: str) -> Path:
    return settings.data_dir / f"binance_{symbol}_{interval}.parquet"


def _default_deribit_path(instrument: str, resolution: str) -> Path:
    return settings.data_dir / f"deribit_{instrument}_{resolution}.parquet"


def _default_dvol_path(currency: str, index_name: str, resolution: str) -> Path:
    return settings.data_dir / f"dvol_{currency}_{index_name}_{resolution}.parquet"


def _latest_states_path() -> Path:
    return settings.data_dir / "latest_states.parquet"


def _latest_backtest_path() -> Path:
    backtest_dir = settings.data_dir / "backtests"
    backtest_dir.mkdir(parents=True, exist_ok=True)
    return backtest_dir / "latest.parquet"


def _latest_metrics_path() -> Path:
    backtest_dir = settings.data_dir / "backtests"
    backtest_dir.mkdir(parents=True, exist_ok=True)
    return backtest_dir / "latest_metrics.json"


@app.command()
def fetch(
    symbol: str = typer.Option("BTCUSDT", help="Trading pair symbol"),
    interval: str = typer.Option("1h", help="Kline interval"),
    days: int = typer.Option(180, help="Trailing days of data to fetch"),
    include_deribit: bool = typer.Option(True, help="Also fetch Deribit candles and DVOL"),
    instrument_name: str = typer.Option("BTC-PERPETUAL", help="Deribit instrument"),
    resolution: str = typer.Option("60", help="Deribit resolution"),
    currency: str = typer.Option("BTC", help="DVOL currency"),
    dvol_index: str = typer.Option(
        settings.deribit_dvol_index, help="Deribit volatility index name"
    ),
    dvol_resolution: str = typer.Option(
        settings.deribit_dvol_resolution, help="DVOL sampling resolution (e.g. 1D)"
    ),
    cache: bool = typer.Option(True, help="Use local cache for API responses"),
):
    """Fetch raw data from Binance and (optionally) Deribit."""

    end = datetime.utcnow()
    start = end - timedelta(days=days)

    typer.echo(f"Fetching Binance {symbol} {interval} from {start} to {end} ...")
    klines = fetch_klines(
        symbol,
        interval,
        start=start,
        end=end,
        use_cache=cache,
    )
    binance_path = _default_binance_path(symbol, interval)
    binance_path.parent.mkdir(parents=True, exist_ok=True)
    klines.to_parquet(binance_path)
    typer.echo(f"Saved Binance data to {binance_path} ({len(klines)} rows)")

    if include_deribit:
        typer.echo(f"Fetching Deribit {instrument_name} {resolution} ...")
        candles = fetch_candles(
            instrument_name=instrument_name,
            resolution=resolution,
            start=start,
            end=end,
            use_cache=cache,
        )
        deribit_path = _default_deribit_path(instrument_name, resolution)
        candles.to_parquet(deribit_path)
        typer.echo(f"Saved Deribit candles to {deribit_path} ({len(candles)} rows)")

        typer.echo(f"Fetching Deribit {dvol_index} for {currency} ...")
        dvol = fetch_dvol(
            currency=currency,
            index_name=dvol_index,
            resolution=dvol_resolution,
            start=start,
            end=end,
            use_cache=cache,
        )
        dvol_path = _default_dvol_path(currency, dvol_index, dvol_resolution)
        dvol.to_parquet(dvol_path)
        typer.echo(f"Saved DVOL to {dvol_path} ({len(dvol)} rows)")


@app.command("fit-hmm")
def fit_hmm(
    n_components: List[int] = typer.Option(
        [3], "-n", "--n-components", help="Candidate number of HMM states"
    ),
    symbol: str = typer.Option("BTCUSDT", help="Symbol of training data"),
    interval: str = typer.Option("1h", help="Interval of training data"),
    data_path: Optional[Path] = typer.Option(
        None, help="Override path for Binance training data"
    ),
    force: bool = typer.Option(False, help="Overwrite existing model"),
):
    """Fit a Gaussian HMM on engineered features."""

    path = data_path or _default_binance_path(symbol, interval)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}. Run fetch first.")

    typer.echo(f"Loading data from {path}")
    df = pd.read_parquet(path)
    feature_set = build_feature_matrix(df)

    candidates = sorted(set(n_components))
    typer.echo(f"Training HMM for candidates: {candidates}")
    if len(candidates) > 1:
        result = select_hmm(
            feature_set.features,
            candidates,
            scaler=feature_set.scaler,
        )
    else:
        result = fit_hmm_model(
            feature_set.features,
            n_components=candidates[0],
            scaler=feature_set.scaler,
        )

    model_path = save_hmm(result, force=force)
    typer.echo(f"Saved HMM model to {model_path}")
    typer.echo(f"AIC: {result.aic:.2f} | BIC: {result.bic:.2f}")

    states_path = settings.data_dir / f"hmm_states_{symbol}_{interval}_{result.model.n_components}c.parquet"
    result.states.to_frame().to_parquet(states_path)
    result.posteriors.to_parquet(
        settings.data_dir
        / f"hmm_posteriors_{symbol}_{interval}_{result.model.n_components}c.parquet"
    )
    result.states.to_frame().to_parquet(_latest_states_path())
    typer.echo(f"Persisted state sequence to {states_path}")


@app.command()
def backtest(
    strategy: str = typer.Option("spot_regime", help="Strategy name"),
    symbol: str = typer.Option("BTCUSDT", help="Symbol to backtest"),
    interval: str = typer.Option("1h", help="Interval to backtest"),
    data_path: Optional[Path] = typer.Option(None, help="Override price data path"),
    model_path: Optional[Path] = typer.Option(None, help="Path to trained HMM joblib"),
    dvol_path: Optional[Path] = typer.Option(None, help="Path to DVOL data"),
    dvol_index: str = typer.Option(
        settings.deribit_dvol_index, help="Deribit volatility index name"
    ),
    dvol_resolution: str = typer.Option(
        settings.deribit_dvol_resolution, help="DVOL resolution used for lookup"
    ),
    start: Optional[datetime] = typer.Option(None, help="Backtest start timestamp"),
    end: Optional[datetime] = typer.Option(None, help="Backtest end timestamp"),
    fee_bps: float = typer.Option(2.0, help="Round-trip fee in basis points"),
):
    """Run backtests for a given strategy."""

    price_path = data_path or _default_binance_path(symbol, interval)
    if not price_path.exists():
        raise FileNotFoundError(f"Price data not found: {price_path}")

    typer.echo(f"Loading price data from {price_path}")
    price_df = pd.read_parquet(price_path)
    price_df = price_df.sort_index()
    if start:
        price_df = price_df[price_df.index >= pd.to_datetime(start, utc=True)]
    if end:
        price_df = price_df[price_df.index <= pd.to_datetime(end, utc=True)]

    bundle = load_hmm_model(model_path)
    model = bundle["model"]
    scaler = bundle["scaler"]

    feature_set = build_feature_matrix(price_df, scaler=scaler)
    states_array = model.predict(feature_set.features.values)
    states = pd.Series(states_array, index=feature_set.features.index, name="state")

    dvol_series = None
    target_dvol_path = dvol_path or _default_dvol_path("BTC", dvol_index, dvol_resolution)
    if target_dvol_path.exists():
        dvol_df = pd.read_parquet(target_dvol_path)
        if not dvol_df.empty and "value" in dvol_df:
            dvol_series = dvol_df["value"]

    price_series = price_df["close"].reindex(feature_set.features.index).ffill()
    returns = feature_set.raw["log_return"]

    result = run_strategy(
        strategy,
        price=price_series,
        returns=returns,
        states=states,
        fee_bps=fee_bps,
        dvol=dvol_series,
    )

    metrics = compute_metrics(result.equity_curve, result.returns)
    typer.echo(json.dumps(metrics, indent=2))

    backtest_dir = settings.data_dir / "backtests"
    backtest_dir.mkdir(parents=True, exist_ok=True)
    out_path = backtest_dir / f"{strategy}_{symbol}_{interval}.parquet"
    output_df = pd.DataFrame(
        {
            "strategy_return": result.returns,
            "equity": result.equity_curve,
            "benchmark": result.benchmark,
            "weight": result.weights,
        }
    )
    output_df.to_parquet(out_path)
    output_df.to_parquet(_latest_backtest_path())
    metrics_path = backtest_dir / f"{strategy}_{symbol}_{interval}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    _latest_metrics_path().write_text(json.dumps(metrics, indent=2))
    typer.echo(f"Saved backtest results to {out_path}")

    states.to_frame().to_parquet(_latest_states_path())


@app.command()
def report(
    symbol: str = typer.Option("BTCUSDT", help="Symbol for report"),
    interval: str = typer.Option("1h", help="Interval for report"),
    backtest_path: Optional[Path] = typer.Option(None, help="Backtest results path"),
    states_path: Optional[Path] = typer.Option(None, help="States parquet path"),
    price_path: Optional[Path] = typer.Option(None, help="Price data path"),
    output: Path = typer.Option(settings.reports_dir / "report.html", help="Report output path"),
):
    """Generate a simple HTML report with plots and metrics."""

    bt_path = backtest_path or _latest_backtest_path()
    st_path = states_path or _latest_states_path()
    px_path = price_path or _default_binance_path(symbol, interval)

    if not bt_path.exists():
        raise FileNotFoundError(f"Backtest data not found: {bt_path}")
    if not st_path.exists():
        raise FileNotFoundError(f"States data not found: {st_path}")
    if not px_path.exists():
        raise FileNotFoundError(f"Price data not found: {px_path}")

    backtest_df = pd.read_parquet(bt_path)
    states_df = pd.read_parquet(st_path)
    price_df = pd.read_parquet(px_path)

    metrics_path = _latest_metrics_path()
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

    equity_fig_path = settings.reports_dir / "equity_curve.png"
    states_fig_path = settings.reports_dir / "states.png"

    plot_equity(backtest_df["equity"], backtest_df["benchmark"], output=equity_fig_path)
    plot_states(price_df["close"], states_df["state"], output=states_fig_path)

    output.parent.mkdir(parents=True, exist_ok=True)
    html = f"""
    <html>
    <head><title>HMM Regime Report</title></head>
    <body>
        <h1>Regime Backtest Report</h1>
        <h2>Metrics</h2>
        <pre>{json.dumps(metrics, indent=2)}</pre>
        <h2>Equity Curve</h2>
        <img src="{os.path.relpath(equity_fig_path.resolve(), output.parent.resolve())}" alt="Equity Curve" width="800" />
        <h2>States Overlay</h2>
        <img src="{os.path.relpath(states_fig_path.resolve(), output.parent.resolve())}" alt="States" width="800" />
    </body>
    </html>
    """
    output.write_text(html)
    typer.echo(f"Report written to {output}")


if __name__ == "__main__":
    app()
