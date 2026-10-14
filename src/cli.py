"""Typer CLI entrypoints for the project."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import numpy as np
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
from src.models.hmm import filter_states, fit_hmm_model, load_hmm_model, save_hmm, select_hmm
from src.viz.plots import plot_equity, plot_states
from src.viz.report import write_report

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


def _ensure_report_outputs_do_not_alias_inputs(
    inputs: list[Path], outputs: list[Path]
) -> None:
    """Refuse report targets that resolve to an input artifact."""
    for output in outputs:
        for input_path in inputs:
            same_target = output.resolve(strict=False) == input_path.resolve(
                strict=False
            )
            same_inode = (
                output.exists()
                and input_path.exists()
                and os.path.samefile(output, input_path)
            )
            if same_target or same_inode:
                raise ValueError(
                    f"Report output must not overwrite an input artifact: {output}"
                )


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
    train_fraction: float = typer.Option(0.7, min=0.1, max=0.9, help="Chronological training fraction; the rest is held out"),
):
    """Fit a Gaussian HMM on engineered features."""

    path = data_path or _default_binance_path(symbol, interval)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}. Run fetch first.")

    typer.echo(f"Loading data from {path}")
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index, utc=True)
    training = df.sort_index().iloc[:int(len(df) * train_fraction)]
    feature_set = build_feature_matrix(training)

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
    fee_bps: float = typer.Option(2.0, min=0, help="Fee per unit of executed turnover, in basis points"),
):
    """Run backtests for a given strategy."""

    price_path = data_path or _default_binance_path(symbol, interval)
    if not price_path.exists():
        raise FileNotFoundError(f"Price data not found: {price_path}")

    typer.echo(f"Loading price data from {price_path}")
    price_df = pd.read_parquet(price_path)
    price_df.index = pd.to_datetime(price_df.index, utc=True)
    price_df = price_df.sort_index()

    bundle = load_hmm_model(model_path)
    model = bundle["model"]
    scaler = bundle["scaler"]

    feature_set = build_feature_matrix(price_df, scaler=scaler)
    states, _ = filter_states(model, feature_set.features)
    evaluation_index = feature_set.features.index[feature_set.features.index > bundle["training_end"]]
    if start:
        evaluation_index = evaluation_index[evaluation_index >= pd.to_datetime(start, utc=True)]
    if end:
        evaluation_index = evaluation_index[evaluation_index <= pd.to_datetime(end, utc=True)]
    if len(evaluation_index) < 2:
        raise ValueError("At least two held-out rows after training_end are required")
    states = states.reindex(evaluation_index)

    dvol_series = None
    target_dvol_path = dvol_path or _default_dvol_path("BTC", dvol_index, dvol_resolution)
    if target_dvol_path.exists():
        dvol_df = pd.read_parquet(target_dvol_path)
        if not dvol_df.empty and "value" in dvol_df:
            dvol_series = dvol_df["value"]

    price_series = price_df["close"].reindex(evaluation_index)
    returns = feature_set.raw["log_return"].reindex(evaluation_index)

    result = run_strategy(
        strategy,
        price=price_series,
        returns=returns,
        states=states,
        fee_bps=fee_bps,
        dvol=dvol_series,
        state_labels=bundle["state_labels"],
    )

    seconds_per_bar = evaluation_index.to_series().diff().dt.total_seconds().dropna().median()
    periods_per_year = max(1, round(365 * 24 * 3600 / seconds_per_bar))
    metrics = compute_metrics(result.equity_curve, result.returns, periods_per_year=periods_per_year)
    provenance = {
        "symbol": symbol, "synthetic": symbol == "DEMO",
        "training_end": str(bundle["training_end"]),
        "evaluation_start": str(evaluation_index[0]),
        "evaluation_end": str(evaluation_index[-1]),
        "evaluation_rows": len(evaluation_index), "state_count": model.n_components,
        "inference": "causal forward filtering", "labels": "training prefix only",
        "position_lag_bars": 1, "fee_bps_per_turnover": fee_bps,
        "periods_per_year": periods_per_year,
        "options_pnl": "not implemented; overlay is a diagnostic signal only",
    }
    (_latest_metrics_path().parent / "latest_provenance.json").write_text(json.dumps(provenance, indent=2))
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
    provenance_path = settings.data_dir / "backtests/latest_provenance.json"

    equity_fig_path = settings.reports_dir / "equity_curve.png"
    states_fig_path = settings.reports_dir / "states.png"
    _ensure_report_outputs_do_not_alias_inputs(
        [bt_path, st_path, px_path, metrics_path, provenance_path],
        [output, equity_fig_path, states_fig_path],
    )

    plot_equity(backtest_df["equity"], backtest_df["benchmark"], output=equity_fig_path)
    plot_states(price_df["close"], states_df["state"], output=states_fig_path)

    output.parent.mkdir(parents=True, exist_ok=True)
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    write_report(output, metrics, [("Strategy and buy-and-hold", equity_fig_path),
                                  ("Price and inferred regimes", states_fig_path)], provenance)
    typer.echo(f"Report written to {output}")


@app.command()
def demo():
    """Run deterministic generated prices through training, holdout backtest, and report."""
    index = pd.date_range("2026-01-01", periods=720, freq="h", tz="UTC")
    rng = np.random.default_rng(2026)
    regime = np.repeat([0, 1, 2] * 4, 60)
    log_returns = np.array([0.001, -0.001, 0.0])[regime] + rng.normal(size=len(index)) * np.array([.002, .004, .010])[regime]
    close = 30000 * np.exp(np.cumsum(log_returns))
    frame = pd.DataFrame({"open": close, "high": close * 1.002, "low": close * .998,
                          "close": close, "volume": rng.uniform(100, 1000, len(index))}, index=index)
    path = _default_binance_path("DEMO", "1h")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    typer.echo("SYNTHETIC OFFLINE DEMO: generated data; no exchange requests or orders.")
    for command in (["fit-hmm", "--symbol", "DEMO", "--force"],
                    ["backtest", "--symbol", "DEMO"],
                    ["report", "--symbol", "DEMO", "--output", str(settings.reports_dir / "demo.html")]):
        subprocess.run([sys.executable, "-m", "src.cli", *command], check=True)


if __name__ == "__main__":
    app()
