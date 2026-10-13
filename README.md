# Regime Research

Regime-aware volatility and options overlay research using only free crypto market data. The project fetches Binance spot candles and Deribit implied volatility series, engineers features, and compares Hidden Markov Model regimes against classical volatility baselines.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[dev]
```

## Offline end-to-end demo

```sh
MPLBACKEND=Agg python -m src.cli demo
```

This uses generated, seeded prices, trains on the first 70% of observations, evaluates only the remaining chronological holdout, and writes `reports/demo.html`. Charts are embedded, so the report works after being moved. No market API or order is called. These results demonstrate the software; they are not trading-performance evidence.

## Market-data workflow

Then:

```bash
python -m src.cli fetch --symbol BTCUSDT --interval 1h --days 30 --dvol-index DVOL --dvol-resolution 1D
python -m src.cli fit-hmm --n-components 3
python -m src.cli backtest --strategy spot_regime
MPLCONFIGDIR=$(pwd)/.cache/mpl MPLBACKEND=Agg python -m src.cli report
```

Artifacts are stored under `artifacts/`, fitted models under `models/`, and generated plots/reports under `reports/`.

Set `HMM_DERIBIT_DVOL_INDEX`/`HMM_DERIBIT_DVOL_RESOLUTION` to override defaults, or pass `--dvol-index`/`--dvol-resolution` to the fetch/backtest commands. If Matplotlib warns about cache permissions, export `MPLCONFIGDIR=$(pwd)/.cache/mpl` before running plotting commands.

## Caveats

- Educational research only; not investment advice.
- Public APIs may rate-limit. Fetchers include basic retries, but you may need to throttle further for larger pulls.
- Deribit DVOL and options overlays are proxies; no live trading connectivity is provided.

## Evaluation semantics

`fit-hmm --train-fraction 0.7` fits the scaler and Gaussian HMM on the training prefix only. The bundle records the training endpoint and regime labels. Backtesting excludes training timestamps and uses forward-filtered state probabilities, which do not revise past labels using future observations. Weights execute one bar later, starting flat; fees are charged on executed position changes. Log returns are converted to simple returns before equity compounding. The DVOL overlay is a diagnostic signal; option P&L is not implemented.

`run_strategy` without supplied training labels remains a retrospective exploration helper. Use the CLI's saved-model path for held-out evaluation. Train and evaluate on the same instrument and interval. Load only trusted locally generated joblib model files.

The `src/models` implementation is versioned; only root `/models/` generated artifacts are ignored. Dependency installation and Python 3.11/3.12 checks are included in CI.

## Validation

```sh
python -m pytest tests -q
```

Regression checks cover model persistence, overwrite protection, causal prefix invariance, position timing and fees, log-return compounding, and the public CLI's complete offline workflow. Live market-source tests use controlled response fixtures; this does not establish current exchange availability.
