# hmm-regimes-crypto

Regime-aware volatility and options overlay research using only free crypto market data. The project fetches Binance spot candles and Deribit implied volatility series, engineers features, and compares Hidden Markov Model regimes against classical volatility baselines.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[dev]
```

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
