"""Offline command-line workflow and persisted model regression checks."""
import json

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from src.cli import app
from src.config import settings
from src.models.hmm import load_hmm_model


def test_offline_fit_backtest_report(tmp_path, monkeypatch):
    for key in ("data_dir", "models_dir", "reports_dir"):
        monkeypatch.setattr(settings, key, tmp_path / key)
    settings.data_dir.mkdir()
    rng = np.random.default_rng(7)
    index = pd.date_range("2024-01-01", periods=500, freq="h", tz="UTC")
    prices = pd.DataFrame({"close": 20000 * np.exp(np.cumsum(rng.normal(0, .002, 500))),
                           "volume": rng.lognormal(7, .2, 500)}, index=index)
    prices.to_parquet(settings.data_dir / "binance_BTCUSDT_1h.parquet")
    runner = CliRunner()
    for args in (["fit-hmm", "-n", "2"], ["backtest"],
                 ["report", "--output", str(settings.reports_dir / "report.html")]):
        result = runner.invoke(app, args)
        assert result.exit_code == 0, (result.output, result.exception)
    bundle = load_hmm_model()
    assert bundle["model"].n_components == 2
    assert bundle["scaler"].n_features_in_ == 4
    metrics = json.loads((settings.data_dir / "backtests/latest_metrics.json").read_text())
    assert metrics
    report = settings.reports_dir / "report.html"
    assert 'src="equity_curve.png"' in report.read_text()
    assert 'src="states.png"' in report.read_text()
    assert (report.parent / "equity_curve.png").is_file()
    second_fit = runner.invoke(app, ["fit-hmm", "-n", "2"])
    assert isinstance(second_fit.exception, FileExistsError)
