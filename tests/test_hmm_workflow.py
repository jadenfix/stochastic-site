"""Offline fitted-model and CLI workflow regression tests."""
from pathlib import Path
import re
import joblib

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from src.cli import app
from src.config import settings
from src.features.transform import build_feature_matrix
from src.models.hmm import fit_hmm_model, load_hmm_model, save_hmm, select_hmm


def prices():
    rng = np.random.default_rng(12)
    index = pd.date_range("2024-01-01", periods=240, freq="h", tz="UTC")
    return pd.DataFrame({"close": 20000 * np.exp(np.cumsum(rng.normal(0, .003, len(index)))),
                         "volume": rng.uniform(900, 1200, len(index))}, index=index)


def test_model_round_trip_and_overwrite_guard(tmp_path, monkeypatch):
    features = build_feature_matrix(prices())
    fitted = fit_hmm_model(features.features, n_components=2, scaler=features.scaler)
    assert np.allclose(fitted.posteriors.sum(axis=1), 1)
    assert fitted.states.index.equals(features.features.index)
    monkeypatch.setattr(settings, "models_dir", tmp_path / "nested")
    path = save_hmm(fitted)
    restored = load_hmm_model(path)
    assert np.array_equal(restored["model"].predict(features.features.values), fitted.states.values)
    with pytest.raises(FileExistsError):
        save_hmm(fitted)
    save_hmm(fitted, force=True)


def test_selection_and_invalid_input():
    features = build_feature_matrix(prices())
    result = select_hmm(features.features, [2, 2], scaler=features.scaler)
    assert result.model.n_components == 2
    with pytest.raises(ValueError, match="candidate"):
        select_hmm(features.features, [], scaler=features.scaler)
    invalid = features.features.copy()
    invalid.iloc[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        fit_hmm_model(invalid, scaler=features.scaler)


def test_load_rejects_invalid_bundle_types(tmp_path):
    path = tmp_path / "invalid.joblib"
    joblib.dump({"model": "placeholder", "scaler": "placeholder"}, path)
    with pytest.raises(ValueError, match="GaussianHMM"):
        load_hmm_model(path)


def test_offline_cli_fit_backtest_report(tmp_path, monkeypatch):
    for name in ("data_dir", "cache_dir", "models_dir", "reports_dir"):
        target = tmp_path / name
        target.mkdir()
        monkeypatch.setattr(settings, name, target)
    data = settings.data_dir / "binance_BTCUSDT_1h.parquet"
    prices().to_parquet(data)
    runner = CliRunner()
    report = tmp_path / "custom-report" / "report.html"
    for command in (["fit-hmm", "--n-components", "2"],
                    ["backtest", "--strategy", "spot_regime"],
                    ["report", "--output", str(report)]):
        result = runner.invoke(app, command)
        assert result.exit_code == 0, (result.output, result.exception)
    backtest = pd.read_parquet(settings.data_dir / "backtests" / "latest.parquet")
    assert np.isfinite(backtest["equity"]).all()
    for image in re.findall(r'<img src="([^"]+)"', report.read_text()):
        assert (report.parent / image).is_file()
