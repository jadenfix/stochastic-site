import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from src.backtest.strategies import _apply_weights
from src.backtest.strategies import _classify_states
from src.eval.metrics import compute_metrics
from src.features.transform import build_feature_matrix, compute_rsi
from src.models.hmm import filter_states, fit_hmm_model, load_hmm_model, save_hmm


def fixture_features():
    rng = np.random.default_rng(91)
    index = pd.date_range("2026-01-01", periods=220, freq="h", tz="UTC")
    frame = pd.DataFrame({"close": 20000 * np.exp(np.cumsum(rng.normal(0, .008, len(index)))),
                          "volume": rng.uniform(200, 500, len(index))}, index=index)
    return build_feature_matrix(frame)


def test_model_round_trip_and_overwrite_guard(tmp_path):
    features = fixture_features()
    model = fit_hmm_model(features.features, n_components=2, scaler=features.scaler)
    path = save_hmm(model, path=tmp_path / "model.joblib")
    loaded = load_hmm_model(path)
    states, probabilities = filter_states(loaded["model"], features.features)
    pd.testing.assert_series_equal(states, model.states)
    assert np.allclose(probabilities.sum(axis=1), 1)
    assert loaded["training_end"] == features.features.index[-1]
    assert loaded["state_labels"] == model.state_labels
    with pytest.raises(FileExistsError):
        save_hmm(model, path=path)


def test_filter_prefix_is_invariant_to_future_observations():
    features = fixture_features()
    model = fit_hmm_model(features.features.iloc[:100], n_components=2, scaler=features.scaler)
    _, prefix = filter_states(model.model, features.features.iloc[:120])
    mutated = features.features.copy()
    mutated.iloc[120:] = 100
    _, full = filter_states(model.model, mutated)
    assert np.allclose(prefix, full.iloc[:120])


def test_returns_execute_later_and_convert_log_returns_before_compounding():
    index = pd.RangeIndex(4)
    returns = pd.Series(np.log([1.1, 1.2, .9, 1.0]), index=index)
    weights = pd.Series([1., 1., 0., 0.], index=index)
    actual, equity = _apply_weights(returns, weights, fee_bps=10)
    assert np.allclose(actual, [0, .199, -.1, -.001])
    assert np.isclose(equity.iloc[-1], 1.199 * .9 * .999)


def test_rsi_monotonic_and_flat_cases():
    assert compute_rsi(pd.Series(range(1, 40))).iloc[-1] == 100
    assert compute_rsi(pd.Series(range(40, 1, -1))).iloc[-1] == 0
    assert compute_rsi(pd.Series([10] * 40)).iloc[-1] == 50


def test_state_roles_do_not_overwrite_bullish_state_with_volatility():
    states = pd.Series([0, 0, 1, 1, 2, 2])
    returns = pd.Series([-.1, .9, -.1, -.2, .02, -.02])
    labels = _classify_states(states, returns)
    assert labels == {0: "calm_up", 1: "bearish", 2: "high_vol"}


def test_metric_json_is_finite_and_drawdown_includes_initial_capital():
    metrics = compute_metrics(pd.Series([.9, .9]), pd.Series([-.1, 0.]))
    assert np.isclose(metrics["MaxDrawdown"], -.1)
    flat = compute_metrics(pd.Series([1., 1.]), pd.Series([0., 0.]))
    assert flat["Sharpe"] is None
    json.dumps(flat, allow_nan=False)


def test_cli_demo_runs_in_isolated_workspace_without_network(tmp_path):
    import os
    environment = dict(os.environ, HMM_DATA_DIR=str(tmp_path / "data"),
                       HMM_MODELS_DIR=str(tmp_path / "models"), HMM_REPORTS_DIR=str(tmp_path / "reports"),
                       MPLBACKEND="Agg", MPLCONFIGDIR=str(tmp_path / "mpl"),
                       OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    result = subprocess.run([sys.executable, "-m", "src.cli", "demo"], env=environment,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    report = (tmp_path / "reports/demo.html").read_text()
    assert "Synthetic offline demonstration" in report
    assert report.count("data:image/png;base64,") == 2
    metadata = json.loads((tmp_path / "data/backtests/latest_provenance.json").read_text())
    assert metadata["evaluation_start"] > metadata["training_end"]
    assert metadata["synthetic"] is True
