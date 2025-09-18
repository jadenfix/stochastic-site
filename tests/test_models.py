"""Tests for model fitting helpers."""
from __future__ import annotations

import numpy as np

import pandas as pd

from src.features.transform import build_feature_matrix
from src.models.hmm import fit_hmm_model


def test_hmm_state_count_matches_configuration() -> None:
    index = pd.date_range("2023-01-01", periods=300, freq="h")
    rng = np.random.default_rng(0)
    price = 20000 + np.cumsum(rng.normal(0, 10, size=len(index)))
    volume = 1000 + rng.normal(0, 5, size=len(index))
    df = pd.DataFrame({"close": price, "volume": volume}, index=index)

    feature_set = build_feature_matrix(df)
    result = fit_hmm_model(feature_set.features, n_components=3, scaler=feature_set.scaler)

    assert result.states.nunique() == 3
    assert result.posteriors.shape[1] == 3
