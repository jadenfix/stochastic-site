"""Feature engineering unit tests."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.transform import FEATURE_COLUMNS, build_feature_matrix


def test_feature_matrix_has_no_nans() -> None:
    index = pd.date_range("2023-01-01", periods=200, freq="h")
    prices = pd.Series(np.linspace(20000, 25000, len(index)) + np.sin(np.linspace(0, 20, len(index))), index=index)
    volume = pd.Series(1000 + 10 * np.cos(np.linspace(0, 10, len(index))), index=index)

    df = pd.DataFrame({"close": prices, "volume": volume})
    feature_set = build_feature_matrix(df)

    assert list(feature_set.features.columns) == FEATURE_COLUMNS
    assert not feature_set.features.isna().any().any()
    assert feature_set.features.index.equals(feature_set.raw.index)
