"""Gaussian HMM fitting, information-criterion selection and persistence."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from src.config import settings


@dataclass
class HMMResult:
    model: GaussianHMM
    scaler: StandardScaler
    states: pd.Series
    posteriors: pd.DataFrame
    aic: float
    bic: float


def fit_hmm_model(
    features: pd.DataFrame,
    *,
    n_components: int = 3,
    scaler: StandardScaler,
    random_state: int = 42,
    n_iter: int = 200,
) -> HMMResult:
    """Fit already-standardized features, preserving their timestamps."""
    values = features.to_numpy(dtype=float)
    if n_components < 1 or len(features) < n_components:
        raise ValueError("Require at least one state and one observation per state.")
    if values.ndim != 2 or values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError("Features must be a finite, nonempty numeric matrix.")
    model = GaussianHMM(
        n_components=n_components,
        covariance_type="diag",
        n_iter=n_iter,
        random_state=random_state,
        min_covar=1e-3,
    ).fit(values)
    return HMMResult(
        model=model,
        scaler=scaler,
        states=pd.Series(model.predict(values), index=features.index, name="state"),
        posteriors=pd.DataFrame(model.predict_proba(values), index=features.index),
        aic=float(model.aic(values)),
        bic=float(model.bic(values)),
    )


def select_hmm(
    features: pd.DataFrame,
    candidates: Iterable[int],
    *,
    scaler: StandardScaler,
) -> HMMResult:
    """Choose the lowest-BIC candidate on the supplied training sample."""
    candidates = sorted(set(candidates))
    if not candidates:
        raise ValueError("At least one candidate state count is required.")
    results = [fit_hmm_model(features, n_components=n, scaler=scaler) for n in candidates]
    return min(results, key=lambda result: result.bic)


def save_hmm(result: HMMResult, path: Path | None = None, *, force: bool = False) -> Path:
    """Save a local model bundle; refuse to overwrite unless requested."""
    target = Path(path) if path is not None else settings.models_dir / "hmm.joblib"
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = "wb" if force else "xb"
    with target.open(mode) as stream:
        joblib.dump({"model": result.model, "scaler": result.scaler,
                     "aic": result.aic, "bic": result.bic}, stream)
    return target


def load_hmm_model(path: Path | None = None) -> dict:
    """Load a trusted local joblib bundle; never load untrusted model files."""
    target = Path(path) if path is not None else settings.models_dir / "hmm.joblib"
    bundle = joblib.load(target)
    if not isinstance(bundle, dict) or not isinstance(bundle.get("model"), GaussianHMM) or not isinstance(bundle.get("scaler"), StandardScaler):
        raise ValueError("Expected a GaussianHMM and fitted scaler bundle.")
    return bundle
