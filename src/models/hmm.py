"""Gaussian regime fitting, model selection, and persistence."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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


def fit_hmm_model(features: pd.DataFrame, *, n_components: int = 3,
                  scaler: StandardScaler, random_state: int = 42) -> HMMResult:
    if features.empty or not np.isfinite(features.to_numpy()).all():
        raise ValueError("HMM features must be nonempty and finite")
    if not 1 <= n_components <= len(features):
        raise ValueError("State count must be between 1 and number of observations")
    model = GaussianHMM(n_components=n_components, covariance_type="full",
                        n_iter=200, random_state=random_state)
    values = features.to_numpy()
    model.fit(values)
    states = pd.Series(model.predict(values), index=features.index, name="state")
    posteriors = pd.DataFrame(model.predict_proba(values), index=features.index,
                              columns=range(n_components))
    return HMMResult(model, scaler, states, posteriors, model.aic(values), model.bic(values))


def select_hmm(features: pd.DataFrame, candidates: list[int], *,
               scaler: StandardScaler) -> HMMResult:
    if not candidates:
        raise ValueError("Provide at least one candidate state count")
    results = [fit_hmm_model(features, n_components=n, scaler=scaler)
               for n in sorted(set(candidates))]
    return min(results, key=lambda result: result.bic)


def save_hmm(result: HMMResult, *, force: bool = False) -> Path:
    path = settings.models_dir / "hmm.joblib"
    if path.exists() and not force:
        raise FileExistsError(f"Model already exists: {path}; pass --force to overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": result.model, "scaler": result.scaler,
                 "feature_columns": list(result.model.feature_names_in_)
                 if hasattr(result.model, "feature_names_in_") else list(result.scaler.get_feature_names_out()),
                 "aic": result.aic, "bic": result.bic}, path)
    return path


def load_hmm_model(path: Path | None = None) -> dict:
    """Load a trusted local joblib file; never load untrusted model artifacts."""
    target = path or settings.models_dir / "hmm.joblib"
    if not target.exists():
        raise FileNotFoundError(f"Model not found: {target}. Run fit-hmm first.")
    bundle = joblib.load(target)
    if not isinstance(bundle, dict) or not {"model", "scaler"}.issubset(bundle):
        raise ValueError("Model artifact is missing model or scaler")
    return bundle
