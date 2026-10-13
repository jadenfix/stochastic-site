"""Gaussian HMM fitting and causal filtering for chronological evaluation."""
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler

from src.config import settings


@dataclass
class HMMResult:
    model: GaussianHMM
    scaler: object
    states: pd.Series
    posteriors: pd.DataFrame
    aic: float
    bic: float
    training_end: pd.Timestamp
    state_labels: dict


def filter_states(model, features):
    """P(state_t | observations through t), never smoothing from future rows."""
    values = np.asarray(features, dtype=float)
    if values.ndim != 2 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Features must be a nonempty finite matrix")
    emissions = np.column_stack([
        multivariate_normal.logpdf(values, mean=model.means_[state],
                                   cov=model.covars_[state], allow_singular=True)
        for state in range(model.n_components)
    ])
    with np.errstate(divide="ignore"):
        transitions = np.log(model.transmat_)
        previous = np.log(model.startprob_)
    probabilities = np.empty_like(emissions)
    for row in range(len(values)):
        prior = previous if row == 0 else logsumexp(previous[:, None] + transitions, axis=0)
        current = prior + emissions[row]
        normalizer = logsumexp(current)
        if not np.isfinite(normalizer):
            raise ValueError("No finite HMM state likelihood for an observation")
        previous = current - normalizer
        probabilities[row] = np.exp(previous)
    index = features.index if hasattr(features, "index") else pd.RangeIndex(len(values))
    states = pd.Series(probabilities.argmax(axis=1), index=index, name="state")
    posteriors = pd.DataFrame(probabilities, index=index, columns=range(model.n_components))
    return states, posteriors


def fit_hmm_model(features, n_components=3, scaler=None, random_state=42):
    from src.backtest.strategies import _classify_states
    if n_components < 1 or len(features) < max(20, n_components * 5):
        raise ValueError("Insufficient training observations for the requested state count")
    values = features.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Training features must be finite")
    model = GaussianHMM(n_components=n_components, covariance_type="diag",
                        n_iter=200, random_state=random_state, min_covar=1e-4)
    model.fit(values)
    states, posteriors = filter_states(model, features)
    raw = scaler.inverse_transform(values) if scaler is not None else values
    training_returns = pd.Series(raw[:, 0], index=features.index)
    labels = _classify_states(states, training_returns)
    return HMMResult(model, scaler, states, posteriors, float(model.aic(values)),
                     float(model.bic(values)), features.index[-1], labels)


def select_hmm(features, candidates, scaler=None):
    candidates = sorted(set(candidates))
    if not candidates:
        raise ValueError("At least one state-count candidate is required")
    return min((fit_hmm_model(features, n, scaler) for n in candidates), key=lambda result: result.bic)


def save_hmm(result, force=False, path=None):
    path = Path(path) if path is not None else settings.models_dir / f"hmm_{result.model.n_components}c.joblib"
    if path.exists() and not force:
        raise FileExistsError(f"Model already exists: {path}. Pass --force to overwrite.")
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {"model": result.model, "scaler": result.scaler,
                 "training_end": result.training_end, "state_labels": result.state_labels,
                 "aic": result.aic, "bic": result.bic}
    with path.open("wb" if force else "xb") as file:
        joblib.dump(bundle, file)
    return path


def load_hmm_model(path=None):
    if path is None:
        candidates = list(settings.models_dir.glob("hmm_*c.joblib"))
        if not candidates:
            raise FileNotFoundError("No trained model. Run fit-hmm first.")
        path = max(candidates, key=lambda candidate: candidate.stat().st_mtime)
    # Model bundles are executable pickle data. Load only locally generated/trusted files.
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or not isinstance(bundle.get("model"), GaussianHMM):
        raise ValueError("Model bundle must contain a fitted GaussianHMM")
    if not isinstance(bundle.get("scaler"), StandardScaler):
        raise ValueError("Model bundle must contain the fitted training StandardScaler")
    if any(key not in bundle for key in ("model", "scaler", "training_end", "state_labels")):
        raise ValueError("Model bundle lacks training metadata; retrain with fit-hmm")
    return bundle
