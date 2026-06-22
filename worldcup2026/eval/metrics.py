"""Scoring metrics for 1X2 probabilistic forecasts.

Convention: probability arrays are shape (n, 3), columns ordered
[P(home), P(draw), P(away)]. ``outcomes`` is an int array with
0=home win, 1=draw, 2=away win (same ordering).

Primary metric is RPS (Ranked Probability Score) — appropriate because the
1X2 outcome is ORDERED (home > draw > away on a home-favorability axis).
Secondary: Brier, log-loss, plus a calibration (reliability) table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OUTCOME_ORDER = ("home", "draw", "away")


def _one_hot(outcomes: np.ndarray, n_classes: int = 3) -> np.ndarray:
    o = np.zeros((len(outcomes), n_classes), dtype=float)
    o[np.arange(len(outcomes)), np.asarray(outcomes, dtype=int)] = 1.0
    return o


def rps_vec(probs: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    """Per-match RPS. probs ordered [home, draw, away]."""
    probs = np.asarray(probs, dtype=float)
    o = _one_hot(outcomes, probs.shape[1])
    cum_p = np.cumsum(probs, axis=1)
    cum_o = np.cumsum(o, axis=1)
    r = probs.shape[1]
    # sum of squared cumulative diffs over the first r-1 categories, normalized
    return np.sum((cum_p[:, : r - 1] - cum_o[:, : r - 1]) ** 2, axis=1) / (r - 1)


def rps(probs: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean(rps_vec(probs, outcomes)))


def brier(probs: np.ndarray, outcomes: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=float)
    o = _one_hot(outcomes, probs.shape[1])
    return float(np.mean(np.sum((probs - o) ** 2, axis=1)))


def log_loss(probs: np.ndarray, outcomes: np.ndarray, eps: float = 1e-15) -> float:
    probs = np.clip(np.asarray(probs, dtype=float), eps, 1.0)
    idx = np.asarray(outcomes, dtype=int)
    picked = probs[np.arange(len(probs)), idx]
    return float(-np.mean(np.log(picked)))


def accuracy(probs: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean(np.argmax(np.asarray(probs), axis=1) == np.asarray(outcomes)))


def calibration_table(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Pooled reliability table over all (predicted_prob, class-occurred) pairs."""
    probs = np.asarray(probs, dtype=float)
    o = _one_hot(outcomes, probs.shape[1])
    p_flat = probs.ravel()
    o_flat = o.ravel()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p_flat, bins) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        rows.append(
            {
                "bin": f"[{bins[b]:.1f},{bins[b+1]:.1f})",
                "n": int(m.sum()),
                "mean_pred": float(p_flat[m].mean()),
                "mean_obs": float(o_flat[m].mean()),
            }
        )
    return pd.DataFrame(rows)


def score_all(probs: np.ndarray, outcomes: np.ndarray) -> dict[str, float]:
    return {
        "rps": rps(probs, outcomes),
        "brier": brier(probs, outcomes),
        "logloss": log_loss(probs, outcomes),
        "accuracy": accuracy(probs, outcomes),
        "n": int(len(outcomes)),
    }


def outcomes_from_goals(home_goals, away_goals) -> np.ndarray:
    """0=home win, 1=draw, 2=away win."""
    hg = np.asarray(home_goals)
    ag = np.asarray(away_goals)
    out = np.where(hg > ag, 0, np.where(hg == ag, 1, 2))
    return out.astype(int)
