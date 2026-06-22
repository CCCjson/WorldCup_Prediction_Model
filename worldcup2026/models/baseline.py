"""M0 — no-ML baselines.

Two baselines, per CLAUDE.md:
  1. Market de-vig: bookmaker odds -> implied 1X2 with the overround removed.
     This is THE bar to beat (validated on club odds in MS1; national-team odds
     wired later).
  2. Elo -> 1X2 via a fitted ORDERED map (ordered logit on the Elo difference),
     refit per walk-forward split (train-only) to avoid leakage.

Probability outputs are ordered [P(home), P(draw), P(away)].
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from worldcup2026.features.elo import DEFAULT_HFA


# --------------------------------------------------------------------------- #
# De-vig (overround removal)
# --------------------------------------------------------------------------- #
def implied_from_odds(odds: np.ndarray) -> np.ndarray:
    """Raw implied probabilities 1/odds (still contain the bookmaker margin)."""
    return 1.0 / np.asarray(odds, dtype=float)


def devig_normalize(odds_h, odds_d, odds_a) -> np.ndarray:
    """Proportional (basic) de-vig: divide implied by the overround.

    Returns (n, 3) array ordered [home, draw, away].
    """
    raw = np.column_stack(
        [implied_from_odds(odds_h), implied_from_odds(odds_d), implied_from_odds(odds_a)]
    )
    return raw / raw.sum(axis=1, keepdims=True)


def devig_shin(odds_h, odds_d, odds_a, iters: int = 100) -> np.ndarray:
    """Shin (1992) de-vig — corrects for the insider-trading component.

    Slightly sharper than proportional when margins are non-trivial. Returns
    (n, 3) ordered [home, draw, away].
    """
    raw = np.column_stack(
        [implied_from_odds(odds_h), implied_from_odds(odds_d), implied_from_odds(odds_a)]
    )
    booksum = raw.sum(axis=1, keepdims=True)
    pi = raw / booksum  # normalized starting point
    z = np.zeros((raw.shape[0], 1))
    for _ in range(iters):
        # p_i = (sqrt(z^2 + 4(1-z) pi_i^2 / booksum) - z) / (2(1-z))  ... iterate z
        sqrt_term = np.sqrt(z**2 + 4 * (1 - z) * (raw**2) / booksum)
        p = (sqrt_term - z) / (2 * (1 - z))
        p = p / p.sum(axis=1, keepdims=True)
        z = np.clip(((p**2).sum(axis=1, keepdims=True) * booksum - 1) / (booksum - 1), 0, 0.5)
    return p


def overround(odds_h, odds_d, odds_a) -> np.ndarray:
    raw = np.column_stack(
        [implied_from_odds(odds_h), implied_from_odds(odds_d), implied_from_odds(odds_a)]
    )
    return raw.sum(axis=1)


# --------------------------------------------------------------------------- #
# Elo -> 1X2 ordered map (ordered logit, weighted MLE)
# --------------------------------------------------------------------------- #
def elo_diff_effective(matches, hfa: float = DEFAULT_HFA) -> np.ndarray:
    """elo_home_pre (+HFA if not neutral) - elo_away_pre, the map's predictor."""
    diff = matches["elo_home_pre"].to_numpy() - matches["elo_away_pre"].to_numpy()
    bump = np.where(matches["neutral"].to_numpy(), 0.0, hfa)
    return diff + bump


class EloOrderedMap:
    """Ordered logit mapping Elo difference -> ordered [home, draw, away].

    Latent ordering away(0) < draw(1) < home(2); two cutpoints. Fit by weighted
    maximum likelihood (supports comp_weight). Predictor is scaled by /400 for
    numerical conditioning.
    """

    SCALE = 400.0

    def __init__(self):
        self.beta_ = None
        self.c1_ = None
        self.c2_ = None

    def _probs(self, x, beta, c1, c2):
        eta = beta * x
        p_away = expit(c1 - eta)
        p_home = 1.0 - expit(c2 - eta)
        p_draw = np.clip(1.0 - p_away - p_home, 1e-12, 1.0)
        return p_home, p_draw, p_away

    def fit(self, elo_diff_eff, outcomes, weights=None):
        x = np.asarray(elo_diff_eff, dtype=float) / self.SCALE
        y = np.asarray(outcomes, dtype=int)  # 0=home,1=draw,2=away
        w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=float)

        def nll(params):
            beta, c1, ddelta = params
            c2 = c1 + np.log1p(np.exp(ddelta))  # softplus -> c2 > c1
            p_home, p_draw, p_away = self._probs(x, beta, c1, c2)
            ll = np.where(y == 0, np.log(p_home), np.where(y == 1, np.log(p_draw), np.log(p_away)))
            return -np.sum(w * ll)

        res = minimize(nll, x0=np.array([1.0, -0.4, 0.0]), method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 5000})
        beta, c1, ddelta = res.x
        self.beta_, self.c1_ = beta, c1
        self.c2_ = c1 + np.log1p(np.exp(ddelta))
        return self

    def predict_proba(self, elo_diff_eff) -> np.ndarray:
        x = np.asarray(elo_diff_eff, dtype=float) / self.SCALE
        p_home, p_draw, p_away = self._probs(x, self.beta_, self.c1_, self.c2_)
        return np.column_stack([p_home, p_draw, p_away])
