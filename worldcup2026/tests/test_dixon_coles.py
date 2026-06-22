"""Unit tests for the Dixon–Coles model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from worldcup2026.models.dixon_coles import DixonColes


def _toy_matches() -> pd.DataFrame:
    # strong team A, weak team C, middling B; repeated over time
    rng = np.random.default_rng(0)
    teams = ["A", "B", "C"]
    strength = {"A": 2.0, "B": 1.2, "C": 0.6}
    rows = []
    dates = pd.date_range("2018-01-01", "2024-12-01", freq="7D")
    for d in dates:
        h, a = rng.choice(teams, 2, replace=False)
        neutral = bool(rng.random() < 0.5)
        home_boost = 1.0 if neutral else 1.35   # genuine HFA only at non-neutral venues
        hg = rng.poisson(strength[h] * home_boost)
        ag = rng.poisson(strength[a])
        rows.append((d, h, a, hg, ag, neutral))
    return pd.DataFrame(rows, columns=["date", "home_team", "away_team",
                                       "home_goals", "away_goals", "neutral"])


def test_fit_converges():
    dc = DixonColes().fit(_toy_matches(), ref_date="2025-01-01")
    assert dc.converged_
    assert -0.2 <= dc.rho_ <= 0.2


def test_score_matrix_is_valid_distribution():
    dc = DixonColes().fit(_toy_matches(), ref_date="2025-01-01")
    mat = dc.score_matrix("A", "C", neutral=True)
    assert np.isclose(mat.sum(), 1.0)
    assert (mat >= 0).all()


def test_1x2_sums_to_one_and_favours_stronger():
    dc = DixonColes().fit(_toy_matches(), ref_date="2025-01-01")
    p = dc.predict_1x2("A", "C", neutral=True)
    assert np.isclose(p.sum(), 1.0)
    assert p[0] > p[2]  # strong A beats weak C more often than loses


def test_home_advantage_zeroed_at_neutral():
    dc = DixonColes().fit(_toy_matches(), ref_date="2025-01-01")
    p_neutral = dc.predict_1x2("A", "B", neutral=True)
    p_home = dc.predict_1x2("A", "B", neutral=False)
    # gamma>0 => playing at home lifts A's win prob vs the same neutral matchup
    assert dc.gamma_ > 0
    assert p_home[0] > p_neutral[0]
