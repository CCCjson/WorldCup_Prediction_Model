"""Leakage unit test for the as-of Elo features (MS0).

The contract: ``elo_home_pre`` / ``elo_away_pre`` on a match row must be
computable from matches strictly earlier in the ordered table only — never from
the match's own result or any later match.

Strategy:
1. Independent recompute: run a clean Elo pass over the FIRST k rows and assert
   the engine's snapshots for those rows match. Because the engine processes in
   date order and snapshots before updating, agreement on a prefix proves no
   future row influenced those snapshots.
2. Chaining: for several teams, assert the pre-rating of their match N equals the
   post-update rating implied by their match N-1 (no gaps, no peeking ahead).
3. Ordering: assert the table is sorted by date ascending.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from worldcup2026.data.ingest import load_matches
from worldcup2026.features.elo import (
    DEFAULT_HFA,
    DEFAULT_K0,
    INIT_RATING,
    comp_weight,
    compute_elo,
    goal_diff_mult,
)


@pytest.fixture(scope="module")
def enriched():
    matches = load_matches()
    out, _ = compute_elo(matches)
    return out


def _reference_pre_ratings(matches: pd.DataFrame, k0=DEFAULT_K0, hfa=DEFAULT_HFA):
    """Deliberately naive, independent Elo recompute (no vectorization tricks)."""
    ratings: dict[str, float] = {}
    home_pre, away_pre = [], []
    for row in matches.itertuples(index=False):
        rh = ratings.get(row.home_team, INIT_RATING)
        ra = ratings.get(row.away_team, INIT_RATING)
        home_pre.append(rh)
        away_pre.append(ra)
        sh = 1.0 if row.home_goals > row.away_goals else (0.0 if row.home_goals < row.away_goals else 0.5)
        eff_hfa = 0.0 if bool(row.neutral) else hfa
        exp_h = 1.0 / (1.0 + 10 ** (-(rh + eff_hfa - ra) / 400.0))
        k = k0 * comp_weight(row.tournament) * goal_diff_mult(row.home_goals - row.away_goals)
        d = k * (sh - exp_h)
        ratings[row.home_team] = rh + d
        ratings[row.away_team] = ra - d
    return np.array(home_pre), np.array(away_pre)


def test_table_sorted_by_date(enriched):
    assert enriched["date"].is_monotonic_increasing


def test_prefix_recompute_matches(enriched):
    """Independent recompute of the first k rows must equal engine snapshots."""
    k = 20_000
    sub = enriched.iloc[:k]
    ref_h, ref_a = _reference_pre_ratings(sub)
    np.testing.assert_allclose(sub["elo_home_pre"].to_numpy(), ref_h, rtol=1e-9, atol=1e-6)
    np.testing.assert_allclose(sub["elo_away_pre"].to_numpy(), ref_a, rtol=1e-9, atol=1e-6)


def test_no_future_dependence(enriched):
    """Truncating the data at row k must not change snapshots for rows < k.

    If any elo_*_pre depended on a later match, recomputing on the truncated
    prefix would differ. This is the core anti-leakage assertion.
    """
    k = 15_000
    truncated = enriched.iloc[:k][
        ["date", "home_team", "away_team", "home_goals", "away_goals", "tournament", "neutral"]
    ].copy()
    re_out, _ = compute_elo(truncated)
    np.testing.assert_allclose(
        re_out["elo_home_pre"].to_numpy(),
        enriched.iloc[:k]["elo_home_pre"].to_numpy(),
        rtol=1e-9, atol=1e-6,
    )
    np.testing.assert_allclose(
        re_out["elo_away_pre"].to_numpy(),
        enriched.iloc[:k]["elo_away_pre"].to_numpy(),
        rtol=1e-9, atol=1e-6,
    )


def test_pre_rating_chaining(enriched):
    """A team's match-N pre-rating == its post-update rating after match N-1."""
    sample_teams = ["Brazil", "Argentina", "England", "Germany", "France"]
    for team in sample_teams:
        mask = (enriched["home_team"] == team) | (enriched["away_team"] == team)
        tm = enriched[mask]
        if len(tm) < 3:
            continue
        prev_post = None
        checked = 0
        for row in tm.itertuples(index=False):
            is_home = row.home_team == team
            pre = row.elo_home_pre if is_home else row.elo_away_pre
            if prev_post is not None:
                assert pre == pytest.approx(prev_post, abs=1e-6), (
                    f"{team}: pre-rating != prior post-rating (gap/leak)"
                )
                checked += 1
            # recompute this match's post-rating for the team
            rh, ra = row.elo_home_pre, row.elo_away_pre
            sh = 1.0 if row.home_goals > row.away_goals else (0.0 if row.home_goals < row.away_goals else 0.5)
            eff_hfa = 0.0 if bool(row.neutral) else DEFAULT_HFA
            exp_h = 1.0 / (1.0 + 10 ** (-(rh + eff_hfa - ra) / 400.0))
            k = DEFAULT_K0 * comp_weight(row.tournament) * goal_diff_mult(row.home_goals - row.away_goals)
            d = k * (sh - exp_h)
            prev_post = (rh + d) if is_home else (ra - d)
        assert checked > 0, f"no chained matches verified for {team}"
