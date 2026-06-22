"""Own, leakage-safe Elo for national teams (World Football Elo style).

Ratings are updated sequentially in strict date order. For every match the
PRE-match ratings of both teams are snapshotted (``elo_home_pre`` /
``elo_away_pre``) BEFORE the result is used to update them. Those snapshots are
the as-of-`t` features: a match at time `t` only ever sees ratings built from
matches with date < `t` (ties broken by stable date sort, i.e. earlier rows).

Formula (CLAUDE.md spec):
    expected_home = 1 / (1 + 10**(-(elo_home + HFA - elo_away) / 400))
    HFA = 0 if neutral else 100
    K   = K0 * comp_weight * goal_diff_mult
    K0  = 40
    goal_diff_mult: 1.0 (margin 1), 1.5 (margin 2), 1.75 (margin 3),
                    then +0.75/8 per extra goal beyond 3.

Run:  python -m worldcup2026.features.elo
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from worldcup2026.data.ingest import load_matches

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
ELO_STATE_PATH = MODELS_DIR / "elo_state.json"
ELO_MATCHES_PARQUET = (
    Path(__file__).resolve().parent.parent / "data" / "cache" / "matches_elo.parquet"
)

INIT_RATING = 1500.0
DEFAULT_K0 = 40.0
DEFAULT_HFA = 100.0

# tournament-string -> competition importance weight (substring match, lowercased)
COMP_WEIGHTS = {
    "friendly": 0.3,
    "qualification": 0.8,   # continental / WC qualifiers
    "fifa world cup": 1.0,
    "uefa euro": 1.0,
    "copa américa": 1.0,
    "copa america": 1.0,
    "african cup of nations": 1.0,
    "afc asian cup": 1.0,
    "gold cup": 1.0,
    "confederations cup": 1.0,
    "nations league": 0.8,
}
DEFAULT_COMP_WEIGHT = 0.6  # "minor" tournaments not otherwise listed


def comp_weight(tournament: str) -> float:
    """Map a tournament string to its importance weight (K multiplier)."""
    t = str(tournament).lower()
    # qualifiers first: "FIFA World Cup qualification" must score 0.8, not 1.0
    if "qualification" in t or "qualifier" in t:
        return COMP_WEIGHTS["qualification"]
    for key, w in COMP_WEIGHTS.items():
        if key in t:
            return w
    return DEFAULT_COMP_WEIGHT


def goal_diff_mult(margin: int) -> float:
    """World Football Elo goal-difference multiplier."""
    m = abs(int(margin))
    if m <= 1:
        return 1.0
    if m == 2:
        return 1.5
    if m == 3:
        return 1.75
    return 1.75 + (m - 3) * 0.75 / 8.0


def compute_elo(
    matches: pd.DataFrame,
    k0: float = DEFAULT_K0,
    hfa: float = DEFAULT_HFA,
    init_rating: float = INIT_RATING,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Return (matches + elo_*_pre columns, final rating state).

    ``matches`` must contain: date, home_team, away_team, home_goals,
    away_goals, tournament, neutral. It is processed in its existing order,
    which ingest guarantees is a stable sort by date ASC.
    """
    ratings: dict[str, float] = {}
    home_pre = []
    away_pre = []

    home_teams = matches["home_team"].to_numpy()
    away_teams = matches["away_team"].to_numpy()
    hg = matches["home_goals"].to_numpy()
    ag = matches["away_goals"].to_numpy()
    neutral = matches["neutral"].to_numpy()
    cw = matches["tournament"].map(comp_weight).to_numpy()

    for i in range(len(matches)):
        h, a = home_teams[i], away_teams[i]
        rh = ratings.get(h, init_rating)
        ra = ratings.get(a, init_rating)

        # snapshot PRE-match ratings (the leakage-safe features)
        home_pre.append(rh)
        away_pre.append(ra)

        # actual result for home
        if hg[i] > ag[i]:
            score_home = 1.0
        elif hg[i] < ag[i]:
            score_home = 0.0
        else:
            score_home = 0.5

        eff_hfa = 0.0 if bool(neutral[i]) else hfa
        expected_home = 1.0 / (1.0 + 10 ** (-(rh + eff_hfa - ra) / 400.0))

        k = k0 * cw[i] * goal_diff_mult(hg[i] - ag[i])
        delta = k * (score_home - expected_home)

        ratings[h] = rh + delta
        ratings[a] = ra - delta

    out = matches.copy()
    out["elo_home_pre"] = home_pre
    out["elo_away_pre"] = away_pre
    out["comp_weight"] = cw
    return out, ratings


def build(save: bool = True) -> tuple[pd.DataFrame, dict[str, float]]:
    matches = load_matches()
    enriched, ratings = compute_elo(matches)
    if save:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        ELO_STATE_PATH.write_text(json.dumps(ratings, indent=2, sort_keys=True))
        enriched.to_parquet(ELO_MATCHES_PARQUET, index=False)
        print(f"[write] {ELO_STATE_PATH}")
        print(f"[write] {ELO_MATCHES_PARQUET} ({len(enriched):,} rows)")
    return enriched, ratings


def top_n(ratings: dict[str, float], n: int = 20) -> pd.DataFrame:
    s = pd.Series(ratings).sort_values(ascending=False).head(n)
    return s.round(1).rename("elo").rename_axis("team").reset_index()


if __name__ == "__main__":
    _, state = build()
    print("\n=== MS0 Elo report: current top-20 ===")
    print(top_n(state, 20).to_string(index=False))
