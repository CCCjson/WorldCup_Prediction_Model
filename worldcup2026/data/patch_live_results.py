"""Patch in live 2026 World Cup results that the Kaggle feed hasn't published yet.

The Kaggle ``martj42`` international-results dataset lags real time by ~3-4 days.
As of the last refresh it covered group matches through 2026-06-18 (28 matches).
This module holds the matches played 2026-06-19 .. 2026-06-21 that were curated
by hand from the per-group Wikipedia pages (one source of truth, cross-checked),
and merges them into the canonical ``matches.parquet``.

Design:
- **Idempotent.** Dedup is by unordered team-pair *among 2026 FIFA World Cup
  matches* (each pair meets exactly once in the group stage), NOT by date —
  web sources and Kaggle disagree on the calendar date of a fixture by up to a
  day (timezone/kickoff-vs-UTC), so a date-based merge would double-insert.
- **Host handling per CLAUDE.md.** Only USA/Canada/Mexico playing in their own
  country are non-neutral; everything else at the 2026 tournament is neutral.
- **Conservative.** Matches without a confirmed final score are omitted (e.g.
  New Zealand vs Egypt, 21 June: Wikipedia showed no score at curation time and
  the only "result" found was from preview/prediction pages — left for the
  simulator to sample rather than risk inserting a wrong score).

Source: en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_{A..L}, curated 2026-06-22.

Run:  python -m worldcup2026.data.patch_live_results
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent / "cache"
MATCHES_PARQUET = CACHE_DIR / "matches.parquet"

HOSTS = {"United States", "Canada", "Mexico"}

# (date, home_team, away_team, home_goals, away_goals)
# Canonical names already (verified against the existing table's team set).
LIVE_RESULTS = [
    # 2026-06-19 — Groups C, D matchday 2
    ("2026-06-19", "Scotland", "Morocco", 0, 1),
    ("2026-06-19", "Brazil", "Haiti", 3, 0),
    ("2026-06-19", "United States", "Australia", 2, 0),  # USA host -> non-neutral
    ("2026-06-19", "Turkey", "Paraguay", 0, 1),
    # 2026-06-20 — Groups E, F matchday 2
    ("2026-06-20", "Germany", "Ivory Coast", 2, 1),
    ("2026-06-20", "Ecuador", "Curaçao", 0, 0),
    ("2026-06-20", "Netherlands", "Sweden", 5, 1),
    ("2026-06-20", "Tunisia", "Japan", 0, 4),
    # 2026-06-21 — Groups G, H matchday 2 (NZ-Egypt omitted: no confirmed score)
    ("2026-06-21", "Belgium", "Iran", 0, 0),
    ("2026-06-21", "Spain", "Saudi Arabia", 4, 0),
    ("2026-06-21", "Uruguay", "Cape Verde", 2, 2),
]


def _new_rows() -> pd.DataFrame:
    rows = []
    for d, h, a, hg, ag in LIVE_RESULTS:
        is_host = h in HOSTS
        rows.append(
            {
                "date": pd.Timestamp(d),
                "home_team": h,
                "away_team": a,
                "home_goals": hg,
                "away_goals": ag,
                "tournament": "FIFA World Cup",
                "neutral": not is_host,
                "host_country": h if is_host else pd.NA,
            }
        )
    df = pd.DataFrame(rows)
    df["home_goals"] = df["home_goals"].astype("Int64")
    df["away_goals"] = df["away_goals"].astype("Int64")
    return df


def patch(df: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is None:
        df = pd.read_parquet(MATCHES_PARQUET)

    # Existing 2026 World Cup team-pairs (unordered) — for dedup. Scope to THIS
    # tournament only: the same pair may have met in a past World Cup (e.g.
    # Scotland 0-3 Morocco in 1998), so an all-history scope would false-skip.
    wc = df[(df["tournament"] == "FIFA World Cup") & (df["date"] >= "2026-06-01")]
    seen = {frozenset((r.home_team, r.away_team)) for r in wc.itertuples(index=False)}

    new = _new_rows()
    mask = [frozenset((r.home_team, r.away_team)) not in seen
            for r in new.itertuples(index=False)]
    fresh = new[mask].copy()

    skipped = len(new) - len(fresh)
    if skipped:
        print(f"[patch] {skipped} match(es) already present — skipped (idempotent)")

    if fresh.empty:
        print("[patch] nothing new to add")
        return df

    out = pd.concat([df, fresh], ignore_index=True)
    out = out.sort_values("date", kind="stable").reset_index(drop=True)
    print(f"[patch] added {len(fresh)} live match(es); table now {len(out):,} rows "
          f"(through {out['date'].max().date()})")
    for r in fresh.itertuples(index=False):
        print(f"        {r.date.date()}  {r.home_team} {r.home_goals}-{r.away_goals} "
              f"{r.away_team}  (neutral={r.neutral})")
    return out


if __name__ == "__main__":
    patched = patch()
    patched.to_parquet(MATCHES_PARQUET, index=False)
    print(f"[write] {MATCHES_PARQUET}")
