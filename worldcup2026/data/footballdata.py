"""football-data.co.uk club-odds ingestion (MS1 de-vig pipeline validation).

These are CLUB leagues, used per CLAUDE.md only to VALIDATE the de-vig + RPS
machinery (national-team odds for the actual WC baseline come from oddsportal
later). We pull big-5 leagues across recent seasons, keep the sharpest closing
odds available per row, and expose results so RPS can be scored.

Closing-odds priority (sharpest first): Pinnacle (PSC*) -> Bet365 (B365C*) ->
average (AvgC*) -> Bet365 pre-close (B365*).

Run:  python -m worldcup2026.data.footballdata
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://www.football-data.co.uk/mmz4281"
LEAGUES = ["E0", "D1", "SP1", "I1", "F1"]  # big-5: EPL, Bundesliga, La Liga, Serie A, Ligue 1
# seasons with usable closing odds (PSC from 15/16); two-digit start+end
SEASONS = ["1516", "1617", "1718", "1819", "1920", "2021", "2122", "2223", "2324", "2425"]

CACHE_DIR = Path(__file__).resolve().parent / "cache" / "footballdata"
CLUB_PARQUET = Path(__file__).resolve().parent / "cache" / "club_odds.parquet"

# odds-column triples in priority order
ODDS_TRIPLES = [
    ("PSCH", "PSCD", "PSCA", "pinnacle_close"),
    ("B365CH", "B365CD", "B365CA", "bet365_close"),
    ("AvgCH", "AvgCD", "AvgCA", "avg_close"),
    ("B365H", "B365D", "B365A", "bet365"),
]


def _download_csv(league: str, season: str) -> pd.DataFrame | None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{season}_{league}.csv"
    if cache_file.exists():
        text = cache_file.read_text(encoding="latin-1")
    else:
        url = f"{BASE_URL}/{season}/{league}.csv"
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"[skip] {season}/{league}: {e}")
            return None
        text = resp.content.decode("latin-1")
        cache_file.write_text(text, encoding="latin-1")
    try:
        df = pd.read_csv(StringIO(text), encoding="latin-1", on_bad_lines="skip")
    except Exception as e:  # noqa: BLE001
        print(f"[skip] parse {season}/{league}: {e}")
        return None
    return df


def _pick_closing_odds(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise select the sharpest available closing-odds triple."""
    n = len(df)
    oh = np.full(n, np.nan)
    od = np.full(n, np.nan)
    oa = np.full(n, np.nan)
    src = np.array(["none"] * n, dtype=object)
    for ch, cd, ca, name in ODDS_TRIPLES:
        if not {ch, cd, ca}.issubset(df.columns):
            continue
        h = pd.to_numeric(df[ch], errors="coerce").to_numpy()
        d = pd.to_numeric(df[cd], errors="coerce").to_numpy()
        a = pd.to_numeric(df[ca], errors="coerce").to_numpy()
        fill = np.isnan(oh) & ~np.isnan(h) & ~np.isnan(d) & ~np.isnan(a)
        oh[fill], od[fill], oa[fill] = h[fill], d[fill], a[fill]
        src[fill] = name
    out = pd.DataFrame(
        {"odds_home": oh, "odds_draw": od, "odds_away": oa, "odds_source": src}
    )
    return out


def build(force: bool = False) -> pd.DataFrame:
    if CLUB_PARQUET.exists() and not force:
        print(f"[cache] {CLUB_PARQUET.name} present — loading")
        return pd.read_parquet(CLUB_PARQUET)

    frames = []
    for season in SEASONS:
        for league in LEAGUES:
            raw = _download_csv(league, season)
            if raw is None or "FTR" not in raw.columns:
                continue
            odds = _pick_closing_odds(raw)
            block = pd.DataFrame(
                {
                    "date": pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce"),
                    "league": league,
                    "season": season,
                    "home_team": raw["HomeTeam"],
                    "away_team": raw["AwayTeam"],
                    "home_goals": pd.to_numeric(raw["FTHG"], errors="coerce"),
                    "away_goals": pd.to_numeric(raw["FTAG"], errors="coerce"),
                    "ftr": raw["FTR"],
                }
            )
            block = pd.concat([block, odds], axis=1)
            frames.append(block)

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["date", "home_goals", "away_goals", "odds_home", "odds_draw", "odds_away"])
    df = df[df["ftr"].isin(["H", "D", "A"])].reset_index(drop=True)
    df.to_parquet(CLUB_PARQUET, index=False)
    print(f"[write] {CLUB_PARQUET} ({len(df):,} rows)")
    return df


if __name__ == "__main__":
    frame = build()
    print("\n=== club odds coverage ===")
    print(f"rows         : {len(frame):,}")
    print(f"date range   : {frame['date'].min().date()} -> {frame['date'].max().date()}")
    print("by source    :")
    print(frame["odds_source"].value_counts().to_string())
    print("by league    :")
    print(frame["league"].value_counts().to_string())
