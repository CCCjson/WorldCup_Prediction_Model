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

# 1X2 收盘三元组(锐度优先:Pinnacle > Bet365 > 均价 > Bet365 盘前)
ODDS_TRIPLES = [
    ("PSCH", "PSCD", "PSCA", "pinnacle_close"),
    ("B365CH", "B365CD", "B365CA", "bet365_close"),
    ("AvgCH", "AvgCD", "AvgCA", "avg_close"),
    ("B365H", "B365D", "B365A", "bet365"),
]

# 1X2 开盘三元组(用于 CLV:开盘下注 vs 收盘比价)。Pinnacle 开 > Bet365 开 > 均价开。
OPEN_TRIPLES = [
    ("PSH", "PSD", "PSA", "pinnacle_open"),
    ("B365H", "B365D", "B365A", "bet365_open"),
    ("AvgH", "AvgD", "AvgA", "avg_open"),
]

# 大小球 2.5 收盘对(over, under)
OU_CLOSE_PAIRS = [
    ("PC>2.5", "PC<2.5", "pinnacle_close"),
    ("B365C>2.5", "B365C<2.5", "bet365_close"),
    ("AvgC>2.5", "AvgC<2.5", "avg_close"),
]

# 大小球 2.5 开盘对
OU_OPEN_PAIRS = [
    ("P>2.5", "P<2.5", "pinnacle_open"),
    ("B365>2.5", "B365<2.5", "bet365_open"),
    ("Avg>2.5", "Avg<2.5", "avg_open"),
]


def _pick(df: pd.DataFrame, groups, out_names, source_col) -> pd.DataFrame:
    """逐行从候选组里选第一个完整可用的赔率(按优先级)。

    groups: [(col_1, ..., col_k, source_name), ...] —— 每组前 k 列对应 out_names;
    out_names: k 个输出赔率列名;source_col: 记录所选来源的列名。
    """
    n = len(df)
    k = len(out_names)
    cols = [np.full(n, np.nan) for _ in range(k)]
    src = np.array(["none"] * n, dtype=object)
    for grp in groups:
        names, sname = grp[:k], grp[k]
        if not set(names).issubset(df.columns):
            continue
        vals = [pd.to_numeric(df[c], errors="coerce").to_numpy() for c in names]
        fill = np.isnan(cols[0])
        for v in vals:
            fill &= ~np.isnan(v)
        for i, v in enumerate(vals):
            cols[i][fill] = v[fill]
        src[fill] = sname
    out = {name: cols[i] for i, name in enumerate(out_names)}
    out[source_col] = src
    return pd.DataFrame(out)


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


def _all_odds(df: pd.DataFrame) -> pd.DataFrame:
    """逐行选取四套赔率:1X2 收盘/开盘 + 大小球 2.5 收盘/开盘。"""
    close_1x2 = _pick(df, ODDS_TRIPLES,
                      ["odds_home", "odds_draw", "odds_away"], "odds_source")
    open_1x2 = _pick(df, OPEN_TRIPLES,
                     ["odds_home_open", "odds_draw_open", "odds_away_open"],
                     "odds_open_source")
    ou_close = _pick(df, OU_CLOSE_PAIRS,
                     ["ou_over", "ou_under"], "ou_source")
    ou_open = _pick(df, OU_OPEN_PAIRS,
                    ["ou_over_open", "ou_under_open"], "ou_open_source")
    return pd.concat([close_1x2, open_1x2, ou_close, ou_open], axis=1)


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
            odds = _all_odds(raw)
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
    import sys
    frame = build(force="--force" in sys.argv)
    print("\n=== club odds coverage ===")
    print(f"rows              : {len(frame):,}")
    print(f"date range        : {frame['date'].min().date()} -> {frame['date'].max().date()}")
    print(f"含开盘 1X2        : {frame['odds_home_open'].notna().sum():,} 行")
    print(f"含收盘大小球 2.5  : {frame['ou_over'].notna().sum():,} 行")
    print(f"含开盘大小球 2.5  : {frame['ou_over_open'].notna().sum():,} 行")
    print("收盘 1X2 来源     :")
    print(frame["odds_source"].value_counts().to_string())
