"""MS0 data ingestion: build the canonical match table.

Downloads the Kaggle international-results dataset (cache-first; never re-hits
the source if the local copy exists) and normalizes it into the canonical
schema defined in CLAUDE.md, written to ``data/cache/matches.parquet``.

Canonical match table (one row per historical match, sorted by date ASC):
    date, home_team, away_team, home_goals, away_goals,
    tournament, neutral, host_country

Run:  python -m worldcup2026.data.ingest
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

# NB: the slug kept its original "-to-2017" suffix even though the dataset is
# maintained to the present day (title: "...from 1872 to 2026").
KAGGLE_DATASET = "martj42/international-football-results-from-1872-to-2017"
RESULTS_CSV = "results.csv"  # the file we use within the dataset

DATA_DIR = Path(__file__).resolve().parent
CACHE_DIR = DATA_DIR / "cache"
ALIAS_PATH = DATA_DIR / "alias_map.json"
MATCHES_PARQUET = CACHE_DIR / "matches.parquet"


def _load_alias_map() -> dict[str, str]:
    raw = json.loads(ALIAS_PATH.read_text())
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def download_raw(force: bool = False) -> Path:
    """Download + unzip the Kaggle dataset into the cache. Cache-first."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = CACHE_DIR / RESULTS_CSV
    if csv_path.exists() and not force:
        print(f"[cache] {csv_path.name} present — skipping download")
        return csv_path

    print(f"[kaggle] downloading {KAGGLE_DATASET} ...")
    subprocess.run(
        [
            "kaggle", "datasets", "download",
            "-d", KAGGLE_DATASET,
            "-p", str(CACHE_DIR),
            "--unzip",
        ],
        check=True,
    )
    if not csv_path.exists():
        raise FileNotFoundError(
            f"expected {csv_path} after download; got {os.listdir(CACHE_DIR)}"
        )
    return csv_path


def canonicalize(csv_path: Path) -> pd.DataFrame:
    """Normalize raw results.csv -> canonical schema."""
    alias = _load_alias_map()
    raw = pd.read_csv(csv_path)

    # raw columns: date, home_team, away_team, home_score, away_score,
    #              tournament, city, country, neutral
    df = pd.DataFrame()
    df["date"] = pd.to_datetime(raw["date"])
    df["home_team"] = raw["home_team"].map(lambda t: alias.get(t, t))
    df["away_team"] = raw["away_team"].map(lambda t: alias.get(t, t))
    df["home_goals"] = raw["home_score"].astype("Int64")
    df["away_goals"] = raw["away_score"].astype("Int64")
    df["tournament"] = raw["tournament"]
    # neutral can be bool or string "TRUE"/"FALSE" depending on dataset version
    df["neutral"] = raw["neutral"].astype(str).str.lower().isin(["true", "1"])

    # host_country: for a non-neutral match, whoever played at home is the host.
    # The raw "country" column is where the match was played; on a non-neutral
    # match that country == the home team (post-alias).
    host = raw["country"].map(lambda t: alias.get(t, t))
    df["host_country"] = host.where(~df["neutral"], other=pd.NA)

    # drop matches with missing scores (unplayed/abandoned rows, if any)
    before = len(df)
    df = df.dropna(subset=["home_goals", "away_goals"]).copy()
    dropped = before - len(df)
    if dropped:
        print(f"[clean] dropped {dropped} rows with missing scores")

    df = df.sort_values("date", kind="stable").reset_index(drop=True)
    return df


def build(force_download: bool = False) -> pd.DataFrame:
    csv_path = download_raw(force=force_download)
    df = canonicalize(csv_path)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(MATCHES_PARQUET, index=False)
    print(f"[write] {MATCHES_PARQUET} ({len(df):,} rows)")
    return df


def load_matches() -> pd.DataFrame:
    """Read the cached canonical table (build it first if missing)."""
    if not MATCHES_PARQUET.exists():
        return build()
    return pd.read_parquet(MATCHES_PARQUET)


def _report(df: pd.DataFrame) -> None:
    teams = pd.unique(pd.concat([df["home_team"], df["away_team"]]))
    print("\n=== MS0 ingestion report ===")
    print(f"rows           : {len(df):,}")
    print(f"date range     : {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"distinct teams : {len(teams):,}")
    print(f"neutral share  : {df['neutral'].mean():.1%}")
    print("tournaments    :")
    print(df["tournament"].value_counts().head(8).to_string())


if __name__ == "__main__":
    force = "--force" in sys.argv
    frame = build(force_download=force)
    _report(frame)
