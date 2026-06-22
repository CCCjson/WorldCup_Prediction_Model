"""External data source interfaces.

MS0 only needs the Kaggle international-results dataset (see ``ingest.py``).
The odds and squad-value feeds are required by later milestones (MS1 market
baseline, feature layer). Per CLAUDE.md's "stub a source behind an interface
and continue" rule, they are defined here as interfaces with stub
implementations that return ``None`` / raise ``NotImplementedError`` until the
real scrapers are wired in.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import Optional

import pandas as pd


# --------------------------------------------------------------------------- #
# Odds (MS1 market baseline + with_odds feature variant)
# --------------------------------------------------------------------------- #
@dataclass
class MatchOdds:
    """De-vig is applied downstream; these are raw decimal odds."""

    home: float
    draw: float
    away: float
    is_closing: bool  # closing -> baseline only; opening -> allowed as feature


class OddsSource(ABC):
    """Interface for 1X2 odds feeds (oddsportal, football-data.co.uk)."""

    @abstractmethod
    def get_odds(
        self, home_team: str, away_team: str, match_date: date
    ) -> Optional[MatchOdds]:
        ...


class StubOddsSource(OddsSource):
    """Placeholder when no scraped odds are available."""

    def get_odds(self, home_team, away_team, match_date):  # noqa: D102
        return None


class OddsportalSource(OddsSource):
    """已落地的 oddsportal 实现(closing 平均 1X2)。

    赔率由 ``scrapers/oddsportal.py`` 抓取并缓存到 parquet(每赛事一个文件);
    本类按无序对 (home, away) 查找,自动对齐主客方向。CLAUDE.md:收盘赔率仅作
    市场基线,不作模型特征。见 ``eval/market_compare.py`` 的端到端配对评估。
    """

    def __init__(self, odds_frame):
        self._idx = {}
        for r in odds_frame.itertuples(index=False):
            self._idx[frozenset((r.home_team, r.away_team))] = (
                r.home_team, r.odds_home, r.odds_draw, r.odds_away)

    def get_odds(self, home_team, away_team, match_date=None):  # noqa: D102
        rec = self._idx.get(frozenset((home_team, away_team)))
        if rec is None:
            return None
        op_home, oh, od, oa = rec
        if op_home != home_team:      # 方向相反 -> 交换主客赔率
            oh, oa = oa, oh
        return MatchOdds(home=oh, draw=od, away=oa, is_closing=True)


# --------------------------------------------------------------------------- #
# Squad value / age (feature layer, when Transfermarkt available)
# --------------------------------------------------------------------------- #
@dataclass
class SquadInfo:
    market_value_eur: float
    mean_age: float


class SquadSource(ABC):
    """Interface for aggregate squad market value / age (Transfermarkt)."""

    @abstractmethod
    def get_squad(self, team: str, as_of: date) -> Optional[SquadInfo]:
        ...


class StubSquadSource(SquadSource):
    """Placeholder until the real scraper is wired (feature layer)."""

    def get_squad(self, team, as_of):  # noqa: D102
        return None


def empty_odds_frame() -> pd.DataFrame:
    """Schema the real odds feed will populate; lets downstream code join safely."""
    return pd.DataFrame(
        columns=[
            "date", "home_team", "away_team",
            "odds_home", "odds_draw", "odds_away", "is_closing",
        ]
    )
