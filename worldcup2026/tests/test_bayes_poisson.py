"""分层贝叶斯泊松模型单测(小数据 + 少量抽样,保证快速)。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from worldcup2026.models.bayes_poisson import BayesPoisson


def _toy_matches() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    teams = ["A", "B", "C"]
    strength = {"A": 2.0, "B": 1.2, "C": 0.6}
    rows = []
    for d in pd.date_range("2020-01-01", "2024-12-01", freq="7D"):
        h, a = rng.choice(teams, 2, replace=False)
        neutral = bool(rng.random() < 0.5)
        boost = 1.0 if neutral else 1.35
        rows.append((d, h, a, rng.poisson(strength[h] * boost), rng.poisson(strength[a]), neutral))
    return pd.DataFrame(rows, columns=["date", "home_team", "away_team",
                                       "home_goals", "away_goals", "neutral"])


@pytest.fixture(scope="module")
def fitted():
    return BayesPoisson(window_years=10).fit(
        _toy_matches(), ref_date="2025-01-01",
        draws=300, tune=300, chains=2, target_accept=0.9)


def test_sampling_runs_and_mixes(fitted):
    # 冒烟测试用小数据 + 少抽样,允许个别发散;真实数据(draws=2000,ta=0.95)为 0 发散
    assert fitted.divergences_ <= 3
    assert fitted.rhat_max_ < 1.1


def test_1x2_sums_to_one_and_favours_stronger(fitted):
    p = fitted.predict_1x2("A", "C", neutral=True)
    assert np.isclose(p.sum(), 1.0)
    assert p[0] > p[2]


def test_score_matrix_valid(fitted):
    mat = fitted.score_matrix("A", "B", neutral=True)
    assert np.isclose(mat.sum(), 1.0)
    assert (mat >= 0).all()


def test_home_field_positive_and_applied(fitted):
    assert fitted.hf_.mean() > 0
    p_neutral = fitted.predict_1x2("A", "B", neutral=True)
    p_home = fitted.predict_1x2("A", "B", neutral=False)
    assert p_home[0] > p_neutral[0]   # 主场抬升主队胜率


def test_unknown_team_does_not_crash(fitted):
    p = fitted.predict_1x2("A", "Nowhereland", neutral=True)
    assert np.isclose(p.sum(), 1.0)
