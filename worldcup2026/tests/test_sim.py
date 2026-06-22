"""锦标赛模拟器单测(用轻量 fake 模型,避免贝叶斯采样开销)。"""

from __future__ import annotations

import numpy as np

from worldcup2026.sim.tournament import (
    _match_thirds,
    config_2022,
    load_2026_config,
    simulate_tournament,
)


class FakeModel:
    """模仿 BayesPoisson 的后验抽样接口:att_/def_/mu_/hf_/idx_。"""

    def __init__(self, teams, seed=1):
        rng = np.random.default_rng(seed)
        self.idx_ = {t: i for i, t in enumerate(teams)}
        S, T = 200, len(teams)
        strv = rng.normal(0, 0.5, T)
        self.att_ = strv[None, :] + rng.normal(0, 0.05, (S, T))
        self.def_ = strv[None, :] + rng.normal(0, 0.05, (S, T))
        self.mu_ = np.full(S, 0.1)
        self.hf_ = np.full(S, 0.25)


def test_match_thirds_respects_constraints():
    slot_allowed = (("C", "D", "F", "G", "H"), ("C", "D", "F", "G", "H"),
                    ("C", "E", "F", "H", "I"), ("E", "H", "I", "J", "K"),
                    ("A", "E", "H", "I", "J"), ("B", "E", "F", "I", "J"),
                    ("E", "F", "G", "I", "J"), ("D", "E", "I", "J", "L"))
    qualified = ("A", "B", "C", "D", "E", "F", "G", "H")
    assign = _match_thirds(qualified, slot_allowed)
    assert len(assign) == 8
    assert len(set(assign)) == 8                      # 每组只用一次
    for slot, g in enumerate(assign):
        assert g in slot_allowed[slot]               # 满足来源组约束
        assert g in qualified


def test_2026_probabilities_valid():
    cfg = load_2026_config()
    teams = [t for g in cfg["groups"].values() for t in g]
    res = simulate_tournament(FakeModel(teams), cfg, n_sims=3000, seed=7)
    assert len(res) == 48
    assert np.isclose(res["P(Winner)"].sum(), 1.0)
    assert np.isclose(res["P(R32)"].sum(), 32.0)     # 32 队进 R32
    assert res["P(Winner)"].max() <= 1.0
    # 轮次单调:R32 >= R16 >= QF >= SF >= Final >= Winner(逐队)
    for a, b in [("R32", "R16"), ("R16", "QF"), ("QF", "SF"), ("SF", "Final"), ("Final", "Winner")]:
        assert (res[f"P({a})"] >= res[f"P({b})"] - 1e-9).all()


def test_2022_format_has_five_stages():
    cfg = config_2022()
    teams = [t for g in cfg["groups"].values() for t in g]
    res = simulate_tournament(FakeModel(teams, seed=2), cfg, n_sims=2000, seed=3)
    assert len(res) == 32
    assert "P(R16)" in res.columns and "P(R32)" not in res.columns
    assert np.isclose(res["P(Winner)"].sum(), 1.0)
    assert np.isclose(res["P(R16)"].sum(), 16.0)     # 16 队进 R16
