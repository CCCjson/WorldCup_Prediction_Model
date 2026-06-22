"""集成/堆叠实验:把 Elo + Dixon–Coles + 贝叶斯三个弱模型按权重组合,
看能否 OOS 超过最好的单模型(用户「按权重分配再训练造新 alpha」的严格版)。

关键诚实点:
  - 集成增益的天花板由模型间**正交性**决定。先测三模型 OOS 预测的相关性;若 ~1,
    增益注定很小。
  - 权重**绝不能在被评分的数据上拟合**。用**留一年交叉验证**:为第 y 年找权重时,
    只用其余年的 OOS 预测拟合,再应用到第 y 年 -> 嵌套 OOS,杜绝 meta 过拟合。
  - 跟「最好的单模型」比,不是跟最差的比;报 bootstrap CI。

三种组合:① 等权平均(无拟合,稳健基准);② 留一年最优凸权重;③ 报相关性诊断。

Run:  python -m worldcup2026.eval.ensemble_stack
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from worldcup2026.eval import metrics
from worldcup2026.eval.walkforward import _bootstrap_ci, BAYES_TEST_YEARS, TRAIN_MIN_YEAR
from worldcup2026.features.elo import build as build_elo
from worldcup2026.models.baseline import EloOrderedMap, elo_diff_effective
from worldcup2026.models.dixon_coles import DixonColes
from worldcup2026.models.bayes_poisson import BayesPoisson


def collect_oos(window_years: int = 8):
    """走步前向收集每场的三模型 OOS 概率 + 结果 + 年份。"""
    enriched, _ = build_elo(save=False)
    enriched = enriched.copy()
    enriched["year"] = enriched["date"].dt.year
    enriched["outcome"] = metrics.outcomes_from_goals(
        enriched["home_goals"], enriched["away_goals"])
    enriched["elo_diff_eff"] = elo_diff_effective(enriched)

    P = {"elo": [], "dc": [], "bayes": []}
    outs, years = [], []
    for y in BAYES_TEST_YEARS:
        split = f"{y}-01-01"
        train_recent = enriched[enriched["date"] < split]
        train_elo = enriched[(enriched["year"] >= TRAIN_MIN_YEAR) & (enriched["date"] < split)]
        test = enriched[enriched["year"] == y]
        out = test["outcome"].to_numpy()

        elo = EloOrderedMap().fit(
            train_elo["elo_diff_eff"].to_numpy(), train_elo["outcome"].to_numpy(),
            weights=train_elo["comp_weight"].to_numpy())
        P["elo"].append(elo.predict_proba(test["elo_diff_eff"].to_numpy()))
        P["dc"].append(DixonColes().fit(train_recent, ref_date=split).predict_proba_frame(test))
        P["bayes"].append(BayesPoisson(window_years=window_years)
                          .fit(train_recent, ref_date=split).predict_proba_frame(test))
        outs.append(out)
        years.append(np.full(len(out), y))
        print(f"  [{y}] n={len(out)} 收集完成")

    return ({k: np.vstack(v) for k, v in P.items()},
            np.concatenate(outs), np.concatenate(years))


def _weighted(P, w):
    return w[0] * P["elo"] + w[1] * P["dc"] + w[2] * P["bayes"]


def _fit_weights(P_sub, out_sub):
    """在子集上找最小化 RPS 的凸权重(单纯形约束)。"""
    keys = ["elo", "dc", "bayes"]

    def obj(w):
        p = w[0] * P_sub["elo"] + w[1] * P_sub["dc"] + w[2] * P_sub["bayes"]
        return metrics.rps(p, out_sub)

    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bnds = [(0, 1)] * 3
    res = minimize(obj, np.array([1 / 3, 1 / 3, 1 / 3]), method="SLSQP",
                   bounds=bnds, constraints=cons, options={"ftol": 1e-9})
    return res.x


def run():
    print("=== 收集三模型 OOS 预测(走步前向 6 年)===")
    P, out, years = collect_oos()
    N = len(out)
    print(f"总场数:{N}\n")

    # ---- 相关性诊断:集成增益天花板 ----
    print("=== 模型间预测相关性(越高 -> 集成增益越小)===")
    flat = {k: P[k].ravel() for k in P}
    for a, b in [("elo", "dc"), ("elo", "bayes"), ("dc", "bayes")]:
        r = np.corrcoef(flat[a], flat[b])[0, 1]
        print(f"  corr(P_{a}, P_{b}) = {r:.3f}")
    # 误差(per-match RPS)相关性
    rps_v = {k: metrics.rps_vec(P[k], out) for k in P}
    print("  误差相关:", {f"{a}-{b}": round(float(np.corrcoef(rps_v[a], rps_v[b])[0, 1]), 3)
                       for a, b in [("elo", "dc"), ("elo", "bayes"), ("dc", "bayes")]})

    # ---- 单模型基准 ----
    single = {k: metrics.rps(P[k], out) for k in P}
    best_name = min(single, key=single.get)
    print("\n=== 单模型 OOS RPS ===")
    for k, v in single.items():
        print(f"  {k:<6} {v:.4f}" + ("  <- 最好" if k == best_name else ""))

    # ---- ① 等权平均 ----
    p_avg = _weighted(P, np.array([1 / 3, 1 / 3, 1 / 3]))
    rps_avg = metrics.rps(p_avg, out)

    # ---- ② 留一年最优凸权重(嵌套 OOS)----
    p_stack = np.zeros((N, 3))
    used_w = []
    for y in BAYES_TEST_YEARS:
        tr = years != y
        te = years == y
        P_tr = {k: P[k][tr] for k in P}
        w = _fit_weights(P_tr, out[tr])
        used_w.append((y, w))
        p_stack[te] = _weighted({k: P[k][te] for k in P}, w)
    rps_stack = metrics.rps(p_stack, out)

    print("\n=== 集成 OOS RPS(对比最好单模型)===")
    best = single[best_name]
    for name, p in [("等权平均", p_avg), ("留一年最优权重", p_stack)]:
        rps = metrics.rps(p, out)
        diff = metrics.rps_vec(p, out) - metrics.rps_vec(P[best_name], out)
        lo, hi = _bootstrap_ci(diff)
        verdict = ("集成显著更好" if hi < 0 else
                   "最好单模型显著更好" if lo > 0 else "无显著差异")
        print(f"  {name:<12} RPS {rps:.4f} | vs 最好单模型({best_name} {best:.4f}) "
              f"delta {diff.mean():+.4f} CI [{lo:+.4f},{hi:+.4f}] -> {verdict}")

    print("\n留一年学到的权重(elo, dc, bayes):")
    for y, w in used_w:
        print(f"  {y}: [{w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f}]")


if __name__ == "__main__":
    run()
