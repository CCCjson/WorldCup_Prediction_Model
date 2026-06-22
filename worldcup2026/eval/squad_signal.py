"""新信号探测:Transfermarkt 阵容身价能否在 Elo/贝叶斯之外带来增量预测力。

核心问题:身价是不是已经被 Elo(从结果学到的实力)包含了?若高度冗余 -> 无独立信号。

两个诚实检验(均无泄漏):
  1. 冗余度:Spearman(log 身价, 当前 Elo) 跨 48 队。越接近 1 越冗余。
  2. 残差检验:用 Elo 的 **as-of** 预测算「实际净得分 − 预期净得分」残差,看它是否被
     身价差解释。slope 显著且为正 -> 高身价队系统性跑赢模型 -> 身价有增量信号。
     测试集:2025-01-01 起所有涉及 48 队的国际赛(ordered map 在 2025 前训练,无泄漏);
     另报本届 39 场子集。

注:身价为 2026-06 当前快照(无历史 as-of),故只作信号探测,不入走步前向训练。

Run:  python -m worldcup2026.eval.squad_signal
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from worldcup2026.eval import metrics
from worldcup2026.features.elo import build as build_elo
from worldcup2026.models.baseline import EloOrderedMap, elo_diff_effective

SV_PATH = Path(__file__).resolve().parent.parent / "data" / "squad_values_2026.json"


def _current_elo(enriched: pd.DataFrame) -> dict:
    """每队最近一次的赛前 Elo 快照。"""
    last = {}
    for r in enriched.sort_values("date").itertuples(index=False):
        last[r.home_team] = r.elo_home_pre
        last[r.away_team] = r.elo_away_pre
    return last


def run():
    sv = json.loads(SV_PATH.read_text())["values"]
    enriched, _ = build_elo(save=False)
    enriched["elo_diff_eff"] = elo_diff_effective(enriched)
    enriched["outcome"] = metrics.outcomes_from_goals(
        enriched["home_goals"], enriched["away_goals"]).astype(int)

    # ---- 1) 冗余度:log 身价 vs 当前 Elo ----
    elo_now = _current_elo(enriched)
    teams = [t for t in sv if t in elo_now]
    miss = [t for t in sv if t not in elo_now]
    log_sv = np.array([np.log(sv[t]) for t in teams])
    elo_v = np.array([elo_now[t] for t in teams])
    rho_s, p_s = spearmanr(log_sv, elo_v)
    print(f"=== 身价 vs Elo 冗余度({len(teams)}/48 队)===")
    if miss:
        print(f"  未匹配 Elo: {miss}")
    print(f"  Spearman(log 身价, 当前 Elo) = {rho_s:.3f}  (p={p_s:.1e})")
    print(f"  -> {'高度冗余,身价基本被 Elo 包含' if rho_s > 0.85 else '有一定独立成分'}")

    # ---- 2) 残差检验 ----
    sv_log = {t: np.log(v) for t, v in sv.items()}
    train = enriched[enriched["date"] < "2025-01-01"]
    elo_map = EloOrderedMap().fit(
        train["elo_diff_eff"].to_numpy(), train["outcome"].to_numpy(),
        weights=train["comp_weight"].to_numpy())

    def residual_test(df, label):
        d = df[df["home_team"].isin(sv) & df["away_team"].isin(sv)].copy()
        if len(d) < 10:
            print(f"\n[{label}] 样本不足({len(d)})")
            return
        p = elo_map.predict_proba(d["elo_diff_eff"].to_numpy())   # as-of,无泄漏
        exp_pts = 3 * p[:, 0] + 1 * p[:, 1]                        # 预期主队净得分
        act_pts = np.where(d["outcome"] == 0, 3.0,
                           np.where(d["outcome"] == 1, 1.0, 0.0))
        resid = act_pts - exp_pts
        sv_diff = d["home_team"].map(sv_log).to_numpy() - d["away_team"].map(sv_log).to_numpy()
        r_p, pp = pearsonr(sv_diff, resid)
        r_s, ps = spearmanr(sv_diff, resid)
        # 身价单独作 1X2(in-sample 拟合,仅粗看是否predictive)vs Elo
        sv_map = EloOrderedMap()
        sv_map.SCALE = 1.0
        sv_map.fit(sv_diff, d["outcome"].to_numpy())
        rps_sv = metrics.rps(sv_map.predict_proba(sv_diff), d["outcome"].to_numpy())
        rps_elo = metrics.rps(p, d["outcome"].to_numpy())
        print(f"\n[{label}] n={len(d)}")
        print(f"  corr(身价差, Elo 残差): Pearson {r_p:+.3f} (p={pp:.3f}) | "
              f"Spearman {r_s:+.3f} (p={ps:.3f})")
        print(f"  解读: {'身价显著解释 Elo 漏掉的部分 -> 有增量信号' if pp < 0.05 and r_p > 0 else '残差与身价无显著关系 -> 身价无独立增量'}")
        print(f"  RPS: Elo(as-of) {rps_elo:.4f} | 身价单独(in-sample) {rps_sv:.4f}")

    played = enriched[(enriched["tournament"] == "FIFA World Cup")
                      & (enriched["date"] >= "2026-06-01")]
    recent = enriched[enriched["date"] >= "2025-01-01"]
    residual_test(recent, "2025-now 全部(涉 48 队)")
    residual_test(played, "本届 39 场子集")

    # 最大分歧:身价 rank vs Elo rank
    print("\n=== 身价 与 Elo 最大分歧队(身价高估/低估 vs Elo)===")
    sv_rank = pd.Series({t: sv[t] for t in teams}).rank(ascending=False)
    elo_rank = pd.Series({t: elo_now[t] for t in teams}).rank(ascending=False)
    gap = (elo_rank - sv_rank).sort_values()
    print("  Elo 看好、身价不看好(便宜却赢球):")
    for t in gap.index[:5]:
        print(f"    {t:<24s} 身价#{int(sv_rank[t]):>2} Elo#{int(elo_rank[t]):>2}")
    print("  身价看好、Elo 不看好(贵却没战绩):")
    for t in gap.index[-5:]:
        print(f"    {t:<24s} 身价#{int(sv_rank[t]):>2} Elo#{int(elo_rank[t]):>2}")


if __name__ == "__main__":
    run()
