"""精确比分(correct score)验证:实测「比分预测对的概率」到底是多少。

1X2 只有三类,容易;**精确比分**有几十种可能,本就低概率——必须实测命中率,并检验
模型给出的概率是否**校准**(说 12% 的比分,是否真有 ~12% 发生)。

两份样本(均无泄漏):
  A) 本届 2026 已踢完比赛:滚动 as-of(每个比赛日把贝叶斯重训到 date<d 再预测当日)。
  B) 2022 卡塔尔世界杯:赛前(2022-11-01)单点训练,预测其后全部 2022 WC 比赛(模型
     未见过)——样本更大,补统计力。

指标:
  - 最可能比分命中率(argmax 比分 == 真实):模型「敢报的那个比分」对的频率。
  - 模型给最可能比分的平均概率:它自报的把握(理想应 ≈ 命中率 → 校准良好)。
  - 模型给真实比分的平均概率;真实比分进前 3 的命中率。
  - 基线:always 最常见比分(历史众数)的命中率。

Run:  python -m worldcup2026.eval.score_validation
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from worldcup2026.features.elo import build as build_elo
from worldcup2026.models.bayes_poisson import BayesPoisson


def _score_stats(model, frame):
    """对一批比赛,返回每场的 (top命中, top概率, 真实比分概率, 真实进前3) 数组。"""
    n = len(frame)
    top_hit = np.zeros(n, bool)
    top_prob = np.zeros(n)
    true_prob = np.zeros(n)
    top3_hit = np.zeros(n, bool)
    for i, r in enumerate(frame.itertuples(index=False)):
        mat = np.asarray(model.score_matrix(r.home_team, r.away_team, bool(r.neutral)))
        cap = mat.shape[0] - 1
        gh, ga = int(min(r.home_goals, cap)), int(min(r.away_goals, cap))
        ti, tj = np.unravel_index(np.argmax(mat), mat.shape)
        top_hit[i] = (ti == gh and tj == ga)
        top_prob[i] = mat[ti, tj]
        true_prob[i] = mat[gh, ga]
        # 真实比分是否在概率前 3 的比分里
        flat_order = np.argsort(mat, axis=None)[::-1][:3]
        top3 = set(map(tuple, np.array(np.unravel_index(flat_order, mat.shape)).T))
        top3_hit[i] = (gh, ga) in top3
    return top_hit, top_prob, true_prob, top3_hit


def _report(name, frame, top_hit, top_prob, true_prob, top3_hit):
    # 基线:always 该样本众数比分
    sc = frame.apply(lambda r: f"{int(r.home_goals)}-{int(r.away_goals)}", axis=1)
    mode = sc.mode().iloc[0]
    base_hit = float((sc == mode).mean())
    print(f"\n=== {name}(n={len(frame)})===")
    print(f"  最可能比分命中率   : {top_hit.mean():.1%}   "
          f"(模型自报平均把握 {top_prob.mean():.1%})")
    print(f"  → 校准:命中率 {top_hit.mean():.1%} vs 自报 {top_prob.mean():.1%} "
          f"({'吻合' if abs(top_hit.mean()-top_prob.mean())<0.03 else '偏差'})")
    print(f"  真实比分进前3命中率: {top3_hit.mean():.1%}")
    print(f"  模型给真实比分平均概率: {true_prob.mean():.1%}")
    print(f"  基线(always {mode}) 命中率: {base_hit:.1%}")
    return {
        "name": name, "n": int(len(frame)),
        "top_score_hit_rate": float(top_hit.mean()),
        "mean_top_score_prob": float(top_prob.mean()),
        "top3_hit_rate": float(top3_hit.mean()),
        "mean_true_score_prob": float(true_prob.mean()),
        "baseline_mode_score": mode, "baseline_hit_rate": base_hit,
    }


def run():
    enriched, _ = build_elo(save=False)
    results = []

    # ---- A) 本届 2026,滚动 as-of ----
    wc = enriched[(enriched["tournament"] == "FIFA World Cup")
                  & (enriched["date"] >= "2026-06-01")].copy().reset_index(drop=True)
    th = np.zeros(len(wc), bool); tp = np.zeros(len(wc))
    rp = np.zeros(len(wc)); t3 = np.zeros(len(wc), bool)
    for d, idx in wc.groupby("date").groups.items():
        idx = list(idx)
        train = enriched[enriched["date"] < d]
        bp = BayesPoisson(window_years=8).fit(train, ref_date=d.strftime("%Y-%m-%d"))
        a, b, c, e = _score_stats(bp, wc.loc[idx])
        for k, ii in enumerate(idx):
            th[ii], tp[ii], rp[ii], t3[ii] = a[k], b[k], c[k], e[k]
    results.append(_report("本届 2026(滚动 as-of)", wc, th, tp, rp, t3))

    # ---- B) 2022 世界杯,赛前单点训练 ----
    cut = "2022-11-01"
    wc22 = enriched[(enriched["tournament"] == "FIFA World Cup")
                    & (enriched["date"] >= "2022-11-15")
                    & (enriched["date"] < "2023-01-01")].copy().reset_index(drop=True)
    bp22 = BayesPoisson(window_years=8).fit(
        enriched[enriched["date"] < cut], ref_date=cut)
    a, b, c, e = _score_stats(bp22, wc22)
    results.append(_report("2022 世界杯(赛前训练)", wc22, a, b, c, e))

    # ---- 合并 ----
    allf = pd.concat([wc, wc22], ignore_index=True)
    ath = np.concatenate([th, a]); atp = np.concatenate([tp, b])
    arp = np.concatenate([rp, c]); at3 = np.concatenate([t3, e])
    results.append(_report("合并", allf, ath, atp, arp, at3))
    return results


if __name__ == "__main__":
    import json
    res = run()
    print("\n" + json.dumps(res, ensure_ascii=False, indent=2))
