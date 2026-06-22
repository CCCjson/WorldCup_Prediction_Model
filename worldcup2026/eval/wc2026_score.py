"""诚实性检查:在**已踢完的 2026 世界杯**比赛上做真实样本外评分。

CLAUDE.md 此前只能用历史大赛(2018/22 WC、2020/24 Euro)代理「模型 vs 市场」;
本届开赛后,我们终于能用**本届真实比赛**直接检验模型——这是最硬的样本外信号。

方法(滚动 walk-forward,严格无泄漏):
  - 取本届已踢完的 FIFA World Cup 比赛(小组赛,date < cutoff)。
  - 按比赛日期分组;对每个日期 d,贝叶斯**重训到 as-of d**(train = date < d),
    再预测当日比赛 -> 第 2、3 轮的预测会用到前几轮信息,但绝不用未来数据。
  - Elo 用 enriched 中本就 as-of 的赛前快照(elo_*_pre),信息口径与贝叶斯一致。
  - neutral/host 取各场实际值(东道主主场不归零)。
  - 市场基线:本届暂无 oddsportal 国家队收盘赔率(未抓),故以**均匀 1/3** 作无技能
    地板;贝叶斯 vs Elo 才是 no_odds 头号对比。

Run:  python -m worldcup2026.eval.wc2026_score
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from worldcup2026.eval import metrics
from worldcup2026.eval.walkforward import _bootstrap_ci
from worldcup2026.features.elo import build as build_elo
from worldcup2026.models.baseline import EloOrderedMap, elo_diff_effective
from worldcup2026.models.bayes_poisson import BayesPoisson

CUTOFF = "2026-06-22"   # 评分截止(含此前所有已踢完的本届比赛)


def run(cutoff: str = CUTOFF):
    enriched, _ = build_elo(save=False)
    enriched["elo_diff_eff"] = elo_diff_effective(enriched)
    enriched["outcome"] = metrics.outcomes_from_goals(
        enriched["home_goals"], enriched["away_goals"]).astype(int)

    played = enriched[(enriched["tournament"] == "FIFA World Cup")
                      & (enriched["date"] >= "2026-06-01")
                      & (enriched["date"] < cutoff)].copy()
    print(f"=== 本届 2026 世界杯样本外评分(截止 {cutoff})===")
    print(f"已踢完比赛 : {len(played)} 场,{played['date'].min().date()} → "
          f"{played['date'].max().date()}")

    # 滚动 walk-forward:每个比赛日把贝叶斯重训到 as-of
    p_bayes = np.zeros((len(played), 3))
    p_elo = np.zeros((len(played), 3))
    played = played.reset_index(drop=True)
    rhats = []
    for d, idx in played.groupby("date").groups.items():
        idx = list(idx)
        ref = d.strftime("%Y-%m-%d")
        train = enriched[enriched["date"] < d]
        bp = BayesPoisson(window_years=8).fit(train, ref_date=ref)
        rhats.append(bp.rhat_max_)
        p_bayes[idx] = bp.predict_proba_frame(played.loc[idx])

        elo_tr = train[train["date"].dt.year >= 1990]
        elo = EloOrderedMap().fit(
            elo_tr["elo_diff_eff"].to_numpy(),
            elo_tr["outcome"].to_numpy(),
            weights=elo_tr["comp_weight"].to_numpy())
        p_elo[idx] = elo.predict_proba(played.loc[idx, "elo_diff_eff"].to_numpy())

    out = played["outcome"].to_numpy()
    p_unif = np.full((len(played), 3), 1 / 3)

    bayes = metrics.score_all(p_bayes, out)
    elo = metrics.score_all(p_elo, out)
    unif = metrics.score_all(p_unif, out)

    # bootstrap CI on per-match RPS 差
    r_bay = metrics.rps_vec(p_bayes, out)
    r_elo = metrics.rps_vec(p_elo, out)
    diff = r_bay - r_elo
    lo, hi = _bootstrap_ci(diff)
    verdict = ("贝叶斯显著更好" if hi < 0 else
               "Elo 显著更好" if lo > 0 else "无显著差异")

    print(f"[采样] r_hat_max(各日)≤ {max(rhats):.3f}")
    print("\n模型         RPS     Brier   logloss  accuracy")
    for name, s in [("贝叶斯 ", bayes), ("Elo    ", elo), ("均匀1/3", unif)]:
        print(f"{name}     {s['rps']:.4f}  {s['brier']:.4f}  "
              f"{s['logloss']:.4f}   {s['accuracy']:.3f}")
    print(f"\n贝叶斯−Elo RPS delta = {diff.mean():+.4f}  "
          f"95%CI [{lo:+.4f},{hi:+.4f}] -> {verdict}")

    # 校准(贝叶斯,样本小仅供参考)
    cal = metrics.calibration_table(p_bayes, out)

    result = {
        "cutoff": cutoff,
        "n_matches": int(len(played)),
        "date_range": [str(played["date"].min().date()), str(played["date"].max().date())],
        "bayes": bayes, "elo": elo, "uniform": unif,
        "bayes_minus_elo_delta": float(diff.mean()),
        "bayes_minus_elo_ci": [float(lo), float(hi)],
        "verdict": verdict,
        "market_note": "本届暂无国家队收盘赔率(oddsportal 未抓),市场基线缺失;以均匀 1/3 作无技能地板。",
        "calibration": cal.to_dict(orient="records"),
    }
    return result


if __name__ == "__main__":
    import json
    res = run()
    print("\n" + json.dumps({k: v for k, v in res.items() if k != "calibration"},
                            ensure_ascii=False, indent=2))
