"""模型 vs 市场:在多届大赛上做配对 RPS 对比(收窄 CI)。

市场 = oddsportal 收盘平均 1X2,de-vig 后隐含概率(CLAUDE.md 强制基线)。
模型 = no_odds 头号贝叶斯泊松(各赛事赛前训练)+ Elo 基线,同一批比赛。
outcome 取自历史数据集比分(口径统一);neutral 取各场实际值(东道主主场不归零)。

赛事:2018/2022 世界杯 + 2020/2024 欧洲杯 ≈ 230 场 -> 更窄的 bootstrap CI。

Run:  python -m worldcup2026.eval.market_compare
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from worldcup2026.eval import metrics
from worldcup2026.eval.walkforward import _bootstrap_ci
from worldcup2026.features.elo import build as build_elo
from worldcup2026.models.baseline import (
    EloOrderedMap, devig_normalize, elo_diff_effective, overround)
from worldcup2026.models.bayes_poisson import BayesPoisson
from worldcup2026.scrapers.oddsportal import scrape_tournament

EVENTS = [
    {"name": "2018 World Cup", "tournament": "FIFA World Cup", "year": 2018,
     "url": "https://www.oddsportal.com/football/world/world-cup-2018/results/",
     "cache": "oddsportal_wc2018.parquet", "ref": "2018-06-14", "expected": 64},
    {"name": "Euro 2020", "tournament": "UEFA Euro", "year": 2021,
     "url": "https://www.oddsportal.com/football/europe/euro-2020/results/",
     "cache": "oddsportal_euro2020.parquet", "ref": "2021-06-11", "expected": 51},
    {"name": "2022 World Cup", "tournament": "FIFA World Cup", "year": 2022,
     "url": "https://www.oddsportal.com/football/world/world-cup-2022/results/",
     "cache": "oddsportal_wc2022.parquet", "ref": "2022-11-20", "expected": 64},
    {"name": "2024 Euro", "tournament": "UEFA Euro", "year": 2024,
     "url": "https://www.oddsportal.com/football/europe/euro-2024/results/",
     "cache": "oddsportal_euro2024.parquet", "ref": "2024-06-14", "expected": 51},
]


def _predict_event(enriched, ev):
    hist = enriched[(enriched["tournament"] == ev["tournament"]) &
                    (enriched["year"] == ev["year"])].copy()
    hist["outcome"] = metrics.outcomes_from_goals(hist["home_goals"], hist["away_goals"])
    hist["elo_diff_eff"] = elo_diff_effective(hist)
    teams = sorted(set(hist["home_team"]) | set(hist["away_team"]))

    op = scrape_tournament(ev["url"], ev["cache"], valid_teams=teams,
                           expected=ev["expected"])
    odds = {frozenset((r.home_team, r.away_team)):
            (r.home_team, r.odds_home, r.odds_draw, r.odds_away)
            for r in op.itertuples(index=False)}

    rows, miss = [], 0
    for r in hist.itertuples(index=False):
        rec = odds.get(frozenset((r.home_team, r.away_team)))
        if rec is None:
            miss += 1
            continue
        op_home, oh, od, oa = rec
        if op_home != r.home_team:        # 方向相反 -> 交换主客赔率
            oh, oa = oa, oh
        rows.append({"home_team": r.home_team, "away_team": r.away_team,
                     "neutral": bool(r.neutral), "outcome": int(r.outcome),
                     "elo_diff_eff": float(r.elo_diff_eff),
                     "oh": oh, "od": od, "oa": oa})
    df = pd.DataFrame(rows)

    out = df["outcome"].to_numpy()
    p_market = devig_normalize(df["oh"], df["od"], df["oa"])
    ov = float(overround(df["oh"], df["od"], df["oa"]).mean())

    train = enriched[enriched["date"] < ev["ref"]]
    bp = BayesPoisson(window_years=8).fit(train, ref_date=ev["ref"])
    p_bayes = bp.predict_proba_frame(df)   # 用各场实际 neutral

    elo_train = enriched[(enriched["year"] >= 1990) & (enriched["date"] < ev["ref"])]
    elo = EloOrderedMap().fit(
        elo_diff_effective(elo_train),
        metrics.outcomes_from_goals(elo_train["home_goals"], elo_train["away_goals"]).astype(int),
        weights=elo_train["comp_weight"].to_numpy())
    p_elo = elo.predict_proba(df["elo_diff_eff"].to_numpy())

    print(f"  [{ev['name']}] 配对 {len(df)} 场(未匹配 {miss}) "
          f"vig {100*(ov-1):.2f}% r_hat={bp.rhat_max_:.3f} div={bp.divergences_}")
    return {"name": ev["name"], "n": len(df), "ov": ov, "outcome": out,
            "p_market": p_market, "p_bayes": p_bayes, "p_elo": p_elo}


def run():
    enriched, _ = build_elo(save=False)
    enriched["year"] = enriched["date"].dt.year

    per = [_predict_event(enriched, ev) for ev in EVENTS]

    # 合并所有赛事的每场 RPS
    out = np.concatenate([e["outcome"] for e in per])
    r_mkt = np.concatenate([metrics.rps_vec(e["p_market"], e["outcome"]) for e in per])
    r_bay = np.concatenate([metrics.rps_vec(e["p_bayes"], e["outcome"]) for e in per])
    r_elo = np.concatenate([metrics.rps_vec(e["p_elo"], e["outcome"]) for e in per])

    result = {"events": [{"name": e["name"], "matches": e["n"],
                          "mean_overround": e["ov"],
                          "market_rps": float(metrics.rps(e["p_market"], e["outcome"])),
                          "bayes_rps": float(metrics.rps(e["p_bayes"], e["outcome"])),
                          "elo_rps": float(metrics.rps(e["p_elo"], e["outcome"]))}
                         for e in per],
              "matches": int(len(out)),
              "market_rps": float(r_mkt.mean()),
              "bayes_rps": float(r_bay.mean()),
              "elo_rps": float(r_elo.mean())}

    print(f"\n=== 模型 vs 市场:合并配对 RPS({len(per)} 届大赛) ===")
    print(f"合并场数            : {len(out)}")
    print(f"市场基线 RPS        : {r_mkt.mean():.4f}")
    print(f"Elo (no_odds) RPS   : {r_elo.mean():.4f}")
    print(f"贝叶斯 (no_odds) RPS : {r_bay.mean():.4f}")
    for name, r in [("bayes", r_bay), ("elo", r_elo)]:
        diff = r - r_mkt
        lo, hi = _bootstrap_ci(diff)
        verdict = ("模型显著更好" if hi < 0 else
                   "市场显著更好" if lo > 0 else "无显著差异")
        result[f"{name}_delta"] = float(diff.mean())
        result[f"{name}_ci"] = [float(lo), float(hi)]
        result[f"{name}_verdict"] = verdict
        print(f"  {name}-市场 delta = {diff.mean():+.4f}  "
              f"95%CI [{lo:+.4f},{hi:+.4f}] -> {verdict}")
    print("\n按赛事:")
    for e in result["events"]:
        print(f"  {e['name']:16s} n={e['matches']:3d} 市场 {e['market_rps']:.4f} "
              f"贝叶斯 {e['bayes_rps']:.4f} Elo {e['elo_rps']:.4f}")
    return result


if __name__ == "__main__":
    run()
