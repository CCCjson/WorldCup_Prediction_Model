"""生成 UI 所需的离线工件,避免 Streamlit 每次重采样/重模拟。

产出(均在 models/):
  - bayes_2026.npz   赛前(2026-06-11)贝叶斯后验抽样
  - sim_2026.csv     50k 次模拟的各阶段晋级概率
  - report.json      市场 de-vig 校准 + RPS 对比(MS1-MS3 已验证数字)

Run:  python -m worldcup2026.build_artifacts
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from worldcup2026.data.footballdata import build as build_club
from worldcup2026.data.ingest import load_matches
from worldcup2026.eval import metrics
from worldcup2026.models.baseline import devig_normalize, overround
from worldcup2026.sim.tournament import (
    _fit_model, build_known_results, load_2026_config, simulate_tournament)

MODELS = Path(__file__).resolve().parent / "models"
# 实时口径:用 < REF_DATE 的全部数据训练(纳入已踢小组赛),并把已踢完的小组赛
# 钉死进模拟。赛事进行中,定期把 REF_DATE 推到“今天”重跑即可滚动更新。
REF_DATE = "2026-06-22"


def main(n_sims: int = 50000):
    MODELS.mkdir(exist_ok=True)

    print(f"[1/3] 训练并保存贝叶斯模型(截止 {REF_DATE})...")
    model = _fit_model(REF_DATE)
    model.save(MODELS / "bayes_2026.npz")
    print(f"      r_hat_max={model.rhat_max_:.3f} div={model.divergences_} "
          f"teams={len(model.teams_)}")

    print(f"[2/3] 跑 {n_sims:,} 次锦标赛模拟(钉死已踢小组赛)...")
    cfg = load_2026_config()
    known, n_known = build_known_results(cfg, load_matches(), REF_DATE)
    print(f"      条件化:钉死 {n_known} 场,采样 {72 - n_known} 场")
    res = simulate_tournament(model, cfg, n_sims=n_sims, known_results=known)
    res.to_csv(MODELS / "sim_2026.csv", index=False)
    print(f"      冠军热门:{res.iloc[0]['team']} {res.iloc[0]['P(Winner)']:.1%}")

    print("[3/3] 生成市场校准 + RPS 报告 ...")
    club = build_club()
    out = np.where(club["ftr"].eq("H"), 0, np.where(club["ftr"].eq("D"), 1, 2)).astype(int)
    p = devig_normalize(club["odds_home"], club["odds_draw"], club["odds_away"])
    cal = metrics.calibration_table(p, out)
    report = {
        "market_club": {
            "matches": int(len(club)),
            "mean_overround": float(overround(club["odds_home"], club["odds_draw"],
                                              club["odds_away"]).mean()),
            "rps": metrics.rps(p, out),
            "calibration": cal.to_dict(orient="records"),
        },
        # MS1-MS3 走步前向已验证的聚合 RPS(见 README;不同样本不可直接横比)
        "rps_table": [
            {"model": "Market de-vig (club, validation)", "rps": 0.1937, "scope": "big-5 club 18,011"},
            {"model": "Elo->1X2 (no_odds)", "rps": 0.1672, "scope": "intl 2019-25"},
            {"model": "Dixon-Coles (no_odds)", "rps": 0.1714, "scope": "intl 2019-25"},
            {"model": "Bayesian Poisson (no_odds, headline)", "rps": 0.1678, "scope": "intl 2019-25"},
        ],
        "honesty_note": (
            "俱乐部市场 RPS 与国际模型 RPS 样本不同,不可直接横比。贝叶斯在 1X2 上与 Elo "
            "无显著差异,其价值在比分分布与后验不确定性传播。"),
    }

    # 真正的「模型 vs 市场」配对(oddsportal 收盘赔率,2022 WC + 2024 Euro 合并)
    try:
        from worldcup2026.eval.market_compare import run as mvm_run
        report["model_vs_market"] = mvm_run()
        print(f"      模型 vs 市场:{report['model_vs_market']['matches']} 场,市场 "
              f"{report['model_vs_market']['market_rps']:.4f} vs 贝叶斯 "
              f"{report['model_vs_market']['bayes_rps']:.4f}")
    except Exception as e:  # noqa: BLE001 — 抓取/网络失败不应阻断其他工件
        print(f"      [跳过] model_vs_market: {e}")
        report["model_vs_market"] = None
    # 本届真实样本外评分(诚实性检查:模型在已踢完的 2026 比赛上的表现)
    try:
        from worldcup2026.eval.wc2026_score import run as wc_run
        report["wc2026_oos"] = wc_run(cutoff=REF_DATE)
        o = report["wc2026_oos"]
        print(f"      WC2026 样本外({o['n_matches']} 场):贝叶斯 RPS "
              f"{o['bayes']['rps']:.4f} vs Elo {o['elo']['rps']:.4f} vs 均匀 "
              f"{o['uniform']['rps']:.4f}")
    except Exception as e:  # noqa: BLE001
        print(f"      [跳过] wc2026_oos: {e}")
        report["wc2026_oos"] = None

    (MODELS / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"      market RPS={report['market_club']['rps']:.4f} "
          f"overround={report['market_club']['mean_overround']:.4f}")
    print("\n工件已生成:bayes_2026.npz / sim_2026.csv / report.json")


if __name__ == "__main__":
    main()
