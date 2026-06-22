"""赛前预测存档:在比赛开打前,用头号贝叶斯模型把当日赛程的预测**固化下来**,
赛后可与真实结果对照(真正的赛前样本外,杜绝事后诸葛)。

为何可信:`models/bayes_2026.npz` 训练截止 = REF_DATE(如 2026-06-22,train=date<截止),
当日比赛在截止当天 → 模型**未见过**这些比赛,存档即赛前快照。

用法:
    python -m worldcup2026.eval.prematch_archive            # 存当前 FIXTURES(默认日期)
存档落地:models/prematch/prematch_<date>.json 与 .csv
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from worldcup2026.models.bayes_poisson import BayesPoisson

MODELS = Path(__file__).resolve().parent.parent / "models"
ARCHIVE = MODELS / "prematch"

# (home, away, neutral, group) —— 2026 除三东道主在本国外均中立场。
# 6-22 四场均在美国本土但美国不参赛,故全部中立。
DEFAULT_DATE = "2026-06-22"
FIXTURES = [
    ("Argentina", "Austria", True, "J"),
    ("France", "Iraq", True, "C"),
    ("Norway", "Senegal", True, "C"),
    ("Jordan", "Algeria", True, "J"),
]


def predict_fixtures(date: str = DEFAULT_DATE, fixtures=FIXTURES):
    model = BayesPoisson.load(MODELS / "bayes_2026.npz")
    rows = []
    for home, away, neutral, grp in fixtures:
        p = model.predict_1x2(home, away, neutral=neutral)          # [H, D, A]
        over, under = model.predict_over_under(home, away, neutral=neutral)
        mat = np.asarray(model.score_matrix(home, away, neutral=neutral))
        i, j = np.unravel_index(np.argmax(mat), mat.shape)
        # 期望进球(对截断比分矩阵求边际均值)
        gh = float((mat.sum(axis=1) * np.arange(mat.shape[0])).sum())
        ga = float((mat.sum(axis=0) * np.arange(mat.shape[1])).sum())
        rows.append({
            "date": date, "group": grp, "home": home, "away": away,
            "neutral": neutral,
            "p_home": round(float(p[0]), 4), "p_draw": round(float(p[1]), 4),
            "p_away": round(float(p[2]), 4),
            "over_2.5": round(float(over), 4), "under_2.5": round(float(under), 4),
            "ml_score": f"{i}-{j}", "ml_score_prob": round(float(mat[i, j]), 4),
            "exp_goals_home": round(gh, 2), "exp_goals_away": round(ga, 2),
        })
    return pd.DataFrame(rows)


def run(date: str = DEFAULT_DATE, fixtures=FIXTURES):
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    df = predict_fixtures(date, fixtures)
    model = BayesPoisson.load(MODELS / "bayes_2026.npz")
    meta = {
        "matchday": date,
        "model": "Bayesian Poisson (no_odds, headline)",
        "trained_cutoff": date,  # bayes_2026.npz 以 REF_DATE 截止训练 = 赛前
        "note": "赛前快照:模型训练截止当日,未见过这些比赛;赛后与真实结果对照。",
        "rhat_max": round(float(model.rhat_max_), 3),
        "predictions": df.to_dict(orient="records"),
    }
    (ARCHIVE / f"prematch_{date}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2))
    df.to_csv(ARCHIVE / f"prematch_{date}.csv", index=False)

    print(f"=== 赛前预测存档 {date}(头号贝叶斯,r̂={model.rhat_max_:.3f})===")
    show = df.copy()
    for c in ["p_home", "p_draw", "p_away", "over_2.5"]:
        show[c] = (show[c] * 100).round(1).astype(str) + "%"
    print(show[["group", "home", "away", "p_home", "p_draw", "p_away",
                "over_2.5", "ml_score", "exp_goals_home", "exp_goals_away"]]
          .to_string(index=False))
    print(f"\n已存档:{ARCHIVE / f'prematch_{date}.json'}（含 .csv)")
    return meta


if __name__ == "__main__":
    run()
