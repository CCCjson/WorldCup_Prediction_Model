"""MS1 evaluation:

  A. Validate the de-vig + RPS pipeline on football-data.co.uk CLUB odds.
     (Sanity: a sharp closing line de-vigged should land ~0.18-0.20 RPS.)
  B. Walk-forward (expanding-window) RPS for the Elo->1X2 ordered map on the
     INTERNATIONAL match data. Map is refit on train-only each split -> no leak.

The market-vs-model paired table on the SAME international matches needs
national-team odds (oddsportal, deferred), so it is NOT produced here; this
establishes the Elo baseline and proves the market machinery is correct.

Run:  python -m worldcup2026.eval.walkforward
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from worldcup2026.data.footballdata import build as build_club
from worldcup2026.features.elo import build as build_elo
from worldcup2026.eval import metrics
from worldcup2026.models.baseline import (
    EloOrderedMap,
    devig_normalize,
    devig_shin,
    elo_diff_effective,
    overround,
)
from worldcup2026.models.dixon_coles import DixonColes
from worldcup2026.models.bayes_poisson import BayesPoisson

TRAIN_MIN_YEAR = 1990   # ignore sparse early Elo history when fitting the map
TEST_YEARS = list(range(2002, 2026))  # expanding-window test blocks


# --------------------------------------------------------------------------- #
# A. Club de-vig validation
# --------------------------------------------------------------------------- #
def validate_devig_on_club() -> pd.DataFrame:
    df = build_club()
    outcomes = np.where(df["ftr"].eq("H"), 0, np.where(df["ftr"].eq("D"), 1, 2)).astype(int)
    oh, od, oa = df["odds_home"], df["odds_draw"], df["odds_away"]

    ov = overround(oh, od, oa)
    rows = []
    for name, fn in [("normalize", devig_normalize), ("shin", devig_shin)]:
        p = fn(oh, od, oa)
        s = metrics.score_all(p, outcomes)
        s["method"] = name
        rows.append(s)

    print("\n=== A. Club de-vig validation (football-data.co.uk) ===")
    print(f"matches            : {len(df):,}")
    print(f"mean overround     : {ov.mean():.4f}  (vig {100*(ov.mean()-1):.2f}%)")
    print(f"outcome base rates : H {np.mean(outcomes==0):.3f} "
          f"D {np.mean(outcomes==1):.3f} A {np.mean(outcomes==2):.3f}")
    table = pd.DataFrame(rows)[["method", "rps", "brier", "logloss", "accuracy", "n"]]
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    cal = metrics.calibration_table(devig_normalize(oh, od, oa), outcomes)
    print("\ncalibration (de-vig normalize, pooled 1X2):")
    print(cal.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return table


# --------------------------------------------------------------------------- #
# B. Elo->1X2 walk-forward on international data
# --------------------------------------------------------------------------- #
def walkforward_elo() -> pd.DataFrame:
    enriched, _ = build_elo(save=False)
    enriched = enriched.copy()
    enriched["year"] = enriched["date"].dt.year
    enriched["outcome"] = metrics.outcomes_from_goals(
        enriched["home_goals"], enriched["away_goals"]
    )
    enriched["elo_diff_eff"] = elo_diff_effective(enriched)

    print("\n[leakage] walk-forward uses strictly train(date < Jan-1 of test year);"
          " Elo snapshots are pre-match. No future rows enter any split.")

    rows = []
    for y in TEST_YEARS:
        train = enriched[(enriched["year"] >= TRAIN_MIN_YEAR) & (enriched["year"] < y)]
        test = enriched[enriched["year"] == y]
        if len(test) < 50 or len(train) < 500:
            continue

        model = EloOrderedMap().fit(
            train["elo_diff_eff"].to_numpy(),
            train["outcome"].to_numpy(),
            weights=train["comp_weight"].to_numpy(),  # downweight friendlies
        )
        p = model.predict_proba(test["elo_diff_eff"].to_numpy())
        s = metrics.score_all(p, test["outcome"].to_numpy())

        # naive baseline: train-set outcome frequencies (constant forecast)
        base = np.bincount(train["outcome"].to_numpy(), minlength=3) / len(train)
        p_base = np.tile(base, (len(test), 1))
        s["rps_baserate"] = metrics.rps(p_base, test["outcome"].to_numpy())
        s["year"] = y
        rows.append(s)

    table = pd.DataFrame(rows)[
        ["year", "n", "rps", "rps_baserate", "brier", "logloss", "accuracy"]
    ]

    # aggregate: match-weighted means
    w = table["n"].to_numpy()
    agg = {
        "rps": np.average(table["rps"], weights=w),
        "rps_baserate": np.average(table["rps_baserate"], weights=w),
        "brier": np.average(table["brier"], weights=w),
        "logloss": np.average(table["logloss"], weights=w),
        "accuracy": np.average(table["accuracy"], weights=w),
        "n": int(w.sum()),
    }

    print("\n=== B. Elo->1X2 walk-forward (international, expanding window) ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\naggregate (match-weighted):")
    print(f"  Elo RPS        : {agg['rps']:.4f}")
    print(f"  base-rate RPS  : {agg['rps_baserate']:.4f}  (naive constant forecast)")
    print(f"  Elo Brier      : {agg['brier']:.4f}")
    print(f"  Elo logloss    : {agg['logloss']:.4f}")
    print(f"  Elo accuracy   : {agg['accuracy']:.3f}")
    print(f"  matches scored : {agg['n']:,}")
    improvement = 100 * (agg["rps_baserate"] - agg["rps"]) / agg["rps_baserate"]
    print(f"  Elo beats base-rate by {improvement:.1f}% RPS")
    return table


# --------------------------------------------------------------------------- #
# C. Paired Dixon–Coles vs Elo baseline (same walk-forward splits)
# --------------------------------------------------------------------------- #
def _bootstrap_ci(diff: np.ndarray, n_boot: int = 2000, seed: int = 7):
    rng = np.random.default_rng(seed)
    n = len(diff)
    means = np.empty(n_boot)
    for b in range(n_boot):
        means[b] = diff[rng.integers(0, n, n)].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def walkforward_compare() -> pd.DataFrame:
    enriched, _ = build_elo(save=False)
    enriched = enriched.copy()
    enriched["year"] = enriched["date"].dt.year
    enriched["outcome"] = metrics.outcomes_from_goals(
        enriched["home_goals"], enriched["away_goals"]
    )
    enriched["elo_diff_eff"] = elo_diff_effective(enriched)

    print("\n[leakage] both models train strictly on date < Jan-1 of the test year.")
    rows = []
    elo_rps_all, dc_rps_all = [], []
    for y in TEST_YEARS:
        split = f"{y}-01-01"
        train = enriched[(enriched["year"] >= TRAIN_MIN_YEAR) & (enriched["date"] < split)]
        test = enriched[enriched["year"] == y]
        if len(test) < 50 or len(train) < 500:
            continue

        elo = EloOrderedMap().fit(
            train["elo_diff_eff"].to_numpy(), train["outcome"].to_numpy(),
            weights=train["comp_weight"].to_numpy(),
        )
        p_elo = elo.predict_proba(test["elo_diff_eff"].to_numpy())

        dc = DixonColes().fit(enriched[enriched["date"] < split], ref_date=split)
        p_dc = dc.predict_proba_frame(test)

        out = test["outcome"].to_numpy()
        r_elo = metrics.rps_vec(p_elo, out)
        r_dc = metrics.rps_vec(p_dc, out)
        elo_rps_all.append(r_elo)
        dc_rps_all.append(r_dc)
        rows.append({
            "year": y, "n": len(test),
            "elo_rps": r_elo.mean(), "dc_rps": r_dc.mean(),
            "delta": r_dc.mean() - r_elo.mean(),
            "dc_conv": dc.converged_,
        })

    table = pd.DataFrame(rows)
    r_elo_all = np.concatenate(elo_rps_all)
    r_dc_all = np.concatenate(dc_rps_all)
    diff = r_dc_all - r_elo_all
    lo, hi = _bootstrap_ci(diff)

    print("\n=== C. Dixon–Coles vs Elo baseline (paired walk-forward) ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\naggregate (match-weighted over all test matches):")
    print(f"  Elo RPS          : {r_elo_all.mean():.4f}")
    print(f"  Dixon-Coles RPS  : {r_dc_all.mean():.4f}")
    print(f"  mean delta (DC-Elo): {diff.mean():+.4f}  "
          f"(negative = DC better)")
    print(f"  bootstrap 95% CI on delta: [{lo:+.4f}, {hi:+.4f}]")
    print(f"  matches scored   : {len(diff):,}")
    verdict = "DC better" if hi < 0 else ("Elo better" if lo > 0 else "no significant difference")
    print(f"  verdict          : {verdict} (95% CI vs 0)")
    return table


# --------------------------------------------------------------------------- #
# D. Hierarchical Bayesian Poisson vs Elo/DC (recent walk-forward splits)
# --------------------------------------------------------------------------- #
BAYES_TEST_YEARS = [2019, 2021, 2022, 2023, 2024, 2025]  # 跳过 2020(COVID 样本少)


def walkforward_bayes(test_years=BAYES_TEST_YEARS, window_years: int = 8) -> pd.DataFrame:
    enriched, _ = build_elo(save=False)
    enriched = enriched.copy()
    enriched["year"] = enriched["date"].dt.year
    enriched["outcome"] = metrics.outcomes_from_goals(
        enriched["home_goals"], enriched["away_goals"]
    )
    enriched["elo_diff_eff"] = elo_diff_effective(enriched)

    print("\n[leakage] 三个模型均严格只用 date < 测试年 1-1 的数据训练;Elo 快照为赛前。")
    rows = []
    elo_all, dc_all, bayes_all, out_all = [], [], [], []
    for y in test_years:
        split = f"{y}-01-01"
        train_recent = enriched[enriched["date"] < split]
        train_elo = enriched[(enriched["year"] >= TRAIN_MIN_YEAR) & (enriched["date"] < split)]
        test = enriched[enriched["year"] == y]
        out = test["outcome"].to_numpy()

        elo = EloOrderedMap().fit(
            train_elo["elo_diff_eff"].to_numpy(), train_elo["outcome"].to_numpy(),
            weights=train_elo["comp_weight"].to_numpy())
        p_elo = elo.predict_proba(test["elo_diff_eff"].to_numpy())

        dc = DixonColes().fit(train_recent, ref_date=split)
        p_dc = dc.predict_proba_frame(test)

        bp = BayesPoisson(window_years=window_years).fit(train_recent, ref_date=split)
        p_bayes = bp.predict_proba_frame(test)

        elo_all.append(metrics.rps_vec(p_elo, out))
        dc_all.append(metrics.rps_vec(p_dc, out))
        bayes_all.append(metrics.rps_vec(p_bayes, out))
        out_all.append(out)
        # 保存贝叶斯概率用于校准
        if y == test_years[0]:
            cal_probs, cal_out = [p_bayes], [out]
        else:
            cal_probs.append(p_bayes); cal_out.append(out)

        rows.append({
            "year": y, "n": len(test),
            "elo_rps": elo_all[-1].mean(), "dc_rps": dc_all[-1].mean(),
            "bayes_rps": bayes_all[-1].mean(),
            "rhat": bp.rhat_max_, "div": bp.divergences_,
        })
        print(f"  [{y}] n={len(test)} elo={rows[-1]['elo_rps']:.4f} "
              f"dc={rows[-1]['dc_rps']:.4f} bayes={rows[-1]['bayes_rps']:.4f} "
              f"(r_hat={bp.rhat_max_:.3f} div={bp.divergences_})")

    table = pd.DataFrame(rows)
    r_elo = np.concatenate(elo_all)
    r_dc = np.concatenate(dc_all)
    r_bayes = np.concatenate(bayes_all)
    diff_be = r_bayes - r_elo
    lo, hi = _bootstrap_ci(diff_be)

    print("\n=== D. 分层贝叶斯泊松 vs Elo/DC(近年走步前向)===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n聚合(按场加权,相同测试集):")
    print(f"  Elo RPS          : {r_elo.mean():.4f}")
    print(f"  Dixon-Coles RPS  : {r_dc.mean():.4f}")
    print(f"  Bayesian RPS     : {r_bayes.mean():.4f}")
    print(f"  delta(Bayes-Elo) : {diff_be.mean():+.4f}  bootstrap 95% CI [{lo:+.4f}, {hi:+.4f}]")
    verdict = "Bayes 显著更好" if hi < 0 else ("Elo 显著更好" if lo > 0 else "无显著差异")
    print(f"  结论             : {verdict}(95% CI vs 0)")
    print(f"  采样诊断         : r_hat_max={table['rhat'].max():.3f}, 总发散={int(table['div'].sum())}")

    cal = metrics.calibration_table(np.vstack(cal_probs), np.concatenate(cal_out))
    print("\n贝叶斯校准(pooled 1X2,10 分箱):")
    print(cal.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return table


if __name__ == "__main__":
    validate_devig_on_club()
    walkforward_elo()
    walkforward_compare()
    walkforward_bayes()
