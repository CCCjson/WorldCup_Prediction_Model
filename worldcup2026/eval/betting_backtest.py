"""下注记分牌:edge / EV + Closing Line Value(CLV)回测引擎。

回答两个比 RPS 更要紧的问题:
  (a) 我方概率在哪个市场、哪些场次真有 edge?
  (b) 按策略下注,ROI 与 CLV 为正吗?

设计:**引擎与模型解耦**。`backtest_market()` 只吃「我方概率 + 开盘/收盘赔率 + 真实
结果」,算 edge、flat/分数凯利 ROI(bootstrap CI)、CLV。概率来源单独提供——本文件用
俱乐部 Dixon–Coles 走步前向作 proving ground;Phase 2 国家队衍生盘可复用同一引擎。

两种下注口径:
  - bet@open :开盘价决策+成交(真实可执行),据此算 **CLV**(开盘价 vs 收盘价)——
               稳定的 +CLV 是职业赌徒公认最强的 edge 信号。
  - bet@close:收盘价(最锐线)决策+成交,量「模型 vs 夏普收盘线」的纯 edge。
               健全性:用收盘隐含概率当「我方概率」回测,ROI 应 ≈ −水位(引擎不高估自己)。

Run:  python -m worldcup2026.eval.betting_backtest
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from worldcup2026.data.footballdata import build as build_club
from worldcup2026.models.baseline import devig_shin
from worldcup2026.models.dixon_coles import DixonColes

MODELS = Path(__file__).resolve().parent.parent / "models"
KELLY_FRACTION = 0.25       # 分数凯利:1/4,控方差
DEFAULT_EDGE_THRESHOLDS = [0.0, 0.02, 0.05]
RNG = np.random.default_rng(2026)


# --------------------------------------------------------------------------- #
# 纯下注数学
# --------------------------------------------------------------------------- #
def edge(my_p: np.ndarray, dec_odds: np.ndarray) -> np.ndarray:
    """EV per 1 unit staked = p·odds − 1。"""
    return my_p * dec_odds - 1.0


def kelly(my_p: np.ndarray, dec_odds: np.ndarray) -> np.ndarray:
    """全凯利下注比例 f* = (b·p − q)/b,b=odds−1。负值截 0(不下注)。"""
    b = dec_odds - 1.0
    f = (b * my_p - (1.0 - my_p)) / b
    return np.clip(f, 0.0, 1.0)


def _roi_ci(profit: np.ndarray, stake: np.ndarray, n_boot: int = 2000):
    """对 ROI = Σprofit/Σstake 做 bootstrap 95% CI(按注重采样)。"""
    if len(profit) == 0:
        return (float("nan"), float("nan"))
    idx = RNG.integers(0, len(profit), size=(n_boot, len(profit)))
    num = profit[idx].sum(axis=1)
    den = stake[idx].sum(axis=1)
    rois = num / np.where(den == 0, np.nan, den)
    return float(np.nanpercentile(rois, 2.5)), float(np.nanpercentile(rois, 97.5))


def _settle(sel_won: np.ndarray, dec_odds: np.ndarray, stake: np.ndarray) -> np.ndarray:
    """每注盈亏:赢 → stake·(odds−1),输 → −stake。"""
    return np.where(sel_won, stake * (dec_odds - 1.0), -stake)


def backtest_market(name, my_p, odds_close, odds_open, sel_won,
                    thresholds=DEFAULT_EDGE_THRESHOLDS, fade=False):
    """对一个市场的所有「选项×场次」候选注做回测。

    入参均为已展平的 1D 数组(每个候选下注项一行):
      my_p       我方概率;odds_close/open 该选项收/开盘赔率;sel_won 该选项是否命中。
    fade=True 时反着买:选 edge < −thr 的选项(回测「跟模型对着干」是否能赢)。
    返回 dict:两种口径(bet@close / bet@open)× 多个 edge 阈值的 ROI/CLV/CI。
    """
    out = {"market": name, "n_candidates": int(len(my_p)), "fade": fade,
           "by_strategy": []}

    for strat, odds_bet in [("bet@close", odds_close), ("bet@open", odds_open)]:
        valid = ~np.isnan(odds_bet) & ~np.isnan(my_p)
        e = edge(my_p, odds_bet)
        for thr in thresholds:
            pick = valid & ((e < -thr) if fade else (e > thr))
            n = int(pick.sum())
            row = {"strategy": strat, "edge_threshold": thr, "n_bets": n}
            if n == 0:
                row.update(roi_flat=None, roi_kelly=None)
                out["by_strategy"].append(row)
                continue
            o = odds_bet[pick]
            won = sel_won[pick]
            # flat 注
            st_f = np.ones(n)
            pf_f = _settle(won, o, st_f)
            roi_f = pf_f.sum() / st_f.sum()
            lo_f, hi_f = _roi_ci(pf_f, st_f)
            # 1/4 凯利(反买全是负 edge,凯利按定义不下注 -> 无意义,置 None)
            st_k = np.zeros(n) if fade else KELLY_FRACTION * kelly(my_p[pick], o)
            keep = st_k > 1e-9
            if keep.any():
                pf_k = _settle(won[keep], o[keep], st_k[keep])
                roi_k = round(pf_k.sum() / st_k[keep].sum(), 4)
                lo_k, hi_k = _roi_ci(pf_k, st_k[keep])
                roi_k_ci = [round(lo_k, 4), round(hi_k, 4)]
            else:
                roi_k, roi_k_ci = None, None
            row.update(
                roi_flat=round(roi_f, 4), roi_flat_ci=[round(lo_f, 4), round(hi_f, 4)],
                roi_kelly=roi_k, roi_kelly_ci=roi_k_ci,
                hit_rate=round(float(won.mean()), 4),
                mean_edge=round(float(e[pick].mean()), 4),
            )
            # CLV 只对 bet@open 有意义(开盘成交 vs 收盘比价)
            if strat == "bet@open":
                oc = odds_close[pick]
                ok = ~np.isnan(oc)
                clv = o[ok] / oc[ok] - 1.0
                row["clv_mean"] = round(float(clv.mean()), 4)
                row["clv_pct_positive"] = round(float((clv > 0).mean()), 4)
            out["by_strategy"].append(row)
    return out


# --------------------------------------------------------------------------- #
# Proving ground:俱乐部 Dixon–Coles 走步前向 → 我方概率
# --------------------------------------------------------------------------- #
def club_model_probs(club: pd.DataFrame, train_years: int = 3) -> pd.DataFrame:
    """按赛季走步前向拟合 Dixon–Coles,给每行填 1X2 + 大小球 2.5 概率(无泄漏)。"""
    df = club.copy()
    df["neutral"] = False                          # 俱乐部:主场优势生效
    df = df.sort_values("date").reset_index(drop=True)
    for c in ["p_home", "p_draw", "p_away", "p_over", "p_under"]:
        df[c] = np.nan

    seasons = sorted(df["season"].unique())
    for s in seasons:
        test_mask = df["season"] == s
        test = df[test_mask]
        start = test["date"].min()
        if pd.isna(start):
            continue
        train = df[(df["date"] < start)
                   & (df["date"] >= start - pd.Timedelta(days=365 * train_years))]
        if len(train) < 500:                        # 头几季训练量不足 -> 跳过(不评分)
            continue
        dc = DixonColes().fit(train, ref_date=start.strftime("%Y-%m-%d"))
        known = set(dc.teams_)
        n_pred = 0
        for i in test.index:
            h, a = df.at[i, "home_team"], df.at[i, "away_team"]
            if h not in known or a not in known:
                continue
            p = dc.predict_1x2(h, a, neutral=False)
            over, under = dc.predict_over_under(h, a, neutral=False, line=2.5)
            df.loc[i, ["p_home", "p_draw", "p_away", "p_over", "p_under"]] = [
                p[0], p[1], p[2], over, under]
            n_pred += 1
        print(f"  季 {s}: 训练 {len(train):,} 场 -> 预测 {n_pred}/{len(test)} 场")
    return df


def _flatten_1x2(df):
    """把每场 3 个选项展平成候选注数组。"""
    out_idx = np.where(df["ftr"].eq("H"), 0, np.where(df["ftr"].eq("D"), 1, 2))
    my_p = df[["p_home", "p_draw", "p_away"]].to_numpy()
    oc = df[["odds_home", "odds_draw", "odds_away"]].to_numpy()
    oo = df[["odds_home_open", "odds_draw_open", "odds_away_open"]].to_numpy()
    won = (np.arange(3)[None, :] == out_idx[:, None])
    return my_p.ravel(), oc.ravel(), oo.ravel(), won.ravel()


def _flatten_ou(df):
    d = df[df["ou_over"].notna() & df["p_over"].notna()]
    over_won = (d["home_goals"] + d["away_goals"]) > 2.5
    my_p = np.column_stack([d["p_over"], d["p_under"]])
    oc = np.column_stack([d["ou_over"], d["ou_under"]])
    oo = np.column_stack([d["ou_over_open"], d["ou_under_open"]])
    won = np.column_stack([over_won, ~over_won])
    return my_p.ravel(), oc.ravel(), oo.ravel(), won.ravel()


SCORED_CACHE = MODELS / "club_scored.parquet"


def run(use_cache=True):
    if use_cache and SCORED_CACHE.exists():
        print(f"[cache] {SCORED_CACHE.name} present — loading scored probs")
        scored = pd.read_parquet(SCORED_CACHE)
    else:
        club = build_club()
        print(f"=== 俱乐部 proving ground:Dixon–Coles 走步前向({len(club):,} 场)===")
        df = club_model_probs(club)
        scored = df[df["p_home"].notna()].copy()
        scored.to_parquet(SCORED_CACHE, index=False)
    print(f"可评分:{len(scored):,} 场\n")

    results = []
    p, oc, oo, won = _flatten_1x2(scored)
    results.append(backtest_market("1X2", p, oc, oo, won))
    results.append(backtest_market("1X2(反买)", p, oc, oo, won, fade=True))
    p, oc, oo, won = _flatten_ou(scored)
    results.append(backtest_market("大小球 2.5", p, oc, oo, won))
    results.append(backtest_market("大小球 2.5(反买)", p, oc, oo, won, fade=True))

    # 健全性:用收盘隐含概率当「我方概率」回测 1X2,ROI 应 ≈ −水位
    imp = devig_shin(scored["odds_home"], scored["odds_draw"], scored["odds_away"])
    out_idx = np.where(scored["ftr"].eq("H"), 0,
                       np.where(scored["ftr"].eq("D"), 1, 2))
    won_s = (np.arange(3)[None, :] == out_idx[:, None]).ravel()
    oc_s = scored[["odds_home", "odds_draw", "odds_away"]].to_numpy().ravel()
    # 全押(threshold=−1 → 恒下注):盲押收盘线,ROI 应 ≈ −水位
    sanity = backtest_market("健全性:盲押收盘线", imp.ravel(), oc_s, oc_s, won_s,
                             thresholds=[-1.0])

    report = {"proving_ground": "club big-5 (football-data)",
              "n_scored": int(len(scored)), "markets": results, "sanity": sanity}
    (MODELS / "betting_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2))
    _print_report(report)
    return report


def _print_report(report):
    for mk in report["markets"]:
        print(f"\n### {mk['market']}(候选注 {mk['n_candidates']:,})")
        print(f"{'口径':<10}{'edge阈值':>8}{'注数':>8}{'flat ROI':>12}"
              f"{'1/4Kelly ROI':>14}{'命中':>8}{'CLV均值':>10}{'CLV>0占比':>10}")
        for r in mk["by_strategy"]:
            if r.get("roi_flat") is None:
                continue
            clv = f"{r.get('clv_mean', ''):>10}" if "clv_mean" in r else f"{'':>10}"
            clvp = (f"{r.get('clv_pct_positive', ''):>10}" if "clv_pct_positive" in r
                    else f"{'':>10}")
            kelly_s = f"{r['roi_kelly']:>14.4f}" if r["roi_kelly"] is not None else f"{'—':>14}"
            print(f"{r['strategy']:<10}{r['edge_threshold']:>8.2f}{r['n_bets']:>8}"
                  f"{r['roi_flat']:>12.4f}{kelly_s}"
                  f"{r['hit_rate']:>8.3f}{clv}{clvp}")
    s = report["sanity"]["by_strategy"][0]
    if s.get("roi_flat") is not None:
        print(f"\n[健全性] 盲押收盘线 1X2:flat ROI = {s['roi_flat']:+.4f} "
              f"(应 ≈ −水位,验证引擎不高估自己)")


if __name__ == "__main__":
    run()
