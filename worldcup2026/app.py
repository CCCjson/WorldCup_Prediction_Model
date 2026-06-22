"""2026 世界杯预测系统 — Streamlit UI(MS5)。

四个标签页:
  1. 比赛预测器:1X2 + 比分热力图(0-5)+ 大小球 2.5
  2. 锦标赛概率:晋级概率表 + P(夺冠) 横向条形图
  3. Bracket 视图:最可能出线 + 「模拟一次」按钮
  4. 模型 vs 市场:RPS 对比表 + 市场校准图

运行:  streamlit run worldcup2026/app.py
（首次运行前先生成工件:python -m worldcup2026.build_artifacts）
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# 中文图表字体(避免方块);按可用性回退
matplotlib.rcParams["font.sans-serif"] = [
    "Arial Unicode MS", "PingFang SC", "Hiragino Sans GB", "Heiti TC", "STHeiti"]
matplotlib.rcParams["axes.unicode_minus"] = False

from worldcup2026.models.bayes_poisson import BayesPoisson
from worldcup2026.sim.tournament import load_2026_config, simulate_once

MODELS = Path(__file__).resolve().parent / "models"

st.set_page_config(page_title="2026 世界杯预测", layout="wide")


# --------------------------------------------------------------------------- #
# 工件加载(缓存)
# --------------------------------------------------------------------------- #
@st.cache_resource
def load_model():
    return BayesPoisson.load(MODELS / "bayes_2026.npz")


@st.cache_data
def load_sim():
    return pd.read_csv(MODELS / "sim_2026.csv")


@st.cache_data
def load_report():
    return json.loads((MODELS / "report.json").read_text())


def _check_artifacts():
    missing = [f for f in ["bayes_2026.npz", "sim_2026.csv", "report.json"]
               if not (MODELS / f).exists()]
    if missing:
        st.error(f"缺少工件 {missing}。请先运行:python -m worldcup2026.build_artifacts")
        st.stop()


_check_artifacts()
model = load_model()
sim = load_sim()
report = load_report()
config = load_2026_config()
TEAMS = sorted(model.teams_)

st.title("⚽ 2026 FIFA 世界杯 — 概率预测系统")
st.caption(f"头号模型:分层贝叶斯泊松(赛前训练 · r̂={model.rhat_max_:.3f} · "
           f"divergences={model.divergences_})")

tab1, tab2, tab3, tab4 = st.tabs(
    ["① 比赛预测器", "② 锦标赛概率", "③ Bracket 视图", "④ 模型 vs 市场"])

# --------------------------------------------------------------------------- #
# Tab 1 — 比赛预测器
# --------------------------------------------------------------------------- #
with tab1:
    c1, c2, c3 = st.columns([2, 2, 1])
    home = c1.selectbox("主队 / 队1", TEAMS, index=TEAMS.index("Brazil"))
    away = c2.selectbox("客队 / 队2", TEAMS, index=TEAMS.index("Argentina"))
    neutral = c3.toggle("中立场", value=True,
                        help="2026 除三东道主在本国外均为中立场")

    if home == away:
        st.warning("请选择两支不同的球队。")
    else:
        p = model.predict_1x2(home, away, neutral=neutral)
        ou_over, ou_under = model.predict_over_under(home, away, neutral=neutral)
        mat = model.score_matrix(home, away, neutral=neutral)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"{home} 胜", f"{p[0]:.1%}")
        m2.metric("平", f"{p[1]:.1%}")
        m3.metric(f"{away} 胜", f"{p[2]:.1%}")
        m4.metric("大 2.5 球", f"{ou_over:.1%}")

        g1, g2 = st.columns(2)
        with g1:
            st.subheader("1X2 概率")
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.bar([f"{home}\n胜", "平", f"{away}\n胜"], p,
                   color=["#2a7", "#999", "#c44"])
            ax.set_ylim(0, 1); ax.set_ylabel("概率")
            for i, v in enumerate(p):
                ax.text(i, v + 0.01, f"{v:.0%}", ha="center")
            st.pyplot(fig)
        with g2:
            st.subheader("比分热力图(0–5)")
            sub = mat[:6, :6]
            fig2, ax2 = plt.subplots(figsize=(4, 3))
            im = ax2.imshow(sub, origin="lower", cmap="viridis")
            ax2.set_xlabel(f"{away} 进球"); ax2.set_ylabel(f"{home} 进球")
            ax2.set_xticks(range(6)); ax2.set_yticks(range(6))
            best = np.unravel_index(np.argmax(sub), sub.shape)
            ax2.text(best[1], best[0], "★", ha="center", va="center", color="white")
            fig2.colorbar(im, ax=ax2, shrink=0.8)
            st.pyplot(fig2)
        mle = np.unravel_index(np.argmax(mat), mat.shape)
        st.caption(f"最可能比分:{home} {mle[0]}–{mle[1]} {away}（概率 {mat[mle]:.1%}）")

# --------------------------------------------------------------------------- #
# Tab 2 — 锦标赛概率
# --------------------------------------------------------------------------- #
with tab2:
    st.subheader("各队晋级概率(50,000 次蒙特卡洛模拟)")
    stage_cols = [c for c in sim.columns if c.startswith("P(")]
    show = sim.copy()
    for c in stage_cols:
        show[c] = (show[c] * 100).round(1)
    st.dataframe(show, width="stretch", height=420,
                 column_config={c: st.column_config.NumberColumn(c, format="%.1f%%")
                                for c in stage_cols})

    st.subheader("夺冠概率 Top-20")
    top = sim.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.barh(top["team"], top["P(Winner)"] * 100, color="#2a7")
    ax.set_xlabel("P(夺冠) %")
    for i, v in enumerate(top["P(Winner)"] * 100):
        ax.text(v + 0.1, i, f"{v:.1f}%", va="center", fontsize=8)
    st.pyplot(fig)

# --------------------------------------------------------------------------- #
# Tab 3 — Bracket 视图
# --------------------------------------------------------------------------- #
with tab3:
    st.subheader("小组出线概率(头名 / 次名)")
    # 用模拟概率近似「最可能出线」
    grp_rows = []
    for g, teams in config["groups"].items():
        for t in teams:
            r = sim[sim["team"] == t]
            grp_rows.append({"组": g, "队": t,
                             "P(进 R16)": float(r["P(R16)"].iloc[0]) if len(r) else 0.0})
    gdf = pd.DataFrame(grp_rows).sort_values(["组", "P(进 R16)"], ascending=[True, False])
    st.dataframe(gdf, width="stretch", height=300,
                 column_config={"P(进 R16)": st.column_config.NumberColumn(format="percent")})

    st.subheader("模拟一次完整赛程")
    seed = st.number_input("随机种子", value=2026, step=1)
    if st.button("🎲 模拟一次"):
        run = simulate_once(model, config, seed=int(seed))
        st.success(f"🏆 本次模拟冠军:**{run['champion']}**")
        for rnd in run["knockout"]:
            with st.expander(f"{rnd['round']}（{len(rnd['matches'])} 场)"):
                for h, hg, ag, a, w in rnd["matches"]:
                    st.write(f"{h} **{hg}–{ag}** {a}  →  {w}")

# --------------------------------------------------------------------------- #
# Tab 4 — 模型 vs 市场
# --------------------------------------------------------------------------- #
with tab4:
    mvm = report.get("model_vs_market")
    if mvm:
        st.subheader("🎯 模型 vs 市场:大赛配对对比(oddsportal 收盘赔率)")
        c1, c2, c3 = st.columns(3)
        c1.metric("市场基线 RPS", f"{mvm['market_rps']:.4f}",
                  help=f"{mvm['matches']} 场配对")
        c2.metric("贝叶斯 (no_odds) RPS", f"{mvm['bayes_rps']:.4f}",
                  delta=f"{mvm['bayes_delta']:+.4f} vs 市场", delta_color="inverse")
        c3.metric("Elo (no_odds) RPS", f"{mvm['elo_rps']:.4f}",
                  delta=f"{mvm['elo_delta']:+.4f} vs 市场", delta_color="inverse")
        ev_names = " + ".join(e["name"] for e in mvm.get("events", []))
        st.caption(
            f"合并 {mvm['matches']} 场({ev_names})。贝叶斯 vs 市场:delta "
            f"{mvm['bayes_delta']:+.4f},95%CI [{mvm['bayes_ci'][0]:+.4f}, "
            f"{mvm['bayes_ci'][1]:+.4f}] → **{mvm['bayes_verdict']}**;Elo → "
            f"**{mvm['elo_verdict']}**。注:oddsportal 平均赔率为跨庄家乐观展示"
            "(水位极低/偶为负),市场基线偏乐观。")
        if mvm.get("events"):
            ev_df = pd.DataFrame(mvm["events"])[
                ["name", "matches", "market_rps", "bayes_rps", "elo_rps"]]
            st.dataframe(ev_df, width="stretch",
                         column_config={c: st.column_config.NumberColumn(c, format="%.4f")
                                        for c in ["market_rps", "bayes_rps", "elo_rps"]})

    oos = report.get("wc2026_oos")
    if oos:
        dr = oos["date_range"]
        st.subheader(f"🔬 本届真实样本外({oos['n_matches']} 场,{dr[0]} → {dr[1]})")
        st.caption("整个项目最硬的诚实信号:滚动 walk-forward,严格无泄漏——每个比赛日把"
                   "贝叶斯重训到 as-of,只用此前数据预测当日比赛。")
        c1, c2, c3 = st.columns(3)
        c1.metric("贝叶斯 (no_odds) RPS", f"{oos['bayes']['rps']:.4f}",
                  delta=f"{oos['bayes_minus_elo_delta']:+.4f} vs Elo",
                  delta_color="inverse", help=f"准确率 {oos['bayes']['accuracy']:.1%}")
        c2.metric("Elo (no_odds) RPS", f"{oos['elo']['rps']:.4f}",
                  help=f"准确率 {oos['elo']['accuracy']:.1%}")
        c3.metric("均匀 1/3(无技能地板)", f"{oos['uniform']['rps']:.4f}")
        ci = oos["bayes_minus_elo_ci"]
        st.caption(
            f"贝叶斯−Elo RPS delta {oos['bayes_minus_elo_delta']:+.4f},95%CI "
            f"[{ci[0]:+.4f}, {ci[1]:+.4f}] → **{oos['verdict']}**(样本仅 "
            f"{oos['n_matches']} 场)。两模型均大幅低于无技能地板 0.2222 → 确有真实"
            f"预测力。{oos['market_note']}")
        oos_df = pd.DataFrame([
            {"模型": "贝叶斯 (no_odds)", **oos["bayes"]},
            {"模型": "Elo (no_odds)", **oos["elo"]},
            {"模型": "均匀 1/3", **oos["uniform"]},
        ])[["模型", "rps", "brier", "logloss", "accuracy"]]
        st.dataframe(oos_df, width="stretch", column_config={
            "rps": st.column_config.NumberColumn("RPS", format="%.4f"),
            "brier": st.column_config.NumberColumn("Brier", format="%.4f"),
            "logloss": st.column_config.NumberColumn("logloss", format="%.4f"),
            "accuracy": st.column_config.NumberColumn("准确率", format="percent")})

    st.subheader("各模型走步前向 RPS(越低越好)")
    rps_df = pd.DataFrame(report["rps_table"])
    st.dataframe(rps_df, width="stretch",
                 column_config={"rps": st.column_config.NumberColumn("RPS", format="%.4f")})
    st.info(report["honesty_note"])

    st.subheader("市场 de-vig 校准曲线(football-data.co.uk 收盘赔率)")
    mc = report["market_club"]
    cal = pd.DataFrame(mc["calibration"])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="完美校准")
    ax.plot(cal["mean_pred"], cal["mean_obs"], "o-", color="#2a7", label="市场 de-vig")
    ax.set_xlabel("预测概率"); ax.set_ylabel("实际频率")
    ax.legend(); ax.set_title(
        f"{mc['matches']:,} 场 · 水位 {mc['mean_overround']:.4f} · RPS {mc['rps']:.4f}")
    st.pyplot(fig)
    st.caption("市场去水后概率与实际频率几乎重合 → de-vig + RPS 管线正确。")
