# ⚽ 2026 FIFA World Cup — Probabilistic Prediction System

端到端的 2026 世界杯概率预测系统:逐场比分分布 → 1X2 / 大小球,蒙特卡洛锦标赛模拟
(各队各阶段晋级概率),Streamlit 交互 UI,以及对标去水后博彩盘口的**诚实 RPS 基准**。

> 核心原则(详见 [`CLAUDE.md`](CLAUDE.md)):**无时间泄漏**(所有特征 as-of-t,只用扩张窗
> 走步前向,绝不随机折)· **RPS 为主指标** · **市场 de-vig 基线是要超越的标杆** · 中立场
> 主场优势归零(仅三东道主在本国为主场)· `no_odds` / `with_odds` 双变体分训分评。

## 🔴 实时预测(截止 2026-06-22,赛事进行中)

灌入已踢完的 39 场小组赛、把它们**钉死**进模拟、只采样剩余 33 场,夺冠概率 Top-10:

| 队 | P(夺冠) | P(决赛) | P(四强) | P(出线 R16) |
|---|---|---|---|---|
| Argentina | **20.2%** | 30.9% | 45.0% | 82.3% |
| Brazil | 15.7% | 26.6% | 40.2% | 71.4% |
| Spain | 11.5% | 19.4% | 34.2% | 68.7% |
| England | 9.7% | 19.0% | 34.1% | 74.1% |
| France | 6.8% | 14.5% | 28.5% | 73.7% |
| Portugal | 5.7% | 10.9% | 21.2% | 62.6% |
| Colombia | 5.6% | 11.2% | 21.7% | 74.0% |
| Belgium | 4.4% | 9.0% | 18.8% | 61.0% |
| Netherlands | 3.0% | 7.0% | 13.9% | 46.2% |
| Germany | 2.6% | 6.9% | 14.8% | 71.1% |

> ✅ 合理性检查:夺冠概率之和=1.000,最大仅 20.2%(无 40%+)。随赛事推进定期把
> `REF_DATE` 推到当天、重跑 `build_artifacts` 即滚动更新。

## 🎯 诚实性结论(头号模型 = `no_odds` 分层贝叶斯泊松)

**① 本届真实样本外(最硬信号,39 场 6-11→6-21,滚动 walk-forward 无泄漏):**

| 模型 | RPS | 准确率 |
|---|---|---|
| **贝叶斯 (no_odds)** | **0.1641** | 61.5% |
| Elo (no_odds) | 0.1779 | 56.4% |
| 均匀 1/3(无技能地板) | 0.2222 | 51.3% |

两模型均大幅低于无技能地板 → 确有真实预测力;贝叶斯−Elo delta −0.0139,95%CI
[−0.0355, +0.0060] → 样本仅 39 场,**统计上无显著差异**。本届暂无国家队收盘赔率,市场基线缺失。

**② 历史大赛配对 vs 去水市场(229 场,2018/2022 WC + 2020/2024 Euro):**

| 模型 | 合并 RPS | vs 市场(95% CI) | 结论 |
|---|---|---|---|
| 市场 de-vig 基线 | **0.1950** | — | — |
| **贝叶斯 (no_odds)** | 0.1994 | +0.0044 [−0.0041, +0.0129] | **无显著差异(追平)** |
| Elo (no_odds) | 0.2062 | +0.0111 [+0.0025, +0.0206] | 市场显著更好 |

> no_odds 贝叶斯**追平**高效盘口但未击败;样本增至 229 场后 Elo 已显著逊于市场——这正是
> 贝叶斯相对 Elo 的价值在大样本下的体现。它选作头号模型的理由是 Elo 给不了的:完整后验
> 比分分布、低样本球队部分汇合收缩、把后验不确定性传播进模拟。

## 🚀 快速开始

```bash
conda env create -f worldcup2026/environment.yml   # 创建 wc2026 环境
conda activate wc2026
# Kaggle 凭据放在 ~/.kaggle/kaggle.json (chmod 600)

python -m worldcup2026.build_artifacts             # 一次性:贝叶斯模型 + 50k 模拟 + 报告
streamlit run worldcup2026/app.py                  # 启动 UI
```

UI 四标签:① 比赛预测器(1X2 + 比分热力图 + 大小球) ② 锦标赛概率 ③ Bracket 视图(模拟一次)
④ 模型 vs 市场(含本届真实样本外 + 校准曲线)。

> ⚠️ 数据缓存(Kaggle / football-data / oddsportal)与浏览器 profile **不入库**,可由
> `worldcup2026/data` 的 ingestion 层重建。详细里程碑数字与设计说明见
> [`worldcup2026/README.md`](worldcup2026/README.md)。

### 🔁 赛事进行中的每日滚动更新

终场比分一落地,三步即可把预测推到当天:

```bash
# 1) 把新确认的终场比分加进 LIVE_RESULTS,再幂等合并进 matches.parquet
#    (Kaggle 通常滞后几天;权威终场可手工补,无确认比分则先略过)
python -m worldcup2026.data.patch_live_results

# 2) 用当天日期重建工件(训练截止 + 钉死已踢小组赛都按 --date 走)
python -m worldcup2026.build_artifacts --date 2026-06-23

# 3) 提交并推送
git add -A && git commit -m "rolling update: 2026-06-23" && git push
```

> `--date` 让训练/条件化截止可参数化,默认仍是 `REF_DATE`,保证旧工件可复现。

## 📦 结构

```
worldcup2026/
  data/      ingest.py(Kaggle→matches.parquet)· patch_live_results.py(实时赛果)· groups_2026.json
  features/  elo.py(自实现无泄漏 Elo)· as_of.py(form/rest)
  models/    baseline · dixon_coles · bayes_poisson(头号)· 工件 .npz/.csv/.json
  eval/      walkforward · metrics · market_compare · wc2026_score(本届样本外)
  sim/       tournament.py(向量化蒙特卡洛 + 已赛条件化)
  app.py     Streamlit UI · tests/  泄漏/平局/模拟测试
```

## ✅ 里程碑

MS0 数据+Elo · MS1 市场基线 · MS2 Dixon–Coles · MS3 贝叶斯(头号)· MS4 锦标赛模拟 · MS5 UI
—— 全部完成,另加实时赛果条件化与本届真实样本外诚实性检查。

---
🤖 与 [Claude Code](https://claude.com/claude-code) 协作构建。
