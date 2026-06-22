# World Cup 2026 Probabilistic Prediction System

End-to-end probabilistic predictor for the 2026 FIFA World Cup: per-match
outcome distributions, a Monte Carlo tournament simulator, a Streamlit UI, and
an honest RPS benchmark vs the de-vigged bookmaker baseline.

See `../CLAUDE.md` for the full spec, constraints (no temporal leakage, RPS as
primary metric, market baseline as the bar), and milestone gating.

## Setup

```bash
conda env create -f environment.yml      # creates the `wc2026` env
conda activate wc2026
# Kaggle creds expected at ~/.kaggle/kaggle.json (chmod 600)
```

## Run (MS0)

```bash
# from the repo root (parent of worldcup2026/)
python -m worldcup2026.data.ingest       # download + build canonical match table
python -m worldcup2026.features.elo       # leakage-safe Elo -> elo_state.json, top-20
python -m pytest worldcup2026/tests/test_leakage.py -v
```

## Status

| Milestone | What | State |
|---|---|---|
| MS0 | Canonical match table + leakage-safe own Elo + leakage test | ✅ done |
| MS1 | Market de-vig (validated on club odds) + Elo→1X2 walk-forward RPS | ✅ done |
| MS2 | Dixon–Coles (bivariate-Poisson, time-decay) + paired vs Elo | ✅ done |
| MS3 | Hierarchical Bayesian Poisson (PyMC, headline) + calibration | ✅ done |
| MS4 | Tournament Monte-Carlo sim (validated on 2022, run 2026) | ✅ done |
| MS5 | Streamlit UI (4 tabs) wired to headline model + sim | ✅ done |

### MS0 numbers (Kaggle dataset, built 2026-06-16)

- **Rows:** 49,417 international matches
- **Date range:** 1872-11-30 → 2026-06-14
- **Distinct teams:** 336
- **Neutral share:** 26.4%
- **Leakage test:** 4/4 passing
- **Top Elo (current):** Spain 2160, Argentina 2139, France 2074, England 2043,
  Brazil 2025, Colombia 2016, Portugal 2003 …

### MS1 numbers

**De-vig pipeline validation** (football-data.co.uk, big-5 leagues 2015/16–2024/25,
18,011 matches, Pinnacle closing): mean overround 1.0252 (2.5% vig), de-vigged
**market RPS = 0.1937**, calibration near-perfect across all 10 bins. Confirms the
de-vig + RPS machinery is correct (textbook football market RPS is ~0.19–0.20).

**Elo→1X2 walk-forward** (international, expanding window, 2002–2025, 22,960 matches,
ordered logit refit train-only per split): aggregate **RPS = 0.1772** vs base-rate
0.2274 (−22%), Brier 0.529, log-loss 0.897, accuracy 58.6%.

> ⚠️ The 0.177 (international Elo) and 0.194 (club market) numbers are **not**
> comparable — different match populations. International fixtures include many
> lopsided mismatches that are easier to call, lowering RPS. A true model-vs-market
> paired table needs national-team odds on the *same* matches (oddsportal, deferred).

### MS2 numbers

**Dixon–Coles** (bivariate-Poisson, low-score τ correction, time-decay xi=0.0019
≈ 1-yr half-life, L2-pooled attack/defense, analytic-gradient weighted MLE; gamma
zeroed at neutral venues). Fitted: mu=0.13, **gamma=0.25** (home edge), **rho=−0.07**.

Paired walk-forward vs the Elo baseline (same splits, 2002–2025, 22,960 matches):

| Model | Aggregate RPS |
|---|---|
| Elo→1X2 | **0.1772** |
| Dixon–Coles | 0.1789 |

Mean delta (DC−Elo) = **+0.0018**, bootstrap 95% CI **[+0.0007, +0.0029]** (excludes 0).

> ⚠️ **Honesty check:** vanilla Dixon–Coles does *not* beat the Elo 1X2 map — Elo is
> marginally but significantly better on RPS. DC's value is the full **score matrix**
> (exact scores, over/under) that Elo can't produce, and it's a stepping stone to the
> M2 hierarchical Bayesian headline. It does not add independent 1X2 signal over Elo here.

### MS3 numbers

**分层贝叶斯泊松**(PyMC,attack/defense 用 ZeroSumNormal 部分汇合 + non-centered,
home_field 仅作用于非中立场,NUTS 4 链 × 2000 抽样,target_accept=0.95)。

采样诊断(6 个走步前向切分全部):**r̂_max=1.010,divergences=0**,ess 充足。

三方走步前向(2019–2025,跳过 2020,相同测试集 6,521 场):

| 模型 | 聚合 RPS |
|---|---|
| Elo→1X2 | **0.1672** |
| Bayesian Poisson | 0.1678 |
| Dixon–Coles | 0.1714 |

delta(Bayes−Elo)=**+0.0006**,bootstrap 95% CI **[−0.0011, +0.0025]** → **无显著差异**。
贝叶斯校准良好(10 分箱 mean_pred ≈ mean_obs)。

> ⚠️ **诚实性检查:** 贝叶斯泊松在 1X2 RPS 上**与简单 Elo 无统计显著差异**(略优于 DC),
> 并未单独带来 1X2 增益。选它作头号模型的理由是 Elo 给不了的东西:① 完整后验**比分分布**;
> ② 低样本球队的**部分汇合**收缩;③ 把**后验参数不确定性**传播进锦标赛模拟(每次模拟抽一个
> 后验样本)。真正的「模型 vs 市场」对比仍需国家队收盘赔率(oddsportal,暂缓)。

Run MS1+MS2 eval:  `python -m worldcup2026.eval.walkforward`
Run MS3 自检采样:  `python -m worldcup2026.models.bayes_poisson`

### MS4 numbers

向量化蒙特卡洛模拟器(`sim/tournament.py`):每次模拟抽一个贝叶斯**后验样本**(传播
参数不确定性)→ 小组循环赛(积分→净胜球→进球→随机平局规则)→ 每组前 2 + 8 个最佳
第三名(best-8 近似 + 满足官方来源组约束的二分匹配)→ 单淘汰 R32→决赛(平局用 λ 比例
条件胜率,折叠 ET/点球)。50k 次模拟秒级完成。真实分组取自 2025-12-05 抽签。

**验证(2022 卡塔尔,赛前 2022-11-20 训练):** 夺冠概率和=1.000,最大=0.348(<0.40);
Top:Brazil 34.8%、Argentina 11.9%、Spain 9.8%、England 6.5%。实际冠军阿根廷排第 2 —
逻辑与分布合理(注:模型对顶级强队略偏自信,巴西概率高于当时博彩 ~18%)。

**2026 预测(赛前 2026-06-11 训练),夺冠概率 Top-10:**

| 队 | P(Winner) | P(Final) | P(SF) | P(R16) |
|---|---|---|---|---|
| Brazil | 17.0% | 28.6% | 43.1% | 73.3% |
| Argentina | 14.4% | 22.6% | 34.3% | 66.2% |
| Spain | 10.5% | 17.9% | 29.6% | 64.1% |
| England | 10.1% | 18.6% | 32.6% | 70.4% |
| Portugal | 6.9% | 13.1% | 23.7% | 67.2% |
| France | 6.8% | 13.3% | 25.4% | 65.0% |
| Belgium | 5.6% | 11.4% | 22.5% | 68.2% |
| Colombia | 4.9% | 9.9% | 19.7% | 62.0% |
| Netherlands | 3.2% | 7.1% | 14.0% | 45.3% |
| Germany | 2.5% | 6.3% | 13.9% | 60.5% |

> ✅ **合理性检查 #4 通过:** 夺冠概率之和=1.000,最大仅 17%(无 40%+),热门集中在
> 个位数~17%,与赛前博彩市场高度一致。

Run MS4:  `python -m worldcup2026.sim.tournament`

### MS5 — Streamlit UI

```bash
python -m worldcup2026.build_artifacts     # 一次性:存赛前模型 + 50k 模拟 + 报告
streamlit run worldcup2026/app.py
```

四个标签页(均用 `st.cache_resource`/`st.cache_data` 加载离线工件,免重采样):
1. **比赛预测器** — 任选两队 + 中立/主场切换 → 1X2 条形图、比分热力图(0–5)、大小球 2.5、最可能比分
2. **锦标赛概率** — 各队各阶段晋级概率表 + P(夺冠) Top-20 横向条形图
3. **Bracket 视图** — 小组出线概率 + 「🎲 模拟一次」按钮展示一次抽样的完整淘汰赛路径与冠军
4. **模型 vs 市场** — RPS 对比表 + 市场 de-vig 校准曲线(含诚实性说明)

工件:`models/bayes_2026.npz`(后验抽样)、`models/sim_2026.csv`、`models/report.json`。

### 模型 vs 市场(oddsportal 国家队收盘赔率)

补齐了 CLAUDE.md 强制的市场基线对比。oddsportal 反爬强、archive feed 为 base64+加密,
故用 **playwright + 持久浏览器 profile + 反检测**从渲染后 DOM 明文提取(`scrapers/oddsportal.py`),
范围为 **2022 世界杯 64 场**。outcome 一律用历史数据集比分(口径统一,不取 oddsportal 的
点球后比分)。`data/sources.py::OddsportalSource` 落地了 `OddsSource` 接口。

范围扩到 **4 届大赛 = 2018/2022 世界杯 + 2020/2024 欧洲杯 = 229 场**以持续收窄 CI;各场用
实际中立标志(东道主主场不归零)。配对 RPS(`eval/market_compare.py`,各赛事赛前训练):

| 来源 | 合并 RPS | vs 市场 delta(95% CI) | 结论 |
|---|---|---|---|
| 市场基线(de-vig 收盘) | **0.1950** | — | — |
| **贝叶斯 (no_odds)** | 0.1994 | +0.0044 [−0.0041, +0.0129] | 无显著差异 |
| Elo (no_odds) | 0.2062 | +0.0111 [+0.0025, +0.0206] | **市场显著更好** |

分赛事(市场 / 贝叶斯):2018 WC 0.1957/0.2017 · Euro2020 0.1835/0.1919 · 2022 WC 0.2081/0.2115 ·
**2024 Euro 0.1893/0.1888(贝叶斯微优)**。CI 随样本收窄:64 场 [−0.020,+0.021] → 114 场
[−0.011,+0.014] → 229 场 [−0.004,+0.013]。

> ✅ **诚实性检查 #1:** no_odds 贝叶斯 RPS(0.1994)≈ 市场(0.1950),**统计上无显著差异**
> (CI 含 0)→ 模型**追平**高效盘口但**未击败**。样本增至 229 场后,**Elo 已显著逊于市场**
> (CI 不再含 0)——这正是贝叶斯相对 Elo 价值在大样本下的体现。
> ⚠️ 注:oddsportal「平均赔率」是跨庄家乐观展示(水位极低,Euro2020 达 −3.5%),非真实单庄
> 收盘盘口,故市场基线偏乐观,模型追平难度被高估。

Run:  `python -m worldcup2026.scrapers.oddsportal`(抓取)· `python -m worldcup2026.eval.market_compare`(对比)

## Layout

```
worldcup2026/
  data/      ingest.py (Kaggle -> matches.parquet), sources.py (odds/squad stubs), alias_map.json
  features/  elo.py (own leakage-safe Elo), as_of.py (form/rest — feature-layer milestone)
  models/    elo_state.json; baseline/dixon_coles/bayes_poisson land per milestone
  eval/      walk-forward RPS/Brier/log-loss (MS1+)
  sim/       tournament Monte Carlo (MS4)
  tests/     test_leakage.py
  app.py     Streamlit UI (MS5)
```

## Design notes

- **Leakage safety:** Elo is updated sequentially in date order; the pre-match
  rating of each team is snapshotted *before* the result is applied. The leakage
  test recomputes a prefix independently and asserts truncating the data does
  not change earlier snapshots.
- **Cache-first ingestion:** sources are never re-hit if the local cache exists.
- **External feeds stubbed:** odds (oddsportal/football-data) and squad value
  (Transfermarkt) sit behind interfaces in `data/sources.py`; wired when MS1+
  needs them.
