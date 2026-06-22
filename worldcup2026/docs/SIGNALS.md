# 信号获取清单（Signal Acquisition Checklist）

> 用途：按本清单逐项获取数据，扩充模型的**正交信号**。每项标了优先级、数据源、获取
> 方式、与 Elo 的正交性、以及"市场是否已定价"。建议按 P0 → P1 → P2 顺序做。

## 0. 判断标准（先读这个）

我们已验证：Elo / Dixon-Coles / 贝叶斯 / 身价 彼此 **0.77~0.94 相关**，都在测同一个
"实力"因子。集成它们只把 RPS 从 0.1672 提到 0.1652（真实但小）。**要再有增量，新信号
必须和"战绩+身价推出的实力"低相关。**

两个目标，别混：
- **提升预测质量**：信号只需和**我们的模型**正交 → 下面多数信号可达。
- **打赢市场（alpha）**：信号需和**市场价格**正交 → 公开数据几乎做不到（市场已吃进
  xG/伤停/赛况）。alpha 靠"私有 + 更快"，是数据竞赛不是建模竞赛。

**取数纪律**：每个新信号入模前，先做两件事（已有脚手架 `eval/squad_signal.py` 可复用）——
① 测它与 Elo 的相关性；② 测它能否解释 Elo 的 as-of 残差（corr + p 值）。**只留下正交且
有残差信号的。**

---

## P0 — 最高优先（正交性高、可得性好、性价比最高）

### ☐ A. 情境 / 赛况（Situational）— **几乎免费，立即可做**
- **测什么**：强队死签轮换、两队都要平、最后一轮"算分"、东道主压力。与"谁强"天然正交。
- **数据源**：**无需外部数据**，从已有的比赛表 + `groups_2026.json` + 赛程**直接计算**：
  - 小组赛第 3 轮前，用前两轮积分算出线情景 → 标记"死签/必须取胜/双方满足于平局"。
  - 已出线/已淘汰 → 轮换风险标记。
- **获取方式**：纯计算（写 `features/situational.py`）。
- **市场定价**：部分；死签/轮换常被低估 → 可能有小 edge。
- **难度**：低。**建议第一个做。**

### ☐ B. xG / 预期进球（underlying performance）— **预测质量上最可能有真增量**
- **测什么**："踢得多好"而非"赢没赢"，修正"赢球却被打爆/输球却占优"的战绩噪声。
- **数据源**：
  - **StatsBomb Open Data**（免费，事件级，含 2018/2022 世界杯）：
    https://github.com/statsbomb/open-data — 直接 clone，质量最高，**优先**。
  - **FBref**（免费，含大赛 xG，2018+ 国家队）：https://fbref.com/en/comps/1/World-Cup-Stats
    — 表格可抓（注意限速/反爬）。
  - **fotmob**（xG + 实时）：https://www.fotmob.com — 非官方 JSON。
  - Understat（仅俱乐部五大联赛）：https://understat.com — 免费 JSON。
- **获取方式**：StatsBomb 直接下载；FBref/fotmob 抓取。
- **市场定价**：是，但较慢；用作模型输入仍能提升预测。
- **难度**：中（国家队大赛样本少是硬约束）。

### ☐ C. 场馆海拔 / 气候 / 旅行（2026 专属）
- **测什么**：墨西哥城海拔 ~2200m、瓜达拉哈拉 ~1560m 显著影响体能/进球；湿热、跨时区
  长途旅行。与实力正交。
- **数据源**：
  - 场馆经纬度/海拔：Wikipedia "2026 FIFA World Cup stadiums" + 维基各场馆页。
  - 天气（历史+预报）：**Open-Meteo**（免费、无需 key）：https://open-meteo.com/en/docs
    — 按 venue 经纬度 + 比赛日期取温度/湿度/降水。
- **获取方式**：建一张 16 场馆 venue 表（名→经纬度/海拔/时区），天气走 Open-Meteo API。
- **市场定价**：部分；海拔/极端气候常被低估。
- **难度**：低-中。

---

## P1 — 高价值但更难（脆弱或需账号）

### ☐ D. 首发 / 伤停 / 停赛（availability）— **原理最强**
- **测什么**：关键球员是否登场。Elo/身价是球队总量，不知道"这场 Mbappé 缺阵"。
- **数据源**：
  - **API-Football**（freemium，100 req/天免费）：https://www.api-football.com
    — 有 lineups / injuries / predictions / 部分 xG。**国家队覆盖好,优先。**
  - Transfermarkt 伤停页：https://www.transfermarkt.com/.../verletzungen（反爬强,WebFetch 被封,需 playwright 持久 profile）。
  - sofascore（首发 + 球员评分）：https://www.sofascore.com — 抓取。
  - 官方/FIFA 赛前公布首发（赛前 ~1 小时）。
- **获取方式**：API-Football 起步；历史伤停回测数据难,主要用于 2026 实时。
- **市场定价**：是,且反应快（赔率秒级调整）→ 难抢 alpha,但能提升预测。
- **难度**：高（晚出、易变、历史数据稀缺）。

### ☐ E. 盘口资金流 / 线路移动（market microstructure）
- **测什么**：线往哪动 = 聪明钱在哪。对**我们模型**正交,直接关系 CLV/择时。
- **数据源**：
  - **the-odds-api.com**（freemium,500 req/月免费,多庄家实时）：https://the-odds-api.com
  - oddsportal 开盘+收盘+历史线（**已有抓取器** `scrapers/oddsportal.py`,可扩开盘价）。
  - football-data.co.uk 俱乐部开+收盘（**已入库**,`data/footballdata.py`）。
  - Betfair Exchange API(成交量/交易价,需账号+key)：https://developer.betfair.com
- **获取方式**：the-odds-api 起步;带时间戳存多个快照算线路移动。
- **市场定价**："它就是市场"——对赢市场是循环,但量 CLV、择时下注必备。
- **难度**：中。

---

## P2 — 补充 / 特定市场（锦上添花）

### ☐ F. 裁判倾向（针对大小球 / 牌 / 点球盘）
- **测什么**：裁判判罚尺度(场均牌/点球/主场偏向)。与队伍实力正交,主要喂大小球/牌盘。
- **数据源**：FBref 裁判页 / Transfermarkt 裁判档案 / footballdata.co.uk（含裁判列）。
- **市场定价**：部分。**难度**：中。

### ☐ G. 教练更替 / 战术变化（new-manager bounce）
- **测什么**：换帅反弹、体系剧变。
- **数据源**：Transfermarkt 教练页 / Wikipedia 各队页。**难度**：中（事件稀疏）。

### ☐ H. 定位球 / 点球率、PPDA 等过程指标
- **测什么**：定位球依赖度、压迫强度——某些与 xG 互补的过程信号。
- **数据源**：StatsBomb Open Data / FBref。**难度**：中（同 xG,样本少）。

### ☐ I. 舆情 / 社媒（低优先，多为噪声）
- **数据源**：Google Trends / 新闻 API / X。**判断**：信号弱、市场秒修,**不建议先做**。

---

## 推荐获取顺序

1. **A 情境**（纯计算,先做,验证"正交信号"流程跑通）。
2. **C 海拔/气候**（场馆表 + Open-Meteo,2026 专属,低成本）。
3. **B xG**（StatsBomb 历史 + FBref,预测质量最可能真增量）。
4. **E 线路移动**（扩 oddsportal 开盘价 + the-odds-api,量 CLV）。
5. **D 伤停**（API-Football,主要服务 2026 实时,历史回测难）。
6. P2 按需。

## 每个信号入模前的验收（复用 `eval/squad_signal.py` 模式）
- [ ] 与 Elo 的 Spearman 相关性 < 0.7（够正交才有意义）。
- [ ] 对 Elo as-of 残差的 corr 显著（p<0.05 且方向合理）。
- [ ] 进集成 stacker（`eval/ensemble_stack.py`）后,留一年 OOS RPS 显著下降。
- [ ] 若目标是 alpha：进 `eval/betting_backtest.py` 看 CLV/ROI 是否转正。
