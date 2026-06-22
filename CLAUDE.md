# CLAUDE-worldcup-2026.md

## ROLE

You are building an end-to-end probabilistic prediction system for the **2026 FIFA World Cup**. Deliver working code, not explanations. Build incrementally, run each component before moving on, and stop to report at every milestone checkpoint defined below.
Respond language : Chinese (defalt)
## OBJECTIVE

Produce, for the 2026 World Cup:
1. Per-match outcome distributions (score-line → 1X2 + over/under).
2. Monte Carlo tournament simulation → each team's probability of reaching R32 / R16 / QF / SF / Final / Winner.
3. A Streamlit UI to inspect predictions and run simulations.
4. An honest benchmark of the model against bookmaker closing odds (de-vigged) using RPS.

## NON-NEGOTIABLE CONSTRAINTS

- **No temporal leakage.** Every feature for a match at time `t` must be computable using only data with timestamp `< t`. Elo, form, rest-days are all as-of-`t`. Never random-shuffle for CV — use walk-forward / expanding-window splits only.
- **Primary metric is RPS** (Ranked Probability Score, ordered home/draw/away). Secondary: Brier, log-loss, calibration curve.
- **Mandatory baseline:** de-vigged bookmaker closing implied probabilities. The model is only "useful" if it matches or beats this baseline on RPS on held-out seasons. Report both numbers side by side, always.
- **Neutral venue handling:** 2026 matches are at neutral venues except for the three hosts (USA/Canada/Mexico) playing in their own country. The `home_advantage` term must be zeroed for all non-host matches. Do not inherit a club-football home coefficient globally.
- **Leakage trap:** if bookmaker odds are used as a model feature, the model will trivially "win" by parroting the market. Keep two model variants: `with_odds` and `no_odds`, trained and evaluated separately, so feature contribution is measurable. The headline model that competes against the market baseline must be `no_odds` (or opening-odds only).

## 2026 TOURNAMENT FORMAT (hard-code this)

```
48 teams, 12 groups of 4 (A..L), round-robin (3 matches each).
Advancement to Round of 32:
  - top 2 of each group  -> 24 teams
  - 8 best 3rd-placed teams across all groups -> 8 teams
Knockout: R32 -> R16 -> QF -> SF -> Final (single elimination, extra time + penalties).
Total 104 matches. Hosts: USA, Canada, Mexico. June 11 – July 19, 2026.
```

- The draw is already complete; **fetch the actual drawn groups** (do not simulate the draw). Provide `data/groups_2026.json` mapping group → [team1..team4], and the official R32 bracket slot mapping.
- **3rd-place qualification** uses FIFA's predetermined bracket table (which group-letter combinations of qualifying 3rd-placed teams map to which R32 slots). Implement the official mapping table if available; otherwise fall back to "best 8 third-placed by points → GD → GF" and flag this as an approximation in code comments.
- Group-stage tiebreakers in order: points → goal difference → goals scored → head-to-head → fair-play → drawing of lots (implement first three; random for the rest).
- Knockout ties: resolve via a model-driven win probability that already integrates ET/penalties (do not separately model penalties; just use match win prob conditional on no draw).

## DATA SOURCES

Build a `data/` ingestion layer. Cache everything to local parquet/sqlite; never re-hit a source if cached.

| Data | Source | Notes |
|---|---|---|
| International match results 1872→present | Kaggle: `martj42/international-football-results-from-1872-to-present` | Core training set. Columns: date, home, away, home_score, away_score, tournament, city, country, neutral. |
| Elo ratings (national teams) | eloratings.net | Either scrape historical or **implement own Elo** (preferred, controllable, leakage-safe). |
| Club/national odds (for baseline + backtests) | football-data.co.uk | Free multi-bookmaker open+close odds (club leagues). Use to validate de-vig + RPS pipeline. |
| National-team match odds | oddsportal (scrape) | Opening + closing. Closing only for baseline; opening allowed as feature. User has anti-bot tooling. |
| Squad value / age | Transfermarkt (scrape) | Per-team aggregate market value, mean age, as-of tournament. |
| 2026 groups + fixtures + bracket | Wikipedia "2026 FIFA World Cup" + FIFA | Drawn groups, match schedule, R32 slot mapping. |

If a source is unreachable, stub it behind an interface and continue; do not block the pipeline on one feed.

## DATA SCHEMA (canonical match table)

```python
# matches.parquet — one row per historical match, sorted by date ASC
date: datetime
home_team: str            # canonical names via a country alias map (England/ENG, etc.)
away_team: str
home_goals: int
away_goals: int
tournament: str           # 'Friendly','FIFA World Cup','UEFA Euro','Copa America',...
neutral: bool
host_country: str | None  # for neutral=False, who is home
# derived (filled by feature layer, all as-of-date):
elo_home_pre: float
elo_away_pre: float
rest_days_home: int
rest_days_away: int
form_home: float          # exp-time-decayed recent result strength
form_away: float
comp_weight: float        # competition importance weight (friendly < qualifier < major)
```

## FEATURE LAYER

Implement a strict as-of feature builder. Required features:

- `elo_diff = elo_home_pre - elo_away_pre` (own Elo; see params below).
- `form_home`, `form_away`: exponential time-decay of recent results, half-life ≈ 180 days.
- `rest_days_diff`.
- `comp_weight`: map tournament → importance (friendly 0.3, minor 0.6, continental qualifier 0.8, continental/WC 1.0). Use as Elo K-multiplier AND/OR sample weight.
- `squad_value_log_diff`, `mean_age_diff` (when Transfermarkt available).
- `neutral`, `is_host_home` flags.
- (optional, `with_odds` variant only) `book_implied_home/draw/away` from opening odds, de-vigged.

## ELO IMPLEMENTATION (own, leakage-safe)

```
expected_home = 1 / (1 + 10**(-(elo_home + HFA - elo_away)/400))
HFA = 0 if neutral else 100            # home-field advantage in Elo points
K = K0 * comp_weight * goal_diff_mult
K0 = 40 (tune 20–60)
goal_diff_mult: 1.0 (margin 1), 1.5 (margin 2), 1.75 (margin 3), then +0.75/8 per extra goal  # World Football Elo style
update sequentially in date order; persist elo_*_pre snapshots before each update.
```

## MODELS

Implement three, in this order. Gate each behind its milestone.

**M0 — Baseline (no ML).** De-vig bookmaker closing odds → implied 1X2. This is the bar to beat. Also: pure Elo → 1X2 via a fitted ordered map. RPS of both on walk-forward test.

**M1 — Dixon–Coles.** Bivariate-Poisson with low-score correlation correction + time-decay weighting.
```
log(lambda_home) = mu + attack[home] - defense[away] + gamma*is_host_home
log(lambda_away) = mu + attack[away] - defense[home]
tau(...) low-score correction for (0,0),(1,0),(0,1),(1,1)
weight(match) = exp(-xi * age_in_days);  xi ≈ 0.0018–0.003 (tune)
fit by weighted MLE.
```
Output: full score matrix (cap 0–9 goals each) → marginalize to 1X2, over/under, exact-score.

**M2 — Hierarchical Bayesian Poisson (recommended headline).** PyMC. Partial pooling on attack/defense so low-sample teams shrink to the mean.
```
attack[team] ~ Normal(0, sigma_att);  defense[team] ~ Normal(0, sigma_def)
sigma_att, sigma_def ~ HalfNormal(1)
home_field ~ Normal(0.25, 0.1)  # applied only when is_host_home
goals ~ Poisson(exp(mu + attack[i] - defense[j] + home_field*is_host_home))
sample: NUTS, 4 chains, 2000 draws, target_accept=0.9; check r_hat<1.01, no divergences.
```
Use posterior predictive for per-match score distributions; propagate full posterior uncertainty into the tournament simulation (sample a parameter draw per simulation run).

Optional **M1b — LightGBM** on 1X2 for a feature-rich comparison; **must** apply isotonic/Platt calibration before scoring, else RPS will be poor.

## TRAINING & VALIDATION

- **Walk-forward / expanding window.** Train on all matches `< split_date`, test on the next block (e.g. each major-tournament cycle or each calendar year). Roll forward. Never random-fold.
- Report per-split and aggregate: **RPS** (primary), Brier, log-loss, accuracy, calibration plot (reliability diagram, 10 bins).
- Always print the **paired table**: `model_RPS` vs `market_baseline_RPS` per split. Compute mean RPS difference and a simple bootstrap CI on the difference.
- For national-team data specifically, weight by `comp_weight` so friendlies don't dominate.
- Save the best model + the fitted Elo state + alias maps under `models/`.

## TOURNAMENT SIMULATION

```
def simulate_tournament(model, groups_2026, n_sims=50000):
    for each sim:
        (if Bayesian) draw one posterior parameter sample
        play all 72 group matches -> sample score from model -> standings
        apply tiebreakers (pts, GD, GF, ...)
        rank 3rd-placed teams, take best 8, map to R32 slots via official table
        run R32->R16->QF->SF->Final; knockout = sample score, if draw use
            conditional win prob (ET/pens folded in) to pick advancer
        record furthest stage reached per team
    return DataFrame: team -> P(R32), P(R16), P(QF), P(SF), P(Final), P(Winner)
```
- Vectorize where possible; 50k sims should run in seconds-to-minutes. Use a seed and expose `n_sims`.
- Validate the simulator on a *past* tournament (e.g. 2022 with that format) before trusting 2026 numbers.

## VISUALIZATION (Streamlit)

`app.py`, single page, tabs:
1. **Match predictor:** pick two teams + neutral/host toggle → 1X2 bar, score-line heatmap (0–5 grid), over/under 2.5.
2. **Tournament odds:** table + horizontal bar chart of P(Winner) and stage-reach probabilities; sortable.
3. **Bracket view:** most-likely bracket and a "simulate once" button showing one sampled run.
4. **Model vs Market:** the RPS comparison table + calibration plot from the backtest.
Keep it functional, not fancy. `st.cache_data` for the heavy sim. A "Run N simulations" slider.

## PROJECT STRUCTURE

```
worldcup2026/
  data/            ingestion + cache (parquet/sqlite), groups_2026.json, alias_map.json
  features/        as_of feature builder, elo.py
  models/          baseline.py, dixon_coles.py, bayes_poisson.py, (lgbm.py), calibrate.py
  eval/            walkforward.py, metrics.py (rps, brier, logloss, calibration)
  sim/             tournament.py
  app.py           streamlit UI
  notebooks/       exploration only
  tests/           leakage test (assert no future data in features), tiebreaker test, sim sanity
  README.md        how to run, current RPS-vs-market numbers
```

## MILESTONES (stop & report at each)

- **MS0 – Data + Elo.** Canonical match table built, own Elo running leakage-safe. Report: row counts, date range, top-20 current Elo. Add the leakage unit test.
- **MS1 – Baseline.** M0 market de-vig + Elo→1X2, walk-forward RPS table. **This sets the bar.** Report market baseline RPS.
- **MS2 – Dixon–Coles.** M1 trained, score matrices working, RPS vs baseline on walk-forward. Report delta.
- **MS3 – Bayesian.** M2 sampled, diagnostics clean, posterior-predictive per-match dists. Report RPS delta + calibration.
- **MS4 – Simulation.** Tournament sim validated on a past edition, then run for 2026. Report top-15 P(Winner).
- **MS5 – UI.** Streamlit app wired to the best model + sim. Report run command.

## HONESTY CHECKS (enforce, do not skip)

1. If `no_odds` model RPS ≥ market baseline RPS → state plainly the model adds no independent signal vs the market; do not dress it up.
2. If `with_odds` beats `no_odds` by a lot → that's odds parroting, not skill; say so.
3. Print the leakage-test result in every training run.
4. 2026 winner probabilities must be sane (favorites single-digit-to-~15% range; no team at 40%+). If not, suspect a bug before believing the number.

## EXECUTION ORDER

Start at MS0. Do not jump ahead. After each milestone, print the report block and wait for go-ahead. Use Chinese only if I ask a conceptual question; otherwise keep output to code, run results, and the milestone report.