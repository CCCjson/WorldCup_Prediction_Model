"""锦标赛蒙特卡洛模拟(向量化)。

用头号贝叶斯模型(BayesPoisson)的后验抽样驱动:**每次模拟抽取一个后验参数
样本**,故参数不确定性被传播进最终的晋级概率(CLAUDE.md 要求)。

流程(每次模拟):
  - 小组赛循环 -> 用模型采样比分 -> 积分榜
  - 平局规则:积分 -> 净胜球 -> 进球(其后用随机打破,代表 H2H/公平竞赛/抽签)
  - 取每组前 2 + 8 个最佳第三名(best-8 近似,见 groups_2026.json 说明)
  - 第三名按官方 slot 来源组约束做二分匹配,填入 R32
  - 单淘汰 R32->...->决赛;平局用条件胜率(λ 比例,已折叠 ET/点球)选晋级方
  - 记录每队达到的最远轮次

输出:team -> P(R32), P(R16), P(QF), P(SF), P(Final), P(Winner)。

Run:  python -m worldcup2026.sim.tournament          # 验证(2022)+ 跑 2026
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

GROUPS_PATH = Path(__file__).resolve().parent.parent / "data" / "groups_2026.json"
GROUP_FIXTURES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]  # 单循环 6 场


# --------------------------------------------------------------------------- #
# 第三名 -> slot 二分匹配(按组字母组合缓存,完美匹配回溯)
# --------------------------------------------------------------------------- #
def _match_thirds(qualified: tuple[str, ...], slot_allowed: tuple[tuple[str, ...], ...]):
    """给定出线第三名的组字母集合,返回 slot 顺序对应的组字母列表(完美匹配)。

    官方表对每种出线组合有确定映射;此处用满足来源组约束的回溯匹配近似。
    """
    n = len(slot_allowed)
    qset = list(qualified)
    assign = [None] * n

    def backtrack(slot):
        if slot == n:
            return True
        for g in slot_allowed[slot]:
            if g in qset and g not in assign:
                assign[slot] = g
                if backtrack(slot + 1):
                    return True
                assign[slot] = None
        return False

    if backtrack(0):
        return tuple(assign)
    # 回退:按出线排名顺序顺次填(极少触发)
    return tuple(sorted(qset)[:n])


# --------------------------------------------------------------------------- #
# 配置加载
# --------------------------------------------------------------------------- #
def load_2026_config():
    cfg = json.loads(GROUPS_PATH.read_text())
    return {
        "groups": cfg["groups"],
        "hosts": set(cfg["hosts"]),
        "round1": cfg["r32"],
        "n_best_third": 8,
    }


def build_known_results(config, matches_df, cutoff_date):
    """从规范比赛表抽取本届(2026)已踢完的小组赛 -> {组: {(i,j): (gi,gj)}}。

    只取 ``date < cutoff_date`` 的 2026 FIFA World Cup 比赛;按组内局部下标 i<j
    存实际进球(gi 为局部 i 队进球)。淘汰赛为跨组对阵(两队不在同组),自动跳过。
    """
    groups = config["groups"]
    loc = {t: (g, k) for g, ts in groups.items() for k, t in enumerate(ts)}
    cutoff = pd.Timestamp(cutoff_date)
    wc = matches_df[(matches_df["tournament"] == "FIFA World Cup")
                    & (matches_df["date"] >= "2026-06-01")
                    & (matches_df["date"] < cutoff)]
    known: dict = {}
    n = 0
    for r in wc.itertuples(index=False):
        if r.home_team not in loc or r.away_team not in loc:
            continue
        gh, ih = loc[r.home_team]
        ga_, ia = loc[r.away_team]
        if gh != ga_ or ih == ia:      # 跨组(淘汰赛)或异常 -> 跳过
            continue
        hg, ag = int(r.home_goals), int(r.away_goals)
        if ih < ia:
            known.setdefault(gh, {})[(ih, ia)] = (hg, ag)
        else:
            known.setdefault(gh, {})[(ia, ih)] = (ag, hg)
        n += 1
    return known, n


def config_2022():
    """卡塔尔 2022(32 队,8 组,前 2 出线,无最佳第三名)—— 用于验证模拟器。"""
    groups = {
        "A": ["Qatar", "Ecuador", "Senegal", "Netherlands"],
        "B": ["England", "Iran", "United States", "Wales"],
        "C": ["Argentina", "Saudi Arabia", "Mexico", "Poland"],
        "D": ["France", "Australia", "Denmark", "Tunisia"],
        "E": ["Spain", "Costa Rica", "Germany", "Japan"],
        "F": ["Belgium", "Canada", "Morocco", "Croatia"],
        "G": ["Brazil", "Serbia", "Switzerland", "Cameroon"],
        "H": ["Portugal", "Ghana", "Uruguay", "South Korea"],
    }
    # 2022 官方 R16 对阵
    r16 = [
        [{"type": "winner", "group": "A"}, {"type": "runner", "group": "B"}],
        [{"type": "winner", "group": "C"}, {"type": "runner", "group": "D"}],
        [{"type": "winner", "group": "E"}, {"type": "runner", "group": "F"}],
        [{"type": "winner", "group": "G"}, {"type": "runner", "group": "H"}],
        [{"type": "winner", "group": "B"}, {"type": "runner", "group": "A"}],
        [{"type": "winner", "group": "D"}, {"type": "runner", "group": "C"}],
        [{"type": "winner", "group": "F"}, {"type": "runner", "group": "E"}],
        [{"type": "winner", "group": "H"}, {"type": "runner", "group": "G"}],
    ]
    return {"groups": groups, "hosts": {"Qatar"}, "round1": r16, "n_best_third": 0}


STAGE_NAMES = {
    16: ["R32", "R16", "QF", "SF", "Final", "Winner"],   # round1 有 16 场 -> 2026
    8: ["R16", "QF", "SF", "Final", "Winner"],            # round1 有 8 场 -> 2022
}


# --------------------------------------------------------------------------- #
# 主模拟
# --------------------------------------------------------------------------- #
def simulate_tournament(model, config, n_sims: int = 20000, seed: int = 2026,
                        known_results: dict | None = None):
    """known_results: {group_letter: {(i, j): (goals_i, goals_j)}} —— 已踢完的小组
    赛场次,按组内局部下标 i<j 给出实际进球。这些场次在每次模拟中被**固定**(不再
    采样),只对剩余场次抽样,从而把已发生的真实结果钉死进晋级概率。None=全部采样
    (赛前/验证用,行为与原先一致)。"""
    rng = np.random.default_rng(seed)
    groups = config["groups"]
    hosts = config["hosts"]
    round1 = config["round1"]
    n_best_third = config["n_best_third"]
    group_letters = list(groups.keys())

    # 全局 48(或 32)队索引
    teams = []
    for g in group_letters:
        teams.extend(groups[g])
    t_idx = {t: j for j, t in enumerate(teams)}
    nT = len(teams)

    # 后验抽样 -> 每次模拟一套队伍强度
    S = model.mu_.shape[0]
    pidx = rng.integers(0, S, n_sims)
    MU = model.mu_[pidx]                          # (n_sims,)
    HF = model.hf_[pidx]
    A = np.zeros((n_sims, nT))
    D = np.zeros((n_sims, nT))
    for j, t in enumerate(teams):
        if t in model.idx_:
            A[:, j] = model.att_[pidx, model.idx_[t]]
            D[:, j] = model.def_[pidx, model.idx_[t]]

    def sample_match(home, away, host_mask):
        """home/away: (n_sims,) 全局队索引;host_mask: (n_sims,) bool 主队主场。"""
        ar = np.arange(n_sims)
        lh = np.exp(MU + A[ar, home] - D[ar, away] + HF * host_mask)
        la = np.exp(MU + A[ar, away] - D[ar, home])
        return rng.poisson(lh), rng.poisson(la), lh, la

    # ---- 小组赛 ----
    winners, runners, thirds = {}, {}, {}
    third_score = {}
    for g in group_letters:
        gteams = np.array([t_idx[t] for t in groups[g]])  # 4 个全局索引
        host_in = [t in hosts for t in groups[g]]
        pts = np.zeros((n_sims, 4)); gf = np.zeros((n_sims, 4)); ga = np.zeros((n_sims, 4))
        gknown = (known_results or {}).get(g, {})
        for (i, j) in GROUP_FIXTURES:
            if (i, j) in gknown:
                # 已踢完 -> 固定实际比分(局部 i,j 视角),不采样
                gi, gj = gknown[(i, j)]
                gi = np.full(n_sims, gi); gj = np.full(n_sims, gj)
                gf[:, i] += gi; ga[:, i] += gj
                gf[:, j] += gj; ga[:, j] += gi
                pts[:, i] += np.where(gi > gj, 3, np.where(gi == gj, 1, 0))
                pts[:, j] += np.where(gj > gi, 3, np.where(gi == gj, 1, 0))
                continue
            hi, hj = host_in[i], host_in[j]
            # 东道主置为主队主场;否则中立
            if hj and not hi:
                home_l, away_l, hm = j, i, np.ones(n_sims, bool)
            else:
                home_l, away_l = i, j
                hm = np.ones(n_sims, bool) if hi else np.zeros(n_sims, bool)
            hg, ag_, _, _ = sample_match(
                np.full(n_sims, gteams[home_l]), np.full(n_sims, gteams[away_l]), hm)
            # 累计(注意 home_l/away_l 的局部下标)
            gf[:, home_l] += hg; ga[:, home_l] += ag_
            gf[:, away_l] += ag_; ga[:, away_l] += hg
            pts[:, home_l] += np.where(hg > ag_, 3, np.where(hg == ag_, 1, 0))
            pts[:, away_l] += np.where(ag_ > hg, 3, np.where(hg == ag_, 1, 0))
        gd = gf - ga
        # 排名分:pts -> gd -> gf -> 随机(代表 H2H/公平竞赛/抽签)
        score = (pts * 1e9 + (gd + 200) * 1e5 + gf * 1e2
                 + rng.random((n_sims, 4)))
        order = np.argsort(-score, axis=1)  # 每行名次(局部下标)
        ar = np.arange(n_sims)
        winners[g] = gteams[order[:, 0]]
        runners[g] = gteams[order[:, 1]]
        thirds[g] = gteams[order[:, 2]]
        # 第三名的排名分(用于 best-8)
        third_score[g] = score[ar, order[:, 2]]

    # ---- 最佳第三名 + slot 匹配 ----
    third_team_for_slot = None
    if n_best_third > 0:
        # 12 组第三名分数矩阵 -> 每 sim 取 top-N 组
        tscore = np.column_stack([third_score[g] for g in group_letters])  # (n_sims,12)
        top_groups_idx = np.argsort(-tscore, axis=1)[:, :n_best_third]      # (n_sims,8)
        qualified_letters = np.array(group_letters)[top_groups_idx]        # (n_sims,8)

        # third slot 的 allowed 组(按 round1 中 type==third 的出现顺序)
        third_slots = []
        for mi, match in enumerate(round1):
            for si, slot in enumerate(match):
                if slot["type"] == "third":
                    third_slots.append((mi, si, tuple(slot["allowed"])))
        slot_allowed = tuple(s[2] for s in third_slots)

        @lru_cache(maxsize=None)
        def cached_match(qkey):
            return _match_thirds(qkey, slot_allowed)

        # 每 sim:出线组集合 -> slot 对应组字母
        third_team_for_slot = np.full((n_sims, len(third_slots)), -1)
        g2col = {g: c for c, g in enumerate(group_letters)}
        thirds_mat = np.column_stack([thirds[g] for g in group_letters])  # (n_sims,12) 队索引
        for s in range(n_sims):
            qk = tuple(sorted(qualified_letters[s]))
            assign = cached_match(qk)  # slot -> group letter
            for slot_i, gl in enumerate(assign):
                third_team_for_slot[s, slot_i] = thirds_mat[s, g2col[gl]]

    # ---- 填充第一轮 bracket ----
    n_matches1 = len(round1)
    bracket = np.empty((n_sims, 2 * n_matches1), dtype=int)
    third_cursor = 0
    for mi, match in enumerate(round1):
        for si, slot in enumerate(match):
            col = 2 * mi + si
            if slot["type"] == "winner":
                bracket[:, col] = winners[slot["group"]]
            elif slot["type"] == "runner":
                bracket[:, col] = runners[slot["group"]]
            else:  # third
                bracket[:, col] = third_team_for_slot[:, third_cursor]
                third_cursor += 1

    # ---- 统计:达到各轮 ----
    stages = STAGE_NAMES[n_matches1]
    reached = {st: np.zeros(nT, dtype=np.int64) for st in stages}

    def count_stage(team_array, stage):
        flat = team_array.ravel()
        np.add.at(reached[stage], flat, 1)

    count_stage(bracket, stages[0])  # 进入第一轮 = 所有 bracket 席位的队

    # ---- 单淘汰 ----
    current = bracket
    for st in stages[1:]:
        m = current.shape[1] // 2
        nxt = np.empty((n_sims, m), dtype=int)
        ar = np.arange(n_sims)
        for k in range(m):
            home = current[:, 2 * k]
            away = current[:, 2 * k + 1]
            hg, ag_, lh, la = sample_match(home, away, np.zeros(n_sims, bool))
            home_win = hg > ag_
            away_win = ag_ > hg
            draw = ~(home_win | away_win)
            # 平局 -> 条件胜率(λ 比例,折叠 ET/点球)
            p_home = lh / (lh + la)
            home_adv = rng.random(n_sims) < p_home
            take_home = home_win | (draw & home_adv)
            nxt[:, k] = np.where(take_home, home, away)
        count_stage(nxt, st)
        current = nxt

    # ---- 汇总为概率表 ----
    out = pd.DataFrame({"team": teams})
    for st in stages:
        out[f"P({st})"] = reached[st] / n_sims
    out = out.sort_values(f"P({stages[-1]})", ascending=False).reset_index(drop=True)
    return out


# --------------------------------------------------------------------------- #
# 单次模拟(供 bracket 视图展示一次抽样运行)
# --------------------------------------------------------------------------- #
def simulate_once(model, config, seed: int = 0):
    rng = np.random.default_rng(seed)
    groups, hosts, round1 = config["groups"], config["hosts"], config["round1"]
    n_best_third = config["n_best_third"]
    S = model.mu_.shape[0]
    s = int(rng.integers(0, S))
    att, dfn, mu, hf = model.att_[s], model.def_[s], float(model.mu_[s]), float(model.hf_[s])

    def col(t):
        return model.idx_.get(t, None)

    def play(home, away, host):
        ah = att[col(home)] if col(home) is not None else 0.0
        dh = dfn[col(home)] if col(home) is not None else 0.0
        aa = att[col(away)] if col(away) is not None else 0.0
        da = dfn[col(away)] if col(away) is not None else 0.0
        lh = np.exp(mu + ah - da + hf * (1.0 if host else 0.0))
        la = np.exp(mu + aa - dh)
        hg, ag_ = int(rng.poisson(lh)), int(rng.poisson(la))
        return hg, ag_, lh, la

    standings, winners, runners, thirds, third_meta = {}, {}, {}, {}, {}
    for g, gteams in groups.items():
        st = {t: [0, 0, 0] for t in gteams}  # pts, gf, ga
        for (i, j) in GROUP_FIXTURES:
            home, away = gteams[i], gteams[j]
            host = (home in hosts) or (away in hosts)
            if away in hosts and home not in hosts:
                home, away = away, home
            hg, ag_, _, _ = play(home, away, host)
            st[home][0] += 3 if hg > ag_ else (1 if hg == ag_ else 0)
            st[away][0] += 3 if ag_ > hg else (1 if hg == ag_ else 0)
            st[home][1] += hg; st[home][2] += ag_
            st[away][1] += ag_; st[away][2] += hg
        ranked = sorted(gteams, key=lambda t: (st[t][0], st[t][1] - st[t][2], st[t][1],
                                               rng.random()), reverse=True)
        standings[g] = [(t, st[t][0], st[t][1] - st[t][2], st[t][1]) for t in ranked]
        winners[g], runners[g], thirds[g] = ranked[0], ranked[1], ranked[2]
        tt = ranked[2]
        third_meta[g] = (st[tt][0], st[tt][1] - st[tt][2], st[tt][1])

    third_for_slot = {}
    if n_best_third > 0:
        best = sorted(groups.keys(), key=lambda g: (third_meta[g], rng.random()),
                      reverse=True)[:n_best_third]
        slot_allowed = tuple(tuple(slot["allowed"]) for m in round1 for slot in m
                             if slot["type"] == "third")
        assign = _match_thirds(tuple(sorted(best)), slot_allowed)
        for k, gl in enumerate(assign):
            third_for_slot[k] = thirds[gl]

    # 填第一轮
    bracket, tc = [], 0
    for m in round1:
        pair = []
        for slot in m:
            if slot["type"] == "winner":
                pair.append(winners[slot["group"]])
            elif slot["type"] == "runner":
                pair.append(runners[slot["group"]])
            else:
                pair.append(third_for_slot[tc]); tc += 1
        bracket.append(pair)

    stages = STAGE_NAMES[len(round1)]
    knockout = []
    current = [t for pair in bracket for t in pair]
    for st_name in stages[:-1]:  # 最后一个 "Winner" 不是一轮比赛
        matches, nxt = [], []
        for k in range(0, len(current), 2):
            home, away = current[k], current[k + 1]
            hg, ag_, lh, la = play(home, away, host=False)
            if hg > ag_:
                w = home
            elif ag_ > hg:
                w = away
            else:
                w = home if rng.random() < lh / (lh + la) else away
            matches.append((home, hg, ag_, away, w))
            nxt.append(w)
        knockout.append({"round": st_name, "matches": matches})
        current = nxt

    return {"standings": standings, "knockout": knockout, "champion": current[0]}


# --------------------------------------------------------------------------- #
# 验证 + 2026
# --------------------------------------------------------------------------- #
def _fit_model(ref_date, window_years=8):
    from worldcup2026.features.elo import build as build_elo
    from worldcup2026.models.bayes_poisson import BayesPoisson
    enriched, _ = build_elo(save=False)
    train = enriched[enriched["date"] < ref_date]
    return BayesPoisson(window_years=window_years).fit(train, ref_date=ref_date)


def validate_2022(n_sims=20000):
    print("\n=== 模拟器验证:2022 卡塔尔(赛前 2022-11-20 训练)===")
    model = _fit_model("2022-11-20")
    print(f"[采样诊断] r_hat_max={model.rhat_max_:.3f} divergences={model.divergences_}")
    res = simulate_tournament(model, config_2022(), n_sims=n_sims)
    print("夺冠概率 Top-10:")
    print(res[["team", "P(Winner)", "P(Final)", "P(SF)"]].head(10)
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n[合理性] 夺冠概率之和={res['P(Winner)'].sum():.3f}(应≈1) "
          f"最大={res['P(Winner)'].max():.3f}(应<0.40)")
    print("[现实对照] 实际冠军=阿根廷,亚军=法国,4强=克罗地亚/摩洛哥")
    return res


def run_2026(n_sims=50000):
    print("\n=== 2026 世界杯预测(赛前 2026-06-11 训练)===")
    model = _fit_model("2026-06-11")
    print(f"[采样诊断] r_hat_max={model.rhat_max_:.3f} divergences={model.divergences_}")
    res = simulate_tournament(model, load_2026_config(), n_sims=n_sims)
    print("\n夺冠/晋级概率 Top-15:")
    cols = ["team", "P(Winner)", "P(Final)", "P(SF)", "P(QF)", "P(R16)", "P(R32)"]
    print(res[cols].head(15).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n[合理性检查#4] 夺冠概率之和={res['P(Winner)'].sum():.3f}(应≈1) "
          f"最大={res['P(Winner)'].max():.3f}(热门应在个位数~15%,无 40%+)")
    return res


def run_2026_live(cutoff_date, n_sims=50000):
    """实时预测:用 ``< cutoff_date`` 的全部数据训练(纳入已踢的小组赛),并把
    本届已踢完的小组赛**钉死**进模拟,只对剩余场次采样。"""
    from worldcup2026.data.ingest import load_matches
    print(f"\n=== 2026 世界杯实时预测(训练截止 {cutoff_date},已完赛小组赛钉死)===")
    model = _fit_model(cutoff_date)
    print(f"[采样诊断] r_hat_max={model.rhat_max_:.3f} divergences={model.divergences_}")
    cfg = load_2026_config()
    known, n_known = build_known_results(cfg, load_matches(), cutoff_date)
    print(f"[条件化] 钉死已踢小组赛 {n_known} 场,剩余 {72 - n_known} 场采样")
    res = simulate_tournament(model, cfg, n_sims=n_sims, known_results=known)
    print("\n夺冠/晋级概率 Top-15:")
    cols = ["team", "P(Winner)", "P(Final)", "P(SF)", "P(QF)", "P(R16)", "P(R32)"]
    print(res[cols].head(15).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n[合理性检查#4] 夺冠概率之和={res['P(Winner)'].sum():.3f}(应≈1) "
          f"最大={res['P(Winner)'].max():.3f}(热门应在个位数~15%,无 40%+)")
    return res


if __name__ == "__main__":
    validate_2022()
    run_2026()
