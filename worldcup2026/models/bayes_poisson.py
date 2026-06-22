"""M2 — 分层贝叶斯泊松(PyMC),推荐的头号模型。

对 attack/defense 做部分汇合(partial pooling),低样本球队向均值收缩:

    sigma_att, sigma_def ~ HalfNormal(1)
    attack[team]  ~ ZeroSumNormal(sigma_att)      # 软性 sum-to-zero,采样更干净
    defense[team] ~ ZeroSumNormal(sigma_def)
    mu            ~ Normal(0, 1)                    # 全局进球水平(log 尺度)
    home_field    ~ Normal(0.25, 0.1)              # 仅在 is_host_home 时施加
    home_goals ~ Poisson(exp(mu + attack[i] - defense[j] + home_field*is_host_home))
    away_goals ~ Poisson(exp(mu + attack[j] - defense[i]))

采样:NUTS,4 链,target_accept=0.9;检查 r_hat<1.01、无发散。

时间相关性:M2 spec 未含时间衰减,这里用「训练窗口截断」(只取切分日前 window_years
年的比赛),既保证近期相关性又控制采样规模。中立场时 home_field 归零(CLAUDE.md 要求)。

预测:对每个后验抽样独立计算 lambda,再用 Skellam(独立泊松之差)精确边缘化到 1X2,
最后对后验抽样平均 —— 由此把参数不确定性传播进每场预测。

Run:  python -m worldcup2026.models.bayes_poisson      # 快速自检采样
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln

MAX_GOALS = 9
DEFAULT_WINDOW_YEARS = 8


class BayesPoisson:
    def __init__(self, window_years: int = DEFAULT_WINDOW_YEARS, max_goals: int = MAX_GOALS):
        self.window_years = window_years
        self.max_goals = max_goals
        self.teams_: list[str] = []
        self.idx_: dict[str, int] = {}
        # 后验抽样(展平 chain×draw -> S):
        self.att_: np.ndarray | None = None   # (S, T)
        self.def_: np.ndarray | None = None   # (S, T)
        self.mu_: np.ndarray | None = None     # (S,)
        self.hf_: np.ndarray | None = None     # (S,)
        self.rho_: np.ndarray | None = None    # (S,) Dixon–Coles 低分相关性
        self.rhat_max_ = None
        self.divergences_ = None

    # ------------------------------------------------------------------ fit
    def fit(self, matches, ref_date, draws=2000, tune=2000, chains=4,
            target_accept=0.95, seed=42, progressbar=False):
        import pymc as pm  # 延迟导入,避免无谓加载

        ref = np.datetime64(ref_date)
        cutoff = ref - np.timedelta64(int(self.window_years * 365.25), "D")
        d = matches["date"].to_numpy().astype("datetime64[D]")
        m = matches[(d >= cutoff) & (d < ref)]

        self.teams_ = sorted(set(m["home_team"]) | set(m["away_team"]))
        self.idx_ = {t: i for i, t in enumerate(self.teams_)}
        hi = m["home_team"].map(self.idx_).to_numpy()
        ai = m["away_team"].map(self.idx_).to_numpy()
        hg = m["home_goals"].to_numpy().astype("int64")
        ag = m["away_goals"].to_numpy().astype("int64")
        is_host = (~m["neutral"].to_numpy()).astype(float)
        # Dixon–Coles 低分格掩码((0,0)(0,1)(1,0)(1,1)),其余比分 τ=1
        m00 = ((hg == 0) & (ag == 0)).astype(float)
        m01 = ((hg == 0) & (ag == 1)).astype(float)
        m10 = ((hg == 1) & (ag == 0)).astype(float)
        m11 = ((hg == 1) & (ag == 1)).astype(float)

        coords = {"team": self.teams_}
        with pm.Model(coords=coords):
            mu = pm.Normal("mu", 0.0, 1.0)
            sigma_att = pm.HalfNormal("sigma_att", 1.0)
            sigma_def = pm.HalfNormal("sigma_def", 1.0)
            # non-centered:标准化的 ZeroSumNormal 再乘以尺度,缓解层次漏斗 -> r_hat 更稳
            attack_z = pm.ZeroSumNormal("attack_z", sigma=1.0, dims="team")
            defense_z = pm.ZeroSumNormal("defense_z", sigma=1.0, dims="team")
            attack = pm.Deterministic("attack", attack_z * sigma_att, dims="team")
            defense = pm.Deterministic("defense", defense_z * sigma_def, dims="team")
            home_field = pm.Normal("home_field", 0.25, 0.1)
            rho = pm.Normal("rho", 0.0, 0.1)        # Dixon–Coles 低分相关性

            log_lh = mu + attack[hi] - defense[ai] + home_field * is_host
            log_la = mu + attack[ai] - defense[hi]
            lh_e = pm.math.exp(log_lh)
            la_e = pm.math.exp(log_la)
            pm.Poisson("home_goals", mu=lh_e, observed=hg)
            pm.Poisson("away_goals", mu=la_e, observed=ag)
            # DC τ 修正:仅 (0,0)(0,1)(1,0)(1,1) 四格偏离独立泊松,τ=1+rho*(...)
            tau = 1.0 + rho * (-lh_e * la_e * m00 + lh_e * m01 + la_e * m10 - m11)
            pm.Potential("dc_lowscore",
                         pm.math.log(pm.math.clip(tau, 1e-9, np.inf)).sum())

            idata = pm.sample(
                draws=draws, tune=tune, chains=chains, target_accept=target_accept,
                random_seed=seed, progressbar=progressbar,
            )

        import arviz as az
        summary = az.summary(idata, var_names=["mu", "home_field", "rho", "sigma_att",
                                               "sigma_def", "attack", "defense"])
        rhat_col = "r_hat" if "r_hat" in summary.columns else "rhat"
        self.summary_ = summary
        self.rhat_col_ = rhat_col
        self.rhat_max_ = float(summary[rhat_col].max())
        self.ess_min_ = float(summary["ess_bulk"].min())
        self.divergences_ = int(idata.sample_stats["diverging"].sum())

        post = idata.posterior
        # 展平 (chain, draw) -> S
        self.att_ = post["attack"].stack(s=("chain", "draw")).transpose("s", "team").values
        self.def_ = post["defense"].stack(s=("chain", "draw")).transpose("s", "team").values
        self.mu_ = post["mu"].stack(s=("chain", "draw")).values
        self.hf_ = post["home_field"].stack(s=("chain", "draw")).values
        self.rho_ = post["rho"].stack(s=("chain", "draw")).values
        # thinning:预测用 ~1000 个均匀抽样足以传播后验不确定性,且大幅加速 Skellam
        S = self.mu_.shape[0]
        if S > 1200:
            sel = np.linspace(0, S - 1, 1000).astype(int)
            self.att_, self.def_ = self.att_[sel], self.def_[sel]
            self.mu_, self.hf_ = self.mu_[sel], self.hf_[sel]
            self.rho_ = self.rho_[sel]
        return self

    # -------------------------------------------------------------- predict
    def _lambda_samples(self, home, away, neutral):
        """返回该场 home/away lambda 的后验抽样数组 (S,)。未知队 -> 0(平均水平)。"""
        S = self.mu_.shape[0]
        ah = self.att_[:, self.idx_[home]] if home in self.idx_ else np.zeros(S)
        dh = self.def_[:, self.idx_[home]] if home in self.idx_ else np.zeros(S)
        aa = self.att_[:, self.idx_[away]] if away in self.idx_ else np.zeros(S)
        da = self.def_[:, self.idx_[away]] if away in self.idx_ else np.zeros(S)
        host = 0.0 if neutral else 1.0
        lh = np.exp(self.mu_ + ah - da + self.hf_ * host)
        la = np.exp(self.mu_ + aa - dh)
        return lh, la

    def predict_1x2(self, home, away, neutral=True) -> np.ndarray:
        # 从 DC 修正后的比分矩阵导出 1X2(与精确比分口径一致;9 球截断尾部可忽略)
        mat = self.score_matrix(home, away, neutral)
        p_home = float(np.tril(mat, -1).sum())   # 主队进球 > 客队 -> 主胜
        p_draw = float(np.trace(mat))
        p_away = float(np.triu(mat, 1).sum())    # 客队进球 > 主队 -> 客胜
        return np.array([p_home, p_draw, p_away])

    def predict_proba_frame(self, frame) -> np.ndarray:
        out = np.empty((len(frame), 3))
        for i, row in enumerate(frame.itertuples(index=False)):
            out[i] = self.predict_1x2(row.home_team, row.away_team, bool(row.neutral))
        return out

    def score_matrix(self, home, away, neutral=True) -> np.ndarray:
        """后验平均比分矩阵(0..max_goals),含 Dixon–Coles 低分修正。"""
        lh, la = self._lambda_samples(home, away, neutral)
        S = lh.shape[0]
        n = self.max_goals + 1
        gk = np.arange(n)
        # (S, n) 泊松 pmf
        logp_h = gk[None, :] * np.log(lh)[:, None] - lh[:, None] - gammaln(gk + 1)[None, :]
        logp_a = gk[None, :] * np.log(la)[:, None] - la[:, None] - gammaln(gk + 1)[None, :]
        ph = np.exp(logp_h)
        pa = np.exp(logp_a)
        M = ph[:, :, None] * pa[:, None, :]        # (S,n,n) 每个后验样本的独立联合
        # DC τ 修正四格(逐样本),再逐样本重归一,最后对后验平均
        rho = self.rho_ if self.rho_ is not None else np.zeros(S)
        M[:, 0, 0] *= 1.0 - lh * la * rho
        M[:, 0, 1] *= 1.0 + lh * rho
        M[:, 1, 0] *= 1.0 + la * rho
        M[:, 1, 1] *= 1.0 - rho
        np.clip(M, 0.0, None, out=M)
        M /= M.sum(axis=(1, 2), keepdims=True)
        return M.mean(axis=0)

    def predict_over_under(self, home, away, neutral=True, line=2.5):
        mat = self.score_matrix(home, away, neutral)
        n = mat.shape[0]
        tot = np.add.outer(np.arange(n), np.arange(n))
        over = float(mat[tot > line].sum())
        return over, 1.0 - over

    # ---------------------------------------------------------- 持久化
    def save(self, path):
        """保存后验抽样,供 UI/模拟免重采样加载。"""
        np.savez_compressed(
            path, att=self.att_, defe=self.def_, mu=self.mu_, hf=self.hf_,
            rho=self.rho_ if self.rho_ is not None else np.zeros_like(self.mu_),
            teams=np.array(self.teams_, dtype=object),
            rhat=float(self.rhat_max_), div=int(self.divergences_))

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=True)
        obj = cls.__new__(cls)
        obj.att_, obj.def_ = d["att"], d["defe"]
        obj.mu_, obj.hf_ = d["mu"], d["hf"]
        obj.rho_ = d["rho"] if "rho" in d.files else np.zeros_like(obj.mu_)
        obj.teams_ = list(d["teams"])
        obj.idx_ = {t: i for i, t in enumerate(obj.teams_)}
        obj.max_goals = MAX_GOALS
        obj.window_years = DEFAULT_WINDOW_YEARS
        obj.rhat_max_ = float(d["rhat"])
        obj.divergences_ = int(d["div"])
        return obj


if __name__ == "__main__":
    import time
    from worldcup2026.features.elo import build as build_elo

    enriched, _ = build_elo(save=False)
    train = enriched[enriched["date"] < "2026-01-01"]
    t0 = time.time()
    bp = BayesPoisson(window_years=8).fit(train, ref_date="2026-01-01")
    print(f"\nfit: {len(bp.teams_)} teams, {time.time()-t0:.1f}s")
    print(f"r_hat_max={bp.rhat_max_:.4f}  divergences={bp.divergences_}  ess_min={bp.ess_min_:.0f}")
    print(f"mu={bp.mu_.mean():.3f}  home_field={bp.hf_.mean():.3f}  "
          f"rho={bp.rho_.mean():+.4f}  (DC 低分相关性,典型 ~−0.1)")
    for h, a in [("Spain", "Brazil"), ("Argentina", "France"), ("Germany", "Japan")]:
        p = bp.predict_1x2(h, a, neutral=True)
        ou = bp.predict_over_under(h, a, neutral=True)
        print(f"{h} vs {a} (neutral): H {p[0]:.3f} D {p[1]:.3f} A {p[2]:.3f} | O2.5 {ou[0]:.3f}")
