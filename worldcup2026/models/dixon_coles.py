"""M1 — Dixon–Coles bivariate-Poisson with time-decay weighting.

    log(lambda_home) = mu + attack[home] - defense[away] + gamma * is_home_adv
    log(lambda_away) = mu + attack[away] - defense[home]
    tau(x,y)         = low-score dependence correction for (0,0),(1,0),(0,1),(1,1)
    weight(match)    = exp(-xi * age_in_days)        # xi ~ 0.0019 => ~1yr half-life

Fit by weighted MLE (L-BFGS-B with an analytic gradient). Identifiability:
sum(attack)=0 and sum(defense)=0 (last team = -sum of the rest), with mu the
global scoring level. `is_home_adv` is `~neutral` for historical matches and the
host flag for 2026 — so gamma is ZEROED at neutral venues, per CLAUDE.md.

Output is the full score matrix (0..MAX_GOALS each) -> 1X2, over/under, exact.

Run:  python -m worldcup2026.models.dixon_coles      # quick self-test fit
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

MAX_GOALS = 9
RHO_BOUND = 0.2
DEFAULT_XI = 0.0019          # ~365-day half-life
TRAIN_WINDOW_DAYS = 365 * 15  # older matches carry negligible weight; drop for speed


def _tau(x, y, lh, la, rho):
    t = np.ones_like(lh)
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)
    t = np.where(m00, 1.0 - lh * la * rho, t)
    t = np.where(m01, 1.0 + lh * rho, t)
    t = np.where(m10, 1.0 + la * rho, t)
    t = np.where(m11, 1.0 - rho, t)
    return t, (m00, m01, m10, m11)


class DixonColes:
    def __init__(self, xi: float = DEFAULT_XI, max_goals: int = MAX_GOALS,
                 reg: float = 1.0):
        self.xi = xi
        self.max_goals = max_goals
        self.reg = reg          # L2 on attack/defense: pools low-data teams, kills flat dirs
        self.teams_: list[str] = []
        self.idx_: dict[str, int] = {}
        self.attack_: np.ndarray | None = None
        self.defense_: np.ndarray | None = None
        self.mu_ = self.gamma_ = self.rho_ = None

    # ------------------------------------------------------------------ fit
    def fit(self, matches, ref_date):
        ref = np.datetime64(ref_date)
        age = (ref - matches["date"].to_numpy().astype("datetime64[D]")).astype("int64")
        keep = age <= TRAIN_WINDOW_DAYS
        m = matches[keep]
        age = age[keep]
        w = np.exp(-self.xi * age)

        self.teams_ = sorted(set(m["home_team"]) | set(m["away_team"]))
        self.idx_ = {t: i for i, t in enumerate(self.teams_)}
        T = len(self.teams_)

        hi = m["home_team"].map(self.idx_).to_numpy()
        ai = m["away_team"].map(self.idx_).to_numpy()
        x = m["home_goals"].to_numpy().astype(float)
        y = m["away_goals"].to_numpy().astype(float)
        hh = (~m["neutral"].to_numpy()).astype(float)

        def unpack(p):
            mu, gamma, rho = p[0], p[1], p[2]
            a_free = p[3:3 + T - 1]
            d_free = p[3 + T - 1:3 + 2 * (T - 1)]
            att = np.concatenate([a_free, [-a_free.sum()]])
            dfn = np.concatenate([d_free, [-d_free.sum()]])
            return mu, gamma, rho, att, dfn

        def objective(p):
            mu, gamma, rho, att, dfn = unpack(p)
            eta_h = mu + att[hi] - dfn[ai] + gamma * hh
            eta_a = mu + att[ai] - dfn[hi]
            lh = np.exp(eta_h)
            la = np.exp(eta_a)
            tau, (m00, m01, m10, m11) = _tau(x, y, lh, la, rho)
            tau = np.clip(tau, 1e-10, None)

            ll = w * (np.log(tau) + x * eta_h - lh + y * eta_a - la)
            nll = -ll.sum() + 0.5 * self.reg * (np.sum(att ** 2) + np.sum(dfn ** 2))

            # --- gradient ---
            dtau_dlh = np.zeros_like(lh)
            dtau_dla = np.zeros_like(lh)
            dtau_drho = np.zeros_like(lh)
            dtau_dlh = np.where(m00, -la * rho, dtau_dlh)
            dtau_dla = np.where(m00, -lh * rho, dtau_dla)
            dtau_drho = np.where(m00, -lh * la, dtau_drho)
            dtau_dlh = np.where(m01, rho, dtau_dlh)
            dtau_drho = np.where(m01, lh, dtau_drho)
            dtau_dla = np.where(m10, rho, dtau_dla)
            dtau_drho = np.where(m10, la, dtau_drho)
            dtau_drho = np.where(m11, -1.0, dtau_drho)

            g_h = w * ((x - lh) + dtau_dlh * lh / tau)
            g_a = w * ((y - la) + dtau_dla * la / tau)
            g_rho = w * (dtau_drho / tau)

            d_mu = -(g_h.sum() + g_a.sum())
            d_gamma = -(g_h * hh).sum()
            d_rho = -g_rho.sum()

            dll_att = np.zeros(T)
            np.add.at(dll_att, hi, g_h)
            np.add.at(dll_att, ai, g_a)
            dll_def = np.zeros(T)
            np.add.at(dll_def, ai, -g_h)
            np.add.at(dll_def, hi, -g_a)
            g_att = -dll_att + self.reg * att  # d NLL / d att_full (+ ridge)
            g_def = -dll_def + self.reg * dfn
            # map to free params: free_j contributes (j) and -(last)
            ga_free = g_att[:T - 1] - g_att[T - 1]
            gd_free = g_def[:T - 1] - g_def[T - 1]

            grad = np.concatenate([[d_mu, d_gamma, d_rho], ga_free, gd_free])
            return nll, grad

        x0 = np.zeros(3 + 2 * (T - 1))
        x0[0] = np.log(max(np.average(np.r_[x, y], weights=np.r_[w, w]), 0.5))
        x0[1] = 0.2
        x0[2] = -0.05
        bounds = [(None, None), (None, None), (-RHO_BOUND, RHO_BOUND)] + \
                 [(None, None)] * (2 * (T - 1))

        res = minimize(objective, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 2000, "maxfun": 50000,
                                "ftol": 1e-10, "gtol": 1e-6})
        mu, gamma, rho, att, dfn = unpack(res.x)
        self.mu_, self.gamma_, self.rho_ = mu, gamma, rho
        self.attack_, self.defense_ = att, dfn
        self.converged_ = res.success
        self.opt_message_ = res.message
        self.grad_max_ = float(np.max(np.abs(res.jac)))
        return self

    # -------------------------------------------------------------- predict
    def _lambdas(self, home, away, neutral):
        a = self.attack_
        d = self.defense_
        ah = a[self.idx_[home]] if home in self.idx_ else 0.0
        dh = d[self.idx_[home]] if home in self.idx_ else 0.0
        aa = a[self.idx_[away]] if away in self.idx_ else 0.0
        da = d[self.idx_[away]] if away in self.idx_ else 0.0
        hh = 0.0 if neutral else 1.0
        lh = np.exp(self.mu_ + ah - da + self.gamma_ * hh)
        la = np.exp(self.mu_ + aa - dh)
        return lh, la

    def score_matrix(self, home, away, neutral=True) -> np.ndarray:
        lh, la = self._lambdas(home, away, neutral)
        n = self.max_goals + 1
        gk = np.arange(n)
        ph = np.exp(gk * np.log(lh) - lh - gammaln(gk + 1))
        pa = np.exp(gk * np.log(la) - la - gammaln(gk + 1))
        mat = np.outer(ph, pa)
        # apply tau correction to the 2x2 low-score corner
        rho = self.rho_
        mat[0, 0] *= 1.0 - lh * la * rho
        mat[0, 1] *= 1.0 + lh * rho
        mat[1, 0] *= 1.0 + la * rho
        mat[1, 1] *= 1.0 - rho
        mat = np.clip(mat, 0.0, None)
        return mat / mat.sum()

    def predict_1x2(self, home, away, neutral=True) -> np.ndarray:
        mat = self.score_matrix(home, away, neutral)
        p_home = np.tril(mat, -1).sum()   # x > y
        p_draw = np.trace(mat)            # x == y
        p_away = np.triu(mat, 1).sum()    # x < y
        return np.array([p_home, p_draw, p_away])

    def predict_over_under(self, home, away, neutral=True, line=2.5) -> tuple[float, float]:
        mat = self.score_matrix(home, away, neutral)
        n = mat.shape[0]
        tot = np.add.outer(np.arange(n), np.arange(n))
        over = mat[tot > line].sum()
        return float(over), float(1.0 - over)

    def predict_proba_frame(self, frame) -> np.ndarray:
        """Vectorized 1X2 for a frame with home_team/away_team/neutral columns."""
        out = np.empty((len(frame), 3))
        for i, row in enumerate(frame.itertuples(index=False)):
            out[i] = self.predict_1x2(row.home_team, row.away_team, bool(row.neutral))
        return out


if __name__ == "__main__":
    import time
    from worldcup2026.features.elo import build as build_elo

    enriched, _ = build_elo(save=False)
    train = enriched[enriched["date"] < "2026-01-01"]
    t0 = time.time()
    dc = DixonColes().fit(train, ref_date="2026-01-01")
    print(f"fit: {len(dc.teams_)} teams, {time.time()-t0:.1f}s, "
          f"mu={dc.mu_:.3f} gamma={dc.gamma_:.3f} rho={dc.rho_:.3f} conv={dc.converged_}")
    for h, a in [("Spain", "Brazil"), ("Argentina", "France"), ("Germany", "Japan")]:
        p = dc.predict_1x2(h, a, neutral=True)
        ou = dc.predict_over_under(h, a, neutral=True)
        print(f"{h} vs {a} (neutral): H {p[0]:.3f} D {p[1]:.3f} A {p[2]:.3f} | "
              f"O2.5 {ou[0]:.3f}")
