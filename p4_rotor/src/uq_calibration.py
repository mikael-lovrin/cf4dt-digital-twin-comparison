"""UQ (round 3): closed-form likelihood of the 2-DOF physics, MAP initialisation, emcee calibration with a
DEMove / DESnookerMove mixture, posterior predictive per sensor position, and P(exceed) risk.

Calibrated vector (13 with discrepancy): the 11 physics parameters the held-out A1B0 experiment can inform
(pm.P2_FREE_CAL: log k_A, log k_B, log k_s, log c_A, log c_B, log e0, rho_p1..3, log G_p2, log G_p3), plus
log sigma_obs and log a_disc. kappa_B, phi and zeta (floor scales) are fixed at the leave-group-out centre.

Observation model. For sensor position p and RPM r_i, z_pi = log(mean Z RMS over the 5 repeats).
    z_p ~ N( m_p(theta), sigma^2 I + a^2 K ),  K_ij = exp(-0.5 ((r_i - r_j)/ELL)^2), ELL = 500 RPM (assumption),
    m_p(theta) = log of the 2-DOF prediction (physics_model.pred2_g).
Positions are independent, each with the same covariance. There is no emulator variance (closed form), so the
round-2 'GP sd / sigma_obs ~ 6' problem does not arise here.
"""
import numpy as np
import emcee
from scipy.optimize import minimize
from scipy.stats import norm
from SALib.analyze import sobol
from SALib.sample import saltelli

import physics_model as pm

ELL = 500.0
LO_S, HI_S = np.log(1e-3), np.log(1.0)     # sigma_obs prior (log g)
LO_A, HI_A = np.log(1e-3), np.log(1.0)     # discrepancy amplitude prior (log g)
NFREE = len(pm.P2_FREE_CAL)                # 11 physics parameters calibrated
LO13 = np.r_[pm.P2_LO[pm.P2_FREE_CAL], LO_S, LO_A]
HI13 = np.r_[pm.P2_HI[pm.P2_FREE_CAL], HI_S, HI_A]
NAMES13 = pm.P2_CAL_NAMES + ["log_sigma_obs", "log_a_disc"]


def K_of(a, b, ell=ELL):
    d = (np.asarray(a, float)[:, None] - np.asarray(b, float)[None, :]) / ell
    return np.exp(-0.5 * d * d)


def prior_std(names_idx):
    """Std of a uniform prior in the calibration (log) space, per calibrated parameter."""
    w = (HI13 - LO13)[names_idx]
    return w / np.sqrt(12.0)


class Calibration:
    """One calibration experiment: a held-out condition with cell means for P positions on a shared RPM set."""

    def __init__(self, base_theta, rpm_o, z_obs, floor_obs, floor_fn, UA, UB, with_disc=True):
        self.base = np.asarray(base_theta, float)          # 16-vector: centre for the fixed parameters
        self.rpm = np.asarray(rpm_o, float)                # (n,)
        self.n = len(self.rpm)
        self.P = z_obs.shape[0]
        self.z = np.asarray(z_obs, float)                  # (P, n) log observed
        self.floor_obs = np.asarray(floor_obs, float)      # (P, n)
        self.floor_fn = floor_fn                           # (pos_array, rpm_array) -> floor [g]
        self.UA, self.UB = float(UA), float(UB)
        self.with_disc = with_disc
        self.K = K_of(self.rpm, self.rpm)
        self.I = np.eye(self.n)
        self.ndim = 13 if with_disc else 12
        self.lo = LO13[:self.ndim].copy()
        self.hi = HI13[:self.ndim].copy()
        self.rpm_flat = np.tile(self.rpm, self.P)
        self.pos_flat = np.repeat(np.arange(self.P), self.n)
        self.floor_flat = self.floor_obs.ravel()
        self.UA_flat = np.full(self.P * self.n, self.UA)
        self.UB_flat = np.full(self.P * self.n, self.UB)

    def theta_full(self, p):
        p = np.atleast_2d(p)
        th = np.tile(self.base, (len(p), 1))
        th[:, pm.P2_FREE_CAL] = p[:, :NFREE]
        return th

    def _a2(self, q):
        if self.with_disc:
            return np.exp(2.0 * q[:, NFREE + 1])
        return np.zeros(len(q))

    def logpost(self, q):
        q = np.atleast_2d(np.asarray(q, float))
        W = len(q)
        inside = np.all((q >= self.lo) & (q <= self.hi), axis=1)
        qq = np.clip(q, self.lo, self.hi)
        th = self.theta_full(qq[:, :NFREE])
        m = np.log(pm.pred2_g(th, self.rpm_flat, self.pos_flat, self.UA_flat, self.UB_flat, self.floor_flat))
        r = (self.z.ravel()[None, :] - m).reshape(W, self.P, self.n)
        s2 = np.exp(2.0 * qq[:, NFREE])
        a2 = self._a2(qq)
        Sig = s2[:, None, None] * self.I[None] + a2[:, None, None] * self.K[None]
        L = np.linalg.cholesky(Sig)
        logdet = 2.0 * np.sum(np.log(np.diagonal(L, axis1=1, axis2=2)), axis=1)
        v = np.linalg.solve(Sig[:, None], r[..., None])[..., 0]
        quad = np.sum(r * v, axis=(1, 2))
        ll = -0.5 * (quad + self.P * logdet + self.P * self.n * np.log(2.0 * np.pi))
        ll = np.where(np.isfinite(ll), ll, -np.inf)
        return np.where(inside, ll, -np.inf)

    def neg_logpost_scalar(self, x):
        v = self.logpost(x[None, :])[0]
        return 1e30 if not np.isfinite(v) else -float(v)

    def map_fit(self, starts):
        best = None
        for x0 in starts:
            x0 = np.clip(np.asarray(x0, float), self.lo + 1e-6, self.hi - 1e-6)
            r = minimize(self.neg_logpost_scalar, x0, method="L-BFGS-B",
                         bounds=list(zip(self.lo, self.hi)), options={"maxiter": 400})
            if best is None or r.fun < best.fun:
                best = r
        return best.x, float(-best.fun)

    def run_mcmc(self, p_map, nsteps, burn, nwalkers=32, seed=0):
        rng = np.random.default_rng(seed)
        width = self.hi - self.lo
        p0 = p_map[None, :] + 1e-3 * width[None, :] * rng.standard_normal((nwalkers, self.ndim))
        p0 = np.clip(p0, self.lo + 1e-9, self.hi - 1e-9)
        lp0 = self.logpost(p0)
        p0[~np.isfinite(lp0)] = p_map
        moves = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]
        sampler = emcee.EnsembleSampler(nwalkers, self.ndim, self.logpost, vectorize=True, moves=moves)
        np.random.seed(seed)          # emcee draws its proposals from the global NumPy RNG: seed it for reproducibility
        sampler.run_mcmc(p0, nsteps, progress=False)
        tau = np.asarray(sampler.get_autocorr_time(discard=burn, quiet=True), float)
        return {"chain": sampler.get_chain(discard=burn, flat=True), "tau": tau,
                "acceptance": float(np.mean(sampler.acceptance_fraction)), "nsteps": nsteps, "burn": burn,
                "nwalkers": nwalkers, "post": nsteps - burn}

    # ------------------------------------------------------------ posterior predictive
    def predictive(self, chain, rpm_t, eps_scale=1.0, nsamp=1000, seed=0):
        """Posterior predictive of log RMS at positions x target RPM. Returns mean (S,P,T) and tau (S,T).
        mean = physics (target) + conditional discrepancy mean; tau^2 = sigma^2 + conditional discrepancy var.
        eps_scale multiplies the unbalance (rebalancing scenario); the observed residuals use eps_scale=1."""
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(chain), size=min(nsamp, len(chain)), replace=False)
        q = chain[idx]
        th = self.theta_full(q[:, :NFREE])
        s2 = np.exp(2.0 * q[:, NFREE])
        a2 = self._a2(q)
        rpm_t = np.asarray(rpm_t, float)
        T = len(rpm_t)
        S = len(q)
        rpm_tf = np.tile(rpm_t, self.P)
        pos_tf = np.repeat(np.arange(self.P), T)
        fl_tf = self.floor_fn(pos_tf, rpm_tf)
        UA_tf = np.full(self.P * T, self.UA)
        UB_tf = np.full(self.P * T, self.UB)
        m_t = np.log(pm.pred2_g(th, rpm_tf, pos_tf, UA_tf, UB_tf, fl_tf, eps_scale)).reshape(S, self.P, T)
        m_o = np.log(pm.pred2_g(th, self.rpm_flat, self.pos_flat, self.UA_flat, self.UB_flat,
                                self.floor_flat)).reshape(S, self.P, self.n)
        r_o = self.z[None] - m_o                                             # (S,P,n)
        Sig = s2[:, None, None] * self.I[None] + a2[:, None, None] * self.K[None]
        Kto = a2[:, None, None] * K_of(rpm_t, self.rpm)[None]                # (S,T,n)
        alpha = np.linalg.solve(Sig[:, None], r_o[..., None])[..., 0]        # (S,P,n)
        mean_d = np.einsum("stn,spn->spt", Kto, alpha)
        B = np.linalg.solve(Sig, np.transpose(Kto, (0, 2, 1)))               # (S,n,T)
        var_d = np.maximum(a2[:, None] - np.einsum("stn,snt->st", Kto, B), 0.0)
        return m_t + mean_d, np.sqrt(s2[:, None] + var_d)

    def intervals(self, chain, rpm_t, levels, seed=0):
        """Central predictive intervals per position (levels: {lev: (2,P,T)}) and for the position mean in
        linear space (levels: {lev: (2,T)}), i.e. the round-1 'position-averaged' observation."""
        mean, tau = self.predictive(chain, rpm_t, seed=seed)
        rng = np.random.default_rng(seed + 1)
        draws = np.exp(mean + tau[:, None, :] * rng.standard_normal(mean.shape))
        per, avg = {}, {}
        dmean = draws.mean(axis=1)
        for lev in levels:
            a = (1 - lev) / 2
            per[lev] = np.quantile(draws, [a, 1 - a], axis=0)
            avg[lev] = np.quantile(dmean, [a, 1 - a], axis=0)
        return per, avg

    def p_exceed(self, chain, rpm_t, limit, eps_scale=1.0, seed=0):
        """P(RMS > limit) per position (P,T) and for 'any position' (T,), averaged over posterior samples.
        Positions are treated as conditionally independent given theta (their discrepancy terms are independent)."""
        mean, tau = self.predictive(chain, rpm_t, eps_scale=eps_scale, seed=seed)
        pe = norm.cdf((mean - np.log(limit)) / tau[:, None, :])               # (S,P,T)
        pany = 1.0 - np.prod(1.0 - pe, axis=1)                                 # (S,T)
        return pany.mean(axis=0), pe.mean(axis=0)


def sobol_indices(names, bounds, fn, N=1024, seed=0):
    """Sobol S1/ST of fn(X) with X in the (log or raw) bounds. Returns (dict, n_evaluations)."""
    problem = {"num_vars": len(names), "names": names, "bounds": np.asarray(bounds).tolist()}
    X = saltelli.sample(problem, N, calc_second_order=False)
    Y = fn(X)
    res = sobol.analyze(problem, Y, calc_second_order=False, seed=seed)
    return {p: {"S1": float(res["S1"][i]), "ST": float(res["ST"][i])} for i, p in enumerate(names)}, len(X)
