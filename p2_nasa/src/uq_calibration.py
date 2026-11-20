"""Bayesian calibration (emcee), posterior predictive checks, RUL propagation and Sobol analysis.

All model evaluations go through the per-case GP emulator (ODEEmulator), built on the design box of the
case's CV fold. Round 3 changes:
  - Likelihood on log(VB + EPS): y = log(VB_obs + EPS) ~ N(mu_log(theta), sd_log(theta)^2 + sig_log^2).
    The noise is multiplicative (relative), so predictive bands are non-negative after exp(.) - EPS.
  - Crossings, RUL and failure risk use the MEDIAN curve exp(mu_log) - EPS, not the lognormal mean.
  - MCMC: initialised at the MAP (L-BFGS-B on the log posterior), DE + DESnooker move mixture.
"""
import numpy as np
import emcee
from scipy.optimize import minimize
from scipy.stats import norm
from SALib.sample import saltelli
from SALib.analyze import sobol

from physics_model import PARAM_NAMES, simulate
from data_loader import VB_CRIT
from surrogate_model import EPS

SIG_LO, SIG_HI = np.log(0.01), np.log(1.0)   # uniform prior on log sigma_log (noise sd in log(VB+EPS) units)
MOVES = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]


def log_prior_fn(lo, hi, center):
    pd_sd = (hi - lo) / 4.0      # Gaussian prior around the fold's physics fit, truncated to the box

    def lp(th, ls):
        inside = np.all((th >= lo) & (th <= hi), axis=1) & (ls >= SIG_LO) & (ls <= SIG_HI)
        val = -0.5 * np.sum(((th - center) / pd_sd) ** 2, axis=1)
        return np.where(inside, val, -np.inf)
    return lp


def prior_std_empirical(center, lo, hi, n=200000, seed=0):
    """Std of the truncated Gaussian prior (used for the posterior-to-prior width ratio)."""
    rng = np.random.default_rng(seed)
    pd_sd = (hi - lo) / 4.0
    s = center + pd_sd * rng.standard_normal((n, 5))
    s = s[np.all((s >= lo) & (s <= hi), axis=1)]
    return s.std(axis=0)


def make_logpost(emu, idx, yl, center):
    lp_fn = log_prior_fn(emu.lo, emu.hi, center)

    def logpost(x):
        x = np.atleast_2d(x)
        th, ls = x[:, :5], x[:, 5]
        lp = lp_fn(th, ls)
        out = np.full(x.shape[0], -np.inf)
        good = np.isfinite(lp)
        if good.any() and len(idx) > 0:
            ml, sl = emu.predict(th[good])
            ml, sl = ml[:, idx], sl[:, idx]
            s2 = sl ** 2 + np.exp(2.0 * ls[good])[:, None]
            ll = -0.5 * np.sum((yl - ml) ** 2 / s2 + np.log(2 * np.pi * s2), axis=1)
            out[good] = lp[good] + ll
        elif good.any():
            out[good] = lp[good]
        return out
    return logpost


def map_estimate(logpost, center, lo, hi):
    """MAP of the log posterior on the 6-D box (physics box x sigma range), L-BFGS-B from the fold centre."""
    x0 = np.r_[np.clip(center, lo, hi), np.log(0.05)]
    bounds = list(zip(np.r_[lo, SIG_LO], np.r_[hi, SIG_HI]))

    def neg(x):
        v = float(logpost(x)[0])
        return 1e300 if not np.isfinite(v) else -v
    r = minimize(neg, x0, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=500))
    return r.x, -float(r.fun), bool(r.success)


def calibrate(emu, idx, yl, center, seed, n_walkers=32, n_steps=12000, burn=2000):
    """Posterior sampling. yl = log(VB_obs + EPS) at the grid indices idx."""
    rng = np.random.default_rng(seed)
    logpost = make_logpost(emu, idx, yl, center)
    x_map, lp_map, map_ok = map_estimate(logpost, center, emu.lo, emu.hi)
    lo6 = np.r_[emu.lo, SIG_LO]
    hi6 = np.r_[emu.hi, SIG_HI]
    scale = np.r_[0.01 * (emu.hi - emu.lo), 0.05]      # small ball around the MAP
    p0, tries = [], 0
    while len(p0) < n_walkers and tries < 20000:
        tries += 1
        x = np.clip(x_map + scale * rng.standard_normal(6), lo6 + 1e-9 * (hi6 - lo6), hi6 - 1e-9 * (hi6 - lo6))
        if np.isfinite(logpost(x)[0]):
            p0.append(x)
    p0 = np.array(p0)
    np.random.seed(seed)          # emcee draws from the global NumPy RNG when no random_state is given
    sampler = emcee.EnsembleSampler(n_walkers, 6, logpost, vectorize=True, moves=MOVES)
    sampler.run_mcmc(p0, n_steps, progress=False)
    chain = sampler.get_chain(discard=burn, flat=True)
    # tol=0 disables emcee's internal "chain too short" exception, so the estimate is always returned.
    # The 50-tau check is NOT silenced: main.py computes chain/tau per parameter and reports it.
    tau_ac = np.asarray(sampler.get_autocorr_time(discard=burn, tol=0), float)
    return dict(chain=chain, tau_ac=tau_ac, acc=float(np.mean(sampler.acceptance_fraction)),
                n_post=int(n_steps - burn), n_walkers=int(n_walkers),
                map=x_map.tolist(), map_logpost=float(lp_map), map_success=map_ok)


def calibrate_job(args):
    """Module-level wrapper so that calibrations can run in a process pool."""
    return calibrate(*args)


def _subsample(chain, n, seed):
    rng = np.random.default_rng(seed)
    return chain[rng.choice(len(chain), size=min(n, len(chain)), replace=False)]


def posterior_predictive(emu, chain, idx, n_draw=400, seed=0, level=0.95):
    """Central predictive interval of the given level and the predictive median, at grid indices idx.
    Sampled in log space (parameter + GP + noise), then mapped back with exp(.) - EPS, so bounds are >= 0."""
    if len(idx) == 0:
        e = np.array([])
        return e, e, e
    rng = np.random.default_rng(seed)
    sub = _subsample(chain, n_draw, seed)
    ml, sl = emu.predict(sub[:, :5])
    ml, sl = ml[:, idx], sl[:, idx]
    s = np.sqrt(sl ** 2 + np.exp(2.0 * sub[:, 5])[:, None])
    yl = ml + s * rng.standard_normal(ml.shape)
    sim = np.maximum(np.exp(yl) - EPS, 0.0)
    a = 100.0 * (1.0 - level) / 2.0
    lo, hi = np.percentile(sim, [a, 100.0 - a], axis=0)
    med = np.percentile(sim, 50.0, axis=0)
    return lo, med, hi


def crossing_time(grid, curves, level=VB_CRIT):
    """First time each curve reaches `level` (linear interpolation on the grid).

    Returns (time, censored) where censored = never reaches level on the grid (time = grid end).
    """
    above = curves >= level
    any_ = above.any(axis=1)
    j = np.clip(np.argmax(above, axis=1), 1, grid.size - 1)
    rows = np.arange(curves.shape[0])
    t0, t1 = grid[j - 1], grid[j]
    v0, v1 = curves[rows, j - 1], curves[rows, j]
    frac = np.clip((level - v0) / np.maximum(v1 - v0, 1e-12), 0.0, 1.0)
    tc = np.where(any_, t0 + frac * (t1 - t0), grid[-1])
    return tc, ~any_


def rul_samples(emu, chain, tau, n_draw=1000, seed=1):
    """Remaining life from cutoff tau per posterior draw, from the MEDIAN emulator curve."""
    sub = _subsample(chain, n_draw, seed)
    med, _ = emu.median_vb(sub[:, :5])
    tc, cens = crossing_time(emu.grid, med)
    return np.maximum(tc - tau, 0.0), cens


def derated_rul(emu, chain, tau, d, n_draw=200, seed=7, dt=0.05):
    """Remaining life from tau when the load is scaled by d after tau (ODE re-simulation per draw).

    Uses the full physics ODE (not the emulator) so that the thermal feedback of a derated load is included.
    Returns (rul samples, censored flags); censored = no crossing before the emulator horizon.
    """
    sub = _subsample(chain, n_draw, seed)
    tg = np.arange(0.0, float(emu.grid[-1]) + dt, dt)
    base = np.interp(tg, emu.case["t_u"], emu.case["u_fit"])     # same masked forcing as the emulator
    u = np.where(tg > tau, base * d, base)
    case_d = dict(emu.case, t_u=tg, u_fit=u)
    curves = np.array([simulate(th[:5], emu.gamma, [case_d], [tg])[0] for th in sub])
    tc, cens = crossing_time(tg, curves)
    return np.maximum(tc - tau, 0.0), cens


def derated_job(args):
    return derated_rul(*args)


def failure_prob_now(emu, chain, idx_tau, n_draw=400, seed=2):
    """Instantaneous P(VB(tau) > VB_crit) under the log-space posterior predictive (median-based)."""
    sub = _subsample(chain, n_draw, seed)
    ml, sl = emu.predict(sub[:, :5])
    ml, sl = ml[:, idx_tau], sl[:, idx_tau]
    s = np.sqrt(sl ** 2 + np.exp(2.0 * sub[:, 5]))
    return float(np.mean(norm.sf((np.log(VB_CRIT + EPS) - ml) / s)))


def sobol_indices(emu, idx_tend, N=1024, seed=0):
    problem = {"num_vars": 5, "names": PARAM_NAMES,
               "bounds": [[float(emu.lo[i]), float(emu.hi[i])] for i in range(5)]}
    np.random.seed(seed)
    X = saltelli.sample(problem, N, calc_second_order=False)
    med, _ = emu.median_vb(X)
    y_vb = med[:, idx_tend]
    tc, cens = crossing_time(emu.grid, med)
    r_vb = sobol.analyze(problem, y_vb, calc_second_order=False, print_to_console=False)
    r_tc = sobol.analyze(problem, tc, calc_second_order=False, print_to_console=False)

    def pack(r):
        return {n: {"S1": float(s1), "ST": float(st)}
                for n, s1, st in zip(PARAM_NAMES, r["S1"], r["ST"])}
    return dict(VB_tend=pack(r_vb), time_to_threshold=pack(r_tc),
                n_samples=int(X.shape[0]), censored_fraction=float(cens.mean()))
