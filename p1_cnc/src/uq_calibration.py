"""High-throughput UQ: Sobol GSA on the GP surrogate, emcee calibration, conditioned RUL propagation.

Round-3 changes:
* c_w removed from the sampler (not identified; the force-observation term is dropped with it);
* prior support restricted to GP-mean VB(horizon) <= VB_PRIOR_MAX (same support as the design filter);
* Sobol analysis on that truncated prior (Saltelli base rows rejected jointly across their blocks);
* emcee with a DE-move / DE-snooker mixture instead of the stretch move only;
* posterior RUL conditioned on the observed VB at the stage time (offset correction of each draw).
"""
import emcee
import numpy as np
from emcee.moves import DEMove, DESnookerMove
from SALib.analyze import sobol
from SALib.sample import saltelli

from surrogate_model import U_NAMES, VB_PRIOR_MAX

VB_CRIT = 0.30
CAL_NAMES = ["k1", "k2", "alpha", "hA", "sigma_obs"]   # k1,k2,hA,sigma_obs sampled in log space; c_w dropped


def rul_from_grid(t_grid, Y, thr=VB_CRIT):
    """First time VB crosses thr (linear interpolation on grid). Returns (t_cross, censored)."""
    Y = np.atleast_2d(Y)
    n = Y.shape[0]
    above = Y >= thr
    has = above.any(axis=1)
    j = np.argmax(above, axis=1)
    j0 = np.maximum(j - 1, 0)
    rows = np.arange(n)
    y0, y1 = Y[rows, j0], Y[rows, j]
    t0, t1 = t_grid[j0], t_grid[j]
    den = np.where(y1 - y0 == 0, 1.0, y1 - y0)
    tc = t0 + (thr - y0) / den * (t1 - t0)
    tc = np.where(j == 0, t_grid[0], tc)
    tc = np.where(has, tc, t_grid[-1])
    return tc, ~has


def conditioned_rul(t_grid, M, t_anchor, y_anchor, thr=VB_CRIT):
    """RUL per posterior draw, conditioned on the observed VB at the stage time.

    M: (k, G) GP mean trajectories on t_grid. Each draw's future trajectory is shifted by the offset
    y_anchor - m(t_anchor) (so it passes through the observation), then the first crossing at or after
    t_anchor is located. Returns (rul measured from t_anchor, censored flag at the grid end).
    """
    m_at = np.array([np.interp(t_anchor, t_grid, row) for row in M])
    delta = y_anchor - m_at
    fut = t_grid > t_anchor
    tg = np.concatenate([[t_anchor], t_grid[fut]])
    G = np.column_stack([np.full(len(M), y_anchor), M[:, fut] + delta[:, None]])
    tc, cens = rul_from_grid(tg, G, thr)
    return tc - t_anchor, cens


def sobol_analysis(gp, lo, hi, t_grid, N=4096, vb_max=VB_PRIOR_MAX):
    """Saltelli/Jansen indices on the truncated prior.

    SALib (1.5) lays the sample out grouped by base point: each group of D+2 rows is
    [A_j, AB_1j, ..., AB_Dj, B_j] and sobol.analyze splits Y with stride D+2. A base point is kept only if
    all D+2 of its rows lie in the prior support (GP-mean VB at the horizon <= vb_max), so the groups stay aligned.
    """
    D = len(U_NAMES)
    problem = {"num_vars": D, "names": U_NAMES,
               "bounds": [[float(a), float(b)] for a, b in zip(lo, hi)]}
    X = saltelli.sample(problem, N, calc_second_order=False)      # N groups of D+2 rows
    X3 = X.reshape(N, D + 2, D)
    m_all, _ = gp.predict(X)
    support = (m_all[:, -1] <= vb_max).reshape(N, D + 2)
    keep = np.all(support, axis=1)
    n_keep = int(keep.sum())
    Xk = X3[keep].reshape(-1, D)
    mk, _ = gp.predict(Xk)
    rul, cens = rul_from_grid(t_grid, mk)
    res = {"n_eval": int(Xk.shape[0]), "n_base": int(N), "n_base_kept": n_keep,
           "support_fraction_base": float(n_keep / N), "RUL_censored_frac": float(cens.mean())}
    for key, Y in [("RUL", rul), ("VB_end", mk[:, -1])]:
        Si = sobol.analyze(problem, Y, calc_second_order=False, print_to_console=False)
        res[key] = {n: {"S1": float(Si["S1"][i]), "ST": float(Si["ST"][i])} for i, n in enumerate(U_NAMES)}
    return res


def cal_bounds(lo, hi):
    """Calibration prior box: uniform in the surrogate box for k1, k2, alpha, hA; log-uniform sigma_obs in [1e-4, 1e-1] mm.
    c_w is no longer sampled (round 3)."""
    lo_c = np.array([lo[0], lo[1], lo[2], lo[3], np.log(1e-4)])
    hi_c = np.array([hi[0], hi[1], hi[2], hi[3], np.log(1e-1)])
    return lo_c, hi_c


def make_logp(gp, t_grid, obs, lo_c, hi_c, vc, vb_max=VB_PRIOR_MAX):
    """Vectorised log-posterior over (k, 5) parameter arrays. Gaussian likelihood on inspected VB with
    sigma_obs and GP variance; prior support requires GP-mean VB(horizon) <= vb_max."""
    t_vb, y_vb = obs["t_vb"], obs["y_vb"]
    G = len(t_grid)

    def interp_w(t):
        j = np.clip(np.searchsorted(t_grid, t, side="right") - 1, 0, G - 2)
        w = (t - t_grid[j]) / (t_grid[j + 1] - t_grid[j])
        return j, w

    jv, wv = interp_w(t_vb)

    def logp_vec(TH):
        TH = np.atleast_2d(TH)
        out = np.full(len(TH), -np.inf)
        ok = np.all((TH >= lo_c) & (TH <= hi_c), axis=1)
        if not ok.any():
            return out
        T = TH[ok]
        U = np.column_stack([T[:, 0], T[:, 1], T[:, 2], T[:, 3], np.full(len(T), vc)])
        M, S = gp.predict(U)
        lsig = T[:, 4:5]
        s_v = S[:, jv] * (1 - wv) + S[:, jv + 1] * wv
        m_v = M[:, jv] * (1 - wv) + M[:, jv + 1] * wv
        var = np.exp(2.0 * lsig) + s_v ** 2
        ll = -0.5 * np.sum((y_vb - m_v) ** 2 / var + np.log(2 * np.pi * var), axis=1)
        ll = np.where(M[:, -1] <= vb_max, ll, -np.inf)
        out[ok] = np.where(np.isfinite(ll), ll, -np.inf)
        return out

    return logp_vec


def run_emcee(logp, lo_c, hi_c, nwalkers=32, nburn=1000, nsteps=5000, seed=42):
    rng = np.random.default_rng(seed)
    ndim = len(lo_c)
    # starting points: uniform in the box, rejection-sampled to finite log-posterior (prior support)
    starts, n_have = [], 0
    while n_have < nwalkers:
        cand = rng.uniform(lo_c, hi_c, size=(4 * nwalkers, ndim))
        good = cand[np.isfinite(logp(cand))]
        starts.append(good)
        n_have += len(good)
    p0 = np.vstack(starts)[:nwalkers]
    moves = [(DEMove(), 0.8), (DESnookerMove(), 0.2)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, logp, vectorize=True, moves=moves)
    state = sampler.run_mcmc(p0, nburn, progress=False)
    sampler.reset()
    sampler.run_mcmc(state, nsteps, progress=False)
    chain = sampler.get_chain(flat=True)
    try:
        tau = np.asarray(sampler.get_autocorr_time(quiet=True), float)
    except Exception:
        tau = np.full(ndim, np.nan)
    acc = float(np.mean(sampler.acceptance_fraction))
    return chain, tau, acc


def posterior_grid(gp, chain_sub, vc):
    """GP mean/std on the grid for each posterior sample, plus observation noise sigma."""
    U = np.column_stack([chain_sub[:, 0], chain_sub[:, 1], chain_sub[:, 2], chain_sub[:, 3],
                         np.full(len(chain_sub), vc)])
    m, s = gp.predict(U)
    sig = np.exp(chain_sub[:, 4])
    return m, s, sig
