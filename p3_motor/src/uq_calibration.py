"""Sobol GSA on the GP emulator; emcee calibration on one held-out session with an AR(1) generalised
least-squares (GLS) likelihood on the three stator nodes; posterior and prior predictive propagation
through the physics model with an AR(1) model-error term (bias-free sd, estimated on training residuals
only). The forecast is initialised from the stator temperatures observed at the decision time."""
import numpy as np
import emcee
from scipy.optimize import minimize
from SALib.sample import saltelli
from SALib.analyze import sobol
import physics_model as pm
import surrogate_model as sm
from data_loader import DT
from decision import T_LIMIT, H_STEPS
import decision as dec

LEVELS = [("0.5", 25.0, 75.0), ("0.8", 10.0, 90.0), ("0.95", 2.5, 97.5)]   # central bands
BOUND = 0.5                                                             # prior box half-width


def _demeaned(seqs, stride):
    subs = [np.asarray(s, float)[::stride] for s in seqs]
    return [s - s.mean() for s in subs]


def integrated_time(seqs, stride=12, max_lag=60):
    """Diagnostic only (not used in the likelihood): integrated autocorrelation time of free-run
    residuals observed every `stride` 5 s steps (12 -> 60 s)."""
    subs = _demeaned(seqs, stride)
    c0 = sum(float(np.sum(s * s)) for s in subs)
    tau = 1.0
    for k in range(1, max_lag + 1):
        ck = sum(float(np.sum(s[:-k] * s[k:])) for s in subs if len(s) > k)
        rho = ck / c0
        if rho <= 0.0:
            break
        tau += 2.0 * rho
    return float(tau)


def ar1_rho(seqs, stride=12):
    """Lag-1 autocorrelation at the observation spacing (60 s), per-session demeaned, pooled."""
    subs = _demeaned(seqs, stride)
    num = sum(float(np.sum(s[:-1] * s[1:])) for s in subs if len(s) > 1)
    den = sum(float(np.sum(s * s)) for s in subs)
    return float(num / den)


def bias_free_sd(seqs, stride=12):
    """Standard deviation of residuals after removing each session's mean (bias-free)."""
    subs = _demeaned(seqs, stride)
    v = np.concatenate(subs)
    return float(np.sqrt(np.mean(v * v)))


def sobol_indices(gp, idx_out, L, N=1024):
    problem = {"num_vars": 6, "names": pm.UNCERT, "bounds": [[-BOUND, BOUND]] * 6}
    np.random.seed(0)
    X = saltelli.sample(problem, N, calc_second_order=False)
    Xf = np.column_stack([X, np.ones(len(X))])          # operating scale d = 1
    mu = gp.predict(sm.to_unit(Xf)).reshape(len(X), -1, 4)
    peak = mu[:, :, pm.M].max(axis=1)
    tt = idx_out * DT / 60.0
    over = mu[:, :, pm.M] > T_LIMIT
    first = np.argmax(over, axis=1)
    ttx = np.where(over.any(axis=1), tt[first], tt[-1])
    out = {}
    for name, y in [("peak_pm", peak), ("time_to_exceed_min", ttx)]:
        si = sobol.analyze(problem, y, calc_second_order=False, print_to_console=False)
        out[name] = {p: {"S1": float(si["S1"][i]), "ST": float(si["ST"][i])}
                     for i, p in enumerate(pm.UNCERT)}
    return out


def make_logpost(gp, Wi, y_obs, sig2, rho, K=sm.K_OUT):
    """AR(1) GLS log-likelihood on the three stator nodes (independent channels).
    Per channel j: residual r_t, variance s2_j (bias-free residual variance + mean emulator inflation,
    a constant), correlation rho_j at the 60-s observation spacing. Exact AR(1) quadratic form and
    log-determinant:
        Q = r_1^2/s2 + sum_{t>=2} (r_t - rho r_{t-1})^2 / (s2 (1 - rho^2)),
        logdet = n log s2 + (n - 1) log(1 - rho^2).
    The GP predictive sd is not evaluated per sample (it is ~0.1 degC against sigma ~3-4 degC)."""
    sig2 = np.asarray(sig2, float)
    rho = np.asarray(rho, float)

    def logp(Th):
        Th = np.atleast_2d(Th)
        inside = np.all(np.abs(Th) <= BOUND, axis=1)
        Xq = np.column_stack([Th, np.ones(len(Th))])
        mu = gp.predict(sm.to_unit(Xq)).reshape(-1, K, 4)[:, :, :3]
        mo = np.einsum("ok,nkv->nov", Wi, mu)                 # (n, S, 3)
        r = y_obs[None] - mo
        n = r.shape[1]
        ll = np.zeros(len(Th))
        for j in range(3):
            rj = r[:, :, j]
            s2, rh = sig2[j], rho[j]
            inn = rj[:, 1:] - rh * rj[:, :-1]
            Q = rj[:, 0] ** 2 / s2 + np.sum(inn ** 2, axis=1) / (s2 * (1.0 - rh * rh))
            logdet = n * np.log(s2) + (n - 1) * np.log(1.0 - rh * rh)
            ll += -0.5 * (Q + logdet + n * np.log(2.0 * np.pi))
        return np.where(inside, ll, -np.inf)
    return logp


def map_estimate(logp):
    """Maximum a posteriori point (L-BFGS-B inside the prior box), used as the sampler start."""
    def f(th):
        return -float(logp(th[None])[0])
    res = minimize(f, np.zeros(6), method="L-BFGS-B", bounds=[(-BOUND, BOUND)] * 6,
                   options=dict(maxiter=80))
    return res.x, -float(res.fun)


def run_mcmc(logp, seed, theta0, nwalkers=32, burn=500, block=500, max_post=3000, tau_mult=50.0):
    """emcee DE-move / DE-snooker mixture started near the MAP. Burn-in, then blocks until every
    parameter has n_post >= tau_mult * tau or n_post reaches max_post (reported as not converged)."""
    rng = np.random.default_rng(seed)
    ndim = 6
    p0 = np.clip(np.asarray(theta0)[None] + 0.01 * rng.normal(size=(nwalkers, ndim)), -0.49, 0.49)
    moves = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, logp, vectorize=True, moves=moves)
    state = sampler.run_mcmc(p0, burn, progress=False)
    sampler.reset()
    n_done = 0
    while True:
        state = sampler.run_mcmc(state, block, progress=False)
        n_done += block
        tau = sampler.get_autocorr_time(discard=0, quiet=True)
        if np.all(n_done >= tau_mult * tau) or n_done >= max_post:
            break
    chain = sampler.get_chain(flat=True)
    acc = float(np.mean(sampler.acceptance_fraction))
    tau = np.asarray(tau, float)
    return dict(chain=chain, tau=tau, acc=acc, n_post=int(n_done),
                ess=nwalkers * n_done / tau, converged=bool(np.all(n_done >= tau_mult * tau)))


def ar1_noise(rng, n, k, sd, rho):
    """Stationary AR(1) error paths (n sample paths, k observation times) with marginal sd and lag-1 rho."""
    e = np.empty((n, k))
    e[:, 0] = rng.normal(0.0, sd, n)
    s = sd * np.sqrt(1.0 - rho * rho)
    for j in range(1, k):
        e[:, j] = rho * e[:, j - 1] + s * rng.normal(size=n)
    return e


def interval_score(lo, hi, y, alpha):
    """Gneiting-Raftery interval score (lower is better) for a central (1-alpha) interval."""
    return float(np.mean((hi - lo) + (2.0 / alpha) * np.maximum(lo - y, 0.0)
                         + (2.0 / alpha) * np.maximum(y - hi, 0.0)))


def band_stats(pred, y):
    out = {}
    for name, a, b in LEVELS:
        lo, hi = np.percentile(pred, [a, b], axis=0)
        out[name] = dict(coverage=float(np.mean((y >= lo) & (y <= hi))),
                         width=float(np.mean(hi - lo)),
                         interval_score=interval_score(lo, hi, y, 1.0 - float(name)),
                         lo=lo, hi=hi)
    return out


def posterior_predict(P_nom, theta, prof, x0, t_c, err, rng, limit=T_LIMIT):
    """Forecast of the physics model from t_c (initialised with the observed stator state) for each
    parameter sample, no action. err = dict(sd, rho) of the pm model error (bias-free, AR(1)).
    The predictive band = parameter spread + AR(1) model error; the parameter-only band omits the error.
    Risk quantities use the decision window [t_c, t_c + H] (same function as decision.policy_table)."""
    P = pm.theta_to_params(P_nom, theta)
    L = len(prof["i2"])
    pre, x_tc = dec.init_state(P, prof, x0, t_c)
    Xf = dec.forecast(P, prof, t_c, L, x_tc)
    X = np.concatenate([pre, Xf], axis=1)                              # (N, L, 4)
    N = X.shape[0]
    steps = np.arange(t_c + 1, L, 12)                                  # 60-s observation grid
    y = prof["meas"][steps, pm.M]
    mean_pred = X[:, steps, pm.M]
    noisy = mean_pred + ar1_noise(rng, N, len(steps), err["sd"], err["rho"])
    pred = band_stats(noisy, y)
    postonly = band_stats(mean_pred, y)
    Lw = min(t_c + H_STEPS + 1, L)
    over_w = X[:, t_c:Lw, pm.M] > limit
    over_s = X[:, t_c:, pm.M] > limit
    return dict(X=X, steps=steps, lo=pred["0.95"]["lo"], hi=pred["0.95"]["hi"],
                pred=pred, postonly=postonly,
                bias=float(np.mean(np.median(mean_pred, axis=0) - y)),
                p_window=float(over_w.any(axis=1).mean()),
                p_session=float(over_s.any(axis=1).mean()),
                peak_w=X[:, t_c:Lw, pm.M].max(axis=1))
