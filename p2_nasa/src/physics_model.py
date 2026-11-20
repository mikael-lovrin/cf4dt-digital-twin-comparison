"""Physics model: coupled flank-wear ODE + lumped cutting-zone thermal balance.

State per case: x = [VB (mm), dT (K above ambient)], integrated with scipy solve_ivp (RK45).

  dVB/dt = A * qn * u(t) * exp(Ea_R * (1/T_inf - 1/(T_inf + min(dT, DT_MAX)))) + B * u(t)
  d dT/dt = ETA * u(t) * qn * (1 + alpha * VB) - hA * dT

  u(t) : spindle-motor DC current (smcDC) normalised by the training median, used as the
         force proxy F_res(t) (no force sensor in this dataset).
  qn   : operating load DOC*feed / (0.75*0.5), the DOC and feed operating inputs.
  A    : 10**log10A * 10**(gamma * material_flag); k1*sigma_n*v_c lumped (v_c is constant
         in this dataset and not separately identifiable).
  B    : 10**log10B, the force-driven (k2) abrasion term.
  ETA  : lumped heat-source gain (K per time unit); C_th is absorbed into ETA and hA.
  T_inf: 300 K reference temperature.
  VB(0) = 0 (new insert), dT(0) = 0.

Round 3 change: the design box is no longer a hard-coded constant. The outer limits PHYS_LO/PHYS_HI are
fixed a priori (not derived from any fit). Within each CV fold the box is derived from TRAINING-fold fits
only (derive_box): stage A fits the training cases on the outer limits with multistart; the range of the
near-optimal stage-A fits, extended on each side, defines the fold box; stage B refits inside that box.
Test cases never enter either stage.
Observation: VB is only used where measured. The force observation link
smcDC = F0 + cw * VB is fitted for documentation (fit_force_observation) and not used in calibration.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.stats import qmc

T_INF = 300.0
ETA = 5.0
DT_MAX = 400.0   # K, cap on the temperature rise inside the Arrhenius factor (cutting-zone temperature saturates;
                 # without it the wear-heat feedback runs away at extreme parameters, giving VB ~ 1e4-1e6 mm)
PARAM_NAMES = ["log10A", "log10B", "Ea_R", "alpha", "log10hA"]

# A-priori outer limits, fixed before any fit. Not derived from data.
PHYS_LO = np.array([-5.0, -7.0, 0.0, 0.0, -4.0])
PHYS_HI = np.array([0.0, -1.0, 2000.0, 5.0, 0.0])
GAMMA_LO, GAMMA_HI = -1.0, 1.0
LO6 = np.r_[PHYS_LO, GAMMA_LO]
HI6 = np.r_[PHYS_HI, GAMMA_HI]

# Fit settings (round 3)
MAX_NFEV = 1000
N_STARTS = 8
# Box derivation rule, fixed before results were seen:
BOX_NEAR_OPT = 1.05    # stage-A fits with cost <= 1.05 x best define the near-optimal set (training data only)
BOX_EXTEND = 0.5       # each side of the near-optimal range is extended by 50% of that range
BOX_MIN_FRAC = 0.2     # minimum range of the box as a fraction of the outer-limit range (guards a degenerate set)


def simulate(theta, gamma, cases, t_eval_list, dt=0.05):
    """Integrate VB for several cases in one solve_ivp call.

    cases: list of case dicts (t_u, u_fit, qn, material). t_eval_list: per-case output times (ascending).
    The force proxy is interpolated from the non-transient samples (t_u, u_fit); before the first
    valid sample and after the last one it is held constant.
    Returns a list of VB arrays aligned with t_eval_list (NaN if the solver failed).
    """
    lA, lB, Ea, al, lh = [float(v) for v in theta]
    n = len(cases)
    tall = np.unique(np.concatenate([np.asarray(t, float) for t in t_eval_list]))
    tmax = float(tall[-1])
    grid = np.arange(0.0, tmax + 2 * dt, dt)
    U = np.stack([np.interp(grid, c["t_u"], c["u_fit"]) for c in cases])   # constant extrapolation
    qn = np.array([c["qn"] for c in cases], float)
    mat = np.array([c["material"] for c in cases], float)
    A = 10.0 ** lA * 10.0 ** (gamma * mat)
    B = 10.0 ** lB
    hA = 10.0 ** lh

    def rhs(t, x):
        VB = np.maximum(x[:n], 0.0)
        dT = np.maximum(x[n:], 0.0)
        k = t / dt
        i0 = min(int(k), grid.size - 2)
        fr = k - i0
        u = (1.0 - fr) * U[:, i0] + fr * U[:, i0 + 1]
        g = np.exp(Ea * (1.0 / T_INF - 1.0 / (T_INF + np.minimum(dT, DT_MAX))))
        return np.concatenate([A * qn * u * g + B * u,
                               ETA * u * qn * (1.0 + al * VB) - hA * dT])

    sol = solve_ivp(rhs, (0.0, tmax), np.zeros(2 * n), t_eval=tall,
                    method="RK45", rtol=1e-4, atol=1e-7)
    if (not sol.success) or sol.y.shape[1] != tall.size:
        return [np.full(len(t), np.nan) for t in t_eval_list]
    out = []
    for i, t in enumerate(t_eval_list):
        idx = np.searchsorted(tall, np.asarray(t, float))
        out.append(np.maximum(sol.y[i, idx], 0.0))
    return out


def _residuals(p6, cases, t_list, obs_list):
    preds = simulate(p6[:5], p6[5], cases, t_list)
    return np.concatenate([np.nan_to_num(p - o, nan=1e3) for p, o in zip(preds, obs_list)])


def lsq_job(job):
    """One least-squares start. Module-level so that it can run in a process pool."""
    cases, t_list, obs, x0, lo, hi, max_nfev = job
    r = least_squares(_residuals, x0, bounds=(lo, hi), args=(cases, t_list, obs),
                      x_scale="jac", max_nfev=max_nfev)
    return dict(x=r.x.tolist(), cost=float(r.cost), nfev=int(r.nfev), status=int(r.status),
                rmse_train=float(np.sqrt(np.mean(r.fun ** 2))), lo=lo.tolist(), hi=hi.tolist())


def make_jobs(cases, lo6, hi6, seed, extra=(), n_starts=N_STARTS, max_nfev=MAX_NFEV):
    """Multistart jobs: centre of the box, the supplied extra points, then Latin-hypercube starts."""
    t_list = [c["t"][c["m"]] for c in cases]
    obs = [c["VB"][c["m"]] for c in cases]
    pts = [0.5 * (lo6 + hi6)] + [np.clip(np.asarray(e, float), lo6, hi6) for e in extra]
    m = n_starts - len(pts)
    pts += list(lo6 + qmc.LatinHypercube(d=6, seed=seed).random(m) * (hi6 - lo6))
    return [(cases, t_list, obs, x0, lo6, hi6, max_nfev) for x0 in pts]


def derive_box(stage_a):
    """Fold box from training-fold stage-A fits only.

    near = stage-A fits with cost <= BOX_NEAR_OPT * best; range = [min, max] of their five physics parameters;
    box = range extended by BOX_EXTEND * range on each side, clipped to the outer limits. A range narrower
    than BOX_MIN_FRAC of the outer range is widened to that minimum before extension.
    """
    best = min(f["cost"] for f in stage_a)
    near = np.array([f["x"][:5] for f in stage_a if f["cost"] <= BOX_NEAR_OPT * best])
    lo_n, hi_n = near.min(axis=0), near.max(axis=0)
    R = np.maximum(hi_n - lo_n, BOX_MIN_FRAC * (PHYS_HI - PHYS_LO))
    lo = np.maximum(PHYS_LO, lo_n - BOX_EXTEND * R)
    hi = np.minimum(PHYS_HI, hi_n + BOX_EXTEND * R)
    return lo, hi, int(near.shape[0])


def best_fit(fits, lo6, hi6):
    """Best start of a fold's stage-B multistart, with convergence and boundary flags."""
    b = min(fits, key=lambda f: f["cost"])
    x = np.array(b["x"])
    tol = 1e-3 * (hi6 - lo6)
    n_conv = sum(1 for f in fits if f["status"] > 0)
    return dict(p6=x.tolist(), rmse_train=b["rmse_train"], cost=b["cost"], nfev=b["nfev"],
                status=b["status"], converged=bool(b["status"] > 0),
                at_bound=bool(np.any((x <= lo6 + tol) | (x >= hi6 - tol))),
                n_starts=len(fits), n_starts_converged=int(n_conv),
                box_lo=list(map(float, lo6[:5])), box_hi=list(map(float, hi6[:5])))


def fit_force_observation(cases):
    """Within-case regression smcDC = F0_c + cw * VB (documents the force-proxy observation model).
    Uses measured, non-transient rows only."""
    xs, ys = [], []
    for c in cases:
        ok = c["m"] & c["keep"]
        v = c["VB"][ok]
        f = c["u"][ok]
        xs.append(v - v.mean())
        ys.append(f - f.mean())
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    cw = float(np.sum(x * y) / np.sum(x * x))
    r2 = float(np.corrcoef(x, y)[0, 1] ** 2) if x.std() > 0 and y.std() > 0 else float("nan")
    return cw, r2
