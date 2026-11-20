"""Coupled physics model: flank wear ODE + lumped cutting-zone thermal ODE.

State y = [VB (mm), T (K)], time t in pseudo-time units (PTU).

  dVB/dt = k1 * sigma_n * vc * exp(-Ea/(R T)) + k2 * F_res(t)
  C_th dT/dt = eta * F_res(t) * vc * (1 + alpha * VB) - hA * (T - T_inf)

Force observation model (used in calibration): F_pred = F0 + c_w * VB.
F_res(t) is driven by the empirical F_res sequence (periodic wrap beyond the record).
Fixed (non-fitted) constants are nominal values documented in summary.md.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

SIGMA_N = 1.0        # normalised normal stress (fixed)
EA_OVER_R = 2000.0   # Ea/R [K] (fixed)
C_TH = 1.0           # lumped heat capacity, normalised (fixed)
T_INF = 300.0        # ambient temperature [K] (fixed)
HA_NOM = 0.02        # nominal convective coefficient [1/PTU]
T_RISE_NOM = 150.0   # nominal steady cutting-zone temperature rise [K]
VC_NOM = 1.0         # nominal cutting speed (normalised)


def make_constants(F_mean):
    """eta chosen so that the nominal steady temperature rise is T_RISE_NOM at mean force."""
    return T_RISE_NOM * HA_NOM / float(F_mean)


def _rhs_factory(k1, k2, alpha, hA, vc, F_series, eta):
    N = len(F_series)
    idx = np.arange(N, dtype=float)

    def rhs(t, y):
        VB = y[0] if y[0] > 0.0 else 0.0   # VB >= 0 enforced in the rates
        T = y[1]
        F = np.interp(np.mod(t, N), idx, F_series)
        dVB = k1 * SIGMA_N * vc * np.exp(-EA_OVER_R / T) + k2 * F
        dT = (eta * F * vc * (1.0 + alpha * VB) - hA * (T - T_INF)) / C_TH
        return [dVB, dT]

    return rhs


def simulate(theta, t_eval, F_series, eta, VB0=0.0):
    """Integrate the coupled ODE and return VB(t_eval). NaN array on solver failure."""
    k1, k2, alpha, hA, vc = [float(x) for x in theta]
    t_eval = np.asarray(t_eval, float)
    rhs = _rhs_factory(k1, k2, alpha, hA, vc, F_series, eta)
    sol = solve_ivp(rhs, (t_eval[0], t_eval[-1]), [float(VB0), T_INF], t_eval=t_eval,
                    method="RK45", rtol=1e-6, atol=1e-9)
    if (not sol.success) or sol.y.shape[1] != t_eval.size:
        return np.full(t_eval.size, np.nan)
    return sol.y[0]


def simulate_batch(thetas, t_grid, F_series, eta, VB0=0.0, dt=1.0):
    """Vectorised fixed-step RK4 (dt = 1 PTU) for many parameter sets at once.

    thetas: (n, 5) array of (k1, k2, alpha, hA, vc). Returns (n, len(t_grid)) VB, linearly
    interpolated onto t_grid. Used for surrogate training data (~50x faster than solve_ivp).
    """
    th = np.atleast_2d(np.asarray(thetas, float))
    k1, k2, alpha, hA, vc = th[:, 0], th[:, 1], th[:, 2], th[:, 3], th[:, 4]
    N = len(F_series)
    idx = np.arange(N, dtype=float)
    t_end = float(t_grid[-1])
    n_steps = int(np.ceil(t_end / dt))
    t_fine = np.arange(n_steps + 1) * dt
    F_at = lambda t: np.interp(np.mod(t, N), idx, F_series)
    VB = np.full(len(th), float(VB0))
    T = np.full(len(th), T_INF)
    out = np.empty((len(th), n_steps + 1))
    out[:, 0] = VB

    def rhs(VBv, Tv, F):
        VBp = np.maximum(VBv, 0.0)
        dVB = k1 * SIGMA_N * vc * np.exp(-EA_OVER_R / Tv) + k2 * F
        dT = (eta * F * vc * (1.0 + alpha * VBp) - hA * (Tv - T_INF)) / C_TH
        return dVB, dT

    for s in range(n_steps):
        t0 = t_fine[s]
        F0, Fm, F1 = F_at(t0), F_at(t0 + 0.5 * dt), F_at(t0 + dt)
        a1, b1 = rhs(VB, T, F0)
        a2, b2 = rhs(VB + 0.5 * dt * a1, T + 0.5 * dt * b1, Fm)
        a3, b3 = rhs(VB + 0.5 * dt * a2, T + 0.5 * dt * b2, Fm)
        a4, b4 = rhs(VB + dt * a3, T + dt * b3, F1)
        VB = VB + dt / 6.0 * (a1 + 2 * a2 + 2 * a3 + a4)
        T = T + dt / 6.0 * (b1 + 2 * b2 + 2 * b3 + b4)
        out[:, s + 1] = VB
    t_grid = np.asarray(t_grid, float)
    return np.array([np.interp(t_grid, t_fine, row) for row in out])


def fit_baseline(t_all, train_idx, vb, F_series, eta):
    """Deterministic least-squares fit on TRAINING rows only (sets prior centres)."""
    x0 = np.array([np.log(0.016), np.log(1e-6), 0.5, np.log(0.02), 0.0])
    lb = np.array([-8.0, -20.0, 0.0, -7.0, 0.0])
    ub = np.array([0.0, -5.0, 2.0, -1.0, 0.1])

    def resid(p):
        th = (np.exp(p[0]), np.exp(p[1]), p[2], np.exp(p[3]), VC_NOM)
        y = simulate(th, t_all, F_series, eta, VB0=p[4])
        r = y[train_idx] - vb[train_idx]
        return np.where(np.isfinite(r), r, 1.0)

    sol = least_squares(resid, x0, bounds=(lb, ub), max_nfev=300, x_scale="jac", diff_step=1e-4)
    p = sol.x
    theta = np.array([np.exp(p[0]), np.exp(p[1]), p[2], np.exp(p[3]), VC_NOM])
    return {
        "theta": theta,                       # k1, k2, alpha, hA, vc
        "VB0": float(p[4]),
        "success": bool(sol.success),
        "nfev": int(sol.nfev),
        "train_rmse": float(np.sqrt(np.mean(sol.fun ** 2))),
        "u": p,                               # log_k1, log_k2, alpha, log_hA, VB0
    }
