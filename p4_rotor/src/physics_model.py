"""Physics models for p4_rotor (round 3).

PRIMARY model: two-DOF lumped rotor, closed-form steady state (no time integration).
  Two disc-carrying coordinates x_A, x_B; each disc mass is normalised to 1 (mass ratio fixed, see summary):
      x_A'' + cA x_A' + (kA + ks) x_A - ks x_B = f_A(t)
      x_B'' + cB x_B' + (kB + ks) x_B - ks x_A = f_B(t)
  kA, kB: bearing/support stiffness per unit mass [1/s^2]; ks: coupling (shaft) stiffness per unit mass [1/s^2];
  cA, cB: viscous damping per unit mass [1/s].
  Unbalance forcing (per unit mass, complex amplitudes, e^{i w t}):
      f_A = w^2 * U_A * e0,            U_A = Unbalance * TypeA  [g-units]
      f_B = w^2 * U_B * e0 * kB_rel * exp(i phi),   U_B = Unbalance * TypeB
  Steady state: Z X = f with Z = [[DA, -ks], [-ks, DB]], DA = kA + ks - w^2 + i cA w, DB = kB + ks - w^2 + i cB w
      X_A = (DB f_A + ks f_B) / (DA DB - ks^2),   X_B = (ks f_A + DA f_B) / (DA DB - ks^2)
  Output at sensor Position p (per-position weights G_p and rho_p on the two coordinates):
      a_p = w^2 |G_p (X_A + rho_p X_B)|,   RMS acceleration [g] = a_p / sqrt(2) / g0
  Healthy background per Position (Normal reference), scaled by zeta_p, added in quadrature:
      RMS_p = sqrt(unb_p^2 + (zeta_p * floor_p(RPM))^2)

Parameter vector theta (16), see P2_NAMES. G at Position 1 is fixed to 1 (scale absorbed by e0).
The closed form is verified against an independent real 4x4 linear solve in main.py (surrogate block).

ABLATION model (round 2 structure, kept as a baseline row): single-DOF Jeffcott rotor,
  a(w) = w^2 * m_u e w^2 / sqrt((k - m w^2)^2 + (c w)^2), m_u = Ueff * 1e-3 kg, Ueff = U * (TypeA + TypeB),
  fitted (round 3) only on the single-disc Mis=0 cells (A1B0, A0B1); the A1B1 configuration is excluded from its fit.
  Healthy floor: pooled over positions (round-2 definition).
"""
import numpy as np
from scipy.optimize import least_squares

from data_loader import GRAV

# ------------------------------------------------------------------ primary: 2-DOF
P2_NAMES = ["log_kA", "log_kB", "log_ks", "log_cA", "log_cB", "log_e0", "log_kappaB", "phi",
            "rho_p1", "rho_p2", "rho_p3", "log_G_p2", "log_G_p3", "log_zeta_p1", "log_zeta_p2", "log_zeta_p3"]
P2_LO = np.array([np.log(1e4), np.log(1e4), np.log(1e2), np.log(1e-1), np.log(1e-1), np.log(1e-9), np.log(1e-2), -np.pi,
                  -6.0, -6.0, -6.0, -4.0, -4.0, -3.0, -3.0, -3.0])
P2_HI = np.array([np.log(3e6), np.log(3e6), np.log(3e6), np.log(1e3), np.log(1e3), np.log(1e-4), np.log(1e2), np.pi,
                  6.0, 6.0, 6.0, 4.0, 4.0, 3.0, 3.0, 3.0])
# parameters the held-out A1B0 experiment can inform (kappa_B and phi act only through disc B; zeta through the floor)
P2_FREE_CAL = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12]
P2_CAL_NAMES = [P2_NAMES[i] for i in P2_FREE_CAL]


def _w(rpm):
    return 2.0 * np.pi * np.asarray(rpm, float)[None, :] / 60.0


def _xab(th, rpm, UA, UB, eps_scale=1.0):
    """Complex steady-state amplitudes (X_A, X_B) in m for parameter rows th (W,16) and N evaluation points."""
    th = np.atleast_2d(np.asarray(th, float))
    w = _w(rpm)
    UA = np.asarray(UA, float)[None, :]
    UB = np.asarray(UB, float)[None, :]
    kA = np.exp(th[:, 0:1])
    kB = np.exp(th[:, 1:2])
    ks = np.exp(th[:, 2:3])
    cA = np.exp(th[:, 3:4])
    cB = np.exp(th[:, 4:5])
    e0 = np.exp(th[:, 5:6]) * eps_scale
    kap = np.exp(th[:, 6:7])
    phi = th[:, 7:8]
    DA = kA + ks - w ** 2 + 1j * cA * w
    DB = kB + ks - w ** 2 + 1j * cB * w
    det = DA * DB - ks ** 2
    fA = w ** 2 * UA * e0
    fB = w ** 2 * UB * e0 * kap * np.exp(1j * phi)
    XA = (DB * fA + ks * fB) / det
    XB = (ks * fA + DA * fB) / det
    return XA, XB, w


def unb2_g(th, rpm, pos, UA, UB, eps_scale=1.0):
    """Unbalance-only RMS acceleration [g] (no healthy floor), shape (W, N)."""
    th = np.atleast_2d(np.asarray(th, float))
    pos = np.asarray(pos, int)
    XA, XB, w = _xab(th, rpm, UA, UB, eps_scale)
    G = np.exp(np.column_stack([np.zeros(len(th)), th[:, 11], th[:, 12]]))[:, pos]
    rho = th[:, 8:11][:, pos]
    amp = np.abs(G * (XA + rho * XB))
    return w ** 2 * amp / np.sqrt(2.0) / GRAV


def pred2_g(th, rpm, pos, UA, UB, floor_row, eps_scale=1.0):
    """Total predicted RMS [g] of the 2-DOF model, shape (W, N). floor_row: healthy floor at each point [g]."""
    th = np.atleast_2d(np.asarray(th, float))
    pos = np.asarray(pos, int)
    u = unb2_g(th, rpm, pos, UA, UB, eps_scale)
    z = np.exp(th[:, 13:16])[:, pos]
    return np.sqrt(u ** 2 + (z * np.asarray(floor_row, float)[None, :]) ** 2)


def steady_state_real4(th, rpm_val, UA, UB):
    """Independent check: solve the full real 4x4 block system for one parameter vector and one speed.
    Returns complex (X_A, X_B). Uses the same physical equations, but no closed-form inverse."""
    th = np.asarray(th, float)
    kA, kB, ks, cA, cB = (np.exp(th[i]) for i in range(5))
    e0, kap, phi = np.exp(th[5]), np.exp(th[6]), th[7]
    w = 2.0 * np.pi * rpm_val / 60.0
    K = np.array([[kA + ks, -ks], [-ks, kB + ks]])
    C = np.diag([cA, cB])
    A = K - w ** 2 * np.eye(2)
    B = w * C
    F = np.array([w ** 2 * UA * e0, w ** 2 * UB * e0 * kap * np.exp(1j * phi)])
    M = np.block([[A, -B], [B, A]])
    rhs = np.concatenate([F.real, F.imag])
    sol = np.linalg.solve(M, rhs)
    X = sol[:2] + 1j * sol[2:]
    return X[0], X[1]


def fit2(rpm, pos, UA, UB, y, floor_row, n_starts=0, seed=0, x0s=()):
    """Least squares on log RMS for the 2-DOF model. Warm starts x0s (list of 16-vectors) plus n_starts random."""
    rng = np.random.default_rng(seed)
    rpm = np.asarray(rpm, float)
    lo, hi = P2_LO.copy(), P2_HI.copy()
    ly = np.log(np.asarray(y, float))

    def resid(x):
        p = pred2_g(x[None, :], rpm, pos, UA, UB, floor_row)[0]
        return np.log(np.maximum(p, 1e-12)) - ly

    starts = [np.clip(np.asarray(x, float), lo + 1e-9, hi - 1e-9) for x in x0s]
    starts += [lo + (hi - lo) * rng.random(len(lo)) for _ in range(n_starts)]
    best = None
    for x0 in starts:
        r = least_squares(resid, x0, bounds=(lo, hi), method="trf", max_nfev=300)
        if best is None or r.cost < best.cost:
            best = r
    return {"theta": best.x, "rmse_log": float(np.sqrt(np.mean(best.fun ** 2))), "n": int(len(y))}


# ------------------------------------------------------------------ ablation: single-DOF Jeffcott (round-2 structure)
PARAMS = ["k", "c", "m", "e"]
PRIOR_LOG = np.array([
    [np.log(1e3), np.log(1e8)],     # k  [N/m]
    [np.log(1.0), np.log(1e5)],     # c  [N s/m]
    [np.log(0.5), np.log(20.0)],    # m  [kg]
    [np.log(1e-5), np.log(1e-1)],   # e  [m]
])
M_FIX = 3.0  # kg, mass fixed during least squares (scale degeneracy of m and e)


def unbalance_rms_g(theta, rpm, ueff_g):
    """RMS acceleration [g] of the single-DOF unbalance response. theta (...,4) SI, broadcastable."""
    theta = np.asarray(theta, float)
    k, c, m, e = (theta[..., i] for i in range(4))
    w = 2.0 * np.pi * np.asarray(rpm, float) / 60.0
    mu = np.asarray(ueff_g, float) * 1e-3
    x = mu * e * w ** 2 / np.sqrt((k - m * w ** 2) ** 2 + (c * w) ** 2)
    return (w ** 2 * x) / np.sqrt(2.0) / GRAV


def total_rms_g(theta, rpm, ueff_g, floor_g):
    """Single-DOF prediction: unbalance response plus healthy background in quadrature."""
    u = unbalance_rms_g(theta, rpm, ueff_g)
    return np.sqrt(u ** 2 + np.asarray(floor_g, float) ** 2)


def fit_physics(rpm, ueff, y, floor, n_starts=8, seed=0):
    """Least-squares fit of single-DOF (k, c, e) with m fixed at M_FIX, on log RMS. Multi-start."""
    rng = np.random.default_rng(seed)
    idx = [0, 1, 3]
    lo = PRIOR_LOG[idx, 0]
    hi = PRIOR_LOG[idx, 1]

    def to_theta(x):
        return np.array([np.exp(x[0]), np.exp(x[1]), M_FIX, np.exp(x[2])])

    def resid(x):
        pred = total_rms_g(to_theta(x), rpm, ueff, floor)
        return np.log(np.maximum(pred, 1e-9)) - np.log(y)

    best = None
    for x0 in lo + (hi - lo) * rng.random((n_starts, 3)):
        r = least_squares(resid, x0, bounds=(lo, hi), method="trf")
        if best is None or r.cost < best.cost:
            best = r
    res = resid(best.x)
    return {"theta": to_theta(best.x), "rmse_log": float(np.sqrt(np.mean(res ** 2))), "n": int(len(y))}
