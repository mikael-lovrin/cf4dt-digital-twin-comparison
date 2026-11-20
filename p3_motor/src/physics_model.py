"""Lumped-parameter thermal network (LPTN) for PMSM temperatures (replaces the flank-wear ODE).

Nodes (state x = T, deg C):  W = stator winding, T = stator tooth, Y = stator yoke, M = magnet (pm).
Boundaries: coolant (housing) and ambient temperatures (measured inputs).

  C_i dT_i/dt = -sum_j G_ij (T_i - T_j) - sum_b G_ib (T_i - T_b) + P_i(t, T_W)

  P_cu  = k_cu * d^2 * (i_d^2 + i_q^2) * (1 + alpha_cu (T_W - 20))   -> winding (temperature-dependent R)
  P_fe  = k_fe1 |n| + k_fe2 n^2   (n in krpm), split 50/50 tooth/yoke   -> iron loss
  P_m   = k_m n^2                                                          -> magnet/rotor eddy loss
  d     = operating scale (current derating factor, 1 = nominal)

Integration: the network is linear in T for fixed losses. The copper term uses the winding
temperature of the previous step (explicit, weak coupling), inputs are averaged over each
5 s step, and the step is advanced EXACTLY with the matrix exponential of the augmented
system (unconditionally stable, no stiffness restriction). Cross-checked against
scipy.integrate.solve_ivp (LSODA) in main.py.
"""
import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from data_loader import DT

PARAM_NAMES = ["C_W", "C_T", "C_Y", "C_M", "G_WT", "G_TY", "G_TM", "G_WC", "G_YC", "G_YA",
               "k_cu", "k_fe1", "k_fe2", "k_m"]
IDX = {n: i for i, n in enumerate(PARAM_NAMES)}
# uncertain parameters (log-multipliers on the nominal calibrated values)
UNCERT = ["C_W", "C_M", "G_WT", "G_TM", "G_YC", "k_cu"]
UNCERT_IDX = [IDX[n] for n in UNCERT]

ALPHA_CU = 0.0039   # 1/K, copper resistance temperature coefficient (fixed)
T_REF = 20.0
W, T, Y, M = 0, 1, 2, 3

# initial guess (SI-like units: J/K, W/K, W/A^2, W/krpm, W/krpm^2)
P_INIT = np.array([1e3, 5e3, 1e4, 1e3,      # capacities
                   50., 100., 20.,          # G_WT, G_TY, G_TM
                   30., 100., 5.,           # G_WC, G_YC, G_YA
                   0.03, 20., 5., 0.5])     # k_cu, k_fe1, k_fe2, k_m
P_LO = np.array([50.] * 4 + [0.1] * 6 + [1e-4, 1e-3, 1e-3, 1e-6])
P_HI = np.array([1e6] * 4 + [1e4] * 6 + [10., 1e4, 1e4, 1e3])


def theta_to_params(P_nom, theta):
    """theta (N,6) log-multipliers -> parameter matrix (N,14)."""
    theta = np.atleast_2d(np.asarray(theta, float))
    P = np.tile(np.asarray(P_nom, float), (theta.shape[0], 1))
    P[:, UNCERT_IDX] *= np.exp(theta)
    return P


def _system(P):
    N = P.shape[0]
    g = lambda name: P[:, IDX[name]]
    Cinv = 1.0 / P[:, [IDX["C_W"], IDX["C_T"], IDX["C_Y"], IDX["C_M"]]]
    K = np.zeros((N, 4, 4))
    for i, j, name in [(W, T, "G_WT"), (T, Y, "G_TY"), (T, M, "G_TM")]:
        gg = g(name)
        K[:, i, i] += gg
        K[:, j, j] += gg
        K[:, i, j] -= gg
        K[:, j, i] -= gg
    for i, name in [(W, "G_WC"), (Y, "G_YC"), (Y, "G_YA")]:
        K[:, i, i] += g(name)
    A = -Cinv[:, :, None] * K
    Braw = np.zeros((N, 4, 2))           # columns: coolant, ambient
    Braw[:, W, 0] = g("G_WC")
    Braw[:, Y, 0] = g("G_YC")
    Braw[:, Y, 1] = g("G_YA")
    Bb = Cinv[:, :, None] * Braw
    Phi = np.empty((N, 4, 4))
    Gam = np.empty((N, 4, 4))
    Mx = np.zeros((8, 8))
    for n in range(N):
        Mx[:] = 0.0
        Mx[:4, :4] = A[n] * DT
        Mx[:4, 4:] = np.eye(4) * DT
        E = expm(Mx)
        Phi[n] = E[:4, :4]
        Gam[n] = E[:4, 4:]
    return dict(Cinv=Cinv, Bb=Bb, Phi=Phi, Gam=Gam, A=A, Braw=Braw,
                kcu=g("k_cu"), kfe1=g("k_fe1"), kfe2=g("k_fe2"), km=g("k_m"))


def simulate(P, prof, x0, d_sched=None):
    """Vectorised simulation for N parameter sets. Returns X (N, L, 4) in deg C.

    d_sched: None (=1), (L,) common, or (N, L) per-sample derating schedule.
    """
    P = np.atleast_2d(np.asarray(P, float))
    S = _system(P)
    N = P.shape[0]
    i2 = prof["i2"]
    nk = prof["n"] / 1000.0
    Tc = prof["coolant"]
    Ta = prof["ambient"]
    L = len(i2)
    d = np.ones((N, L)) if d_sched is None else np.broadcast_to(np.asarray(d_sched, float), (N, L))
    d2 = d * d
    x = np.broadcast_to(np.asarray(x0, float), (N, 4)).copy()
    X = np.empty((N, L, 4))
    X[:, 0] = x
    Cinv, Bb, Phi, Gam = S["Cinv"], S["Bb"], S["Phi"], S["Gam"]
    kcu, kfe1, kfe2, km = S["kcu"], S["kfe1"], S["kfe2"], S["km"]
    for k in range(L - 1):
        I2 = 0.5 * (i2[k] * d2[:, k] + i2[k + 1] * d2[:, k + 1])
        nm = 0.5 * (nk[k] + nk[k + 1])
        um = np.array([0.5 * (Tc[k] + Tc[k + 1]), 0.5 * (Ta[k] + Ta[k + 1])])
        pcu = kcu * I2 * (1.0 + ALPHA_CU * (x[:, W] - T_REF))
        pfe = kfe1 * abs(nm) + kfe2 * nm * nm
        pmg = km * nm * nm
        pv = np.stack([pcu, 0.5 * pfe, 0.5 * pfe, pmg], axis=1)
        f = Cinv * pv + Bb @ um
        x = np.einsum("nij,nj->ni", Phi, x) + np.einsum("nij,nj->ni", Gam, f)
        X[:, k + 1] = x
    return X


def simulate_reference(p, prof, x0, d=1.0, method="LSODA"):
    """Reference continuous integration of the same ODE with scipy solve_ivp (one parameter set)."""
    S = _system(np.asarray(p, float)[None, :])
    A0, Bb0, Cinv0 = S["A"][0], S["Bb"][0], S["Cinv"][0]
    kcu, kfe1, kfe2, km = S["kcu"][0], S["kfe1"][0], S["kfe2"][0], S["km"][0]
    tg = prof["t"]

    def rhs(t, x):
        i2 = np.interp(t, tg, prof["i2"])
        nt = np.interp(t, tg, prof["n"]) / 1000.0
        Tc = np.interp(t, tg, prof["coolant"])
        Ta = np.interp(t, tg, prof["ambient"])
        pcu = kcu * d * d * i2 * (1.0 + ALPHA_CU * (x[W] - T_REF))
        pfe = kfe1 * abs(nt) + kfe2 * nt * nt
        pmg = km * nt * nt
        pv = np.array([pcu, 0.5 * pfe, 0.5 * pfe, pmg])
        return A0 @ x + Bb0 @ np.array([Tc, Ta]) + Cinv0 * pv

    sol = solve_ivp(rhs, (tg[0], tg[-1]), np.asarray(x0, float), method=method,
                    t_eval=tg, rtol=1e-6, atol=1e-6, max_step=DT)
    return sol.y.T


def fit_nominal(train_profiles, sigma_node, max_nfev=40):
    """Nominal calibration: least_squares on log-parameters, free-run residuals on all 4 nodes."""
    def batch_resid(LP):
        P = np.exp(LP)
        out = []
        for prof in train_profiles:
            X = simulate(P, prof, prof["meas"][0])
            out.append(((X - prof["meas"][None]) / sigma_node).reshape(P.shape[0], -1))
        return np.concatenate(out, axis=1)

    def fun(x):
        return batch_resid(x[None])[0]

    def jac(x):
        h = 1e-5
        Xb = np.vstack([x[None], x[None] + h * np.eye(len(x))])
        R = batch_resid(Xb)
        return ((R[1:] - R[0]) / h).T

    res = least_squares(fun, np.log(P_INIT), jac=jac,
                        bounds=(np.log(P_LO), np.log(P_HI)), method="trf", max_nfev=max_nfev)
    return np.exp(res.x), res
