"""Replacement-time optimisation (expected cost per unit time) and derating factor.

Renewal-reward cost rate for replacement interval T (replace at T or at failure, whichever first):
    C(T) = [c_p * P(RUL > T) + c_f * P(RUL <= T)] / E[min(RUL, T)]
c_p = premature replacement cost, c_f = failure penalty (incl. unplanned stop).
Both are ASSUMED normalised values (c_p = 1, c_f = 5); they are not from the data.
Units of T: PTU (pseudo-time units). Cost unit: c_p per PTU.

Round-3 rule (fixed before the run, see interior_check): an optimum is reported as INTERIOR only if the
argmin is not at either grid end AND the cost rises by at least TOL_REL (relative) somewhere after the argmin.
Otherwise the result is 'no interior optimum' and no replacement time is recommended.
"""
import numpy as np

TOL_REL = 0.01


def cost_rate_curve(rul, T_grid, c_p, c_f):
    rul = np.asarray(rul, float)
    P_fail = np.array([np.mean(rul <= T) for T in T_grid])
    E_min = np.array([np.mean(np.minimum(rul, T)) for T in T_grid])
    cost = (c_p * (1.0 - P_fail) + c_f * P_fail) / np.maximum(E_min, 1e-9)
    return cost, P_fail


def interior_check(cost, T_grid, tol=TOL_REL):
    """Interior iff argmin is strictly inside the grid and max(cost after argmin) >= (1 + tol) * min(cost)."""
    cost = np.asarray(cost, float)
    i = int(np.argmin(cost))
    n = len(cost)
    if i >= n - 1:
        rise = None
    else:
        rise = float(np.max(cost[i + 1:]) / cost[i] - 1.0)
    interior = bool(0 < i < n - 1 and rise is not None and rise >= tol)
    return {"i_argmin": i, "T_argmin": float(T_grid[i]), "interior": interior,
            "rise_after_argmin_rel": rise}


def optimal_replacement(rul, c_p=1.0, c_f=5.0, T_grid=None):
    if T_grid is None:
        T_grid = np.linspace(5.0, 2000.0, 800)
    cost, P = cost_rate_curve(rul, T_grid, c_p, c_f)
    chk = interior_check(cost, T_grid)
    i = chk["i_argmin"]
    return {"T_star": chk["T_argmin"] if chk["interior"] else None,   # None = no interior optimum
            "interior": chk["interior"], "T_argmin": chk["T_argmin"],
            "rise_after_argmin_rel": chk["rise_after_argmin_rel"],
            "cost_rate": float(cost[i]), "P_fail_at_T": float(P[i]),
            "T_grid": T_grid, "cost": cost}


def derating_factor(risk, gamma=0.5, d_min=0.7):
    """Multiplicative feed/speed derating: d = max(d_min, 1 - gamma * risk), risk = P(fail within T*)."""
    return float(max(d_min, 1.0 - gamma * risk))
