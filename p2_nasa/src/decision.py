"""Decision layer: replacement timing and derating under elevated wear risk.

For a replacement at time t after the cutoff, with remaining life T (posterior samples), a cycle
ends at min(T, t). A planned replacement costs c_p; a failure costs c_f (failure penalty).
  P_f(t)       = P(T <= t)
  cycle cost   = c_p * (1 - P_f(t)) + c_f * P_f(t)
  cycle length = E[min(T, t)]                      (time, the renewal-reward cycle length)
Derating by factor d (load scaled after the cutoff) lengthens life to T_d (obtained by re-simulating
the ODE, so thermal and Arrhenius effects are included) and reduces the output rate by d.
Objective: expected cost per unit time (age-replacement form)
  J(t, d) = cycle cost / (d * E[min(T_d, t)])
Derating is only considered when the instantaneous failure risk exceeds risk_thresh.

Round 3 additions:
  - bootstrap_gain: 1000 bootstrap resamples of the remaining-life samples (independently per derate level)
    give a 95% interval for the derating gain 1 - J*(with derating)/J*(no derating). Derating is kept only
    if the interval's lower bound is above 0.
  - post_stats: P(T <= t) and remaining-life quantiles at a given replacement time.
  - The recommendation rule is applied in main.py (not here): a replacement time is recommended only if it
    is no later than the lower bound of the observed VB crossing bracket after the cutoff.
"""
import numpy as np


def expected_cost_per_time(rul, d, c_p, c_f, t_grid):
    """Vectorised over the replacement grid via sorted samples (equal to the direct form
    mean(rul <= t) and mean(min(rul, t)); verified in main.py)."""
    T = np.sort(np.asarray(rul, float))
    n = T.size
    k = np.searchsorted(T, t_grid, side="right")          # number of samples with T <= t
    csum = np.concatenate([[0.0], np.cumsum(T)])
    Pf = k / n
    Emin = (csum[k] + t_grid * (n - k)) / n               # E[min(T, t)]
    return (c_p * (1.0 - Pf) + c_f * Pf) / (d * np.maximum(Emin, 1e-9))


def expected_cost_direct(rul, d, c_p, c_f, t_grid):
    """Reference implementation (direct averages) used for the equality check."""
    Pf = (rul[:, None] <= t_grid[None, :]).mean(axis=0)
    Emin = np.minimum(rul[:, None], t_grid[None, :]).mean(axis=0)
    return (c_p * (1.0 - Pf) + c_f * Pf) / (d * np.maximum(Emin, 1e-9))


def post_stats(rul, t):
    """P(T <= t) and remaining-life quantiles of the samples."""
    rul = np.asarray(rul, float)
    return dict(P_T_le_t=float(np.mean(rul <= t)),
                quantiles={str(q): float(np.percentile(rul, q)) for q in (5, 25, 50, 75, 95)})


def _first_min(j):
    """Index of the first grid point attaining the minimum (ties on a plateau resolve to its start)."""
    i = int(np.argmin(j))
    return int(np.argmax(j <= j[i] * (1.0 + 1e-9)))


def _is_interior(j, i):
    return bool(0 < i < j.size - 1 and j[-1] > j[i] * (1.0 + 1e-6))


def optimise(rul_by_d, c_p, c_f, risk_now, risk_thresh, t_max, n_grid=400):
    """rul_by_d: dict {derate factor d: array of remaining-life samples under that d}; must include 1.0.
    t_max: last replacement time evaluated (censoring horizon after the cutoff)."""
    t_grid = np.linspace(0.5, float(t_max), n_grid)
    considered = sorted(rul_by_d.keys(), reverse=True) if risk_now > risk_thresh else [1.0]
    J, idx = {}, {}
    best = None
    for dl in considered:
        j = expected_cost_per_time(rul_by_d[dl], dl, c_p, c_f, t_grid)
        i = _first_min(j)
        J[float(dl)] = j
        idx[float(dl)] = i
        if best is None or j[i] < best[2]:
            best = (float(t_grid[i]), float(dl), float(j[i]))
    d_star = best[1]
    i_star = idx[d_star]
    j1 = J[1.0] if 1.0 in J else expected_cost_per_time(rul_by_d[1.0], 1.0, c_p, c_f, t_grid)
    i1 = _first_min(j1)
    return dict(t_star=best[0], d_star=d_star, J_star=best[2],
                t_star_no_derate=float(t_grid[i1]), J_no_derate=float(j1[i1]),
                derating_considered=len(considered) > 1,
                interior=_is_interior(J[d_star], i_star) if d_star in J else False,
                interior_no_derate=_is_interior(j1, i1),
                t_max=float(t_max), t_grid=t_grid, J=J)


def bootstrap_gain(rul_by_d, c_p, c_f, risk_now, risk_thresh, t_grid, n_boot=1000, seed=11):
    """Bootstrap distribution of the derating gain 1 - min_d J_d / J_1 over the replacement grid.

    Each bootstrap resample draws the remaining-life samples with replacement, independently for each
    derate level, and re-optimises t for each level. Returns the array of gains (n_boot,).
    """
    rng = np.random.default_rng(seed)
    considered = sorted(rul_by_d.keys(), reverse=True) if risk_now > risk_thresh else [1.0]
    gains = np.empty(n_boot)
    for b in range(n_boot):
        jmin = {}
        for dl in considered:
            R = rul_by_d[dl]
            Rb = R[rng.integers(0, R.size, R.size)]
            jmin[dl] = float(np.min(expected_cost_per_time(Rb, dl, c_p, c_f, t_grid)))
        gains[b] = 1.0 - min(jmin.values()) / jmin[1.0]
    return gains
