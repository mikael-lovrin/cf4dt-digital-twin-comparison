"""Decision layer: choose the speed setpoint (derating) and whether to rebalance, by expected cost per hour.

There is no time/wear axis in this dataset, so the brief's "action time" optimisation is
replaced by a speed/rebalance decision: the cost rate of operating at setpoint r with action a is

    J(r, a) = V_full * (1 - r / r_max)                       throughput loss from derating
            + C_fail * h_fail * P_exceed(r, a)               expected failure cost rate
            + a * C_rb / H_amort                             rebalance cost amortised over H_amort hours

with a = 1 for rebalance (residual unbalance = resid_frac * U) and a = 0 otherwise.
All cost values are ASSUMPTIONS (not from the data) and are varied in a sensitivity table.
"""
import numpy as np

COSTS = {
    "V_full": 10.0,        # cost units / h of lost throughput at r_max (scaled by 1 - r/r_max)
    "C_fail": 500.0,       # cost units per failure event
    "h_fail": 0.1,         # failure rate per operating hour while the limit is exceeded
    "C_rb": 200.0,         # cost units per rebalance
    "H_amort": 200.0,      # operating hours over which the rebalance cost is amortised
    "resid_frac": 0.1,     # residual unbalance after rebalancing, as a fraction of U
    "p_safe": 0.05,        # "safe" = P(exceed) <= p_safe without action
    "p_elev": 0.20,        # elevated risk: derate if P(exceed) at r_max exceeds this
}


def cost_rates(rpm_grid, pf_keep, pf_reb, costs=COSTS, rpm_max=None):
    rpm_max = float(np.max(rpm_grid)) if rpm_max is None else rpm_max
    thr = costs["V_full"] * (1.0 - np.asarray(rpm_grid, float) / rpm_max)
    J_keep = thr + costs["C_fail"] * costs["h_fail"] * np.asarray(pf_keep)
    J_reb = thr + costs["C_fail"] * costs["h_fail"] * np.asarray(pf_reb) + costs["C_rb"] / costs["H_amort"]
    return J_keep, J_reb


def optimise(rpm_grid, pf_keep, pf_reb, costs=COSTS):
    rpm_grid = np.asarray(rpm_grid, float)
    rpm_max = rpm_grid.max()
    J_keep, J_reb = cost_rates(rpm_grid, pf_keep, pf_reb, costs, rpm_max)
    J = np.vstack([J_keep, J_reb])
    a, i = np.unravel_index(np.argmin(J), J.shape)
    safe = np.where(np.asarray(pf_keep) <= costs["p_safe"])[0]
    max_safe = float(rpm_grid[safe].max()) if len(safe) else float("nan")
    r_star = float(rpm_grid[i])
    return {
        "rpm_star": r_star,
        "action": "rebalance" if a == 1 else "continue",
        "J_star": float(J[a, i]),
        "derate_factor": float(r_star / rpm_max),
        "elevated_risk_at_rmax": bool(pf_keep[-1] > costs["p_elev"]),
        "max_safe_rpm_no_action": max_safe,
        "J_keep_at_argmin": J_keep.tolist(),
        "J_reb_at_argmin": J_reb.tolist(),
    }
