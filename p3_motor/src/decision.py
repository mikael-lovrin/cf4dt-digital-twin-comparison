"""Decision layer: when to start derating and by how much (optimal derating policy).

Policy (tau, d): at t_c + tau (minutes after the calibration instant) the current scale is
multiplied by d (torque/copper loss reduced) until the end of the planning horizon H.
Per posterior sample i, over the horizon [t_c, t_c + H]:
    J_i = C_ACT * 1[d<1] + C_THR * (1-d) * (H - tau) + C_FAIL * 1[pm > limit in horizon]
Expected cost per unit time = E[J] / H  (cost units per minute). Minimised over the grid.

Forecast: the simulation is initialised at the decision instant t_c with the stator temperatures
OBSERVED at t_c (winding, tooth, yoke) and the model magnet temperature; the future inputs
(current, speed, coolant, ambient) are the measured ones of the session (perfect foresight).

Headline scenario = nominal limit (T_LIMIT). The stress scenario is PRE-REGISTERED (round 3, fixed
before any stressed result was computed): limit unchanged, measured currents x STRESS_CURRENT_SCALE,
coolant + STRESS_COOLANT_K, applied only over the decision window. It does not use the posterior to
set any threshold.

Informativeness (pre-registered): an action is worth taking only if the exceedance reduction exceeds
the break-even dP = (C_ACT + C_THR*(1-d)*(H-tau)) / C_FAIL, computed from the constants below.
Cost constants are illustrative (no economic data was provided).
"""
import numpy as np
import physics_model as pm
from data_loader import DT

T_LIMIT = 100.0          # deg C pm limit (documented choice; data are in deg C, pm max 113.6)
H_MIN = 30.0             # planning horizon (minutes)
H_STEPS = int(round(H_MIN * 60.0 / DT))   # horizon in 5 s steps (360)
D_LEVELS = np.round(np.arange(0.5, 1.0 + 1e-9, 0.05), 2)   # derating factors
TAU_MIN = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0])      # action delays (min)
C_FAIL, C_THR, C_ACT = 1000.0, 20.0, 5.0

# pre-registered stress (round 3)
STRESS_CURRENT_SCALE = 1.2     # measured i_d, i_q x1.2 (copper loss x1.44) inside the window
STRESS_COOLANT_K = 10.0        # coolant temperature +10 K inside the window


def tail(prof, a, b):
    """Input arrays of a profile restricted to steps [a, b)."""
    return {k: prof[k][a:b] for k in ["i2", "n", "coolant", "ambient"]}


def stress_profile(prof, t_c):
    """Pre-registered stress: applied only over the decision window [t_c, t_c + H_STEPS]."""
    a, b = t_c, t_c + H_STEPS + 1
    sp = dict(prof)
    sp["i2"] = prof["i2"].copy()
    sp["i2"][a:b] *= STRESS_CURRENT_SCALE ** 2
    sp["coolant"] = prof["coolant"].copy()
    sp["coolant"][a:b] += STRESS_COOLANT_K
    return sp


def init_state(P, prof, x0, t_c):
    """Prefix trajectory before t_c (N, t_c, 4) and the state at t_c: stator nodes = observed
    temperatures at t_c, magnet node = model value (magnet temperature is not measured)."""
    Xh = pm.simulate(P, tail(prof, 0, t_c + 1), x0)
    x_tc = Xh[:, -1].copy()
    x_tc[:, :3] = prof["meas"][t_c, :3]
    return Xh[:, :t_c], x_tc


def forecast(P, prof, t_c, L, x_tc, dsched=None):
    """Forecast from t_c to L (exclusive) with an optional full-length derating schedule (L,)."""
    ds = None if dsched is None else np.asarray(dsched, float)[t_c:L]
    return pm.simulate(P, tail(prof, t_c, L), x_tc, ds)           # (N, L - t_c, 4)


def breakeven_dP(d, tau_min):
    """Exceedance-probability reduction needed for a derating policy to pay (pre-registered rule)."""
    return (C_ACT + C_THR * (1.0 - d) * (H_MIN - tau_min)) / C_FAIL


def policy_table(P, prof, x0, t_c, limit=T_LIMIT):
    """Exceedance probability (over parameter samples) for every policy; cost-independent.
    Row 0 is no action (d = 1)."""
    L = t_c + H_STEPS + 1
    if L > len(prof["i2"]):
        raise ValueError("profile too short for the decision horizon at this stage")
    _, x_tc = init_state(P, prof, x0, t_c)

    def p_exc(dsched):
        Xf = forecast(P, prof, t_c, L, x_tc, dsched)
        return float(np.any(Xf[:, :, pm.M] > limit, axis=1).mean())

    rows = [dict(tau_min=np.nan, d=1.0, act=0, lost_min=0.0, p_exc=p_exc(None), limit=float(limit))]
    for tm in TAU_MIN:
        ts = t_c + int(round(tm * 60.0 / DT))
        for d in D_LEVELS[D_LEVELS < 1.0 - 1e-9]:
            ds = np.ones(L)
            ds[ts:] = d
            rows.append(dict(tau_min=float(tm), d=float(d), act=1,
                             lost_min=float((1.0 - d) * (H_MIN - tm)), p_exc=p_exc(ds),
                             limit=float(limit)))
    return rows


def cost_rates(rows, c_fail=C_FAIL, c_thr=C_THR, c_act=C_ACT):
    return np.array([(c_act * r["act"] + c_thr * r["lost_min"] + c_fail * r["p_exc"]) / H_MIN
                     for r in rows])


def optimum(rows, **kw):
    c = cost_rates(rows, **kw)
    i = int(np.argmin(c))
    return rows[i], float(c[i]), c


def find_row(rows, tau_min, d):
    return next(r for r in rows if r["act"] == 1 and r["tau_min"] == tau_min and abs(r["d"] - d) < 1e-9)
