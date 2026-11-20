"""p1_cnc, round 3 (final): hybrid physics + data-driven digital twin for CNC flank wear.

Run from p1_cnc\\ :  ..\\.venv\\Scripts\\python.exe -I main.py

Every number in results/summary.md is read back from results/metrics.json (build_summary).
Round-1 and round-2 reference values are read from results/metrics_round1.json and results/metrics_round2.json.

Pre-registered pass/fail criteria (written before the round-3 run; not tuned to its output):
  C1 coverage nominal:  |empirical - nominal| <= 0.05 for nominal 0.5, 0.8, 0.95 on held-out rows.
  C2 MCMC converged:    min steps/tau >= 50 over parameters and stages.
  C3 interior optimum:  decision.interior_check: argmin strictly inside [5, 2000] PTU AND the cost rises by
                        >= 1 % (relative) somewhere after the argmin. Otherwise 'no interior optimum'.
  C4 stressed action:   interior AND 0.05 < P(fail by T*) < 0.95 (75 % life, c_f = 20).
  C5 hybrid vs linear:  leave-one-segment-out jackknife 95 % CI (t with k-1 df, k = held-out segments) of
                        RMSE(hybrid) - RMSE(linear). 'better' if the CI is below 0, 'worse' if above 0.
  C6 band scale:        scale = 95 % quantile of |standardised forecast residual| / 1.96, fitted on TRAIN rows
                        beyond the 75 % stage (never used in calibration, never held-out).
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, t as t_dist
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, KFold, cross_val_score

from data_loader import (TARGET, default_csv_path, feature_uniformity, inspect_dataset, load_raw,
                         reconstruct_pseudo_trajectory)
from physics_model import fit_baseline, make_constants, simulate, VC_NOM
from surrogate_model import (GPSurrogate, U_NAMES, VB_PRIOR_MAX, design_bounds, lhs_design,
                             prior_support_mask, run_ode_batch, u_to_theta)
from uq_calibration import (CAL_NAMES, VB_CRIT, cal_bounds, conditioned_rul, make_logp, posterior_grid,
                            run_emcee, sobol_analysis)
from decision import TOL_REL, cost_rate_curve, derating_factor, interior_check, optimal_replacement

T_START = time.time()
SEED = 42
RES = os.path.join(HERE, "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

C_P, C_F = 1.0, 5.0                       # assumed normalised costs
CF_SWEEP = [2.0, 5.0, 10.0, 20.0]         # c_f sensitivity sweep (c_p fixed), 50 % life
GAMMA, D_MIN = 0.5, 0.7                   # assumed derating law
H_GRID = np.linspace(0.0, 2000.0, 101)    # round 3: horizon = record length (2000 PTU), 20 PTU step
STAGES = [("25%", 0.25), ("50%", 0.50), ("75%", 0.75)]
INSPECT_EVERY = 25                        # intermittent VB inspections (training rows only)
N_WALKERS, N_BURN, N_STEPS = 32, 1000, 5000
N_POST = 400
SOBOL_N = 4096
T_DECISION = np.linspace(5.0, 2000.0, 800)
RK_CHECK_N = 30                           # designs used for the RK4-vs-RK45 check and timing
N_CAND_TR, N_TR = 600, 400                # prior-predictive candidates -> accepted designs used for fitting
N_CAND_TE, N_TE = 200, 100
COVER_TOL = 0.05                          # C1
CONV_SOT = 50.0                           # C2
# round-1 settings: 32 walkers, 600 burn-in + 1000 steps (summary_round1.md section 7); horizon 3000 PTU
R1_SETTINGS = {"nwalkers": 32, "nburn": 600, "nsteps": 1000}
R1_HORIZON = 3000.0


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def _rm(e):
    return float(np.sqrt(np.mean(np.asarray(e, float) ** 2)))


def loso_diff(e_a, e_b, seg):
    """RMSE(e_a) - RMSE(e_b) on the held-out rows with a leave-one-segment-out jackknife CI.

    Units are the k held-out segments. Leave-one-out estimates drop one segment and use the others.
    SE = sqrt((k-1)/k * sum_s (theta_(-s) - mean)^2); CI = point +/- t_{0.975, k-1} * SE.
    """
    e_a, e_b, seg = np.asarray(e_a, float), np.asarray(e_b, float), np.asarray(seg)
    segs = np.unique(seg)
    k = len(segs)
    point = _rm(e_a) - _rm(e_b)
    loo = np.array([_rm(e_a[seg != s]) - _rm(e_b[seg != s]) for s in segs])
    se = float(np.sqrt((k - 1) / k * np.sum((loo - loo.mean()) ** 2)))
    tc = float(t_dist.ppf(0.975, k - 1))
    return {"diff": float(point), "ci95": [float(point - tc * se), float(point + tc * se)],
            "k_segments": int(k), "se_jackknife": se, "t_crit": tc,
            "leave_one_out_diffs": [float(x) for x in loo],
            "per_segment_diff": {int(s): float(_rm(e_a[seg == s]) - _rm(e_b[seg == s])) for s in segs}}


def _cmp_word(ci):
    if ci[1] < 0:
        return "first model better (CI below 0)"
    if ci[0] > 0:
        return "first model worse (CI above 0)"
    return "no distinguishable difference (CI includes 0)"


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    raise TypeError(f"not serialisable: {type(o)}")


def load_json(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def load_round1():
    r = load_json("metrics_round1.json")
    if r is None:
        return None
    return {
        "heldout_nrmse": r["heldout_nrmse"],
        "surrogate_speedup_x": r["surrogate"]["speedup_x"],
        "surrogate_rmse": r["surrogate"]["rmse"],
        "surrogate_horizon_PTU": R1_HORIZON,
        "mcmc": R1_SETTINGS,
        "mean_autocorr_time": r["calibration"]["mean_autocorr_time"],
        "acceptance_fraction": r["calibration"]["acceptance_fraction"],
        "coverage95_heldout": r["calibration"]["coverage95_heldout"],
        "c_w_ratio": r["calibration"]["posterior_to_prior_std_ratio"]["c_w"],
        "T_star_PTU": r["decision"]["optimal_value"],
        "risk_P_fail": r["risk"]["value"],
        "runtime_sec": r["runtime_sec"],
    }


def load_round2():
    r = load_json("metrics_round2.json")
    if r is None:
        return None
    return {
        "heldout_nrmse": r["heldout_nrmse"],
        "surrogate_rmse": r["surrogate"]["rmse"],
        "coverage": r["interval_calibration"]["coverage"],
        "coverage95": r["interval_calibration"]["coverage"]["0.95"],
        "min_steps_over_tau": r["mcmc"]["min_steps_over_tau"],
        "ess_min": r["mcmc"]["ess_min"],
        "T_star_PTU": r["decision"]["optimal_value"],
        "cost_sweep_range": r["decision_checks"]["cost_sweep_range"],
        "stressed_nontrivial": r["decision_checks"]["stressed_scenario_action_nontrivial"],
        "runtime_sec": r["runtime_sec"],
    }


def main():
    # ------------------------------------------------------------------ 1. data
    csv = default_csv_path(HERE)
    raw = load_raw(csv)
    info = inspect_dataset(raw)
    uni = feature_uniformity(raw)
    ds = reconstruct_pseudo_trajectory(raw)
    N = ds["n"]
    t_all = ds["t"]                       # pseudo-time index = rank of VB_mm (target-derived)
    vb, F, X = ds["vb"], ds["F"], ds["X"]
    ho, tr = ds["holdout"], ~ds["holdout"]
    tr_idx = np.where(tr)[0]
    n_ho, n_tr = int(ho.sum()), int(tr.sum())
    print(f"[data] rows={N} train={n_tr} heldout={n_ho} VB range=[{vb.min():.4f},{vb.max():.4f}]")

    # feature-only diagnostics (no time input): how much do the features explain VB?
    feat_table = []
    for j, name in enumerate(ds["feature_names"]):
        feat_table.append({
            "feature": name,
            "pearson": float(pearsonr(X[:, j], vb)[0]),
            "spearman": float(spearmanr(X[:, j], vb)[0]),
            "ks_p_uniform": float(uni[name]) if name in uni else None,
        })
    max_abs_corr = max(abs(r["pearson"]) for r in feat_table)
    rf_probe = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=SEED, n_jobs=-1)
    cv_random = float(cross_val_score(rf_probe, X, vb, cv=KFold(5, shuffle=True, random_state=SEED),
                                      scoring="r2").mean())
    cv_blocked = float(cross_val_score(rf_probe, X, vb, cv=GroupKFold(5), groups=ds["segment"],
                                       scoring="r2").mean())
    print(f"[diag] max|pearson|={max_abs_corr:.3f} RF CV R2 random={cv_random:.3f} blocked={cv_blocked:.3f}")

    # standardisation with TRAINING statistics only; shared time input for ALL held-out models
    mu_x, sd_x = X[tr].mean(axis=0), X[tr].std(axis=0) + 1e-12
    Xs = (X - mu_x) / sd_x
    t_mu, t_sd = t_all[tr].mean(), t_all[tr].std()
    tz = (t_all - t_mu) / t_sd
    Z = np.column_stack([Xs, tz])

    # ------------------------------------------------------------------ 2. physics (trend fit)
    F_mean, F_sd = float(F[tr].mean()), float(F[tr].std())
    eta = make_constants(F_mean)
    ls = fit_baseline(t_all, tr_idx, vb, F, eta)
    VB0 = ls["VB0"]
    phys_all = simulate(ls["theta"], t_all, F, eta, VB0=VB0)
    print(f"[physics] LS fit success={ls['success']} nfev={ls['nfev']} theta={np.round(ls['theta'], 5)} "
          f"VB0={VB0:.4f} train_rmse={ls['train_rmse']:.4f}")

    # ------------------------------------------------------------------ 3. held-out comparison
    std_ho = float(vb[ho].std())
    y_ho = vb[ho]
    ph_ho = phys_all[ho]
    seg_ho = ds["segment"][ho]
    lin = LinearRegression().fit(t_all[tr, None], vb[tr])
    lin_ho = lin.predict(t_all[ho, None])
    rf_kw = dict(n_estimators=300, min_samples_leaf=5, random_state=SEED, n_jobs=-1)
    rf_dd = RandomForestRegressor(**rf_kw).fit(Z[tr], vb[tr])
    dd_ho = rf_dd.predict(Z[ho])
    rf_res = RandomForestRegressor(**rf_kw).fit(Z[tr], vb[tr] - phys_all[tr])
    hy_ho = ph_ho + rf_res.predict(Z[ho])
    rf_feat = RandomForestRegressor(**rf_kw).fit(Xs[tr], vb[tr])
    feat_ho = rf_feat.predict(Xs[ho])

    rmse_p, rmse_d, rmse_h = rmse(y_ho, ph_ho), rmse(y_ho, dd_ho), rmse(y_ho, hy_ho)
    rmse_lin, rmse_feat = rmse(y_ho, lin_ho), rmse(y_ho, feat_ho)
    r2_d = float(1 - np.sum((y_ho - dd_ho) ** 2) / np.sum((y_ho - y_ho.mean()) ** 2))
    loso_hp = loso_diff(y_ho - hy_ho, y_ho - ph_ho, seg_ho)     # hybrid - physics
    loso_hl = loso_diff(y_ho - hy_ho, y_ho - lin_ho, seg_ho)    # hybrid - linear trend
    loso_dp = loso_diff(y_ho - dd_ho, y_ho - ph_ho, seg_ho)     # data-driven - physics
    loso_pl = loso_diff(y_ho - ph_ho, y_ho - lin_ho, seg_ho)    # physics - linear trend
    print(f"[heldout] RMSE physics={rmse_p:.4f} data(RF+t)={rmse_d:.4f} hybrid={rmse_h:.4f} "
          f"linear(t)={rmse_lin:.4f} RF(features only)={rmse_feat:.4f} (std={std_ho:.4f})")
    print(f"[heldout] LOSO hybrid-linear={loso_hl['diff']:+.5f} CI={loso_hl['ci95']} | "
          f"hybrid-physics={loso_hp['diff']:+.5f} CI={loso_hp['ci95']}")

    # ------------------------------------------------------------------ 4. GP surrogate of the ODE
    center = np.array([ls["u"][0], ls["u"][1], ls["u"][2], ls["u"][3], VC_NOM])
    lo, hi = design_bounds(center)
    # prior-predictive support (data-independent): keep designs with simulated VB(2000) <= VB_PRIOR_MAX
    U_c = lhs_design(lo, hi, N_CAND_TR, seed=1)
    Y_c = run_ode_batch(U_c, H_GRID, F, eta, VB0)
    sup_c = prior_support_mask(Y_c)
    U_tr, Y_tr = U_c[sup_c][:N_TR], Y_c[sup_c][:N_TR]
    U_e = lhs_design(lo, hi, N_CAND_TE, seed=2)
    Y_e = run_ode_batch(U_e, H_GRID, F, eta, VB0)
    sup_e = prior_support_mask(Y_e)
    U_te, Y_te = U_e[sup_e][:N_TE], Y_e[sup_e][:N_TE]
    surr_filter = {
        "prior_VB_max_mm": float(VB_PRIOR_MAX), "horizon_PTU": float(H_GRID[-1]),
        "train_candidates": int(N_CAND_TR), "train_accepted": int(sup_c.sum()), "train_used": int(len(U_tr)),
        "train_rejection_rate": float(1.0 - sup_c.mean()),
        "test_candidates": int(N_CAND_TE), "test_accepted": int(sup_e.sum()), "test_used": int(len(U_te)),
        "test_rejection_rate": float(1.0 - sup_e.mean()),
    }
    print(f"[filter] train accepted {surr_filter['train_accepted']}/{N_CAND_TR} "
          f"(rejection {surr_filter['train_rejection_rate']:.3f}); test {surr_filter['test_accepted']}/{N_CAND_TE}")
    gp = GPSurrogate(seed=SEED).fit(U_tr, Y_tr)
    m_te, s_te = gp.predict(U_te)
    yv, mv = Y_te.ravel(), m_te.ravel()
    sur_r2 = float(1 - np.sum((yv - mv) ** 2) / np.sum((yv - yv.mean()) ** 2))
    sur_rmse = rmse(yv, mv)
    cov2 = float(np.mean(np.abs(Y_te - m_te) <= 2 * s_te))
    end_r2 = float(1 - np.sum((Y_te[:, -1] - m_te[:, -1]) ** 2) / np.sum((Y_te[:, -1] - Y_te[:, -1].mean()) ** 2))
    per_design = np.sqrt(np.mean((Y_te - m_te) ** 2, axis=1))
    sur_err = {"per_design_rmse_median": float(np.median(per_design)),
               "per_design_rmse_p95": float(np.percentile(per_design, 95)),
               "per_design_rmse_max": float(per_design.max()),
               "abs_err_median": float(np.median(np.abs(yv - mv))),
               "abs_err_max": float(np.max(np.abs(yv - mv)))}

    # RK4 (training targets) vs RK45 (reference ODE) on the test designs
    th_chk = u_to_theta(U_te[:RK_CHECK_N])
    t1 = time.time()
    Y45 = np.array([simulate(th, H_GRID, F, eta, VB0=VB0) for th in th_chk])
    t_rk45 = (time.time() - t1) / len(th_chk)
    ok45 = np.all(np.isfinite(Y45), axis=1)
    dif = np.abs(Y_te[:RK_CHECK_N][ok45] - Y45[ok45])
    rk_check = {
        "n_designs": int(ok45.sum()), "n_rk45_failures": int((~ok45).sum()),
        "max_abs_mm": float(dif.max()), "rmse_mm": float(np.sqrt(np.mean(dif ** 2))),
        "max_abs_over_VB_crit": float(dif.max() / VB_CRIT),
        "rk45_rtol": 1e-6, "rk4_dt_PTU": 1.0,
    }
    # speed: per-run RK45 vs per-sample GP; RK4 batch timing (round-1 reference) reported too
    t1 = time.time()
    run_ode_batch(U_te[:RK_CHECK_N], H_GRID, F, eta, VB0)
    t_rk4 = (time.time() - t1) / RK_CHECK_N
    U_big = lhs_design(lo, hi, 2000, seed=3)
    t1 = time.time()
    gp.predict(U_big)
    t_gp = (time.time() - t1) / len(U_big)
    speedup_rk45 = t_rk45 / max(t_gp, 1e-12)
    speedup_rk4 = t_rk4 / max(t_gp, 1e-12)
    print(f"[surrogate] n_train={len(U_tr)} R2(flat)={sur_r2:.4f} R2(VB_end)={end_r2:.4f} rmse={sur_rmse:.4f} "
          f"median per-design RMSE={sur_err['per_design_rmse_median']:.4f} p95={sur_err['per_design_rmse_p95']:.4f} "
          f"max={sur_err['per_design_rmse_max']:.4f} cov2sig={cov2:.3f} "
          f"RK4-RK45 max={rk_check['max_abs_mm']:.4f} mm nugget={gp.noise_level:.2e} at_bound={gp.noise_at_bound}")
    print(f"[surrogate] speed-up vs RK4 batch={speedup_rk4:.0f}x (round-1 reference) vs RK45={speedup_rk45:.0f}x")

    # ------------------------------------------------------------------ 5. Sobol (truncated prior)
    np.random.seed(SEED)
    sob = sobol_analysis(gp, lo, hi, H_GRID, N=SOBOL_N, vb_max=VB_PRIOR_MAX)
    print(f"[sobol] n_eval={sob['n_eval']} base kept={sob['n_base_kept']}/{sob['n_base']} "
          f"censored={sob['RUL_censored_frac']:.3f} RUL S1/ST: " +
          ", ".join(f"{k}={sob['RUL'][k]['S1']:.2f}/{sob['RUL'][k]['ST']:.2f}" for k in U_NAMES))

    # ------------------------------------------------------------------ 6. calibration, RUL, decision
    lo_c, hi_c = cal_bounds(lo, hi)
    prior_sd = (hi_c - lo_c) / np.sqrt(12.0)
    true_cross = int(np.argmax(vb >= VB_CRIT))   # circular 'empirical' crossing (first sorted index, VB >= 0.30)
    stage_out, chains, rul_by_stage = {}, {}, {}
    cov_levels, interval = {}, {}
    cov_info = None
    rng = np.random.default_rng(SEED)
    for name, frac in STAGES:
        t_s = frac * N
        anchor = int(tr_idx[tr_idx <= t_s][-1])          # last TRAINING row at or before the stage time
        t_anchor, y_anchor = float(t_all[anchor]), float(vb[anchor])
        vb_idx = tr_idx[(tr_idx <= t_s) & (tr_idx % INSPECT_EVERY == 0)]
        obs = {"t_vb": t_all[vb_idx], "y_vb": vb[vb_idx]}
        logp = make_logp(gp, H_GRID, obs, lo_c, hi_c, VC_NOM, VB_PRIOR_MAX)
        chain, tau, acc = run_emcee(logp, lo_c, hi_c, nwalkers=N_WALKERS, nburn=N_BURN, nsteps=N_STEPS,
                                    seed=SEED)
        chains[name] = chain
        post_sd = chain.std(axis=0)
        ratio = post_sd / prior_sd
        sot = N_STEPS / tau
        ess = N_WALKERS * N_STEPS / tau

        sub = chain[rng.choice(len(chain), N_POST, replace=False)]
        m, s, sig = posterior_grid(gp, sub, VC_NOM)
        rul, cens = conditioned_rul(H_GRID, m, t_anchor, y_anchor)
        rul_by_stage[name] = rul
        opt = optimal_replacement(rul, C_P, C_F, T_DECISION)
        derate_raw = 1.0 - GAMMA * opt["P_fail_at_T"]
        derate = derating_factor(opt["P_fail_at_T"], GAMMA, D_MIN)
        emp_cross = float(true_cross)
        pred_cross = t_anchor + float(np.median(rul))
        rul_true = emp_cross - t_anchor
        p25, p975 = float(np.percentile(rul, 2.5)), float(np.percentile(rul, 97.5))
        stage_out[name] = {
            "t_stage_PTU": float(t_s), "anchor_index_PTU": anchor, "anchor_VB_obs_mm": y_anchor,
            "n_vb_obs": int(len(vb_idx)),
            "acceptance_fraction": float(acc),
            "autocorr_time": {n: float(v) for n, v in zip(CAL_NAMES, tau)},
            "steps_over_tau": {n: float(v) for n, v in zip(CAL_NAMES, sot)},
            "ess": {n: float(v) for n, v in zip(CAL_NAMES, ess)},
            "steps_over_tau_min": float(np.min(sot)),
            "ess_min": float(np.min(ess)),
            "posterior_mean": {n: float(v) for n, v in zip(CAL_NAMES, chain.mean(axis=0))},
            "posterior_sd": {n: float(v) for n, v in zip(CAL_NAMES, post_sd)},
            "prior_sd": {n: float(v) for n, v in zip(CAL_NAMES, prior_sd)},
            "post_to_prior_sd": {n: float(v) for n, v in zip(CAL_NAMES, ratio)},
            "rul_pred_median": float(np.median(rul)),
            "rul_pred_95": [p25, p975],
            "pred_crossing_PTU": float(pred_cross),
            "empirical_crossing_PTU_circular": emp_cross,
            "miss_PTU_circular": float(pred_cross - emp_cross),
            "rul_true_PTU_circular": float(rul_true),
            "rul_true_in_95_circular": bool(p25 <= rul_true <= p975),
            "censored_frac": float(cens.mean()),
            "T_star_PTU": opt["T_star"], "T_argmin_PTU": opt["T_argmin"], "T_star_interior": opt["interior"],
            "rise_after_argmin_rel": opt["rise_after_argmin_rel"],
            "expected_cost_rate": opt["cost_rate"],
            "P_fail_within_T_star": opt["P_fail_at_T"],
            "derating_raw": float(derate_raw), "derating_factor": derate,
            "_rul": rul, "_T": opt["T_grid"], "_cost": opt["cost"],
        }
        print(f"[cal {name}] acc={acc:.2f} tau_max={np.max(tau):.1f} steps/tau_min={np.min(sot):.1f} "
              f"RUL pred={np.median(rul):.0f} [{p25:.0f},{p975:.0f}] empirical crossing={emp_cross:.0f} "
              f"miss={pred_cross - emp_cross:+.0f} T*={opt['T_star']} argmin={opt['T_argmin']:.0f} "
              f"interior={opt['interior']}")

        if name == "75%":
            sub2 = chain[rng.choice(len(chain), 1000, replace=False)]
            m2, s2, sig2 = posterior_grid(gp, sub2, VC_NOM)
            # C6: band scale from TRAIN rows beyond the 75% stage (forecasts, not used in calibration)
            fore = tr & (t_all > t_s)
            mu_f = np.array([np.interp(t_all[fore], H_GRID, row) for row in m2])
            sd_f = np.array([np.interp(t_all[fore], H_GRID, row) for row in s2])
            mu_bar = mu_f.mean(axis=0)
            sd_bar = np.sqrt(np.mean(sig2 ** 2) + np.mean(sd_f ** 2, axis=0))
            r_f = (vb[fore] - mu_bar) / sd_bar
            scale_c = float(np.quantile(np.abs(r_f), 0.95) / 1.959964)
            # held-out band: latent GP sd + sigma_obs, times the scale (held-out rows never used)
            th = t_all[ho]
            mu_h = np.array([np.interp(th, H_GRID, row) for row in m2])
            sd_h = np.array([np.interp(th, H_GRID, row) for row in s2])
            sd_tot = scale_c * np.sqrt(sig2[:, None] ** 2 + sd_h ** 2)
            draws = mu_h + rng.standard_normal(mu_h.shape) * sd_tot
            bands = {}
            for lev in (0.5, 0.8, 0.95):
                a = 100.0 * (1.0 - lev) / 2.0
                lo_q, hi_q = np.percentile(draws, [a, 100.0 - a], axis=0)
                cov_levels[f"{lev:g}"] = float(np.mean((y_ho >= lo_q) & (y_ho <= hi_q)))
                bands[lev] = (lo_q, hi_q)
            w95 = float(np.mean(bands[0.95][1] - bands[0.95][0]))
            interval = {"coverage": dict(cov_levels), "mean_width95": w95,
                        "scale_c": scale_c, "scale_fit_rows": int(fore.sum()),
                        "scale_fit_mean_abs_std_residual": float(np.mean(np.abs(r_f))),
                        "median_sigma_obs_post": float(np.median(sig2)),
                        "median_gp_sd_heldout": float(np.median(sd_h)),
                        "median_gp_sd_heldout_over_sigma_obs": float(np.median(sd_h) / np.median(sig2)),
                        "coverage_nominal": bool(all(abs(cov_levels[f"{lv:g}"] - lv) <= COVER_TOL
                                                     for lv in (0.5, 0.8, 0.95)))}
            cov_info = {"lo95": bands[0.95][0], "hi95": bands[0.95][1], "m_med": np.median(m2, axis=0)}
            print(f"[coverage] held-out (75% stage) coverage 50/80/95 = "
                  f"{cov_levels['0.5']:.3f}/{cov_levels['0.8']:.3f}/{cov_levels['0.95']:.3f}; "
                  f"mean width95={w95:.4f} mm; scale c={scale_c:.3f}")

    final, mid, early = stage_out["75%"], stage_out["50%"], stage_out["25%"]
    tau_max = {n: float(max(stage_out[k]["autocorr_time"][n] for k in stage_out)) for n in CAL_NAMES}
    min_steps_over_tau = float(min(stage_out[k]["steps_over_tau_min"] for k in stage_out))
    ess_min = float(min(stage_out[k]["ess_min"] for k in stage_out))
    mean_tau = float(np.mean([v for k in stage_out for v in stage_out[k]["autocorr_time"].values()]))
    mean_acc = float(np.mean([stage_out[k]["acceptance_fraction"] for k in stage_out]))

    # c_f sensitivity sweep for T* (50 % life, conditioned RUL); interior flag by the pre-registered rule
    rul_mid = rul_by_stage["50%"]
    cf_sweep = {}
    for cf in CF_SWEEP:
        cost, P = cost_rate_curve(rul_mid, T_DECISION, C_P, cf)
        chk = interior_check(cost, T_DECISION)
        i = chk["i_argmin"]
        cf_sweep[f"{cf:g}"] = {"c_f": float(cf), "T_argmin_PTU": chk["T_argmin"],
                               "T_star_PTU": chk["T_argmin"] if chk["interior"] else None,
                               "interior": chk["interior"], "rise_after_argmin_rel": chk["rise_after_argmin_rel"],
                               "expected_cost_rate": float(cost[i]), "P_fail_at_argmin": float(P[i])}
    argmins = [v["T_argmin_PTU"] for v in cf_sweep.values()]
    # stressed scenario: 75 % life with the largest failure penalty in the sweep (definition unchanged from round 2)
    cf_s = CF_SWEEP[-1]
    cost_s, P_s = cost_rate_curve(rul_by_stage["75%"], T_DECISION, C_P, cf_s)
    chk_s = interior_check(cost_s, T_DECISION)
    i_s = chk_s["i_argmin"]
    stressed = {"stage": "75%", "c_f": float(cf_s), "T_argmin_PTU": chk_s["T_argmin"],
                "T_star_PTU": chk_s["T_argmin"] if chk_s["interior"] else None,
                "P_fail_within_T_star": float(P_s[i_s]), "interior": chk_s["interior"]}
    stressed["action_nontrivial"] = bool(stressed["interior"] and 0.05 < stressed["P_fail_within_T_star"] < 0.95)
    decision_checks = {
        "optimum_interior": bool(mid["T_star_interior"]),
        "cost_sweep_range": [float(min(argmins)), float(max(argmins))],
        "cost_sweep_interior_count": int(sum(v["interior"] for v in cf_sweep.values())),
        "stressed_scenario_action_nontrivial": stressed["action_nontrivial"],
        "cf_sweep": cf_sweep, "stressed_scenario": stressed,
        "definition": ("interior (C3): argmin of the cost-rate curve strictly inside "
                       f"[{T_DECISION[0]:g}, {T_DECISION[-1]:g}] PTU AND max cost after the argmin >= "
                       f"(1 + {TOL_REL:g}) x min cost; cost_sweep_range: min/max argmin T over c_f in {CF_SWEEP} "
                       "at 50% life (argmin reported even when not interior); stressed action non-trivial (C4): "
                       "interior and 0.05 < P(fail by T*) < 0.95 at 75% life with the largest c_f"),
    }
    print(f"[decision] 50% life c_f={C_F:g}: interior={decision_checks['optimum_interior']} "
          f"argmin={mid['T_argmin_PTU']:.1f} sweep argmin range={decision_checks['cost_sweep_range']} "
          f"interior in sweep={decision_checks['cost_sweep_interior_count']}/{len(CF_SWEEP)} "
          f"stressed nontrivial={decision_checks['stressed_scenario_action_nontrivial']}")

    # ------------------------------------------------------------------ 7. figures
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 9})
    colors = ["tab:blue", "tab:green", "tab:red"]
    # F1 data overview
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.scatter(t_all[tr], vb[tr], s=3, c="tab:blue", label="training rows")
    ax.scatter(t_all[ho], vb[ho], s=3, c="tab:orange", label="held-out segments (20%)")
    for thr, lab in [(0.10, "Healthy/Moderate"), (0.25, "Moderate/Worn"), (VB_CRIT, "VB_crit = 0.30 mm")]:
        ax.axhline(thr, ls="--", lw=0.8, c="gray")
        ax.text(10, thr + 0.004, lab, fontsize=7, color="gray")
    ax.set_xlabel("pseudo-time (PTU, rows sorted by VB_mm: time index is target-derived)")
    ax.set_ylabel("VB_mm (mm)")
    ax.set_title("Reconstructed pseudo-trajectory (monotone by construction)")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig01_pseudo_trajectory.png"))
    plt.close(fig)

    # F2 feature diagnostics
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.6))
    names = [r["feature"] for r in feat_table]
    axs[0].barh(range(len(names)), [abs(r["pearson"]) for r in feat_table], color="tab:gray")
    axs[0].set_yticks(range(len(names)))
    axs[0].set_yticklabels(names, fontsize=6)
    axs[0].set_xlabel("|Pearson r| with VB_mm")
    axs[0].set_title("Feature-VB correlation")
    axs[1].scatter(y_ho, dd_ho, s=4, c="tab:orange", label="RF (features + time index)")
    axs[1].scatter(y_ho, feat_ho, s=4, c="tab:gray", alpha=0.5, label="RF (features only)")
    axs[1].plot([vb.min(), vb.max()], [vb.min(), vb.max()], "k--", lw=0.8)
    axs[1].set_xlabel("VB_mm observed (held-out)")
    axs[1].set_ylabel("predicted")
    axs[1].set_title(f"Held-out: RF+t R2={r2_d:.2f}")
    axs[1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig02_feature_diagnostics.png"))
    plt.close(fig)

    # F3 physics (trend fit) + model comparison with the shared time index
    fig, axs = plt.subplots(1, 2, figsize=(8.5, 3.6))
    axs[0].plot(t_all, vb, ".", ms=1.5, c="gray", label="VB_mm (sorted)")
    axs[0].plot(t_all, phys_all, "r-", lw=1, label="physics ODE (LS trend fit)")
    axs[0].plot(t_all, lin.predict(t_all[:, None]), "b--", lw=0.8, label="linear VB~t")
    axs[0].set_xlabel("pseudo-time (PTU)")
    axs[0].set_ylabel("VB (mm)")
    axs[0].legend(fontsize=6)
    axs[0].set_title("Trend fits on the shared time index")
    labels = ["physics\n(trend fit)", "data-driven\n(RF+t)", "hybrid\n(phys+RF+t)", "linear\nVB~t"]
    vals = [rmse_p, rmse_d, rmse_h, rmse_lin]
    axs[1].bar(labels, vals, color=["tab:red", "tab:orange", "tab:green", "tab:blue"])
    axs[1].axhline(std_ho, ls="--", c="k", lw=0.8)
    axs[1].text(0, std_ho * 1.02, "std(target)", fontsize=7)
    axs[1].set_ylabel("held-out RMSE (mm)")
    axs[1].set_title("Held-out comparison (same time input)")
    axs[1].tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig03_physics_and_model_comparison.png"))
    plt.close(fig)

    # F4 GP surrogate parity (0-2000 PTU; the emulated window is the whole grid)
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.scatter(yv, mv, s=1, alpha=0.3)
    ax.plot([yv.min(), yv.max()], [yv.min(), yv.max()], "k--", lw=0.8)
    ax.set_xlabel("ODE VB (mm), RK4 training solver")
    ax.set_ylabel("GP VB (mm)")
    ax.set_title(f"GP surrogate, prior-filtered LHS test set (R2={sur_r2:.3f})")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig04_gp_surrogate_parity.png"))
    plt.close(fig)

    # F5 Sobol
    fig, ax = plt.subplots(figsize=(6, 3.2))
    xk = np.arange(len(U_NAMES))
    ax.bar(xk - 0.2, [sob["RUL"][k]["S1"] for k in U_NAMES], 0.4, label="S1 (RUL)")
    ax.bar(xk + 0.2, [sob["RUL"][k]["ST"] for k in U_NAMES], 0.4, label="ST (RUL)")
    ax.set_xticks(xk)
    ax.set_xticklabels(U_NAMES, fontsize=8)
    ax.set_title(f"Sobol indices of RUL (GP, truncated prior, horizon {H_GRID[-1]:.0f} PTU)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig05_sobol_rul.png"))
    plt.close(fig)

    # F6 posterior narrowing
    fig, axs = plt.subplots(2, 3, figsize=(9, 5))
    flat = axs.ravel()
    for k in range(len(CAL_NAMES)):
        ax = flat[k]
        for (name, _), c in zip(STAGES, colors):
            ax.hist(chains[name][:, k], bins=40, density=True, alpha=0.45, color=c, label=f"{name} life")
        ax.axvline(lo_c[k], c="k", lw=0.6, ls=":")
        ax.axvline(hi_c[k], c="k", lw=0.6, ls=":")
        ax.set_title(CAL_NAMES[k] + (" (log)" if CAL_NAMES[k] in ("k1", "k2", "hA", "sigma_obs") else ""))
    flat[len(CAL_NAMES)].axis("off")
    flat[0].legend(fontsize=6)
    fig.suptitle("Posterior narrowing with sequential data (dotted = prior bounds; c_w dropped)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig06_posterior_stages.png"))
    plt.close(fig)

    # F7 RUL posterior per stage (truth is circular)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for (name, _), c in zip(STAGES, colors):
        so = stage_out[name]
        ax.hist(so["_rul"], bins=50, density=True, alpha=0.45, color=c,
                label=f"{name} life (circular truth {so['rul_true_PTU_circular']:.0f})")
        ax.axvline(so["T_argmin_PTU"] - 0, c=c, ls=("-" if so["T_star_interior"] else "--"), lw=1)
    ax.set_xlabel("RUL (PTU), conditioned on observed VB at the stage time")
    ax.set_ylabel("posterior predictive density")
    ax.set_title("RUL distributions; vertical = grid argmin T (c_f=5); solid = interior")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig07_rul_distribution.png"))
    plt.close(fig)

    # F8 cost rate
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for (name, _), c in zip(STAGES, colors):
        so = stage_out[name]
        tag = f"T*={so['T_star_PTU']:.0f}" if so["T_star_interior"] else "no interior optimum"
        ax.plot(so["_T"], so["_cost"], c=c, lw=1, label=f"{name} life ({tag})")
        ax.plot(so["T_argmin_PTU"], so["expected_cost_rate"], "o", c=c, ms=4)
    ax.set_yscale("log")
    ax.set_xlabel("replacement interval T (PTU)")
    ax.set_ylabel("expected cost rate (c_p per PTU)")
    ax.set_xlim(0, 2000)
    ax.set_title(f"Expected cost per unit time (c_p={C_P:g}, c_f={C_F:g}, assumed)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig08_cost_rate.png"))
    plt.close(fig)

    # F9 predictive band on held-out rows (75% life)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.scatter(t_all[ho], y_ho, s=3, c="tab:orange", zorder=3, label="held-out VB_mm")
    ax.plot(t_all[ho], np.interp(t_all[ho], H_GRID, cov_info["m_med"]), "k.", ms=1,
            label="posterior median (75% life)")
    ax.fill_between(t_all[ho], cov_info["lo95"], cov_info["hi95"], color="tab:blue", alpha=0.2,
                    label="95% predictive")
    ax.set_xlabel("pseudo-time (PTU)")
    ax.set_ylabel("VB (mm)")
    ax.set_title(f"Held-out 95% band: coverage {cov_levels['0.95']:.2f}, mean width "
                 f"{interval['mean_width95']:.3f} mm")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig09_heldout_predictive.png"))
    plt.close(fig)

    # ------------------------------------------------------------------ 8. metrics.json + stage file
    runtime = time.time() - T_START
    stage_clean = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in stage_out.items()}
    with open(os.path.join(RES, "stage_diagnostics.json"), "w", encoding="utf-8") as fh:
        json.dump(stage_clean, fh, indent=2, default=_jsonable)

    coverage95 = cov_levels["0.95"]
    n_blocks = int(ds["segment"].max()) + 1
    r2ref = load_round2()
    criteria = {
        "COVER_TOL": COVER_TOL, "CONV_STEPS_OVER_TAU": CONV_SOT, "TOL_REL": TOL_REL,
        "coverage_nominal": bool(interval["coverage_nominal"]),
        "converged_50tau": bool(min_steps_over_tau >= CONV_SOT),
        "optimum_interior": bool(decision_checks["optimum_interior"]),
        "stressed_scenario_action_nontrivial": bool(decision_checks["stressed_scenario_action_nontrivial"]),
        "hybrid_vs_linear": _cmp_word(loso_hl["ci95"]),
        "hybrid_vs_linear_ci95": loso_hl["ci95"],
    }
    limitations = [
        f"Data appear SYNTHETIC (suspected): uniform features, no tool/time/unit ID, {info['n_negative_VB']} "
        f"negative VB_mm values; Wear_Class is a deterministic binning of VB_mm. Every conclusion concerns the "
        "method on this file, not a real machine tool.",
        "Pseudo-time is the rank of VB_mm, so VB is monotone in pseudo-time by construction. The time index is "
        "target-derived and is the SAME input for physics, data-driven and hybrid models; none of the three is an "
        "independent predictive test. The physics row is a trend fit.",
        f"Features explain almost nothing: max|Pearson r|={max_abs_corr:.3f}, RF CV R2 random={cv_random:.3f}, "
        f"blocked={cv_blocked:.3f} (features only, no time input).",
        f"Held-out evaluation uses {n_ho} rows in {len(np.unique(seg_ho))} contiguous segments. The hybrid-vs-linear "
        "CI is a leave-one-segment-out jackknife with only k segments as units (t with k-1 df): it is wide and "
        "should be read as indicative only.",
        "RUL is in pseudo-time units (PTU, 1 PTU = one sorted sample index), not hours or cycles. The record has "
        f"{N} PTU, so the emulated horizon is the record length; the forcing is not wrapped inside the horizon.",
        f"Prior-predictive support: designs with simulated VB at {H_GRID[-1]:.0f} PTU above {VB_PRIOR_MAX:g} mm were "
        f"rejected before fitting ({100 * surr_filter['train_rejection_rate']:.1f} % of training candidates). The "
        "prior is therefore truncated, not the design box; the GP, the calibration prior and the Sobol analysis all "
        "use the same truncation (GP-mean VB at the horizon). The truncation is data-independent, but the results "
        "hold only for this truncated prior.",
        "RUL truth (first sorted index with VB >= 0.30 minus stage time) is CIRCULAR: it comes from the same target "
        "the models were fitted to. The 'miss' values compare the conditioned prediction with that circular crossing; "
        "they are not validation.",
        f"c_w (force per mm of wear) is NOT IDENTIFIED and was dropped from the sampler. The force-observation term "
        "is therefore absent from the calibration; forces do not enter the posterior. No posterior ratio is reported for it.",
        f"MCMC: {N_WALKERS} walkers, {N_BURN} burn-in, {N_STEPS} steps per stage with a DE-move/DE-snooker mixture. "
        f"Minimum steps/tau = {min_steps_over_tau:.1f} (target >= {CONV_SOT:g}): "
        + ("met." if min_steps_over_tau >= CONV_SOT else "NOT met; chains are unconverged and posterior widths are provisional."),
        f"RUL is censored at the {H_GRID[-1]:.0f}-PTU horizon in {100 * sob['RUL_censored_frac']:.1f} % of Sobol "
        f"samples and {100 * mid['censored_frac']:.1f} % of 50%-life posterior draws; censored values enter the "
        "cost curves as the horizon value, so T* depends on the censoring.",
        f"Sobol indices use the truncated prior with {sob['n_base_kept']} of {sob['n_base']} base rows kept "
        "(a base row is kept only if all its D+2 Saltelli rows lie in the support). Indices are noisy at that size.",
        f"Surrogate trained on fixed-step RK4 (dt=1 PTU). Against the reference RK45 ODE the max |difference| on "
        f"{rk_check['n_designs']} designs is {rk_check['max_abs_mm']:.4f} mm "
        f"({rk_check['max_abs_over_VB_crit']:.3f} x VB_crit). The headline speed-up is against the same RK4 batch as round 1; "
        f"the RK45 figure is reported separately.",
        f"GP nugget (standardised units, cap 1e-6) = {gp.noise_level:.2e}; at its bound: {gp.noise_at_bound}. The "
        f"emulator is near-interpolating because the ODE is deterministic, so the GP sd is tiny (median held-out "
        f"{interval['median_gp_sd_heldout']:.2e} mm) and the band is set mainly by sigma_obs and the residual scale.",
        f"Band scale c = {interval['scale_c']:.3f} is fitted on forecast residuals of TRAIN rows beyond the 75% stage "
        f"({interval['scale_fit_rows']} rows). Held-out rows are not used for the scale. The physics trend-fit "
        "centres of the prior box use all training rows, a small data dependence.",
        "Posterior RUL is conditioned on the observed VB at each stage time (offset correction of each GP mean "
        "trajectory). GP variance and sigma_obs enter the likelihood and the coverage band, not the RUL draws.",
        f"Costs c_p={C_P:g}, c_f={C_F:g} and derating law (gamma={GAMMA:g}, d_min={D_MIN:g}) are assumptions. "
        "Derating values are computed at the grid argmin and are meaningful only when an interior optimum exists.",
        "Nominal constants (sigma_n, Ea/R, C_th, T_inf, eta, T rise) are fixed, not fitted; k1, k2, alpha, hA are "
        "fitted in the trend fit and calibrated, and vc is fixed at 1 in calibration.",
        "F_res = sqrt(CF_1^2+CF_2^2+CF_3^2) is a combination of feature columns, not measured Fx,Fy,Fz.",
        "Vibration/AE are single snapshot values per row; no windowed RMS/std.",
    ]
    mcmc_block = {
        "tau_per_param": tau_max, "n_steps": int(N_STEPS), "n_walkers": int(N_WALKERS), "n_burn": int(N_BURN),
        "min_steps_over_tau": min_steps_over_tau, "ess_min": ess_min,
        "converged_50tau": criteria["converged_50tau"],
        "sampler": "emcee DEMove 0.8 + DESnookerMove 0.2",
    }
    metrics = {
        "project": "p1_cnc",
        "round": 3,
        "n_rows": int(N),
        "target": TARGET,
        "target_unit": "mm",
        "has_real_time_axis": False,
        "has_unit_ids": False,
        "data_synthetic_suspected": True,
        "heldout_rmse": {"physics": rmse_p, "data_driven": rmse_d, "hybrid": rmse_h},
        "heldout_nrmse": {"physics": rmse_p / std_ho, "data_driven": rmse_d / std_ho, "hybrid": rmse_h / std_ho},
        "surrogate": {"r2": sur_r2, "rmse": sur_rmse, "coverage_2sigma": cov2, "speedup_x": float(speedup_rk4),
                      "speedup_reference": "per-sample RK4 batch (same reference as round 1)"},
        "sobol": {k: sob["RUL"][k] for k in U_NAMES},
        "sobol_VB_end": {k: sob["VB_end"][k] for k in U_NAMES},
        "calibration": {
            "posterior_to_prior_std_ratio": {**{k: final["post_to_prior_sd"][k] for k in CAL_NAMES},
                                             "c_w": "not identified"},
            "mean_autocorr_time": mean_tau,
            "acceptance_fraction": mean_acc,
            "coverage95_heldout": coverage95,
        },
        "risk": {
            "description": ("P(RUL <= T*) at 50% life: probability of failure (VB >= 0.30 mm) within the "
                            "replacement interval at the grid argmin (T* if interior), posterior predictive, "
                            "conditioned on observed VB"),
            "value": float(mid["P_fail_within_T_star"]),
        },
        "decision": {
            "description": ("Replacement interval minimising expected cost rate C(T)=[c_p P(RUL>T)+c_f P(RUL<=T)]/"
                            "E[min(RUL,T)], c_p=1, c_f=5 (assumed); 50% life, conditioned RUL; PTU. Reported as an "
                            "optimum only if interior (C3); otherwise no recommendation."),
            "optimal_value": mid["T_star_PTU"],
            "optimal_value_status": ("interior optimum" if mid["T_star_interior"]
                                     else "no interior optimum in [5, 2000] PTU"),
            "argmin_grid_PTU": mid["T_argmin_PTU"],
            "unit": "PTU",
            "expected_cost": mid["expected_cost_rate"],
        },
        "runtime_sec": float(time.time() - T_START),
        "limitations": limitations,
        "mcmc": mcmc_block,
        "interval_calibration": interval,
        "decision_checks": decision_checks,
        "criteria": criteria,
        "baselines": {
            "linear_trend_rmse": rmse_lin, "linear_trend_nrmse": rmse_lin / std_ho,
            "data_driven_features_only_rmse": rmse_feat, "data_driven_features_only_nrmse": rmse_feat / std_ho,
            "hybrid_minus_physics_rmse": loso_hp,
            "hybrid_minus_linear_rmse": loso_hl,
            "data_driven_minus_physics_rmse": loso_dp,
            "physics_minus_linear_rmse": loso_pl,
            "heldout_r2_data_driven": r2_d,
            "std_heldout_VB_mm": std_ho,
        },
        "physics": {
            "label": "trend fit (LS-fitted ODE on the target-derived time index)",
            "theta": {"k1": float(ls["theta"][0]), "k2": float(ls["theta"][1]), "alpha": float(ls["theta"][2]),
                      "hA": float(ls["theta"][3]), "vc": float(ls["theta"][4])},
            "VB0": VB0, "train_rmse": ls["train_rmse"], "ls_success": ls["success"], "ls_nfev": ls["nfev"],
        },
        "surrogate_checks": {
            "rk4_vs_rk45": rk_check,
            "error_quantiles": sur_err,
            "design_filter": surr_filter,
            "gp_kernel": str(gp.gp.kernel_),
            "gp_noise_level": gp.noise_level, "gp_noise_at_bound": gp.noise_at_bound,
            "n_train": int(len(U_tr)), "n_test": int(len(U_te)), "horizon_PTU": float(H_GRID[-1]),
            "grid_step_PTU": float(H_GRID[1] - H_GRID[0]),
            "end_of_record_R2": end_r2,
            "t_rk45_per_run_sec": float(t_rk45), "t_rk4_batch_per_sample_sec": float(t_rk4),
            "t_gp_per_sample_sec": float(t_gp),
            "speedup_vs_rk4_batch": float(speedup_rk4),
            "speedup_vs_rk45": float(speedup_rk45),
        },
        "sobol_meta": {"n_eval": sob["n_eval"], "n_base": sob["n_base"], "n_base_kept": sob["n_base_kept"],
                       "RUL_censored_frac": sob["RUL_censored_frac"]},
        "stages": stage_clean,
        "data_inspection": {
            "shape": info["shape"], "n_missing": info["n_missing"], "duplicate_rows": info["duplicate_rows"],
            "label_counts": info["label_counts"], "label_VB_ranges": info["label_VB_ranges"],
            "VB_min": info["VB_min"], "VB_max": info["VB_max"], "VB_mean": info["VB_mean"],
            "n_negative_VB": info["n_negative_VB"], "id_or_time_like_columns": info["id_or_time_like_columns"],
        },
        "feature_table": feat_table,
        "data_diagnostics": {"max_abs_pearson": max_abs_corr, "rf_cv_r2_random_features_only": cv_random,
                             "rf_cv_r2_blocked_features_only": cv_blocked},
        "pseudo_time": {"n_rows": int(N), "n_train": n_tr, "n_heldout": n_ho, "segment_len_PTU": 100,
                        "n_segments": n_blocks, "n_heldout_segments": int(len(np.unique(seg_ho)))},
        "assumptions": {
            "VB_crit_mm": VB_CRIT, "c_p": C_P, "c_f": C_F, "cf_sweep": CF_SWEEP, "gamma": GAMMA, "d_min": D_MIN,
            "vb_prior_max_mm": VB_PRIOR_MAX, "decision_tol_rel": TOL_REL,
            "sigma_obs_bounds": [1e-4, 1e-1], "cal_params": CAL_NAMES,
            "prior_sd": {n: float(v) for n, v in zip(CAL_NAMES, prior_sd)},
            "stage_fractions": {n: f for n, f in STAGES}, "inspect_every": INSPECT_EVERY,
            "n_walkers": N_WALKERS, "n_burn": N_BURN, "n_steps": N_STEPS, "n_post": N_POST,
            "horizon_PTU": float(H_GRID[-1]), "decision_grid_PTU": [float(T_DECISION[0]), float(T_DECISION[-1])],
            "sobol_N": SOBOL_N, "seed": SEED,
        },
        "round1_reference": load_round1(),
        "round2_reference": r2ref,
        "changes_r3": [
            "GP and calibration/RUL grid reverted to 0-2000 PTU (record length); RUL beyond 2000 PTU is not modelled.",
            f"Design prior-predictive filter: only designs with simulated VB(2000) <= {VB_PRIOR_MAX:g} mm are used "
            "(train and test); the same support is imposed on the calibration posterior and on the Sobol sample.",
            "GP outputs standardised per grid column; WhiteKernel nugget capped at 1e-6 (was 1e-1).",
            "Surrogate error reported as per-design quantiles as well as RMSE/R2.",
            "Held-out 50/80/95 band = GP latent sd + sigma_obs, times a scale fitted on TRAIN forecast residuals beyond "
            "the 75% stage (held-out rows not used).",
            "Decision: interior optimum only if the cost rises >= 1 % after the argmin; otherwise 'no interior optimum'. "
            "The cost sweep reports the argmin for each c_f and the interior flags.",
            "RUL conditioned on the observed VB at each stage time (offset correction of each draw); predicted vs "
            "empirical crossing and the miss are reported per stage.",
            "c_w removed from the sampler and reported as 'not identified' (no ratio).",
            "MCMC: DE-move/DE-snooker mixture, rejection-sampled starts inside the prior support.",
            "Hybrid and physics compared with the linear trend and with leave-one-segment-out jackknife CIs.",
            "Round-1 reference settings corrected to 32 walkers / 600 burn-in / 1000 steps (round 2 had 24/300/600).",
            "Speed-up headline is against the same RK4 batch as round 1 (round 2 headline used RK45).",
            "Summary generated from metrics.json only; round-1 and round-2 columns read from their snapshot JSON files.",
        ],
        "r2_vs_r3": {
            "heldout_nrmse": {"r2": r2ref["heldout_nrmse"], "r3": {"physics": rmse_p / std_ho,
                                                                    "data_driven": rmse_d / std_ho,
                                                                    "hybrid": rmse_h / std_ho}},
            "coverage95": [r2ref["coverage95"], coverage95],
            "min_steps_over_tau": [r2ref["min_steps_over_tau"], min_steps_over_tau],
        },
    }
    metrics["verdict"] = make_verdict(metrics)
    metrics_path = os.path.join(RES, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, default=_jsonable)

    # ------------------------------------------------------------------ 9. summary.md from metrics.json only
    with open(metrics_path, encoding="utf-8") as fh:
        m_file = json.load(fh)
    summary = build_summary(m_file)
    with open(os.path.join(RES, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write(summary)

    print(f"[done] runtime={metrics['runtime_sec']:.1f}s metrics and summary written to {RES}")


# ---------------------------------------------------------------------- verdict (generated from metrics)
def _tt(x):
    return "no interior optimum" if x is None else f"{x:.0f} PTU"


def make_verdict(m):
    sg, se = m["surrogate"], m["surrogate_checks"]["error_quantiles"]
    sf = m["surrogate_checks"]["design_filter"]
    ic, mc, dc, bl = m["interval_calibration"], m["mcmc"], m["decision_checks"], m["baselines"]
    cov, cl, st = ic["coverage"], m["criteria"], m["stages"]
    word = cl["hybrid_vs_linear"]
    ci = cl["hybrid_vs_linear_ci95"]
    v1 = ("1. Data are SYNTHETIC (suspected): every conclusion below concerns the METHOD on this file "
          "(pipeline behaviour), not a real CNC machine. The RUL 'truth' is circular.")
    v2 = (f"2. Surrogate and band: GP on 0-{m['surrogate_checks']['horizon_PTU']:.0f} PTU, prior restricted to "
          f"VB(horizon) <= {sf['prior_VB_max_mm']:g} mm (rejects {100 * sf['train_rejection_rate']:.0f} % of candidates); "
          f"R2 = {sg['r2']:.4f}, per-design RMSE median {se['per_design_rmse_median']:.4f} mm, p95 "
          f"{se['per_design_rmse_p95']:.4f} mm. Held-out coverage 50/80/95 % = {cov['0.5']:.2f}/{cov['0.8']:.2f}/"
          f"{cov['0.95']:.2f}, mean 95 % width {ic['mean_width95']:.3f} mm: "
          + ("PASS (all levels within 0.05 of nominal)." if cl["coverage_nominal"]
             else "FAIL (not nominal at one or more levels)."))
    v3 = (f"3. Calibration: min steps/tau = {mc['min_steps_over_tau']:.1f} (required >= {cl['CONV_STEPS_OVER_TAU']:g}): "
          + ("PASS." if cl["converged_50tau"] else "FAIL, chains NOT converged; posterior widths are provisional.")
          + f" Min ESS = {mc['ess_min']:.0f}.")
    miss = {k: st[k]["miss_PTU_circular"] for k in ("25%", "50%", "75%")}
    v4 = (f"4. RUL and decision: conditioned RUL, predicted minus empirical crossing (circular) = "
          f"{miss['25%']:+.0f} / {miss['50%']:+.0f} / {miss['75%']:+.0f} PTU at 25/50/75 % life. "
          + (f"50 % life, c_f = 5: interior optimum T* = {m['decision']['optimal_value']:.0f} PTU."
             if dc["optimum_interior"]
             else "50 % life, c_f = 5: no interior optimum in [5, 2000] PTU, so no replacement time is recommended.")
          + f" RUL censored at the horizon for {100 * st['50%']['censored_frac']:.0f} % of 50 %-life draws.")
    bb = bl["physics_minus_linear_rmse"]
    v5 = (f"5. Hybrid vs linear VB~t trend (leave-one-segment-out CI, k = {m['pseudo_time']['n_heldout_segments']} "
          f"segments): hybrid minus linear = {bl['hybrid_minus_linear_rmse']['diff']:+.5f} mm, CI "
          f"[{ci[0]:+.5f}, {ci[1]:+.5f}]: {word}. Physics-only minus linear = {bb['diff']:+.5f} mm. "
          + ("The physics adds nothing measurable over the linear trend on this file."
             if ci[1] >= 0 else "The hybrid beats the linear trend on this file (synthetic data only)."))
    return [v1, v2, v3, v4, v5]


# ---------------------------------------------------------------------- summary builder
def _f(x, spec=".3f"):
    return "n/a" if x is None else format(x, spec)


def _yn(b):
    return "yes" if b else "no"


def _pf(b):
    return "PASS" if b else "FAIL"


def _tstr(x):
    return "no interior optimum" if x is None else f"{x:.0f} PTU"


def build_summary(m):
    di, dd, pt, A = m["data_inspection"], m["data_diagnostics"], m["pseudo_time"], m["assumptions"]
    ph, hr, hn, bl = m["physics"], m["heldout_rmse"], m["heldout_nrmse"], m["baselines"]
    sg, sc, mc, ic = m["surrogate"], m["surrogate_checks"], m["mcmc"], m["interval_calibration"]
    dc, st, cal, cl = m["decision_checks"], m["stages"], m["calibration"], m["criteria"]
    r1, r2 = m["round1_reference"], m["round2_reference"]
    S25, S50, S75 = st["25%"], st["50%"], st["75%"]
    rk, se, flt = sc["rk4_vs_rk45"], sc["error_quantiles"], sc["design_filter"]
    cov = ic["coverage"]
    prior_sd = A["prior_sd"]
    CAL = A["cal_params"]
    hl, hp, dp, pl = bl["hybrid_minus_linear_rmse"], bl["hybrid_minus_physics_rmse"], bl["data_driven_minus_physics_rmse"], bl["physics_minus_linear_rmse"]
    lines = []
    w = lines.append

    w("# p1_cnc: CNC flank-wear digital twin (round 3, final)")
    w("")
    w("This file is generated by `main.py` from `results/metrics.json`. Every number is an f-string over that file. "
      "Only the descriptive prose is typed.")
    w("")
    w("## 0. Verdict (5 lines; failures included)")
    for line in m["verdict"]:
        w(f"- {line}")
    w("")
    w("Pre-registered pass/fail criteria (defined in the code before the run):")
    w(f"- C1 coverage nominal (|empirical - nominal| <= {cl['COVER_TOL']:g} at 50/80/95 %): **{_pf(cl['coverage_nominal'])}**.")
    w(f"- C2 MCMC converged (min steps/tau >= {cl['CONV_STEPS_OVER_TAU']:g}): **{_pf(cl['converged_50tau'])}**.")
    w(f"- C3 interior optimum (argmin inside the grid and cost rises >= {100 * cl['TOL_REL']:g} % after it): "
      f"**{_pf(cl['optimum_interior'])}**.")
    w(f"- C4 stressed action non-trivial: **{_pf(cl['stressed_scenario_action_nontrivial'])}**.")
    w(f"- C5 hybrid vs linear, leave-one-segment-out CI excludes 0: **{cl['hybrid_vs_linear']}**.")
    w("")

    w("## 1. Data inspection")
    w(f"- Shape {di['shape'][0]} x {di['shape'][1]}; missing values {di['n_missing']}; duplicate rows {di['duplicate_rows']}.")
    w("- No tool, run, unit or timestamp column exists (id/time-like columns found: "
      f"{di['id_or_time_like_columns'] or 'none'}). Rows are unordered snapshots.")
    w(f"- Wear_Class counts: {di['label_counts']}. Class VB ranges (mm): {di['label_VB_ranges']}. The ranges do not "
      "overlap, so Wear_Class is a deterministic binning of VB_mm, not an independent label.")
    w(f"- VB_mm: min {di['VB_min']:.4f}, max {di['VB_max']:.4f}, mean {di['VB_mean']:.4f}; "
      f"**{di['n_negative_VB']} negative values** (physically impossible for flank wear).")
    w("- Raw features are close to uniform on their range (KS p-values below). This is consistent with synthetic data; "
      "it is not a proof.")
    w("")
    w("| feature | Pearson r | Spearman rho | KS p (uniform) |")
    w("|---|---|---|---|")
    for r in m["feature_table"]:
        w(f"| {r['feature']} | {r['pearson']:+.3f} | {r['spearman']:+.3f} | {_f(r['ks_p_uniform'])} |")
    w("")

    w("## 2. Pseudo-time reconstruction (assumption)")
    w(f"- Rows are sorted on VB_mm ascending, so **VB is monotone in pseudo-time by construction**. Unit: PTU "
      f"(1 PTU = one sorted sample index, 0..{pt['n_rows'] - 1}). **All RUL values are in PTU, not hours or cycles.**")
    w(f"- Blocked split: {pt['n_segments']} contiguous segments of {pt['segment_len_PTU']} PTU; "
      f"{pt['n_heldout_segments']} segments ({pt['n_heldout']} rows) are held out; {pt['n_train']} rows train.")
    w("- F_res = sqrt(CF_Feature_1^2 + CF_Feature_2^2 + CF_Feature_3^2): feature columns, not measured Fx, Fy, Fz.")
    w("")

    w("## 3. Do the features explain VB_mm?")
    w(f"- Max |Pearson r| with VB_mm over the 16 inputs: **{dd['max_abs_pearson']:.3f}**.")
    w(f"- Random forest CV R2 on features only (no time): **{dd['rf_cv_r2_random_features_only']:.3f}** shuffled 5-fold, "
      f"**{dd['rf_cv_r2_blocked_features_only']:.3f}** blocked 5-fold by segment.")
    w("- The features carry essentially no measurable information about wear in this file.")
    w("")

    w("## 4. Models and held-out comparison (shared time input)")
    w("- **Time input.** Physics, data-driven and hybrid all receive the same time index t (PTU). t is the rank of "
      "VB_mm, so it is target-derived.")
    w(f"- **Physics** (trend fit): RK45 coupled wear/thermal ODE, least-squares fit on training rows. Fitted "
      f"k1={ph['theta']['k1']:.4g}, k2={ph['theta']['k2']:.4g}, alpha={ph['theta']['alpha']:.4g}, "
      f"hA={ph['theta']['hA']:.4g}, VB0={ph['VB0']:.4g}; vc fixed at 1.")
    w("- **Data-driven**: random forest (300 trees) on 16 features + t, standardised with training statistics.")
    w("- **Hybrid**: physics + random forest on the training residual VB_obs - VB_phys, same inputs.")
    w("- **Baselines**: linear VB~t (trend reference); random forest on features only (no time).")
    w("")
    w("| model | held-out RMSE (mm) | NRMSE = RMSE/std(target) |")
    w("|---|---|---|")
    w(f"| Physics (trend fit) | {hr['physics']:.4f} | {hn['physics']:.3f} |")
    w(f"| Data-driven (RF + t) | {hr['data_driven']:.4f} | {hn['data_driven']:.3f} |")
    w(f"| Hybrid (physics + RF + t) | {hr['hybrid']:.4f} | {hn['hybrid']:.3f} |")
    w(f"| Linear VB~t | {bl['linear_trend_rmse']:.4f} | {bl['linear_trend_nrmse']:.3f} |")
    w(f"| RF features only (no t) | {bl['data_driven_features_only_rmse']:.4f} | {bl['data_driven_features_only_nrmse']:.3f} |")
    w("")
    w(f"std(held-out VB_mm) = {bl['std_heldout_VB_mm']:.4f} mm; RF (features + t) held-out R2 = {bl['heldout_r2_data_driven']:.3f}.")
    w("")
    w(f"**Leave-one-segment-out jackknife CIs** (units = {pt['n_heldout_segments']} held-out segments, t with "
      f"{hl['k_segments'] - 1} df, t = {hl['t_crit']:.2f}):")
    w("")
    w("| comparison (RMSE difference, mm) | point | 95% CI | reading |")
    w("|---|---|---|---|")
    w(f"| hybrid - linear VB~t | {hl['diff']:+.5f} | [{hl['ci95'][0]:+.5f}, {hl['ci95'][1]:+.5f}] | {_cmp_text(hl['ci95'])} |")
    w(f"| hybrid - physics | {hp['diff']:+.5f} | [{hp['ci95'][0]:+.5f}, {hp['ci95'][1]:+.5f}] | {_cmp_text(hp['ci95'])} |")
    w(f"| physics - linear VB~t | {pl['diff']:+.5f} | [{pl['ci95'][0]:+.5f}, {pl['ci95'][1]:+.5f}] | {_cmp_text(pl['ci95'])} |")
    w(f"| data-driven - physics | {dp['diff']:+.5f} | [{dp['ci95'][0]:+.5f}, {dp['ci95'][1]:+.5f}] | {_cmp_text(dp['ci95'])} |")
    w("")
    w("Per-segment RMSE differences (hybrid - linear) are in `metrics.json` under `baselines.hybrid_minus_linear_rmse`.")
    w("")

    w("## 5. GP surrogate of the ODE (prior-predictive filter, 0-2000 PTU)")
    w(f"- **Design filter (data-independent).** Simulated VB at {sc['horizon_PTU']:.0f} PTU must be <= {A['vb_prior_max_mm']:g} mm. "
      f"Training candidates {flt['train_candidates']}, accepted {flt['train_accepted']}, used {flt['train_used']} "
      f"(**rejection {100 * flt['train_rejection_rate']:.1f} %**); test candidates {flt['test_candidates']}, accepted "
      f"{flt['test_accepted']}, used {flt['test_used']} (rejection {100 * flt['test_rejection_rate']:.1f} %).")
    w(f"- Input u = (log k1, log k2, alpha, log hA, vc), LHS box around the LS fit. Output = VB on a "
      f"{sc['grid_step_PTU']:.0f}-PTU grid to {sc['horizon_PTU']:.0f} PTU (round 1: {r1['surrogate_horizon_PTU']:.0f} PTU; "
      f"round 2: 6000 PTU). Training: RK4 (dt = 1 PTU).")
    w(f"- Test R2 (flattened) = **{sg['r2']:.4f}**; R2 at end of horizon = {sc['end_of_record_R2']:.4f}; "
      f"RMSE = {sg['rmse']:.4f} mm; fraction within +/-2 sigma = {sg['coverage_2sigma']:.3f}.")
    w(f"- **Error is summarised per design as well as globally** (the 0-2000 window is the whole grid, so the "
      f"window RMSE equals the global RMSE): per-design RMSE median {se['per_design_rmse_median']:.4f} mm, "
      f"p95 {se['per_design_rmse_p95']:.4f} mm, max {se['per_design_rmse_max']:.4f} mm; absolute error median "
      f"{se['abs_err_median']:.4f} mm, max {se['abs_err_max']:.4f} mm.")
    w(f"- GP kernel: `{sc['gp_kernel']}`. Nugget = {sc['gp_noise_level']:.2e} (cap 1e-6); at bound: {_yn(sc['gp_noise_at_bound'])}.")
    w(f"- **Solver check (RK4 training vs RK45 reference)** on {rk['n_designs']} test designs over the horizon: "
      f"max |difference| = {rk['max_abs_mm']:.4f} mm ({rk['max_abs_over_VB_crit']:.3f} x VB_crit), RMSE = {rk['rmse_mm']:.4f} mm, "
      f"RK45 failures = {rk['n_rk45_failures']}.")
    w(f"- **Speed-up** (like-for-like with round 1): **{sg['speedup_x']:.0f}x against the batched RK4** used to build the "
      f"training data (round 1: {r1['surrogate_speedup_x']:.0f}x). Against per-run RK45 (`solve_ivp`, rtol 1e-6): "
      f"{sc['speedup_vs_rk45']:.0f}x (reported separately, not compared with round 1).")
    w("- Times per sample: RK45 "
      f"{sc['t_rk45_per_run_sec']:.4f} s, RK4 batch {sc['t_rk4_batch_per_sample_sec']:.5f} s, GP "
      f"{sc['t_gp_per_sample_sec']:.2e} s.")
    w("")

    w(f"## 6. Sobol sensitivity (Saltelli N = {A['sobol_N']}, truncated prior)")
    w(f"Base rows kept after the joint support filter: {m['sobol_meta']['n_base_kept']} of {m['sobol_meta']['n_base']}; "
      f"{m['sobol_meta']['n_eval']} GP evaluations. RUL is censored at the {sc['horizon_PTU']:.0f} PTU horizon in "
      f"{100 * m['sobol_meta']['RUL_censored_frac']:.1f} % of samples.")
    w("")
    w("| parameter | S1 (RUL) | ST (RUL) | S1 (VB_end) | ST (VB_end) |")
    w("|---|---|---|---|---|")
    for k in ["log_k1", "log_k2", "alpha", "log_hA", "vc"]:
        w(f"| {k} | {m['sobol'][k]['S1']:.3f} | {m['sobol'][k]['ST']:.3f} | "
          f"{m['sobol_VB_end'][k]['S1']:.3f} | {m['sobol_VB_end'][k]['ST']:.3f} |")
    w("")

    w(f"## 7. Bayesian calibration (emcee, {mc['n_walkers']} walkers, {mc['n_burn']} burn-in + {mc['n_steps']} steps per stage)")
    w(f"- Sampler: {mc['sampler']}; starts rejection-sampled inside the prior support.")
    w(f"- Likelihood: Gaussian on inspected VB (every {A['inspect_every']}th training row up to the stage; sigma_obs plus "
      "GP variance). **c_w is not identified and is dropped**: the force observations do not enter the calibration.")
    w("- Prior: uniform in the surrogate box for k1, k2, alpha, hA; log-uniform sigma_obs in [1e-4, 1e-1] mm; "
      f"support requires GP-mean VB at {sc['horizon_PTU']:.0f} PTU <= {A['vb_prior_max_mm']:g} mm.")
    w(f"- Stages: 25%, 50%, 75% of the {pt['n_rows']}-row record ({S25['t_stage_PTU']:.0f}, {S50['t_stage_PTU']:.0f}, "
      f"{S75['t_stage_PTU']:.0f} PTU).")
    w("")
    w("**MCMC diagnostics (worst case over the three stages)**")
    w("")
    w("| parameter | tau (max over stages) | steps/tau | ESS (min over stages) |")
    w("|---|---|---|---|")
    for n in CAL:
        tmax = mc["tau_per_param"][n]
        w(f"| {n} | {tmax:.1f} | {mc['n_steps'] / tmax:.1f} | "
          f"{min(st[k]['ess'][n] for k in st):.0f} |")
    w("")
    w(f"- Minimum steps/tau = {mc['min_steps_over_tau']:.1f}; converged (>= {cl['CONV_STEPS_OVER_TAU']:g}) = "
      f"**{_yn(mc['converged_50tau'])}**. Mean tau = {cal['mean_autocorr_time']:.1f} steps; mean acceptance = "
      f"{cal['acceptance_fraction']:.2f}.")
    w(f"- Round 1: {r1['mcmc']['nwalkers']} walkers, {r1['mcmc']['nburn']} burn-in, {r1['mcmc']['nsteps']} steps "
      f"(steps/tau about {r1['mcmc']['nsteps'] / r1['mean_autocorr_time']:.1f}). Round 2: min steps/tau "
      f"{r2['min_steps_over_tau']:.1f}.")
    w("")
    w("**Posterior vs prior at 75% life** (absolute posterior sd and ratio to prior sd; ratio 1.0 = no information)")
    w("")
    w("| parameter | posterior mean | posterior sd | prior sd | ratio 75% | ratio 50% | ratio 25% |")
    w("|---|---|---|---|---|---|---|")
    for n in CAL:
        w(f"| {n} | {S75['posterior_mean'][n]:.4g} | {S75['posterior_sd'][n]:.4g} | {prior_sd[n]:.4g} | "
          f"{S75['post_to_prior_sd'][n]:.3f} | {S50['post_to_prior_sd'][n]:.3f} | {S25['post_to_prior_sd'][n]:.3f} |")
    w("| c_w | not identified | - | - | - | - | - |")
    w("")
    w("**Held-out band (75% life): latent GP sd + sigma_obs, times a scale from TRAIN forecast residuals**")
    w("")
    w(f"Scale c = {ic['scale_c']:.3f} (fitted on {ic['scale_fit_rows']} training rows beyond the 75% stage, "
      f"mean |standardised residual| {ic['scale_fit_mean_abs_std_residual']:.3f}). Held-out rows were not used for the scale.")
    w("")
    w("| nominal level | empirical coverage on held-out rows | within 0.05 of nominal |")
    w("|---|---|---|")
    for lev in ["0.5", "0.8", "0.95"]:
        w(f"| {lev} | {cov[lev]:.3f} | {_yn(abs(cov[lev] - float(lev)) <= cl['COVER_TOL'])} |")
    w("")
    w(f"Mean 95% interval width = {ic['mean_width95']:.4f} mm. Median posterior sigma_obs = "
      f"{ic['median_sigma_obs_post']:.4f} mm; median GP sd on held-out rows = {ic['median_gp_sd_heldout']:.2e} mm. "
      f"Coverage nominal (C1): **{_pf(cl['coverage_nominal'])}**.")
    w("")

    w("## 8. RUL, risk and replacement decision (RUL in PTU, conditioned on observed VB at the stage time)")
    w(f"**RUL truth is circular.** VB_crit = {A['VB_crit_mm']:.2f} mm. The 'empirical' crossing is the first sorted index "
      "with VB >= 0.30. It comes from the same target the models were fitted to, so the miss is not validation.")
    w(f"Costs c_p = {A['c_p']:g}, c_f = {A['c_f']:g} (assumed). Derating d = max({A['d_min']:g}, 1 - {A['gamma']:g} P(fail by T*)) (assumed law), "
      "evaluated at the grid argmin.")
    w("")
    w("| stage | t (PTU) | observed VB at stage (mm) | RUL pred median [95%] | predicted crossing | empirical crossing (circular) | miss (PTU) | censored | T* | cost rate | P(fail by T*) | derating |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, s in [("25%", S25), ("50%", S50), ("75%", S75)]:
        w(f"| {name} | {s['t_stage_PTU']:.0f} | {s['anchor_VB_obs_mm']:.4f} | "
          f"{s['rul_pred_median']:.0f} [{s['rul_pred_95'][0]:.0f}, {s['rul_pred_95'][1]:.0f}] | "
          f"{s['pred_crossing_PTU']:.0f} | {s['empirical_crossing_PTU_circular']:.0f} | {s['miss_PTU_circular']:+.0f} | "
          f"{100 * s['censored_frac']:.1f} % | {_tstr(s['T_star_PTU'])} | "
          f"{s['expected_cost_rate']:.4f} | {s['P_fail_within_T_star']:.3f} | {s['derating_factor']:.3f} |")
    w("")
    w(f"- Headline (50% life, c_f = {A['c_f']:g}): **{_tstr(S50['T_star_PTU'])}**"
      + (f"; expected cost rate {S50['expected_cost_rate']:.4f} c_p per PTU, failure risk at the argmin = "
         f"{S50['P_fail_within_T_star']:.3f}." if S50["T_star_interior"]
         else f". Grid argmin {S50['T_argmin_PTU']:.0f} PTU is not an interior optimum (cost rise after it "
              f"{_f(S50['rise_after_argmin_rel'], '.4f')} relative, below the {100 * cl['TOL_REL']:g} % rule or at the grid end). "
              "No replacement time is recommended from this stage."))
    w("")
    w(f"**c_f sensitivity (50% life; c_p = {A['c_p']:g}).** Each row gives the argmin of the cost curve; "
      "'T*' is shown only when the argmin is interior.")
    w("")
    w("| c_f | argmin T (PTU) | T* (interior only) | cost rate at argmin | P(fail) at argmin | rise after argmin (rel.) |")
    w("|---|---|---|---|---|---|")
    for k, v in dc["cf_sweep"].items():
        w(f"| {v['c_f']:g} | {v['T_argmin_PTU']:.1f} | {_tstr(v['T_star_PTU'])} | {v['expected_cost_rate']:.4f} | "
          f"{v['P_fail_at_argmin']:.3f} | {_f(v['rise_after_argmin_rel'], '.4f')} |")
    w("")
    sn = dc["stressed_scenario"]
    w(f"- Interior optima across the sweep: {dc['cost_sweep_interior_count']} of {len(dc['cf_sweep'])}. "
      f"Argmin range [{dc['cost_sweep_range'][0]:.1f}, {dc['cost_sweep_range'][1]:.1f}] PTU.")
    w(f"- Stressed scenario ({sn['stage']} life, c_f = {sn['c_f']:g}): argmin {sn['T_argmin_PTU']:.1f} PTU, "
      f"interior = {_yn(sn['interior'])}, P(fail by argmin) = {sn['P_fail_within_T_star']:.3f}; "
      f"action non-trivial (C4) = **{_yn(dc['stressed_scenario_action_nontrivial'])}**.")
    w(f"- Interior optimum at 50% life (C3): **{_yn(dc['optimum_interior'])}**.")
    w("")

    w("## 9. Changes vs round 1 (round-1 column read from `metrics_round1.json`)")
    w("")
    w("| item | round 1 | round 3 |")
    w("|---|---|---|")
    w(f"| Time input to data-driven / hybrid | none (physics had target rank) | same t as physics |")
    w(f"| Linear VB~t baseline | absent | held-out RMSE {bl['linear_trend_rmse']:.4f} mm |")
    w(f"| Held-out NRMSE physics / data / hybrid | {r1['heldout_nrmse']['physics']:.3f} / "
      f"{r1['heldout_nrmse']['data_driven']:.3f} / {r1['heldout_nrmse']['hybrid']:.3f} | "
      f"{hn['physics']:.3f} / {hn['data_driven']:.3f} / {hn['hybrid']:.3f} |")
    w(f"| MCMC walkers / burn-in / steps | {r1['mcmc']['nwalkers']} / {r1['mcmc']['nburn']} / {r1['mcmc']['nsteps']} | "
      f"{mc['n_walkers']} / {mc['n_burn']} / {mc['n_steps']} |")
    w(f"| min steps/tau | {r1['mcmc']['nsteps'] / r1['mean_autocorr_time']:.1f} (from mean tau) | "
      f"{mc['min_steps_over_tau']:.1f} (min over params and stages) |")
    w(f"| c_w | +/-1000 prior, ratio {r1['c_w_ratio']:.3f} | not identified, dropped |")
    w(f"| Surrogate horizon | {r1['surrogate_horizon_PTU']:.0f} PTU | {sc['horizon_PTU']:.0f} PTU |")
    w(f"| Surrogate RMSE (mm) | {r1['surrogate_rmse']:.4f} | {sg['rmse']:.4f} |")
    w(f"| Speed-up | {r1['surrogate_speedup_x']:.0f}x vs RK4 batch | {sg['speedup_x']:.0f}x vs RK4 batch "
      f"({sc['speedup_vs_rk45']:.0f}x vs RK45, separately) |")
    w(f"| Solver check RK4 vs RK45 | none | max {rk['max_abs_mm']:.4f} mm |")
    w(f"| 95% held-out coverage | {r1['coverage95_heldout']:.3f} | {cov['0.95']:.3f} (mean width {ic['mean_width95']:.3f} mm) |")
    w(f"| Coverage 50 / 80 % | not reported | {cov['0.5']:.3f} / {cov['0.8']:.3f} |")
    w(f"| T* at 50% life (c_f = 5) | {r1['T_star_PTU']:.1f} PTU | {_tstr(m['decision']['optimal_value'])} (argmin {m['decision']['argmin_grid_PTU']:.1f}) |")
    w(f"| Stage diagnostics file | not written | results/stage_diagnostics.json |")
    w(f"| Summary | typed by hand, mismatches | generated from metrics.json only |")
    w("")
    w("## 9b. Round 2 to round 3 (round-2 column read from `metrics_round2.json`)")
    w("")
    w("| item | round 2 | round 3 |")
    w("|---|---|---|")
    rr = m["r2_vs_r3"]
    w(f"| Held-out NRMSE physics / data / hybrid | {rr['heldout_nrmse']['r2']['physics']:.3f} / "
      f"{rr['heldout_nrmse']['r2']['data_driven']:.3f} / {rr['heldout_nrmse']['r2']['hybrid']:.3f} | "
      f"{rr['heldout_nrmse']['r3']['physics']:.3f} / {rr['heldout_nrmse']['r3']['data_driven']:.3f} / "
      f"{rr['heldout_nrmse']['r3']['hybrid']:.3f} |")
    w(f"| 95% coverage (held-out) | {rr['coverage95'][0]:.3f} | {rr['coverage95'][1]:.3f} |")
    w(f"| min steps/tau | {rr['min_steps_over_tau'][0]:.1f} | {rr['min_steps_over_tau'][1]:.1f} |")
    w(f"| Surrogate RMSE (mm) | {r2['surrogate_rmse']:.4f} (6000 PTU grid, unfiltered prior) | {sg['rmse']:.4f} (2000 PTU, filtered) |")
    w(f"| T* at 50% life | {_tstr(r2['T_star_PTU'])} (argmin at grid edge) | {_tstr(m['decision']['optimal_value'])} |")
    w(f"| Cost sweep T* range (PTU) | {r2['cost_sweep_range'][0]:.1f} to {r2['cost_sweep_range'][1]:.1f} | "
      f"{dc['cost_sweep_range'][0]:.1f} to {dc['cost_sweep_range'][1]:.1f} (argmin; interior count {dc['cost_sweep_interior_count']}) |")
    w(f"| Stressed action non-trivial | {_yn(r2['stressed_nontrivial'])} | {_yn(dc['stressed_scenario_action_nontrivial'])} |")
    w(f"| Runtime (s) | {r2['runtime_sec']:.0f} | {m['runtime_sec']:.0f} |")
    w("")
    w("**Changes in round 3:**")
    w("")
    for x in m["changes_r3"]:
        w(f"- {x}")
    w("")

    w("## 10. Known limitations")
    for x in m["limitations"]:
        w(f"- {x}")
    w("")
    w("## 11. Figures and runtime")
    w("`results/figures/fig01` to `fig09`: pseudo-trajectory, feature diagnostics, trend fits and model comparison, "
      "GP parity, Sobol, posterior narrowing, RUL distributions (circular truth), cost rate with c_f = 5, held-out band.")
    w("")
    w(f"Runtime: {m['runtime_sec']:.0f} s.")
    return "\n".join(lines) + "\n"


def _cmp_text(ci):
    if ci[1] < 0:
        return "first model better (CI below 0)"
    if ci[0] > 0:
        return "first model worse (CI above 0)"
    return "no distinguishable difference"


if __name__ == "__main__":
    main()
