"""p3_motor (round 3): LPTN digital twin of PMSM magnet (pm) temperature.

Pipeline: data inspection -> physics (LPTN) nominal fit + two out-of-fold physics fits -> held-out
comparison (physics / data-driven / hybrid with out-of-fold residual RF / extra models / baselines)
-> GP emulator (+ emulator-variance inflation) and Sobol -> AR(1) GLS calibration on held-out session
68 at 25/50/75 % (MAP start, DE-move sampler, burn-in 500, post-burn cap 3000) -> predictive band with
bias-free AR(1) model error -> nominal decision (headline) and a PRE-REGISTERED stress (limit 100 degC,
measured currents x1.2, coolant +10 K over the decision window), replicated on held-out session 12 with
the posterior of session 68.
"""
import os
import sys
import json
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import data_loader as dl
import physics_model as pm
import surrogate_model as sm
import uq_calibration as uq
import decision as dec
from data_loader import DT

# relative to the project folder: <repo>/04_Project/_data/p3_motor/measures_v2.csv (read-only)
DATA = os.path.normpath(os.path.join(HERE, "..", "_data", "p3_motor", "measures_v2.csv"))
FIG = os.path.join(HERE, "results", "figures")
METRICS = os.path.join(HERE, "results", "metrics.json")
SEED = 2026
STAGES = [0.25, 0.5, 0.75]
HEADLINE = 0.5                 # headline stage (50 % of the calibration session)
REPLICATE_PID = 12             # second held-out session, replicates the stress (posterior of 68 transferred)
N_POST = 300
PRIOR_STD = 1.0 / np.sqrt(12.0)   # uniform on [-0.5, 0.5]
C_THR_SWEEP = [5.0, 10.0, 20.0, 40.0, 60.0]
RUNTIME_BUDGET = 600.0
NOMINAL_TOL = 0.05             # pass/fail rule for predictive coverage: |coverage - nominal| <= 0.05

# ---- pre-registered criteria (fixed in code before any result of this round was produced) ----
# stress informative  <=>  0 < P(exceed | no action) < 1  AND  the cost-minimising policy is an action
#                         whose exceedance reduction exceeds the break-even dP(d, tau) (decision.breakeven_dP)
# predictive band ok  <=>  |coverage - nominal| <= NOMINAL_TOL at 50, 80 and 95 %
# runtime ok          <=>  runtime_sec <= RUNTIME_BUDGET
# MCMC converged      <=>  n_post >= 50 * tau for every parameter (reported as is; cap 3000 post-burn steps)


def log(msg):
    print(msg, flush=True)


def load_round2():
    with open(METRICS, encoding="utf-8") as fh:
        m = json.load(fh)
    assert m.get("round") == 2, "round-2 metrics.json needed for the r2_vs_r3 block"
    return m


def fig_data(profiles, train, test):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ids = sorted(profiles)
    hours = [profiles[i]["raw_len"] * dl.RAW_DT / 3600 for i in ids]
    cols = ["tab:blue" if i in train else "tab:orange" if i in test else "0.7" for i in ids]
    ax[0].bar(range(len(ids)), hours, color=cols)
    ax[0].set_xticks(range(len(ids)))
    ax[0].set_xticklabels([str(i) for i in ids], rotation=90, fontsize=6)
    ax[0].set_ylabel("session duration (h, 2 Hz nominal)")
    ax[0].set_title("Sessions: blue = train (8), orange = held-out (3), grey = unused")
    allpm = np.concatenate([profiles[i]["meas"][:, 3] for i in train + test])
    allc = np.concatenate([profiles[i]["coolant"] for i in train + test])
    ax[1].hist(allpm, bins=80, alpha=0.6, label="pm", density=True)
    ax[1].hist(allc, bins=80, alpha=0.6, label="coolant", density=True)
    ax[1].set_xlabel("deg C")
    ax[1].set_title("Temperature distributions (decimated, train + held-out)")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_data_overview.png"), dpi=150)
    plt.close(fig)


def fig_fit(prof, sim):
    t = prof["t"] / 3600
    fig, ax = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for j, a in enumerate(ax.ravel()):
        a.plot(t, prof["meas"][:, j], "k", lw=1, label="measured")
        a.plot(t, sim[:, j], "tab:blue", lw=1, label="LPTN free run (in-sample)")
        a.set_title(dl.NODE_NAMES[j])
        a.set_ylabel("deg C")
    ax[0, 0].legend()
    ax[1, 0].set_xlabel("h")
    ax[1, 1].set_xlabel("h")
    fig.suptitle("Nominal LPTN fit, training session %d" % prof["pid"])
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_physics_fit_train.png"), dpi=150)
    plt.close(fig)


def fig_heldout(test, phys, dd, hy, hyreg, pf):
    fig, ax = plt.subplots(len(test), 1, figsize=(12, 3.2 * len(test)), squeeze=False)
    for r, p in enumerate(test):
        a = ax[r, 0]
        t = p["t"] / 3600
        a.plot(t, p["meas"][:, 3], "k", lw=1.2, label="pm measured")
        a.plot(t, phys[r][:, 3], "tab:blue", lw=0.9, label="physics (free run)")
        a.plot(t, dd[r], "tab:orange", lw=0.9, label="data-driven (RF)")
        a.plot(t, hy[r], "tab:green", lw=0.9, label="hybrid (OOF residual RF)")
        a.plot(t, hyreg[r], "tab:olive", lw=0.9, ls="--", label="hybrid, regularised (OOF)")
        a.plot(t, pf[r], "tab:purple", lw=0.9, ls=":", label="physics-feature RF")
        a.axhline(dec.T_LIMIT, color="r", ls="--", lw=0.8)
        a.set_title("held-out session %d" % p["pid"])
        a.set_ylabel("pm, deg C")
        a.set_xlabel("h")
    ax[0, 0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_heldout_predictions.png"), dpi=150)
    plt.close(fig)


def fig_surrogate(val):
    Yt = val["Ytrue"].reshape(len(val["Ytrue"]), -1, 4)[:, :, 3].ravel()
    Ym = val["Ymean"].reshape(len(val["Ymean"]), -1, 4)[:, :, 3].ravel()
    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    ax.scatter(Yt, Ym, s=2, alpha=0.3)
    lo, hi = min(Yt.min(), Ym.min()), max(Yt.max(), Ym.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    ax.set_xlabel("LPTN ODE (truth), pm deg C")
    ax.set_ylabel("GP emulator mean, pm deg C")
    ax.set_title("GP emulator, 100 held-out LHS points\nR2=%.4f, 2-sigma coverage=%.2f (nominal 0.95)" %
                 (val["r2"], val["coverage_2sigma"]))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_surrogate_validation.png"), dpi=150)
    plt.close(fig)


def fig_sobol(sob):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for a, key in zip(ax, ["peak_pm", "time_to_exceed_min"]):
        names = pm.UNCERT
        x = np.arange(len(names))
        s1 = [sob[key][n]["S1"] for n in names]
        st = [sob[key][n]["ST"] for n in names]
        a.bar(x - 0.2, s1, 0.4, label="S1")
        a.bar(x + 0.2, st, 0.4, label="ST")
        a.set_xticks(x)
        a.set_xticklabels(names)
        a.set_title("Sobol indices (GP emulator): " + key.replace("_", " "))
        a.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_sobol.png"), dpi=150)
    plt.close(fig)


def fig_posterior(stage_res):
    fig, ax = plt.subplots(2, 3, figsize=(13, 7))
    for j, name in enumerate(pm.UNCERT):
        a = ax.ravel()[j]
        a.axvspan(-0.5, 0.5, color="0.93", zorder=0, label="prior box")
        for f, col in zip(STAGES, ["tab:blue", "tab:orange", "tab:green"]):
            ch = stage_res[f]["chain"][:, j]
            a.hist(ch, bins=40, range=(-0.5, 0.5), density=True, histtype="step", lw=1.6,
                   color=col, label="%d%% of session" % round(100 * f))
        a.set_title(name + " (log-multiplier)")
        a.set_xlim(-0.5, 0.5)
    ax.ravel()[0].legend(fontsize=8)
    fig.suptitle("AR(1) GLS posterior by stage (calibration on held-out session 68)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig6_posterior.png"), dpi=150)
    plt.close(fig)


def fig_calib(calib, stage_res):
    fig, ax = plt.subplots(len(STAGES), 1, figsize=(12, 3.4 * len(STAGES)), sharex=True)
    t = calib["t"] / 3600
    for a, f in zip(ax, STAGES):
        sr = stage_res[f]
        post = sr["post"]
        med = np.median(post["X"][:, :, pm.M], axis=0)
        a.plot(t, calib["meas"][:, pm.M], "k", lw=1, label="pm measured (not used in fit)")
        a.plot(t, med, "tab:blue", lw=1, label="posterior median (no action, no noise)")
        a.fill_between(t[post["steps"]], post["lo"], post["hi"], color="tab:blue", alpha=0.25,
                       label="95% predictive (parameters + AR(1) model error)")
        a.fill_between(t[post["steps"]], post["postonly"]["0.95"]["lo"], post["postonly"]["0.95"]["hi"],
                       color="tab:green", alpha=0.25, label="95% parameter-only band")
        a.axvspan(0, sr["t_c"] * DT / 3600, color="0.9", zorder=0, label="calibration window")
        a.axvspan(sr["t_c"] * DT / 3600, (sr["t_c"] + dec.H_STEPS) * DT / 3600, color="r", alpha=0.06,
                  zorder=0, label="30-min decision window")
        a.axhline(dec.T_LIMIT, color="r", ls="--", lw=0.8, label="pm limit")
        a.set_ylabel("deg C")
        a.set_title("calibrated at %d%% of session %d" % (round(100 * f), calib["pid"]))
    ax[-1].set_xlabel("h")
    ax[0].legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig7_calibration_predictive.png"), dpi=150)
    plt.close(fig)


def fig_decision(panels):
    """panels: list of (title, rows, cvec, best, cost) -> one axis per scenario."""
    fig, ax = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 5))
    for a, (title, rows, cvec, best, cost) in zip(np.atleast_1d(ax), panels):
        a.axhline(cvec[0], color="k", ls="--", label="no action (d=1)")
        for tm in dec.TAU_MIN:
            idx = [i for i, r in enumerate(rows) if r["act"] == 1 and r["tau_min"] == tm]
            a.plot([rows[i]["d"] for i in idx], cvec[idx], marker="o", ms=3,
                   label="derate from +%d min" % tm)
        a.plot([best["d"]], [cost], "r*", ms=14, label="optimum: d=%.2f, tau=%s min" %
               (best["d"], ("%g" % best["tau_min"]) if best["act"] else "none"))
        a.set_xlabel("derating factor d (current scale)")
        a.set_ylabel("expected cost per unit time (cost units / min)")
        a.set_title(title, fontsize=9)
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig8_decision.png"), dpi=150)
    plt.close(fig)


def policy_grid(rows, cvec):
    """Markdown table of the expected cost per minute of every derating policy (no-action = cvec[0])."""
    ds = [float(d) for d in dec.D_LEVELS if d < 1 - 1e-9]
    lut = {}
    for r, c in zip(rows, cvec):
        if r["act"] == 1:
            lut[(float(r["tau_min"]), round(float(r["d"]), 2))] = float(c)
    out = ["| start tau [min] \\ derating d | " + " | ".join("%.2f" % d for d in ds) + " |",
           "|---" * (len(ds) + 1) + "|"]
    for tm in dec.TAU_MIN:
        out.append("| %g | " % tm + " | ".join("%.3f" % lut[(float(tm), round(d, 2))] for d in ds) + " |")
    return out


def main():
    t_start = time.time()
    os.makedirs(FIG, exist_ok=True)
    rng = np.random.default_rng(SEED)
    OLD = load_round2()

    # ---------- 1. data -----------------------------------------------------------------
    log("[1] loading and inspecting CSV (%s) ..." % DATA)
    rep, profiles = dl.load_and_prepare(DATA)
    train_ids, test_ids = dl.split_profiles(profiles, SEED)
    assert REPLICATE_PID in test_ids
    train = [profiles[i] for i in train_ids]
    test = [profiles[i] for i in test_ids]
    calib = max(test, key=lambda p: len(p["i2"]))              # session 68
    log("    rows=%d profiles=%d contiguous_runs=%d nan=%d" %
        (rep["n_rows"], rep["n_profiles"], rep["n_contiguous_runs"], rep["n_nan_total"]))
    log("    train ids %s  held-out ids %s  calibration session %d" % (train_ids, test_ids, calib["pid"]))
    n_dec = sum(len(p["i2"]) for p in profiles.values())
    fig_data(profiles, train_ids, test_ids)

    # ---------- 2. physics: nominal fit + out-of-fold fits ------------------------------
    log("[2] LPTN nominal least-squares fit on training sessions, plus 2 session-grouped out-of-fold fits ...")
    sigma_node = np.vstack([p["meas"] for p in train]).std(axis=0)
    t0 = time.time()
    P_nom, fit = pm.fit_nominal(train, sigma_node)
    log("    nominal fit: status=%d nfev=%d (%.1f s)" % (fit.status, fit.nfev, time.time() - t0))
    phys_train = [sm.physics_states(P_nom, p) for p in train]
    rms_node = np.sqrt(np.mean(np.vstack([X - p["meas"] for X, p in zip(phys_train, train)]) ** 2, axis=0))
    order = np.random.default_rng(SEED + 9).permutation(len(train))
    half = len(train) // 2
    fold_A = [train[i] for i in order[:half]]
    fold_B = [train[i] for i in order[half:]]
    t0 = time.time()
    P_A, fit_A = pm.fit_nominal(fold_A, sigma_node)
    P_B, fit_B = pm.fit_nominal(fold_B, sigma_node)
    oof = {}
    for p in fold_A:
        oof[p["pid"]] = sm.physics_states(P_B, p)              # fold-B model predicts fold-A sessions
    for p in fold_B:
        oof[p["pid"]] = sm.physics_states(P_A, p)              # fold-A model predicts fold-B sessions
    phys_oof = [oof[p["pid"]] for p in train]
    rms_oof = np.sqrt(np.mean(np.vstack([X - p["meas"] for X, p in zip(phys_oof, train)]) ** 2, axis=0))
    log("    out-of-fold fits: folds %s / %s (%.1f s); OOF free-run RMSE per node (W,T,Y,M) %s vs in-sample %s" %
        (sorted(p["pid"] for p in fold_A), sorted(p["pid"] for p in fold_B), time.time() - t0,
         np.round(rms_oof, 3), np.round(rms_node, 3)))
    log("    train free-run RMSE per node (W,T,Y,M) [degC]: %s" % np.round(rms_node, 3))

    # error model: bias-free sd and AR(1) lag-1 correlation at 60 s, TRAINING residuals only
    res_nodes = [[X[:, j] - p["meas"][:, j] for X, p in zip(phys_train, train)] for j in range(4)]
    IF = np.array([max(uq.integrated_time(res_nodes[j]), 1.0) for j in range(4)])
    sd_bf = np.array([uq.bias_free_sd(res_nodes[j]) for j in range(4)])
    rho = np.array([uq.ar1_rho(res_nodes[j]) for j in range(4)])
    err_pm = dict(sd=float(sd_bf[3]), rho=float(rho[3]))
    log("    bias-free residual sd (W,T,Y,M) %s degC; AR(1) rho at 60 s %s; integrated time (diagnostic) %s" %
        (np.round(sd_bf, 3), np.round(rho, 3), np.round(IF, 1)))
    ref = pm.simulate_reference(P_nom, train[0], train[0]["meas"][0])
    scheme_vs_ivp = float(np.max(np.abs(phys_train[0] - ref)))
    log("    max |exact-step scheme - solve_ivp(LSODA)| on train session: %.4f degC" % scheme_vs_ivp)
    fig_fit(train[0], phys_train[0])

    # ---------- 3. held-out comparison --------------------------------------------------
    log("[3] held-out comparison: physics / data-driven / hybrid (OOF residual RF) / extra models / baselines ...")
    phys_test = [sm.physics_states(P_nom, p) for p in test]
    Xd_tr = np.vstack([sm.data_features(p) for p in train])
    y_tr = np.concatenate([p["meas"][:, 3] for p in train])
    rf_dd = sm.fit_rf(Xd_tr, y_tr, SEED)
    Xh_tr = np.vstack([np.column_stack([sm.data_features(p), X]) for p, X in zip(train, phys_oof)])
    r_tr = np.concatenate([p["meas"][:, 3] - X[:, 3] for p, X in zip(train, phys_oof)])
    rf_res = sm.fit_rf(Xh_tr, r_tr, SEED + 1)                              # hybrid, OOF residuals
    rf_reg = sm.fit_rf(Xh_tr, r_tr, SEED + 2, min_leaf=200, max_depth=4)   # regularised, OOF residuals
    rf_pf = sm.fit_rf(Xh_tr, y_tr, SEED + 3)                               # physics-feature RF, OOF features
    Xh_te = [np.column_stack([sm.data_features(p), X]) for p, X in zip(test, phys_test)]
    pred_phys = [X[:, 3] for X in phys_test]
    pred_dd = [rf_dd.predict(sm.data_features(p)) for p in test]
    pred_hy = [X[:, 3] + rf_res.predict(F) for X, F in zip(phys_test, Xh_te)]
    pred_hyreg = [X[:, 3] + rf_reg.predict(F) for X, F in zip(phys_test, Xh_te)]
    pred_pf = [rf_pf.predict(F) for F in Xh_te]
    const_mean = float(y_tr.mean())
    pred_const = [np.full(len(p["meas"]), const_mean) for p in test]
    L_max = max(len(p["meas"]) for p in train)
    pad = np.full((len(train), L_max), np.nan)
    for i, p in enumerate(train):
        pad[i, :len(p["meas"])] = p["meas"][:, 3]
    tcurve = np.nanmean(pad, axis=0)

    def time_curve(Lt):
        c = tcurve if Lt <= L_max else np.concatenate([tcurve, np.full(Lt - L_max, tcurve[-1])])
        return c[:Lt]
    pred_time = [time_curve(len(p["meas"])) for p in test]
    y_te = np.concatenate([p["meas"][:, 3] for p in test])

    def score(preds):
        e = np.concatenate([pr - p["meas"][:, 3] for pr, p in zip(preds, test)])
        r = float(np.sqrt(np.mean(e ** 2)))
        return r, r / float(y_te.std()), float(np.mean(e))

    main_models = {"physics": pred_phys, "data_driven": pred_dd, "hybrid": pred_hy}
    extra_models = {"hybrid_regularised_rf": pred_hyreg, "physics_feature_rf": pred_pf,
                    "baseline_constant_mean": pred_const, "baseline_time_only": pred_time}
    all_models = {**main_models, **extra_models}
    rm = {k: score(v) for k, v in main_models.items()}
    rm_extra = {k: score(v) for k, v in extra_models.items()}
    per_profile = {k: [float(np.sqrt(np.mean((pr - p["meas"][:, 3]) ** 2))) for pr, p in zip(v, test)]
                   for k, v in all_models.items()}
    for k, v in {**rm, **rm_extra}.items():
        log("    %-24s pooled held-out RMSE=%.3f degC  NRMSE=%.4f  bias=%+.2f degC" % (k, v[0], v[1], v[2]))
    fig_heldout(test, phys_test, pred_dd, pred_hy, pred_hyreg, pred_pf)

    # ---------- 4. GP emulator ----------------------------------------------------------
    log("[4] GP emulator on calibration session %d (7-D LHS design, 300 runs) ..." % calib["pid"])
    x0c = calib["meas"][0]
    L = len(calib["i2"])
    idx_out = sm.output_index(L)
    K = sm.K_OUT
    t0 = time.time()
    Xd = sm.lhs_design(300, SEED)
    Yd = sm.reference_outputs(P_nom, calib, x0c, Xd, idx_out)
    gp = sm.fit_gp(Xd, Yd, SEED)
    log("    GP (PCA, %d components, %.5f of output variance) fitted in %.1f s" %
        (gp.n_comp, gp.var_ratio_, time.time() - t0))
    val_pre = sm.validate(gp, P_nom, calib, x0c, idx_out)
    log("    before emulator-variance inflation: R2=%.5f coverage2s=%.3f" % (val_pre["r2"], val_pre["coverage_2sigma"]))
    Xc = sm.lhs_design(100, SEED + 2)                       # separate design for the inflation term
    Yc = sm.reference_outputs(P_nom, calib, x0c, Xc, idx_out)
    gp.calibrate_emulator_inflation(sm.to_unit(Xc), Yc)
    val = sm.validate(gp, P_nom, calib, x0c, idx_out)       # coverage on a third, unseen design
    infl_node = gp.inflation_var_.reshape(K, 4).mean(axis=0)
    sig2 = sd_bf[:3] ** 2 + infl_node[:3]                   # GLS variance per stator node (constant)
    log("    emulator: R2=%.5f RMSE=%.4f degC coverage2s=%.3f (50/80/95: %s) inflation scale=%.3f speed-up=%.0fx" %
        (val["r2"], val["rmse"], val["coverage_2sigma"],
         " / ".join("%.3f" % val["coverage_levels"][k] for k in ["0.5", "0.8", "0.95"]),
         gp.inflation_scale_, val["speedup_x"]))
    fig_surrogate(val)

    # ---------- 5. Sobol ----------------------------------------------------------------
    log("[5] Sobol indices on the GP emulator (N=1024, 6 uncertain parameters) ...")
    sob = uq.sobol_indices(gp, idx_out, L, N=1024)
    for key in sob:
        log("    %s: " % key + ", ".join("%s S1=%.3f ST=%.3f" % (p, sob[key][p]["S1"], sob[key][p]["ST"])
                                       for p in pm.UNCERT))
    fig_sobol(sob)

    # ---------- 6. calibration, propagation, decision at 25/50/75 % ---------------------
    stage_res = {}
    for f in STAGES:
        t_c = int(round(f * L))
        steps = np.arange(12, t_c, 12)
        Wi = np.array([np.interp(steps, idx_out, np.eye(K)[k]) for k in range(K)]).T
        y_obs = calib["meas"][steps, :3]
        logp = uq.make_logpost(gp, Wi, y_obs, sig2, rho[:3])
        t0 = time.time()
        th_map, lp_map = uq.map_estimate(logp)
        mc = uq.run_mcmc(logp, seed=SEED + int(100 * f), theta0=th_map)
        chain = mc["chain"]
        theta_s = chain[rng.choice(len(chain), N_POST, replace=False)]
        post = uq.posterior_predict(P_nom, theta_s, calib, x0c, t_c, err_pm, rng)
        prior = uq.posterior_predict(P_nom, rng.uniform(-0.5, 0.5, (N_POST, 6)), calib, x0c, t_c, err_pm, rng)
        rows = dec.policy_table(pm.theta_to_params(P_nom, theta_s), calib, x0c, t_c)
        best, cbest, cvec = dec.optimum(rows)
        stage_res[f] = dict(t_c=t_c, n_obs=len(steps), chain=chain, mc=mc, tau=mc["tau"], acc=mc["acc"],
                            theta=theta_s, map=th_map, map_lp=lp_map, post=post, prior=prior, rows=rows,
                            best=best, cost=cbest, cvec=cvec)
        log("[6] stage %d%%: %d obs, MAP logp=%.1f, MCMC %.1f s, %d post-burn steps, acc=%.3f, tau_max=%.1f, "
            "converged(>=50 tau)=%s, min steps/tau=%.1f" %
            (round(100 * f), len(steps), lp_map, time.time() - t0, mc["n_post"], mc["acc"], np.max(mc["tau"]),
             mc["converged"], np.min(mc["n_post"] / mc["tau"])))
        log("    pm cov95 pred=%.3f (param-only %.3f), bias=%+.2f degC, width95=%.2f degC, IS95=%.2f, "
            "P(pm>%g in 30 min|no action)=%.4f, P(pm>%g before session end|no action)=%.4f" %
            (post["pred"]["0.95"]["coverage"], post["postonly"]["0.95"]["coverage"], post["bias"],
             post["pred"]["0.95"]["width"], post["pred"]["0.95"]["interval_score"], dec.T_LIMIT,
             post["p_window"], dec.T_LIMIT, post["p_session"]))
        log("    decision (nominal): d*=%.2f  tau*=%s  cost rate=%.3f  (no-action cost rate=%.3f)" %
            (best["d"], ("%g" % best["tau_min"]) if best["act"] else "none", cbest, cvec[0]))
    fig_posterior(stage_res)
    fig_calib(calib, stage_res)

    # ---------- 6b. pre-registered stress, session 68 and replicate on session 12 --------
    hs = stage_res[HEADLINE]
    tcH = hs["t_c"]
    P_h = pm.theta_to_params(P_nom, hs["theta"])
    calib_st = dec.stress_profile(calib, tcH)
    rows_st = dec.policy_table(P_h, calib_st, x0c, tcH)
    best_st, c_st, cvec_st = dec.optimum(rows_st)
    s12 = profiles[REPLICATE_PID]
    t12 = int(round(HEADLINE * len(s12["i2"])))
    x012 = s12["meas"][0]
    rows12 = dec.policy_table(P_h, s12, x012, t12)
    best12, c12, cvec12 = dec.optimum(rows12)
    rows12_st = dec.policy_table(P_h, dec.stress_profile(s12, t12), x012, t12)
    best12_st, c12_st, cvec12_st = dec.optimum(rows12_st)
    post12 = uq.posterior_predict(P_nom, hs["theta"], s12, x012, t12, err_pm, rng)

    def informative(rows, best, c, cvec):
        p0 = rows[0]["p_exc"]
        pays = bool(best["act"] == 1 and c < cvec[0] - 1e-12)
        dP = float(p0 - best["p_exc"]) if best["act"] == 1 else 0.0
        be = dec.breakeven_dP(best["d"], best["tau_min"]) if best["act"] == 1 else float("nan")
        return dict(p_no_action=float(p0), informative=bool(0.0 < p0 < 1.0 and pays),
                    best_is_action=bool(best["act"] == 1), dP_best=dP, breakeven_best=be,
                    d_opt=float(best["d"]), tau_opt=(float(best["tau_min"]) if best["act"] else None),
                    cost=float(c), no_action_cost=float(cvec[0]))

    inf68 = informative(rows_st, best_st, c_st, cvec_st)
    inf12 = informative(rows12_st, best12_st, c12_st, cvec12_st)
    r08 = dec.find_row(rows_st, 0.0, 0.8)
    dP08 = float(rows_st[0]["p_exc"] - r08["p_exc"])
    be08 = float(dec.breakeven_dP(0.8, 0.0))
    log("[6b] PRE-REGISTERED stress (limit %g degC, currents x%.1f, coolant +%g K over the 30-min window), 50%% stage:"
        % (dec.T_LIMIT, dec.STRESS_CURRENT_SCALE, dec.STRESS_COOLANT_K))
    log("    session 68 (calibration): P(exceed|no action)=%.3f, d*=%.2f tau*=%s cost=%.3f (no-action %.3f), informative=%s"
        % (inf68["p_no_action"], inf68["d_opt"], inf68["tau_opt"], inf68["cost"], inf68["no_action_cost"], inf68["informative"]))
    log("    session 12 (replicate, posterior of 68): P(exceed|no action)=%.3f, d*=%.2f tau*=%s cost=%.3f (no-action %.3f), informative=%s"
        % (inf12["p_no_action"], inf12["d_opt"], inf12["tau_opt"], inf12["cost"], inf12["no_action_cost"], inf12["informative"]))
    log("    break-even dP at d=0.8, tau=0 (from code constants) = %.4f; stress dP at d=0.8, tau=0 (session 68) = %.4f"
        % (be08, dP08))
    log("    session 12 nominal: P(exceed|no action)=%.3f; posterior-68 forecast coverage95 on session 12 = %.3f"
        % (rows12[0]["p_exc"], post12["pred"]["0.95"]["coverage"]))

    panels = [("nominal, limit %g degC, %d%% stage, session %d" % (dec.T_LIMIT, round(100 * HEADLINE), calib["pid"]),
               hs["rows"], hs["cvec"], hs["best"], hs["cost"]),
              ("stress (pre-reg.), limit %g, session %d" % (dec.T_LIMIT, calib["pid"]),
               rows_st, cvec_st, best_st, c_st),
              ("stress (pre-reg.), limit %g, replicate session %d" % (dec.T_LIMIT, REPLICATE_PID),
               rows12_st, cvec12_st, best12_st, c12_st)]
    fig_decision(panels)

    # sensitivity of the optimum to C_thr (policy probabilities are cost-independent)
    sens = {}
    for name, rows in [("nominal", hs["rows"]), ("stress", rows_st)]:
        sens[name] = {}
        for c_thr in C_THR_SWEEP:
            b, c, _ = dec.optimum(rows, c_thr=c_thr)
            sens[name][str(c_thr)] = dict(d_opt=float(b["d"]), tau_opt_min=(float(b["tau_min"]) if b["act"] else None),
                                          cost=c, act=int(b["act"]))
            log("    sensitivity (%s, session %d) C_thr=%g -> d*=%.2f tau*=%s cost=%.3f" %
                (name, calib["pid"], c_thr, b["d"], b["tau_min"] if b["act"] else "none", c))

    # ---------- 7. metrics.json ----------------------------------------------------------
    s75 = stage_res[0.75]
    ratio75 = {p: float(np.std(s75["chain"][:, j]) / PRIOR_STD) for j, p in enumerate(pm.UNCERT)}
    cal_cov95 = float(np.mean([stage_res[f]["post"]["pred"]["0.95"]["coverage"] for f in STAGES]))
    cov_pooled = {lvl: float(np.mean([stage_res[f]["post"]["pred"][lvl]["coverage"] for f in STAGES]))
                  for lvl, _, _ in uq.LEVELS}
    is95 = float(np.mean([stage_res[f]["post"]["pred"]["0.95"]["interval_score"] for f in STAGES]))
    cov_postonly = {lvl: float(np.mean([stage_res[f]["post"]["postonly"][lvl]["coverage"] for f in STAGES]))
                    for lvl, _, _ in uq.LEVELS}
    width95 = float(np.mean([stage_res[f]["post"]["pred"]["0.95"]["width"] for f in STAGES]))
    prior_cov = {lvl: float(np.mean([stage_res[f]["prior"]["pred"][lvl]["coverage"] for f in STAGES]))
                 for lvl, _, _ in uq.LEVELS}
    prior_width95 = float(np.mean([stage_res[f]["prior"]["pred"]["0.95"]["width"] for f in STAGES]))
    cov12 = {lvl: float(post12["pred"][lvl]["coverage"]) for lvl, _, _ in uq.LEVELS}
    corr75 = np.corrcoef(s75["chain"].T)
    ridge_pairs = []
    for i in range(len(pm.UNCERT)):
        for j in range(i + 1, len(pm.UNCERT)):
            if abs(corr75[i, j]) > 0.7:
                ridge_pairs.append([pm.UNCERT[i], pm.UNCERT[j], float(corr75[i, j])])
    n_prior_like = int(sum(1 for p in pm.UNCERT if ratio75[p] >= 0.8))
    cov_ok = all(abs(cov_pooled[k] - nom) <= NOMINAL_TOL for k, nom in [("0.5", 0.5), ("0.8", 0.8), ("0.95", 0.95)])
    runtime_partial = time.time() - t_start
    interior = bool(best_st["act"] == 1 and best_st["d"] > float(dec.D_LEVELS.min()) + 1e-9
                    and best_st["tau_min"] < float(dec.TAU_MIN.max()))
    sweep_costs = [sens["stress"][k]["cost"] for k in sens["stress"]]
    nontrivial = bool(best_st["act"] == 1 and c_st < cvec_st[0] - 1e-12)
    risk_check = abs(hs["rows"][0]["p_exc"] - hs["post"]["p_window"]) < 1e-12
    mcmc_by_stage = {str(round(100 * f)): {"tau_per_param": {p: float(stage_res[f]["tau"][j]) for j, p in enumerate(pm.UNCERT)},
                                           "n_post": int(stage_res[f]["mc"]["n_post"]),
                                           "min_steps_over_tau": float(np.min(stage_res[f]["mc"]["n_post"] / stage_res[f]["tau"])),
                                           "ess_min": float(np.min(stage_res[f]["mc"]["ess"])),
                                           "acceptance": float(stage_res[f]["acc"]),
                                           "converged_50tau": bool(stage_res[f]["mc"]["converged"])}
                     for f in STAGES}
    n_conv = int(sum(1 for f in STAGES if stage_res[f]["mc"]["converged"]))
    min_steps_75 = float(np.min(s75["mc"]["n_post"] / s75["tau"]))
    limitations = [
        "Data: one public test-bench dataset (69 sessions, 2 Hz nominal, no timestamp column). Time axis is "
        "row order at an assumed 2 Hz; not verified against a clock.",
        "Units: values are NOT standardised/normalised as the brief suggested (speed in rpm, currents in A, "
        "temperatures in degC); no rescaling was applied beyond the train-only statistics for sigma.",
        "Split: 8 training and 3 held-out sessions (2, 12, 68; lengths >= 2.36 h); with so few sessions the physics "
        "fit, the out-of-fold physics fits (4 sessions each) and the residual/data-driven models are weakly constrained.",
        "Perfect foresight of inputs: the forecasts and the decision use the MEASURED future torque, speed, currents, "
        "coolant and ambient of the session. The decision is therefore conditional on known future load, which a "
        "real controller does not have; the derating benefit is an upper bound on what a forecast-based policy could achieve.",
        "Session 68 is used for calibration (MCMC, emulator, Sobol) AND as the forecast session for the coverage and the "
        "decision; it is not an independent test set. Session 12 (not used for calibration) is the transfer check: "
        "it applies the posterior of session 68 to a different session and gives a single additional forecast.",
        "Forecasts are initialised at the decision instant with the MEASURED stator temperatures at that instant (the "
        "magnet temperature is not measured and comes from the model). The calibration itself is a free run from the "
        "session start; the calibration likelihood does not use the stator observations to re-initialise the state.",
        "Decimation to 5 s: fine for the thermal time constants used (dt/tau < 1 for the winding), inputs averaged per step; "
        "the matrix-exponential step is stable but first-order accurate in the input averaging.",
        "LPTN is a 4-node simplification (no separate rotor/air-gap/end-winding nodes); no magnet heat from eddy "
        "currents beyond k_m n^2; iron loss is a speed polynomial without flux dependence.",
        "Copper temperature coefficient alpha_cu fixed at 0.0039 1/K; winding feedback uses the previous step (explicit).",
        "Initial state of every node is taken from the measurement at session start, including the magnet temperature "
        "(an assumption: in practice the magnet temperature is not measured; this favours the physics model).",
        "Likelihood: AR(1) GLS on the three stator nodes with independent channels; rho is estimated from training residuals "
        "at 60 s and sigma is the bias-free training residual sd. The cross-node correlation is ignored. The GLS posterior is "
        "conditional on this error model; if the residuals are not AR(1), the posterior is overconfident (see coverage).",
        "Emulator-variance inflation: per-output variance added to the emulator's own predictive variance, estimated on a "
        "separate 100-point design at one calibration session; it is an emulator-error scale, not a model-discrepancy term, and "
        "it is a constant in the likelihood (its magnitude is far below the observation variance).",
        "Predictive pm band: bias-free AR(1) model error (sd and rho from training residuals only). The posterior-median bias "
        "of the no-action forecast is NOT included in the band, so coverage is limited by that bias.",
        "Interval coverage is pooled over the three calibration stages of session 68, which share the same session and are "
        "not independent; the session-12 transfer coverage is a single forecast per level.",
        "Stress decision: the limit is 100 degC (chosen operating limit, not a manufacturer rating); the stress (currents x1.2, "
        "coolant +10 K over the window) is pre-registered and not calibrated to make the action non-trivial. Cost constants are "
        "illustrative; no economic data was provided.",
        "Break-even rule: the review's ~0.017 break-even at d=0.8, tau=0 does not reproduce from the code constants; the value "
        "computed from them is reported and used.",
        "Derating acts on the current scale; the torque and copper-loss reduction are modelled, the speed-dependent iron "
        "loss is unchanged.",
        "Held-out evaluation uses only 3 sessions; the RF, regularised hybrid, physics-feature RF and baseline comparisons "
        "carry large uncertainty.",
        "MCMC: MAP start, DE-move / DE-snooker mixture, burn-in 500 and a post-burn cap of 3000 steps for every stage (runtime "
        "budget of 600 s). The 50-tau criterion is reported per stage; where it is false, the chain is too short for it.",
    ]
    if not cov_ok:
        limitations.append("Predictive coverage of the 50/80/95 bands differs from nominal by more than %g (see section 7); "
                           "this is reported as a failure, not corrected." % NOMINAL_TOL)
    metrics = {
        "project": "p3_motor",
        "n_rows": int(rep["n_rows"]),
        "target": "pm",
        "target_unit": "degC",
        "has_real_time_axis": False,
        "has_unit_ids": True,
        "data_synthetic_suspected": False,
        "heldout_rmse": {k: float(rm[k][0]) for k in main_models},
        "heldout_nrmse": {k: float(rm[k][1]) for k in main_models},
        "heldout_nrmse_extra": {k: float(rm_extra[k][1]) for k in rm_extra},
        "surrogate": {"r2": float(val["r2"]), "rmse": float(val["rmse"]),
                      "coverage_2sigma": float(val["coverage_2sigma"]), "speedup_x": float(val["speedup_x"])},
        "sobol": {p: {"S1": float(sob["peak_pm"][p]["S1"]), "ST": float(sob["peak_pm"][p]["ST"])} for p in pm.UNCERT},
        "calibration": {"posterior_to_prior_std_ratio": ratio75,
                        "mean_autocorr_time": float(np.mean(s75["tau"])),
                        "acceptance_fraction": float(s75["acc"]),
                        "coverage95_heldout": cal_cov95},
        "risk": {"description": "P(pm > %g degC at any time within the 30-min decision window [t_c, t_c+30 min]), "
                                "no action, nominal limit, posterior samples, headline stage %d%% of held-out session %d "
                                "(same window and function as the decision layer). P(exceed before session end) is in key "
                                "risk_p_session_end." % (dec.T_LIMIT, round(100 * HEADLINE), calib["pid"]),
                 "value": float(hs["post"]["p_window"])},
        "decision": {"description": "NOMINAL scenario (headline; limit %g degC, %d%% stage, session %d). Optimal derating "
                                    "(current scale d, start tau after calibration) minimising expected cost per unit time: "
                                    "C_act*1[d<1] + C_thr*(1-d)*(H-tau) + C_fail*P(pm>limit), H=%g min, C_fail=%g, C_thr=%g, "
                                    "C_act=%g; optimal start tau*=%s min. Stress scenario: decision_stress_prereg."
                                    % (dec.T_LIMIT, round(100 * HEADLINE), calib["pid"], dec.H_MIN, dec.C_FAIL, dec.C_THR,
                                       dec.C_ACT, ("%g" % hs["best"]["tau_min"]) if hs["best"]["act"] else "no derating"),
                     "optimal_value": float(hs["best"]["d"]),
                     "unit": "derating factor d (fraction of nominal current scale)",
                     "expected_cost": float(hs["cost"])},
        "runtime_sec": float(runtime_partial),
        "limitations": limitations,
        # ----- round 2 keys (kept) -----
        "mcmc": {"tau_per_param": {p: float(s75["tau"][j]) for j, p in enumerate(pm.UNCERT)},
                 "n_steps": int(s75["mc"]["n_post"]),
                 "min_steps_over_tau": min_steps_75,
                 "ess_min": float(np.min(s75["mc"]["ess"])),
                 "converged_50tau": bool(s75["mc"]["converged"]),
                 "stage": "75%"},
        "interval_calibration": {"coverage": cov_pooled, "mean_width95": width95,
                                 "interval_score_95": is95,
                                 "bias_pm_degC": float(np.mean([stage_res[f]["post"]["bias"] for f in STAGES])),
                                 "posterior_only_coverage": cov_postonly,
                                 "prior_predictive_baseline": {"coverage": prior_cov, "mean_width95": prior_width95},
                                 "nominal_tolerance": NOMINAL_TOL,
                                 "pass": bool(cov_ok),
                                 "session12_transfer": {"coverage": cov12, "interval_score_95": float(post12["pred"]["0.95"]["interval_score"]),
                                                        "bias_pm_degC": float(post12["bias"])}},
        "decision_checks": {"optimum_interior": interior,
                            "cost_sweep_range": [float(min(sweep_costs)), float(max(sweep_costs))],
                            "stressed_scenario_action_nontrivial": nontrivial,
                            "scenario": "pre-registered stress (limit %g degC, currents x%.1f, coolant +%g K), session %d, 50%% stage"
                                         % (dec.T_LIMIT, dec.STRESS_CURRENT_SCALE, dec.STRESS_COOLANT_K, calib["pid"]),
                            "risk_matches_policy_noaction": bool(risk_check),
                            "stress_informative_session68": inf68["informative"],
                            "stress_informative_session12": inf12["informative"],
                            "breakeven_dP_d08_tau0": be08,
                            "stress_dP_d08_tau0_session68": dP08},
        "round": 3,
        # ----- round 3 additions -----
        "changes_r3": CHANGES_R3,
        "r2_vs_r3": {"heldout_nrmse": {"r2": OLD["heldout_nrmse"], "r3": {k: float(rm[k][1]) for k in main_models}},
                     "coverage95": [float(OLD["calibration"]["coverage95_heldout"]), cal_cov95],
                     "min_steps_over_tau": [float(OLD["mcmc"]["min_steps_over_tau"]), min_steps_75]},
        "mcmc_by_stage": mcmc_by_stage,
        "mcmc_n_converged_stages": n_conv,
        "map": {str(round(100 * f)): {"theta": [float(x) for x in stage_res[f]["map"]], "logpost": float(stage_res[f]["map_lp"])}
                for f in STAGES},
        "error_model": {"sd_bias_free_degC_WTYM": [float(x) for x in sd_bf],
                        "rho_60s_WTYM": [float(x) for x in rho],
                        "pm_predictive": err_pm,
                        "likelihood_sigma2_stator": [float(x) for x in sig2],
                        "emulator_inflation_mean_var_stator": [float(x) for x in infl_node[:3]],
                        "integrated_time_60s_diagnostic_WTYM": [float(x) for x in IF],
                        "raw_rms_degC_WTYM": [float(x) for x in rms_node]},
        "surrogate_extra": {"coverage_2sigma_before_inflation": float(val_pre["coverage_2sigma"]),
                            "r2_before_inflation": float(val_pre["r2"]),
                            "coverage_levels_after_inflation": val["coverage_levels"],
                            "inflation_scale_factor": float(gp.inflation_scale_),
                            "pca_variance_ratio": float(gp.var_ratio_)},
        "posterior_correlation": {"order": list(pm.UNCERT), "matrix_75pct": [[float(x) for x in r] for r in corr75],
                                  "ridge_pairs_abs_gt_0p7": ridge_pairs},
        "prior_like_count_ratio_ge_0p8": n_prior_like,
        "decision_nominal": {"limit_degC": float(dec.T_LIMIT), "optimal_value": float(hs["best"]["d"]),
                             "tau_min": (float(hs["best"]["tau_min"]) if hs["best"]["act"] else None),
                             "expected_cost": float(hs["cost"]), "no_action_cost": float(hs["cvec"][0]),
                             "p_window_no_action": float(hs["post"]["p_window"])},
        "decision_stress_prereg": {"limit_degC": float(dec.T_LIMIT), "current_scale": dec.STRESS_CURRENT_SCALE,
                                   "coolant_offset_K": dec.STRESS_COOLANT_K, "window_min": dec.H_MIN,
                                   "stage": round(100 * HEADLINE), "session68": inf68, "session12_replicate": inf12,
                                   "breakeven_rule": "dP > (C_act + C_thr*(1-d)*(H-tau))/C_fail"},
        "decision_sensitivity": sens,
        "risk_p_session_end": float(hs["post"]["p_session"]),
        "heldout_extra": {"nrmse": {k: float(v[1]) for k, v in rm_extra.items()},
                          "rmse": {k: float(v[0]) for k, v in rm_extra.items()},
                          "bias_degC": {**{k: float(v[2]) for k, v in rm.items()},
                                        **{k: float(v[2]) for k, v in rm_extra.items()}}},
        "oof_physics_rmse_WTYM": [float(x) for x in rms_oof],
    }
    # ---------- 8. write files ------------------------------------------------------------
    runtime = time.time() - t_start
    metrics["runtime_sec"] = float(runtime)
    with open(METRICS, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else o.tolist())
    write_summary(dict(rep=rep, train_ids=train_ids, test_ids=test_ids, P_nom=P_nom, rms_node=rms_node,
                       rms_oof=rms_oof, scheme_vs_ivp=scheme_vs_ivp, rm=rm, rm_extra=rm_extra, per_profile=per_profile,
                       val=val, val_pre=val_pre, sob=sob, stage_res=stage_res, calib=calib, sens=sens, gp=gp,
                       metrics=metrics, runtime=runtime, n_dec=n_dec, fit=fit, fit_A=fit_A, fit_B=fit_B,
                       sig2=sig2, infl_node=infl_node, sd_bf=sd_bf, rho=rho, IF=IF, err_pm=err_pm,
                       rows12=rows12, rows12_st=rows12_st, best12=best12, best12_st=best12_st,
                       c12=c12, c12_st=c12_st, cvec12=cvec12, cvec12_st=cvec12_st,
                       rows_st=rows_st, best_st=best_st, c_st=c_st, cvec_st=cvec_st,
                       inf68=inf68, inf12=inf12, dP08=dP08, be08=be08, post12=post12, cov12=cov12,
                       ratio75=ratio75, corr75=corr75, ridge_pairs=ridge_pairs, n_prior_like=n_prior_like,
                       cov_pooled=cov_pooled, cov_postonly=cov_postonly, prior_cov=prior_cov,
                       width95=width95, prior_width95=prior_width95, is95=is95, cal_cov95=cal_cov95,
                       interior=interior, nontrivial=nontrivial, risk_check=risk_check, n_conv=n_conv,
                       min_steps_75=min_steps_75, cov_ok=cov_ok, s75=s75, hs=hs, t12=t12, L12=len(s12["i2"])))
    log("done in %.1f s; wrote results/metrics.json, results/summary.md, 8 figures" % runtime)


CHANGES_R3 = [
    "Stress replaced: the limit is no longer the posterior median (T_STRESS removed). The pre-registered stress is limit 100 degC, "
    "measured currents x1.2 and coolant +10 K, applied only over the 30-min decision window, on session 68 and replicated on session 12.",
    "Headline decision block (metrics decision) is the NOMINAL scenario at the 50% stage; the stress is in decision_stress_prereg.",
    "Informativeness: the break-even dP is computed from the code constants. At d=0.8, tau=0 it is 0.125, not the review's ~0.017 "
    "(the review value does not reproduce from C_fail=1000, C_thr=20, C_act=5, H=30); the computed value is used.",
    "MCMC: MAP start (L-BFGS-B), emcee DE-move 0.8 / DE-snooker 0.2 mixture, burn-in 500, blocks of 500, post-burn cap 3000. "
    "converged_50tau is reported truthfully per stage; the 6000-step cap of round 2 is gone.",
    "Likelihood: the sqrt(integrated-time) variance inflation (round 2) is replaced by an AR(1) GLS likelihood on the three stator "
    "nodes (rho and bias-free sd from training residuals at 60 s). The emulator variance is a constant term in sigma^2.",
    "GP term renamed emulator-variance inflation; its scale factor and 50/80/95 emulator coverage are reported. It is not "
    "described as model discrepancy and not as a calibration improvement.",
    "Predictive band: bias-free AR(1) model error (sd, rho on training residuals only) instead of i.i.d. RMS noise. Coverage at "
    "50/80/95, interval score 95, parameter-only and prior-predictive baselines reported. The 'calibration adds' claim is removed.",
    "Hybrid residual RF trained on out-of-fold residuals: two session-grouped folds of the 8 training sessions, each LPTN fitted on "
    "the other fold (both physics features and residual target are out of fold). Test-time physics is the nominal model.",
    "Baselines added to the held-out table: constant mean and time-only (training mean curve).",
    "Forecast initialised at the decision instant with the measured stator temperatures (magnet from the model); the same state is "
    "used by the policy table and the predictive band.",
    "Runtime: GP restarts 2 -> 1, GP used as mean only inside the MCMC (its variance is a constant term).",
    "Limitations added: perfect-foresight of future inputs; session 68 used for calibration and forecast; session 12 transfer check.",
]


def write_summary(C):
    rep, calib, metrics, stage_res = C["rep"], C["calib"], C["metrics"], C["stage_res"]
    rm, rm_extra, val, val_pre, sob = C["rm"], C["rm_extra"], C["val"], C["val_pre"], C["sob"]
    hs = stage_res[HEADLINE]
    f = lambda x: "%.4g" % x
    r = rep["ranges"]
    mc = metrics
    names = ["winding", "tooth", "yoke", "magnet"]
    lines = []
    a = lines.append
    cov = mc["interval_calibration"]["coverage"]
    c12 = mc["interval_calibration"]["session12_transfer"]
    i68, i12 = C["inf68"], C["inf12"]
    rt = C["runtime"]
    best = mc["decision"]
    st_desc = lambda inf: ("derate d=%.2f from +%s min" % (inf["d_opt"], ("%g" % inf["tau_opt"]) if inf["tau_opt"] is not None else "0"))\
        if inf["best_is_action"] else "no action"
    surr_ok = abs(val["coverage_2sigma"] - 0.95) <= 0.03
    hy, dd, ph = rm["hybrid"][1], rm["data_driven"][1], rm["physics"][1]

    a("# p3_motor: LPTN digital twin of PMSM magnet temperature (round 3)")
    a("")
    a("Runtime %.0f s (budget %g s: %s). Target: pm (magnet) temperature, deg C. Nominal limit %g degC. Every number below is written by "
      "main.py from computed values (metrics.json holds the same numbers)." %
      (rt, RUNTIME_BUDGET, "PASS" if rt <= RUNTIME_BUDGET else "FAIL", dec.T_LIMIT))
    a("")
    a("## 0. Verdict (5 lines, failures included)")
    a("1. Held-out pm NRMSE: physics %.3f, data-driven %.3f, hybrid with out-of-fold residual RF %.3f. The hybrid %s the "
      "data-driven model; the three held-out sessions give weak evidence for any ranking." %
      (ph, dd, hy, "beats" if hy < dd else "does NOT beat"))
    a("2. Predictive band (session %d, 3 stages pooled, bias-free AR(1) model error): coverage 50/80/95 = %.3f / %.3f / %.3f "
      "vs nominal 0.50 / 0.80 / 0.95 (%s; tolerance %g). Mean posterior-median bias %+.2f degC is outside the band; interval score 95 = %.2f."
      % (calib["pid"], cov["0.5"], cov["0.8"], cov["0.95"], "PASS" if mc["interval_calibration"]["pass"] else "FAIL",
         NOMINAL_TOL, mc["interval_calibration"]["bias_pm_degC"], C["is95"]))
    a("3. Pre-registered stress (limit %g degC, currents x1.2, coolant +10 K, 50%% stage): session %d P(exceed | no action) = %.3f, "
      "optimum %s, informative = %s; replicate session %d P = %.3f, optimum %s, informative = %s. Break-even dP at d=0.8, tau=0 = %.3f "
      "(stress dP = %.3f); the review's ~0.017 does not reproduce from the constants."
      % (dec.T_LIMIT, calib["pid"], i68["p_no_action"], st_desc(i68), i68["informative"],
         REPLICATE_PID, i12["p_no_action"], st_desc(i12), i12["informative"], C["be08"], C["dP08"]))
    a("4. MCMC (AR(1) GLS, MAP start, DE-move mixture): converged (>= 50 tau) in %d of 3 stages; the 75%% stage has min steps/tau = %.1f. "
      "Runtime is %s. Where the 50-tau rule fails, the posterior is a short-chain approximation and is reported as such."
      % (C["n_conv"], C["min_steps_75"], "within the 600 s budget" if rt <= RUNTIME_BUDGET else "OVER the 600 s budget"))
    a("5. Surrogate 2-sigma coverage %.3f after emulator-variance inflation (scale x%.2f on the emulator sd; %.3f before). The inflation "
      "is not a calibration improvement and not model discrepancy. Forecasts use the measured future inputs (perfect foresight), and session "
      "68 is both the calibration and the forecast session; session 12 is the only transfer check."
      % (val["coverage_2sigma"], C["metrics"]["surrogate_extra"]["inflation_scale_factor"], val_pre["coverage_2sigma"]))
    a("")
    a("## Changes vs round 2")
    for s in mc["changes_r3"]:
        a("- " + s)
    a("")
    a("## 1. Data inspection")
    a("- File: `_data/p3_motor/measures_v2.csv` (outside the project folder, read-only), %d rows x %d columns, %d NaN, %d duplicate rows." %
      (rep["n_rows"], rep["n_cols"], rep["n_nan_total"], rep["n_duplicate_rows"]))
    a("- Columns: %s." % ", ".join(rep["columns"]))
    a("- Sessions: %d `profile_id`, %d contiguous runs (rows of a session are contiguous). Raw length min/median/max: %s samples." %
      (rep["n_profiles"], rep["n_contiguous_runs"], rep["profile_raw_len_min_median_max"]))
    a("- Ranges (min, max): " + "; ".join("%s [%s, %s]" % (c, f(r[c][0]), f(r[c][1])) for c in
                                         ["i_d", "i_q", "motor_speed", "torque", "coolant", "ambient", "stator_winding", "pm"]) + ".")
    a("- Units check: the brief said torque/speed/currents are normalised. They are not: speed spans %s to %s rpm, "
      "temperatures are plausible degC (ambient %s to %s, pm max %s). No standardisation was applied." %
      (f(r["motor_speed"][0]), f(r["motor_speed"][1]), f(r["ambient"][0]), f(r["ambient"][1]), f(r["pm"][1])))
    a("- Time axis: no timestamp column. Row order is chronological (pm lag-1 autocorrelation %.4f, 99th percentile "
      "of |pm step| per raw sample %s degC). A 2 Hz sampling rate is assumed from the dataset documentation, not verified. "
      "`has_real_time_axis` is set to false." % (rep["pm_lag1_autocorr"], f(rep["pm_abs_step_p99_per_raw_sample"])))
    a("- Unit IDs: `profile_id` is a measurement session. Splits are by session (no leakage).")
    a("- High-frequency sensor noise estimate (MAD of second differences, degC): " +
      ", ".join("%s %s" % (k, f(v)) for k, v in rep["hf_noise_std_est_degC"].items()) + ".")
    a("- Synthetic structure: none suspected, `data_synthetic_suspected` = false.")
    a("- Decimation: every 10th raw sample (%.0f s step), %d decimated rows in total." % (DT, C["n_dec"]))
    a("")
    a("## 2. Split")
    a("- Training (8 sessions): %s." % C["train_ids"])
    a("- Held-out (3 sessions): %s. Random choice with seed %d. Replicate session for the stress: %d." % (C["test_ids"], SEED, REPLICATE_PID))
    a("- Calibration session for emulator, MCMC and decision: held-out session %d (longest, %d decimated steps)." % (calib["pid"], len(calib["i2"])))
    a("")
    a("## 3. Physics model (LPTN)")
    a("- Nodes: winding W, stator tooth T, stator yoke Y, magnet M. Boundaries: coolant (to W and Y), ambient (to Y).")
    a("- Copper loss k_cu d^2 (i_d^2+i_q^2)(1+0.0039(T_W-20)); iron loss k_fe1|n|+k_fe2 n^2 (50/50 T/Y); magnet loss k_m n^2.")
    a("- Integration: exact matrix-exponential step on the 5 s grid, inputs averaged per step, winding feedback lagged by one step.")
    a("- Check against `scipy.integrate.solve_ivp` (LSODA, rtol 1e-6): max abs. deviation on the first training session = %s degC." % f(C["scheme_vs_ivp"]))
    a("- Nominal parameters: least_squares on log-parameters, free-run residuals of all four nodes on the 8 training sessions "
      "(status %d, %d function evaluations). Out-of-fold fits (4 sessions each, status %d and %d) are used only for the hybrid residual target."
      % (C["fit"].status, C["fit"].nfev, C["fit_A"].status, C["fit_B"].status))
    a("")
    a("| parameter | nominal value |")
    a("|---|---|")
    from physics_model import PARAM_NAMES
    for nme, v in zip(PARAM_NAMES, C["P_nom"]):
        a("| %s | %s |" % (nme, f(v)))
    a("")
    a("- Training in-sample free-run RMSE per node (winding, tooth, yoke, magnet) [degC]: %s." % ", ".join(f(x) for x in C["rms_node"]))
    a("- Out-of-fold free-run RMSE per node on the training sessions [degC]: %s." % ", ".join(f(x) for x in C["rms_oof"]))
    a("")
    a("## 4. Held-out comparison (pm, free run from session start, 3 sessions pooled)")
    a("| model | RMSE [degC] | NRMSE (RMSE/std) | bias [degC] | per-session RMSE |")
    a("|---|---|---|---|---|")
    for k in ["physics", "data_driven", "hybrid"]:
        a("| %s | %s | %s | %+.2f | %s |" % (k, f(rm[k][0]), f(rm[k][1]), rm[k][2],
                                             ", ".join(f(x) for x in C["per_profile"][k])))
    for k in ["hybrid_regularised_rf", "physics_feature_rf", "baseline_constant_mean", "baseline_time_only"]:
        a("| %s (extra) | %s | %s | %+.2f | %s |" % (k, f(rm_extra[k][0]), f(rm_extra[k][1]), rm_extra[k][2],
                                                      ", ".join(f(x) for x in C["per_profile"][k])))
    a("")
    a("- Data-driven: random forest on measured inputs and exponentially weighted averages of the losses and coolant/ambient (no temperatures, no initial state).")
    a("- Hybrid: physics free run (nominal, fitted on all 8 training sessions) plus RF residual trained on OUT-OF-FOLD residuals of the physics model "
      "(2 session-grouped folds). Regularised hybrid: same target, min_samples_leaf 200, max_depth 4. Physics-feature RF: RF on the same OOF features predicting pm.")
    a("- Baselines: constant mean (training mean of pm) and time-only (mean training pm trajectory against time since session start).")
    a("- The physics model receives the measured initial state of all nodes at session start; the other models do not.")
    a("")
    hr_n = rm_extra["hybrid_regularised_rf"][1]
    pf_n = rm_extra["physics_feature_rf"][1]
    a("**Hybrid (NRMSE %s) vs RF (%s) and physics (%s).** The residual RF now learns OOF residuals. The hybrid %s the data-driven model. "
      "The dominant error is the free-run drift of the unobserved magnet node (held-out physics bias %+.2f degC), which a residual "
      "correction on the current features does not remove. Regularised hybrid NRMSE %s, physics-feature RF NRMSE %s."
      % (f(hy), f(dd), f(ph), "beats" if hy < dd else "does NOT beat", rm["physics"][2], f(hr_n), f(pf_n)))
    a("")
    a("## 5. GP emulator and emulator-variance inflation")
    a("- Design: 300-point LHS over 6 uncertain log-multipliers (C_W, C_M, G_WT, G_TM, G_YC, k_cu; +-0.5 in log) and the derating scale d in [0.5, 1].")
    a("- Output: trajectories of all four nodes at %d time points, PCA to %d components (%.4f of variance). Kernel: Constant x Matern(2.5, ARD) + White, normalised outputs, one restart." %
      (sm.K_OUT, C["gp"].n_comp, C["gp"].var_ratio_))
    a("- Emulator-variance inflation: a per-output variance (mean squared emulator error minus mean GP variance on a separate 100-point design) "
      "added to the GP predictive variance. Scale factor of the emulator sd on that design: x%s. It measures the emulator's own error against the LPTN ODE, not model-data discrepancy."
      % f(C["gp"].inflation_scale_))
    a("- Validation on a third set of 100 LHS points: R2 = %s, RMSE = %s degC, speed-up vs solve_ivp per run = %sx (solve_ivp %s s/run, GP %s s/run)." %
      (f(val["r2"]), f(val["rmse"]), f(val["speedup_x"]), f(val["t_ode_s"]), f(val["t_gp_s"])))
    a("- Coverage on the validation design, before inflation: 2-sigma %s. After inflation: 2-sigma %s, 50 / 80 / 95 %% = %s / %s / %s (nominal 0.50 / 0.80 / 0.95)." %
      (f(val_pre["coverage_2sigma"]), f(val["coverage_2sigma"]), f(val["coverage_levels"]["0.5"]),
       f(val["coverage_levels"]["0.8"]), f(val["coverage_levels"]["0.95"])))
    a("- Status of the 2-sigma coverage: %s (outside 0.03 of nominal = over-coverage, the emulator sd is too wide after inflation; "
      "the inflation does not change the posterior, see section 7)." % ("PASS (within 0.03 of nominal)" if surr_ok else "FAIL (over/under-coverage outside 0.03 of nominal)"))
    a("- Inflation variance relative to the observation variance on the stator nodes: %s."
      % ", ".join("%.4f" % (C["infl_node"][j] / C["sig2"][j]) for j in range(3)))
    a("")
    a("## 6. Sobol sensitivity (GP emulator, operating scale d=1)")
    a("| parameter | S1 peak pm | ST peak pm | S1 time-to-exceed | ST time-to-exceed |")
    a("|---|---|---|---|---|")
    for p in pm.UNCERT:
        a("| %s | %s | %s | %s | %s |" % (p, f(sob["peak_pm"][p]["S1"]), f(sob["peak_pm"][p]["ST"]),
                                          f(sob["time_to_exceed_min"][p]["S1"]), f(sob["time_to_exceed_min"][p]["ST"])))
    a("- Output 1 is the peak pm over the session; output 2 is the first time pm exceeds the limit (censored at session end). N=1024 Saltelli base samples.")
    a("")
    a("## 7. Calibration, predictive band and risk (AR(1) GLS, 32 walkers) on held-out session %d" % calib["pid"])
    a("Observations: stator winding, tooth, yoke every 1 min up to the stage instant; pm is not observed in the fit. "
      "Likelihood: AR(1) GLS per stator node, sigma (bias-free residual sd) = %s degC, rho at 60 s = %s, emulator inflation included as a constant. "
      "Start: MAP (L-BFGS-B). Sampler: DE-move 0.8 / DE-snooker 0.2, burn-in 500, blocks of 500, cap 3000 post-burn steps."
      % (", ".join(f(x) for x in C["sig2"] ** 0.5), ", ".join("%.3f" % x for x in C["rho"][:3])))
    a("")
    a("| stage | obs | post-burn steps | min steps / tau | converged (>=50 tau) | acceptance | tau max | ESS min |")
    a("|---|---|---|---|---|---|---|---|")
    for fr in STAGES:
        sr = stage_res[fr]
        a("| %d%% | %d | %d | %s | %s | %s | %s | %s |" %
          (round(100 * fr), sr["n_obs"], sr["mc"]["n_post"], f(np.min(sr["mc"]["n_post"] / sr["tau"])),
           sr["mc"]["converged"], f(sr["acc"]), f(np.max(sr["tau"])), f(np.min(sr["mc"]["ess"]))))
    a("")
    a("| stage | bias pm [degC] | 95%% band width [degC] | cov50 | cov80 | cov95 (predictive) | cov95 (parameter only) | IS95 | P(pm>%g in 30 min, no action) | P(pm>%g before session end, no action) |" % (dec.T_LIMIT, dec.T_LIMIT))
    a("|---|---|---|---|---|---|---|---|---|---|")
    for fr in STAGES:
        po = stage_res[fr]["post"]
        a("| %d%% | %+.2f | %s | %s | %s | %s | %s | %s | %s | %s |" %
          (round(100 * fr), po["bias"], f(po["pred"]["0.95"]["width"]), f(po["pred"]["0.5"]["coverage"]),
           f(po["pred"]["0.8"]["coverage"]), f(po["pred"]["0.95"]["coverage"]), f(po["postonly"]["0.95"]["coverage"]),
           f(po["pred"]["0.95"]["interval_score"]), f(po["p_window"]), f(po["p_session"])))
    a("")
    a("- Pooled over stages (session %d): predictive coverage 0.5/0.8/0.95 = %s / %s / %s (nominal 0.5/0.8/0.95); mean 95%% width %s degC; interval score 95 %s."
      % (calib["pid"], f(C["cov_pooled"]["0.5"]), f(C["cov_pooled"]["0.8"]), f(C["cov_pooled"]["0.95"]), f(C["width95"]), f(C["is95"])))
    a("- Parameter-only coverage (no model error): 0.5/0.8/0.95 = %s / %s / %s." %
      (f(C["cov_postonly"]["0.5"]), f(C["cov_postonly"]["0.8"]), f(C["cov_postonly"]["0.95"])))
    a("- Prior-predictive baseline (parameters from the prior box, same error model): 0.5/0.8/0.95 = %s / %s / %s, mean 95%% width %s degC." %
      (f(C["prior_cov"]["0.5"]), f(C["prior_cov"]["0.8"]), f(C["prior_cov"]["0.95"]), f(C["prior_width95"])))
    a("- Transfer to held-out session %d (posterior of session %d at 50%%, not calibrated on session %d): coverage 0.5/0.8/0.95 = %s / %s / %s, bias %+.2f degC."
      % (REPLICATE_PID, calib["pid"], REPLICATE_PID, f(C["cov12"]["0.5"]), f(C["cov12"]["0.8"]), f(C["cov12"]["0.95"]), c12["bias_pm_degC"]))
    a("- Bias: posterior median of the no-action forecast minus measured pm is %+.2f degC at 25 %% (not in the band)." % stage_res[0.25]["post"]["bias"])
    a("- Posterior-to-prior std ratio (prior uniform on [-0.5,0.5], std 0.289), 75 % stage: " +
      ", ".join("%s %s" % (p, f(C["ratio75"][p])) for p in pm.UNCERT) +
      ". Parameters with ratio >= 0.8 are called prior-like: %d of 6." % C["n_prior_like"])
    a("- Per-stage ratios (25/50/75 %):")
    for p_j, p in enumerate(pm.UNCERT):
        a("  - %s: %s" % (p, ", ".join(f(np.std(stage_res[fr]["chain"][:, p_j]) / PRIOR_STD) for fr in STAGES)))
    a("- Per-parameter tau at 75%% stage: %s." % ", ".join("%s %s" % (p, f(C["s75"]["tau"][j])) for j, p in enumerate(pm.UNCERT)))
    a("")
    a("Posterior correlation matrix (75% stage):")
    a("")
    a("| | " + " | ".join(pm.UNCERT) + " |")
    a("|---" * (len(pm.UNCERT) + 1) + "|")
    for j, p in enumerate(pm.UNCERT):
        a("| %s | " % p + " | ".join("%+.2f" % C["corr75"][j, k] for k in range(len(pm.UNCERT))) + " |")
    a("")
    if C["ridge_pairs"]:
        a("Ridge (|rho| > 0.7): " + "; ".join("%s-%s (%+.2f)" % (x, y, z) for x, y, z in C["ridge_pairs"]) +
          ". k_cu is the parameter with the clearest posterior reduction (ratio %s); its partners on the ridge are the G and C terms named above."
          % f(C["ratio75"]["k_cu"]))
    else:
        a("No parameter pair has |rho| > 0.7.")
    a("")
    a("## 8. Decision")
    a("- Policy: start derating d (fraction of current scale) at tau minutes after the calibration instant, for a 30-min horizon. "
      "Expected cost per unit time = [C_act 1[d<1] + C_thr (1-d)(H-tau) + C_fail P(pm>limit)] / H, C_fail=%g, C_thr=%g, C_act=%g (illustrative)." %
      (dec.C_FAIL, dec.C_THR, dec.C_ACT))
    a("- Forecast: initialised at the decision instant with the measured stator temperatures and the model magnet temperature; measured future inputs (perfect foresight).")
    a("- **Headline (nominal, limit %g degC, session %d, 50%% stage):** P(exceed in window | no action) = %s. Recommendation: %s."
      % (dec.T_LIMIT, calib["pid"], f(hs["post"]["p_window"]),
         ("derate d=%.2f from +%g min" % (hs["best"]["d"], hs["best"]["tau_min"])) if hs["best"]["act"] and hs["cost"] < hs["cvec"][0] - 1e-12
         else "no action (no derating policy beats no action)"))
    a("- **Pre-registered stress (limit %g degC, currents x%.1f, coolant +%g K over the window):** session %d P(exceed | no action) = %s, "
      "replicate session %d P(exceed | no action) = %s. Break-even dP(d=0.8, tau=0) = %s; stress dP(d=0.8, tau=0) on session %d = %s."
      % (dec.T_LIMIT, dec.STRESS_CURRENT_SCALE, dec.STRESS_COOLANT_K, calib["pid"], f(i68["p_no_action"]),
         REPLICATE_PID, f(i12["p_no_action"]), f(C["be08"]), calib["pid"], f(C["dP08"])))
    a("")
    a("| scenario | session | stage | optimum d* | tau* [min] | expected cost [per min] | no-action cost [per min] | action pays |")
    a("|---|---|---|---|---|---|---|---|")
    for fr in STAGES:
        sr = stage_res[fr]
        b = sr["best"]
        a("| nominal %g degC | %d | %d%% | %.2f | %s | %s | %s | %s |" %
          (dec.T_LIMIT, calib["pid"], round(100 * fr), b["d"], ("%g" % b["tau_min"]) if b["act"] else "none",
           f(sr["cost"]), f(sr["cvec"][0]), "yes" if sr["cost"] < sr["cvec"][0] - 1e-12 else "no"))
    a("| stress (pre-reg.) %g degC | %d | 50%% | %.2f | %s | %s | %s | %s |" %
      (dec.T_LIMIT, calib["pid"], C["best_st"]["d"], ("%g" % C["best_st"]["tau_min"]) if C["best_st"]["act"] else "none",
       f(C["c_st"]), f(C["cvec_st"][0]), "yes" if C["nontrivial"] else "no"))
    a("| stress (pre-reg.) %g degC | %d | 50%% | %.2f | %s | %s | %s | %s |" %
      (dec.T_LIMIT, REPLICATE_PID, C["best12_st"]["d"], ("%g" % C["best12_st"]["tau_min"]) if C["best12_st"]["act"] else "none",
       f(C["c12_st"]), f(C["cvec12_st"][0]), "yes" if C["c12_st"] < C["cvec12_st"][0] - 1e-12 else "no"))
    a("")
    a("**Decision checks:** stress optimum interior (session %d) = %s; stress action non-trivial = %s; informative (session %d / %d) = %s / %s; "
      "C_thr sweep range of the optimal stress cost = [%s, %s]; risk (no action, nominal) equals the policy table value = %s."
      % (calib["pid"], C["interior"], C["nontrivial"], calib["pid"], REPLICATE_PID, i68["informative"], i12["informative"],
         f(mc["decision_checks"]["cost_sweep_range"][0]), f(mc["decision_checks"]["cost_sweep_range"][1]), C["risk_check"]))
    a("")
    a("Every policy's expected cost per minute, nominal limit %g degC, session %d, 50%% stage:" % (dec.T_LIMIT, calib["pid"]))
    a("")
    for line in policy_grid(hs["rows"], hs["cvec"]):
        a(line)
    a("")
    a("No-action cost (nominal) = %s per min. Lowest policy cost = %s." % (f(hs["cvec"][0]), f(hs["cost"])))
    a("")
    a("Every policy's expected cost per minute, pre-registered stress (limit %g degC), session %d, 50%% stage:" % (dec.T_LIMIT, calib["pid"]))
    a("")
    for line in policy_grid(C["rows_st"], C["cvec_st"]):
        a(line)
    a("")
    a("No-action cost (stress) = %s per min. Lowest policy cost = %s." % (f(C["cvec_st"][0]), f(C["c_st"])))
    a("")
    a("Every policy's expected cost per minute, pre-registered stress (limit %g degC), replicate session %d, 50%% stage:" % (dec.T_LIMIT, REPLICATE_PID))
    a("")
    for line in policy_grid(C["rows12_st"], C["cvec12_st"]):
        a(line)
    a("")
    a("No-action cost (stress, session %d) = %s per min. Lowest policy cost = %s." % (REPLICATE_PID, f(C["cvec12_st"][0]), f(C["c12_st"])))
    a("")
    a("- Sensitivity to C_thr (5 / 10 / 20 / 40 / 60), 50 %% stage, session %d:" % calib["pid"])
    for name in ["nominal", "stress"]:
        for k, v in C["sens"][name].items():
            a("  - %s, C_thr=%s: d*=%.2f, tau*=%s, cost %s" %
              (name, k, v["d_opt"], ("%g" % v["tau_opt_min"]) if v["tau_opt_min"] is not None else "none", f(v["cost"])))
    a("")
    a("## 9. Figures")
    for fn in ["fig1_data_overview.png", "fig2_physics_fit_train.png", "fig3_heldout_predictions.png",
               "fig4_surrogate_validation.png", "fig5_sobol.png", "fig6_posterior.png",
               "fig7_calibration_predictive.png", "fig8_decision.png"]:
        a("- `results/figures/%s`" % fn)
    a("")
    a("## 10. Known limitations")
    for s in mc["limitations"]:
        a("- " + s)
    a("")
    with open(os.path.join(HERE, "results", "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
