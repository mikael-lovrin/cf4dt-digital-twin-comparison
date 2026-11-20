"""p2_nasa (round 3): hybrid physics + data-driven digital twin for milling-tool flank wear (NASA mill data).

Pipeline
  1. Load + clean the CSV (degenerate sensor row and single-run case 6 dropped; run-in transient flagged).
  2. Setting-grouped 4-fold CV (primary; 8 operating settings = (DOC, feed, material), 2 settings per fold).
     Secondary: case-grouped 5-fold CV. Per fold, the physics design box is derived from training-fold fits only
     (stage A on the a-priori outer limits, multistart; stage B refit in the derived box), max_nfev = 1000.
     Models: physics ODE | data-driven GP | hybrid | constant mean | time-only power law
     (round-2 qn log-linear kept as reference only).
  3. Per-case GP emulator of log(VB+EPS) on the case's fold box, with a VB_CAP = 3 mm build and an uncapped
     ablation build.
  4. emcee calibration on cases 1, 11, 13 at 25/50/75% of life (log-VB likelihood, MAP start, DE/snooker moves),
     prior centre = the case's fold fit. Predictive intervals at 50/80/95%, computed on the median curve.
  5. RUL to VB = 0.45 mm (median curve); failure risk; Sobol (SALib, Saltelli N=1024) on case 1.
  6. Decision: expected cost per unit time over replacement time and derating. Derating kept only if the 1000-draw
     bootstrap interval of its gain excludes 0. A replacement time is recommended only if it is no later than the
     lower bound of the observed crossing bracket; otherwise the output is "no recommendation".

PRE-REGISTERED PASS/FAIL CRITERIA (fixed in code before the round-3 run; they are not tuned afterwards)
  C1 physics_converged_all_setting_folds : every setting-CV fold's best stage-B start has status > 0
  C2 physics_beats_best_simple_baseline  : primary nRMSE(physics) < min(nRMSE constant mean, nRMSE time power law)
  C3 hybrid_beats_best_simple_baseline   : same comparison for the hybrid model
  C4 mcmc_50tau_met                      : min over the 9 calibration runs and 6 parameters of chain/tau >= 50
  C5 interval_calibrated                 : |coverage - nominal| <= 0.05 at 50%, 80% and 95%
  C6 recommendation_issued               : t_policy <= lower bound of the observed crossing bracket (bracket exists)
  C7 derating_kept                       : 95% bootstrap interval of the derating gain has lower bound > 0
  C8 optimum_interior                    : policy optimum strictly below its plateau and not the first grid point
  C9 box_training_only                   : every stage-A/B fit job holds training-fold cases only (asserted in code)

Outputs: results/metrics.json, results/details.json, results/summary.md, results/run.log, results/figures/*.png
"""
import os

# Single-threaded BLAS in every worker (the parallelism is across fits and calibrations, see ProcessPoolExecutor).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import corner  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402

from data_loader import (DATA_CSV, VB_CRIT, load_raw, clean, make_folds,  # noqa: E402
                         make_setting_folds, case_table, rows_xy, scaled_cases)
from physics_model import (simulate, fit_force_observation, lsq_job, make_jobs, derive_box,  # noqa: E402
                           best_fit, LO6, HI6, GAMMA_LO, GAMMA_HI, PARAM_NAMES, MAX_NFEV, N_STARTS,
                           PHYS_LO, PHYS_HI)
from surrogate_model import (DataDriven, Hybrid, ConstantMean, TimePowerLaw, LogLinearBaseline,  # noqa: E402
                             build_emulator, VB_CAP, EPS)
from uq_calibration import (calibrate_job, posterior_predictive, failure_prob_now, rul_samples,  # noqa: E402
                            derated_job, sobol_indices, prior_std_empirical, crossing_time)
from decision import (optimise, expected_cost_per_time, expected_cost_direct, post_stats,  # noqa: E402
                      bootstrap_gain)

ROUND = 3
RESULTS = os.environ.get("P2_RESULTS_DIR") or os.path.join(HERE, "results")   # override only for test runs
FIGS = os.path.join(RESULTS, "figures")
os.makedirs(FIGS, exist_ok=True)
QUICK = os.environ.get("P2_QUICK") == "1"     # smoke test only (reduced MCMC); the reported run is not quick

SEED = 0
N_FOLDS_SETTING = 4            # 8 settings -> 4 folds of 2 settings
N_FOLDS_CASE = 5               # secondary, case-grouped
STAGES = (0.25, 0.50, 0.75)
LEVELS = (0.5, 0.8, 0.95)
EMCEE_CASES = (1, 11, 13)
PRIMARY = 1
N_WALKERS = 32
N_STEPS, BURN = (1200, 400) if QUICK else (12000, 2000)
TAU_TARGET = 50                # emcee criterion: chain length >= 50 integrated autocorrelation times
DERATE_LEVELS = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6)
DERATE_DRAWS = 200
C_P, C_F = 1.0, 5.0            # relative cost units: planned change 1, failure 5 (assumption)
C_F_SWEEP = (2.0, 5.0, 10.0, 20.0)
C_F_STRESS = 10.0              # stressed scenario: failure penalty doubled relative to the base case
RISK_THRESH = 0.10             # derating considered only above this instantaneous risk
N_BOOT = 1000
N_WORKERS = 12
PNAME = {"log10A": r"$\log_{10}A$", "log10B": r"$\log_{10}B$", "Ea_R": r"$E_a/R$",
         "alpha": r"$\alpha$", "log10hA": r"$\log_{10}hA$", "log_sigma": r"$\log\sigma_{obs}$"}
MODELS = ["physics", "data_driven", "hybrid", "constant_mean", "time_power_law"]
REF_MODEL = "log_linear_qn"    # round-2 comparator, reference only
MODEL_LABELS = {"physics": "Pure physics", "data_driven": "Pure data-driven (GP)",
                "hybrid": "Hybrid (physics + GP residual)", "constant_mean": "Constant mean (baseline)",
                "time_power_law": "Time-only power law (baseline)",
                "log_linear_qn": "Log-linear with qn (round-2 reference, not a comparator)"}


def phys_rows(p6, cs, cases):
    cl = sorted(cases)
    preds = simulate(p6[:5], p6[5], [cs[c] for c in cl], [cs[c]["t"][cs[c]["m"]] for c in cl])
    return np.concatenate(preds)


def _nrmse_block(df, models, mask=None):
    sub = df if mask is None else df[mask]
    y = sub["VB"].to_numpy()
    sd = float(np.std(y))
    rmse = {m: float(np.sqrt(np.mean((sub[m].to_numpy() - y) ** 2))) for m in models}
    return dict(rmse=rmse, nrmse={m: rmse[m] / sd for m in models}, n=int(len(y)), sd=sd)


def run_cv(pool, d, tables, cases, folds):
    """Grouped CV. Per fold: box from training-fold fits only, then physics refit in that box; baselines and
    data-driven models on training folds only."""
    cs_fold, train_of, jobs_a = {}, {}, []
    for k, test in enumerate(folds):
        train = [c for c in cases if c not in test]
        assert not (set(train) & set(test))
        ref = float(np.median(d.loc[d["case"].isin(train) & ~d["transient"], "smcDC"]))
        cs = scaled_cases(tables, cases, ref)
        cs_fold[k], train_of[k] = cs, train
        # C9: only training-fold cases enter stage A and stage B of fold k
        jobs_a.append(make_jobs([cs[c] for c in train], LO6, HI6, seed=SEED))
    res_a = list(pool.map(lsq_job, [j for js in jobs_a for j in js]))
    stage_a = {k: res_a[k * N_STARTS:(k + 1) * N_STARTS] for k in range(len(folds))}

    boxes, jobs_b = {}, []
    for k in range(len(folds)):
        lo5, hi5, n_near = derive_box(stage_a[k])
        best_a = min(stage_a[k], key=lambda f: f["cost"])
        lo6 = np.r_[lo5, GAMMA_LO]
        hi6 = np.r_[hi5, GAMMA_HI]
        boxes[k] = (lo6, hi6, n_near, best_a)
        jobs_b.append(make_jobs([cs_fold[k][c] for c in train_of[k]], lo6, hi6, seed=SEED,
                                extra=[best_a["x"]]))
    res_b = list(pool.map(lsq_job, [j for js in jobs_b for j in js]))

    oof_parts, fold_info, fits = [], [], {}
    for k, test in enumerate(folds):
        lo6, hi6, n_near, best_a = boxes[k]
        bf = best_fit(res_b[k * N_STARTS:(k + 1) * N_STARTS], lo6, hi6)
        p6 = np.array(bf["p6"])
        fits[k] = bf
        cs, train = cs_fold[k], train_of[k]
        Xtr, ytr, ctr = rows_xy(d, train)
        Xte, yte, cte = rows_xy(d, test)
        vb_tr = phys_rows(p6, cs, train)
        vb_te = phys_rows(p6, cs, test)
        dd = DataDriven().fit(Xtr, ytr, seed=SEED)
        dd_te, _ = dd.predict(Xte)
        hy = Hybrid().fit(Xtr, ytr, vb_tr, seed=SEED)
        hy_te = hy.predict(Xte, vb_te)
        cm_te = ConstantMean().fit(ytr).predict(len(yte))
        tpl = TimePowerLaw().fit(Xtr[:, 0], ytr)
        tpl_te = tpl.predict(Xte[:, 0])
        qn_tr = np.array([tables[c]["qn"] for c in ctr], float)
        qn_te = np.array([tables[c]["qn"] for c in cte], float)
        lb = LogLinearBaseline().fit(qn_tr, Xtr[:, 0], ytr)
        lb_te = lb.predict(qn_te, Xte[:, 0])
        oof_parts.append(pd.DataFrame({
            "fold": k, "case": cte, "time": Xte[:, 0], "VB": yte, "converged": bf["converged"],
            "physics": vb_te, "data_driven": dd_te, "hybrid": hy_te,
            "constant_mean": cm_te, "time_power_law": tpl_te, REF_MODEL: lb_te}))
        fold_info.append(dict(
            fold=k, test_cases=test, train_cases=train, ref=float(np.median(d.loc[d["case"].isin(train)
                                                                                    & ~d["transient"], "smcDC"])),
            box_lo=bf["box_lo"], box_hi=bf["box_hi"], n_near_optimal_stageA=n_near,
            stageA_best_cost=float(best_a["cost"]), stageB_best_cost=float(bf["cost"]),
            p6=[float(v) for v in p6], physics_train_rmse=bf["rmse_train"], nfev=bf["nfev"],
            status=bf["status"], converged=bf["converged"], at_bound=bf["at_bound"],
            n_starts=bf["n_starts"], n_starts_converged=bf["n_starts_converged"],
            n_test_rows=int(len(yte)), time_power_law=tpl.params, baseline_n_fit=tpl.n_fit,
            log_linear_qn_params=lb.params))
        print(f"fold {k}: test={test} box lo={np.round(bf['box_lo'], 2)} hi={np.round(bf['box_hi'], 2)} "
              f"physics train RMSE={bf['rmse_train']:.4f} nfev={bf['nfev']} converged={bf['converged']} "
              f"at_bound={bf['at_bound']} p6={np.round(p6, 3)}")
    oof = pd.concat(oof_parts, ignore_index=True)
    allm = _nrmse_block(oof, MODELS + [REF_MODEL])
    conv_mask = oof["converged"].to_numpy(bool)
    conv_block = _nrmse_block(oof, MODELS + [REF_MODEL], mask=conv_mask)
    conv_folds = [k for k in fits if fits[k]["converged"]]
    fold_rmse = [{m: float(np.sqrt(np.mean((g[m].to_numpy() - g["VB"].to_numpy()) ** 2))) for m in MODELS}
                 for _, g in oof.groupby("fold")]
    return dict(oof=oof, fold_info=fold_info, cs_fold=cs_fold, boxes={k: (boxes[k][0], boxes[k][1])
                                                                       for k in boxes},
                all=allm, conv=conv_block, conv_folds=conv_folds, n_conv=len(conv_folds),
                rmse=allm["rmse"], nrmse=allm["nrmse"], sd_y=allm["sd"], fold_rmse=fold_rmse,
                fits=fits, n_folds=len(folds))


def fig_trajectories(oof, cases, tables, fold_info, cs_fold, fold_of):
    fig, axes = plt.subplots(3, 5, figsize=(16, 9), sharey=True)
    for ax, c in zip(axes.ravel(), cases):
        k = fold_of[c]
        cs = cs_fold[k]
        p6 = np.array(fold_info[k]["p6"])
        tt = np.linspace(0, tables[c]["tlast"], 200)
        vb_line = simulate(p6[:5], p6[5], [cs[c]], [tt])[0]
        ax.plot(tt, vb_line, "-", color="C0", label="physics (out-of-fold)")
        o = oof[oof["case"] == c].sort_values("time")
        ax.plot(o["time"], o["VB"], "ko", ms=4, label="measured VB")
        ax.plot(o["time"], o["hybrid"], "s", color="C2", ms=3, label="hybrid (OOF)")
        ax.axhline(VB_CRIT, color="r", ls="--", lw=0.8)
        mat = "steel" if tables[c]["material"] else "cast iron"
        ax.set_title(f"case {c}: {mat}, DOC {tables[c]['DOC']}, f {tables[c]['feed']}", fontsize=8)
        ax.set_xlabel("time (CSV units)", fontsize=7)
    axes[0, 0].legend(fontsize=7)
    fig.supylabel("VB (mm)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig1_vb_trajectories.png"), dpi=150)
    plt.close(fig)


def fig_comparison(oof, nrmse):
    models = MODELS + [REF_MODEL]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    y = oof["VB"].to_numpy()
    for ax, m in zip(axes.ravel(), models):
        p = oof[m].to_numpy()
        ax.scatter(y, p, s=14, alpha=0.7)
        lim = [0, max(y.max(), p.max()) * 1.05]
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.set_xlabel("measured VB (mm)")
        ax.set_ylabel("predicted VB (mm)")
        ax.set_title(f"{MODEL_LABELS[m]}: nRMSE {nrmse[m]:.3f}", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig2_model_comparison.png"), dpi=150)
    plt.close(fig)


def fig_emulator(emus):
    fig, axes = plt.subplots(1, len(emus), figsize=(4.5 * len(emus), 4.2))
    for ax, (c, emu) in zip(np.atleast_1d(axes), emus.items()):
        _, Yt, mu, sd = emu.test
        ax.errorbar(Yt.ravel(), mu.ravel(), yerr=2 * sd.ravel(), fmt=".", ms=2, alpha=0.3, elinewidth=0.4)
        lim = [0, max(Yt.max(), mu.max()) * 1.05]
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.set_xlabel("ODE VB (mm), capped")
        ax.set_ylabel("GP median (+/- 2 sd, delta method)")
        ax.set_title(f"case {c} emulator, R2={emu.metrics['r2']:.4f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig3_gp_emulator.png"), dpi=150)
    plt.close(fig)


def fig_calibration(emu, c, stage_out, tables):
    tc = tables[c]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, f in zip(axes, STAGES):
        so = stage_out[(c, f)]
        idx = np.arange(emu.grid.size)
        lo, med, hi = posterior_predictive(emu, so["chain"], idx)
        ax.fill_between(emu.grid, lo, hi, color="C0", alpha=0.25, label="95% posterior predictive (log-noise)")
        ax.plot(emu.grid, med, "-", color="C0", lw=1, label="predictive median")
        used = tc["m"] & (tc["t"] > 0) & (tc["t"] <= so["tau"])
        later = tc["m"] & (tc["t"] > so["tau"])
        ax.plot(tc["t"][used], tc["VB"][used], "ko", label="calibration data")
        ax.plot(tc["t"][later], tc["VB"][later], "o", mfc="none", mec="r", label="later measured (held out)")
        ax.axvline(so["tau"], color="gray", ls=":", lw=1)
        ax.axhline(VB_CRIT, color="r", ls="--", lw=0.8)
        ax.set_title(f"case {c}, {int(f*100)}% of life (tau={so['tau']:.1f}); n_data={so['n_data']}", fontsize=9)
        ax.set_xlabel("time (CSV units)")
    axes[0].set_ylabel("VB (mm)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig4_calibration_case%d.png" % c), dpi=150)
    plt.close(fig)


def fig_posteriors(c, stage_out, emu):
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.6))
    for i, ax in enumerate(axes):
        for f in STAGES:
            ch = stage_out[(c, f)]["chain"][:, i]
            ax.hist(ch, bins=40, density=True, histtype="step", lw=1.4, label=f"{int(f*100)}%")
        ax.set_title(PNAME[PARAM_NAMES[i]])
        ax.set_xlim(emu.lo[i], emu.hi[i])
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Posterior marginals, case {c} (prior box = its fold's training-derived box)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig5_posteriors_case%d.png" % c), dpi=150)
    plt.close(fig)


def fig_corner(c, stage_out):
    ch = stage_out[(c, 0.75)]["chain"]
    fig = corner.corner(ch, labels=[PNAME[n] for n in PARAM_NAMES] + [PNAME["log_sigma"]],
                        quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt=".2f",
                        title_kwargs={"fontsize": 8})
    fig.savefig(os.path.join(FIGS, "fig6_corner_case%d_75pct.png" % c), dpi=120)
    plt.close(fig)


def fig_rul(stage_out, brackets):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = 0
    ticks, labels = [], []
    for c in EMCEE_CASES:
        for f in STAGES:
            so = stage_out[(c, f)]
            ax.errorbar(x, so["rul_median"], yerr=[[so["rul_median"] - so["rul_lo"]], [so["rul_hi"] - so["rul_median"]]],
                        fmt="o", color="C0", capsize=4)
            b = brackets[c]
            if b is not None and so["tau"] < b[1]:
                lo_b, hi_b = max(b[0] - so["tau"], 0), b[1] - so["tau"]
                ax.fill_between([x - 0.3, x + 0.3], lo_b, hi_b, color="C2", alpha=0.35)
            ticks.append(x)
            labels.append(f"c{c} {int(f*100)}%")
            x += 1
        x += 0.6
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("RUL to VB = 0.45 mm (CSV time units)")
    ax.set_title("RUL posterior of the median curve (median, 95%) vs observed crossing bracket (green)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig7_rul.png"), dpi=150)
    plt.close(fig)


def fig_sobol(sob):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key, title in zip(axes, ["VB_tend", "time_to_threshold"],
                              ["VB at end of case", "time to VB = 0.45 mm"]):
        d = sob[key]
        names = PARAM_NAMES[:5]
        xs = np.arange(len(names))
        s1 = [d[n]["S1"] for n in names]
        st = [d[n]["ST"] for n in names]
        ax.bar(xs - 0.2, s1, 0.4, label="S1")
        ax.bar(xs + 0.2, st, 0.4, label="ST")
        ax.set_xticks(xs)
        ax.set_xticklabels(names, rotation=20)
        ax.set_title(f"Sobol indices: {title}", fontsize=10)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig8_sobol.png"), dpi=150)
    plt.close(fig)


def fig_decision(dec, title, lab):
    fig, ax = plt.subplots(figsize=(8, 4))
    for dl, j in sorted(dec["J"].items(), reverse=True):
        ax.plot(dec["t_grid"], j, label=f"derate d={dl:.2f}")
    ax.plot(dec["t_star"], dec["J_star"], "r*", ms=12,
            label=f"grid optimum t={dec['t_star']:.2f}, d={dec['d_star']:.2f} ({lab})")
    ax.set_xlabel("replacement time after cutoff (CSV time units)")
    ax.set_ylabel("expected cost per unit time (relative)")
    ax.set_yscale("log")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig9_decision.png"), dpi=150)
    plt.close(fig)


def crossing_check(emu, emu_nc, cs_case, p6, grid):
    """Crossing-time error of the GP median curve against the ODE at the fold's physics centre, with and without
    the VB_CAP training cap."""
    center = np.clip(p6[:5], emu.lo, emu.hi)
    ode = simulate(center, p6[5], [cs_case], [grid])[0]
    med_c, _ = emu.median_vb(center)
    med_n, _ = emu_nc.median_vb(center)
    t_ode, c_ode = crossing_time(grid, np.nan_to_num(ode, nan=0.0)[None, :])
    t_c, c_c = crossing_time(grid, med_c)
    t_n, c_n = crossing_time(grid, med_n)
    return dict(t_ode=float(t_ode[0]), t_gp_capped=float(t_c[0]), t_gp_uncapped=float(t_n[0]),
                abs_err_capped=float(abs(t_ode[0] - t_c[0])), abs_err_uncapped=float(abs(t_ode[0] - t_n[0])),
                delta_cap_minus_uncapped=float(t_c[0] - t_n[0]),
                censored_ode=bool(c_ode[0]), censored_gp=bool(c_c[0]))


def main():
    t_wall = time.perf_counter()
    raw = load_raw(DATA_CSV)
    d, clog = clean(raw)
    cases = sorted(d["case"].unique().tolist())
    tables = {c: case_table(d, c) for c in cases}
    folds = make_setting_folds(tables, cases, N_FOLDS_SETTING)
    folds_case = make_folds(cases, N_FOLDS_CASE)
    fold_of = {c: k for k, f in enumerate(folds) for c in f}
    print("usable cases:", cases)
    print("setting folds (primary):", folds)
    print("case folds (secondary):", folds_case)
    print("transient rows masked:", clog["transient_rows_masked"])
    print("quick mode:", QUICK, "results dir:", RESULTS, flush=True)

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        # ---- 1. grouped CV ----------------------------------------------------------------
        P = run_cv(pool, d, tables, cases, folds)
        print("setting-grouped held-out RMSE:", P["rmse"], "nRMSE:", P["nrmse"],
              "converged folds:", P["conv_folds"], f"t={time.perf_counter()-t_wall:.0f}s", flush=True)
        S = run_cv(pool, d, tables, cases, folds_case)
        print("case-grouped held-out RMSE (secondary):", S["rmse"], f"t={time.perf_counter()-t_wall:.0f}s",
              flush=True)
        oof, fold_info, cs_fold = P["oof"], P["fold_info"], P["cs_fold"]
        rmse, nrmse, sd_y = P["rmse"], P["nrmse"], P["sd_y"]

        # ---- 2. force-proxy observation model and forcing diagnostics ---------------------
        ref_all = float(np.median(d.loc[~d["transient"], "smcDC"]))
        cs_all = scaled_cases(tables, cases, ref_all)
        cw, cw_r2 = fit_force_observation([cs_all[c] for c in cases])
        forcing_cv = {}
        for c in cases:
            u = tables[c]["u_raw"][tables[c]["keep"]]
            forcing_cv[c] = float(np.std(u) / np.mean(u))
        cvs = np.array(list(forcing_cv.values()))
        n_transient = int(sum((~tables[c]["keep"]).sum() for c in cases))

        # ---- 3. emulators (capped primary and uncapped ablation), calibration, RUL -------
        emu_jobs, grids, centers = [], {}, {}
        for c in EMCEE_CASES:
            k = fold_of[c]
            p6 = np.array(fold_info[k]["p6"])
            tc = cs_fold[k][c]
            lo, hi = P["boxes"][k][0][:5], P["boxes"][k][1][:5]
            center = np.clip(p6[:5], lo, hi)
            tlast = tc["tlast"]
            H = max(1.6 * tlast, tlast + 20.0)
            taus = {f: f * tlast for f in STAGES}
            meas_t = tc["t"][tc["m"]]
            grid = np.unique(np.r_[meas_t[meas_t > 0], list(taus.values()), tlast,
                                   np.linspace(H / 40.0, H, 40)])
            grids[c], centers[c] = grid, center
            emu_jobs.append((tc, p6[5], grid, lo, hi, VB_CAP, SEED))
            emu_jobs.append((tc, p6[5], grid, lo, hi, None, SEED))
        built = list(pool.map(build_emulator, emu_jobs))
        emus = {c: built[2 * i] for i, c in enumerate(EMCEE_CASES)}
        emus_nc = {c: built[2 * i + 1] for i, c in enumerate(EMCEE_CASES)}
        xcheck, brackets, cal_jobs, cal_meta = {}, {}, [], []
        for c in EMCEE_CASES:
            k = fold_of[c]
            p6 = np.array(fold_info[k]["p6"])
            cs = cs_fold[k]
            tc = cs[c]
            xcheck[c] = crossing_check(emus[c], emus_nc[c], tc, p6, grids[c])
            print(f"emulator case {c}: {emus[c].metrics}", flush=True)
            vb_m = tc["VB"][tc["m"]]
            meas_t = tc["t"][tc["m"]]
            jj = np.where(vb_m >= VB_CRIT)[0]
            brackets[c] = None
            if jj.size:
                j = int(jj[0])
                brackets[c] = (float(meas_t[j - 1]) if j > 0 else 0.0, float(meas_t[j]))
            for f in STAGES:
                tau = f * tc["tlast"]
                sel = tc["m"] & (tc["t"] > 0) & (tc["t"] <= tau)
                idx = np.searchsorted(grids[c], tc["t"][sel])
                yl = np.log(tc["VB"][sel] + EPS)
                cal_jobs.append((emus[c], idx, yl, centers[c], SEED, N_WALKERS, N_STEPS, BURN))
                cal_meta.append((c, f, float(tau), int(sel.sum())))
        cals = list(pool.map(calibrate_job, cal_jobs))
        print(f"calibrations done t={time.perf_counter()-t_wall:.0f}s", flush=True)

        stage_out = {}
        for (c, f, tau, n_data), cal in zip(cal_meta, cals):
            emu, grid, tc = emus[c], grids[c], cs_fold[fold_of[c]][c]
            chain = cal["chain"]
            post = tc["m"] & (tc["t"] > tau)
            idx_post = np.searchsorted(grid, tc["t"][post])
            y_post = tc["VB"][post]
            n_post = int(y_post.size)
            hits_by_level, width95_sum, hits95, bias95 = {}, 0.0, 0, float("nan")
            for L in LEVELS:
                lo, med, hi = posterior_predictive(emu, chain, idx_post, level=L)
                if n_post:
                    h = (y_post >= lo) & (y_post <= hi)
                    hits_by_level[str(L)] = int(h.sum())
                    if abs(L - 0.95) < 1e-9:
                        width95_sum = float(np.sum(hi - lo))
                        hits95 = int(h.sum())
                        bias95 = float(np.mean(y_post - med))
                else:
                    hits_by_level[str(L)] = 0
            rul, cens = rul_samples(emu, chain, tau)
            rul_nc, _ = rul_samples(emus_nc[c], chain, tau)           # same posterior, uncapped emulator
            pnow = failure_prob_now(emu, chain, int(np.searchsorted(grid, tau)))
            b = brackets[c]
            rb = None
            if b is not None and tau < b[1]:
                rb = (max(b[0] - tau, 0.0), b[1] - tau)
            rul_lo, rul_med, rul_hi = np.percentile(rul, [2.5, 50, 97.5])
            tau_ac = np.asarray(cal["tau_ac"], float)
            chain_over_tau = (N_STEPS - BURN) / tau_ac
            stage_out[(c, f)] = dict(
                case=c, frac=f, tau=float(tau), n_data=int(n_data), chain=chain,
                tau_ac=tau_ac.tolist(), chain_over_tau=chain_over_tau.tolist(), acc=cal["acc"], n_post=n_post,
                map=cal["map"], map_logpost=cal["map_logpost"], map_success=cal["map_success"],
                hits=hits95, hits_by_level=hits_by_level, width95_sum=width95_sum, bias95=bias95,
                coverage_post=float(hits95 / n_post) if n_post else float("nan"),
                rul_median=float(rul_med), rul_lo=float(rul_lo), rul_hi=float(rul_hi),
                rul_median_uncapped=float(np.median(rul_nc)),
                rul_censored_fraction=float(cens.mean()), risk_now=pnow,
                bracket_rul=rb, post_mean=chain.mean(axis=0).tolist(), post_std=chain.std(axis=0).tolist(),
                corr=np.corrcoef(chain[:, :5].T).tolist() if len(chain) > 5 else None,
                already_crossed=bool(b is not None and tau >= b[1]),
                ci_overlaps=(bool(rul_lo <= rb[1] and rul_hi >= rb[0]) if rb is not None else None),
                median_in_bracket=(bool(rb[0] <= rul_med <= rb[1]) if rb is not None else None),
            )
            print(f"case {c} {int(f*100)}%: tau={tau:.1f} n_data={n_data} RUL med={rul_med:.1f} "
                  f"[{rul_lo:.1f},{rul_hi:.1f}] bracket={rb} risk={pnow:.3f} cov95={stage_out[(c, f)]['coverage_post']} "
                  f"acc={cal['acc']:.2f} tau_max={tau_ac.max():.0f} steps/tau_min={chain_over_tau.min():.0f} "
                  f"MAP ok={cal['map_success']}", flush=True)

        # ---- 4. Sobol on the primary emulator -----------------------------------------------
        emu_p = emus[PRIMARY]
        tlast_p = cs_fold[fold_of[PRIMARY]][PRIMARY]["tlast"]
        idx_tend = int(np.searchsorted(grids[PRIMARY], tlast_p))
        sob = sobol_indices(emu_p, idx_tend, N=1024, seed=SEED)
        print("Sobol (VB_tend):", sob["VB_tend"])
        print(f"Sobol done t={time.perf_counter()-t_wall:.0f}s", flush=True)

        # ---- 5. decision on the primary case at 75% of life -------------------------------
        so75 = stage_out[(PRIMARY, 0.75)]
        risk75 = so75["risk_now"]
        tau75 = so75["tau"]
        t_max = float(grids[PRIMARY][-1]) - tau75           # censoring horizon after the cutoff
        deriv_jobs = [(emu_p, so75["chain"], tau75, dl, DERATE_DRAWS, 7) for dl in DERATE_LEVELS]
        deriv = list(pool.map(derated_job, deriv_jobs))
        rul_by_d = {dl: deriv[i][0] for i, dl in enumerate(DERATE_LEVELS)}
        cens_by_d = {dl: deriv[i][1] for i, dl in enumerate(DERATE_LEVELS)}
        for dl in DERATE_LEVELS:
            print(f"derate d={dl}: RUL median={np.median(rul_by_d[dl]):.2f} censored={cens_by_d[dl].mean():.2f}")
    cens_max = float(max(v.mean() for v in cens_by_d.values()))
    dec = optimise(rul_by_d, C_P, C_F, risk75, RISK_THRESH, t_max=t_max)
    tg = dec["t_grid"]
    # the fast sorted-sample cost must equal the direct averages
    fast_err = float(np.max(np.abs(expected_cost_per_time(rul_by_d[1.0], 1.0, C_P, C_F, tg)
                                   - expected_cost_direct(rul_by_d[1.0], 1.0, C_P, C_F, tg))))
    assert fast_err < 1e-9, fast_err
    sweep = {cf: optimise(rul_by_d, C_P, cf, risk75, RISK_THRESH, t_max=t_max) for cf in C_F_SWEEP}
    stress = sweep[C_F_STRESS]
    sweep_ts = [sweep[cf]["t_star"] for cf in C_F_SWEEP]
    sens = {str(cf): dict(t_star=float(r["t_star"]), d_star=float(r["d_star"]), J_star=float(r["J_star"]),
                          interior=bool(r["interior"])) for cf, r in sweep.items()}
    boot = bootstrap_gain(rul_by_d, C_P, C_F, risk75, RISK_THRESH, tg, n_boot=N_BOOT, seed=11)
    gain_pt = float(1.0 - dec["J_star"] / dec["J_no_derate"])
    gain_lo, gain_hi = (float(v) for v in np.percentile(boot, [2.5, 97.5]))
    derate_kept = bool(dec["derating_considered"] and gain_lo > 0.0)        # C7
    if derate_kept:
        d_pol, t_pol, J_pol, int_pol = dec["d_star"], dec["t_star"], dec["J_star"], dec["interior"]
    else:
        d_pol, t_pol, J_pol, int_pol = 1.0, dec["t_star_no_derate"], dec["J_no_derate"], dec["interior_no_derate"]
    st = post_stats(rul_by_d[d_pol], t_pol)
    rb75 = so75["bracket_rul"]
    t_le_bracket = bool(rb75 is not None and t_pol <= rb75[0])               # C6
    recommend = t_le_bracket
    t_le_median = bool(t_pol <= so75["rul_median"])
    print("decision:", {k: v for k, v in dec.items() if k not in ("t_grid", "J")}, "gain CI", gain_lo, gain_hi,
          "kept", derate_kept, "policy t", t_pol, "P(T<=t)", st["P_T_le_t"], "recommend", recommend, flush=True)

    # ---- 6. aggregate metrics -----------------------------------------------------------
    cal_cases = EMCEE_CASES
    sur_r2 = float(r2_score(np.concatenate([emus[c].test[1].ravel() for c in cal_cases]),
                            np.concatenate([emus[c].test[2].ravel() for c in cal_cases])))
    Yt_all = np.concatenate([emus[c].test[1].ravel() for c in cal_cases])
    mu_all = np.concatenate([emus[c].test[2].ravel() for c in cal_cases])
    sd_all = np.concatenate([emus[c].test[3].ravel() for c in cal_cases])
    sur_rmse = float(np.sqrt(np.mean((Yt_all - mu_all) ** 2)))
    sur_cov = float(np.mean([emus[c].metrics["coverage_2sigma"] for c in cal_cases]))   # log-space, per-case mean
    m06 = Yt_all <= 0.6
    sur_rmse_06 = float(np.sqrt(np.mean((Yt_all[m06] - mu_all[m06]) ** 2)))
    sur_speed = float(np.mean([emus[c].metrics["speedup_x"] for c in cal_cases]))
    cap_frac = float(np.mean([emus[c].metrics["cap_fraction"] for c in cal_cases]))
    cap_ablation = {str(c): dict(xcheck[c], delta_rul_median_capped_minus_uncapped=float(
        np.mean([stage_out[(c, f)]["rul_median"] - stage_out[(c, f)]["rul_median_uncapped"] for f in STAGES])),
        max_abs_delta_rul_median=float(max(abs(stage_out[(c, f)]["rul_median"] - stage_out[(c, f)]["rul_median_uncapped"])
                                           for f in STAGES)))
        for c in cal_cases}
    max_abs_dt_cross = float(max(abs(xcheck[c]["delta_cap_minus_uncapped"]) for c in cal_cases))
    max_abs_drul = float(max(v["max_abs_delta_rul_median"] for v in cap_ablation.values()))

    all_runs = list(stage_out.values())
    n_post_tot = int(sum(r["n_post"] for r in all_runs))
    cov = {str(L): float(sum(r["hits_by_level"][str(L)] for r in all_runs) / n_post_tot) for L in LEVELS}
    width95 = float(sum(r["width95_sum"] for r in all_runs) / n_post_tot)
    cov95 = cov["0.95"]
    acc_mean = float(np.mean([r["acc"] for r in all_runs]))
    tau_stage = np.array([r["tau_ac"] for r in all_runs], float)          # (runs, 6)
    chain_tau = np.array([r["chain_over_tau"] for r in all_runs], float)  # (runs, 6)
    n_post_mc = N_STEPS - BURN
    names6 = PARAM_NAMES + ["log_sigma"]
    tau_per_param = {n: float(np.max(tau_stage[:, i])) for i, n in enumerate(names6)}
    tau_per_run = {f"case{r['case']}_{int(r['frac']*100)}": {n: float(r["tau_ac"][i]) for i, n in enumerate(names6)}
                   for r in all_runs}
    min_steps_over_tau = float(np.min(chain_tau))
    ess_min = float(np.min(N_WALKERS * n_post_mc / tau_stage))
    mac = float(np.mean(tau_stage[:, :5]))
    converged_50 = bool(min_steps_over_tau >= TAU_TARGET)                 # C4
    lo_p5, hi_p5 = P["boxes"][fold_of[PRIMARY]][0][:5], P["boxes"][fold_of[PRIMARY]][1][:5]
    prior_sd_p = prior_std_empirical(centers[PRIMARY], lo_p5, hi_p5)
    ratio = {n: float(so75["post_std"][i] / prior_sd_p[i]) for i, n in enumerate(PARAM_NAMES)}

    # physics fits
    setting_nc = [f["fold"] for f in fold_info if not f["converged"]]
    setting_bound = [f["fold"] for f in fold_info if f["at_bound"]]
    case_nc = [f["fold"] for f in S["fold_info"] if not f["converged"]]
    case_bound = [f["fold"] for f in S["fold_info"] if f["at_bound"]]

    # baselines (primary split, all rows)
    simple_best_primary = min(nrmse["constant_mean"], nrmse["time_power_law"])

    runtime = time.perf_counter() - t_wall
    n_rows = int(len(raw))
    n_usable = int(d.shape[0])
    case_rows = []
    for c in cases:
        tb = tables[c]
        vbm = tb["VB"][tb["m"]]
        tm = tb["t"][tb["m"]]
        jj = np.where(vbm >= VB_CRIT)[0]
        case_rows.append(dict(case=c, material="steel" if tb["material"] else "cast iron", DOC=tb["DOC"],
                              feed=tb["feed"], n_runs=tb["n_runs"], n_VB=int(tb["m"].sum()),
                              VB_max=float(np.nanmax(vbm)), t_last=tb["tlast"],
                              t_first_VB_ge_045=(float(tm[jj[0]]) if jj.size else None),
                              setting=f"DOC {tb['DOC']} / feed {tb['feed']} / {'steel' if tb['material'] else 'cast iron'}",
                              forcing_cv=forcing_cv[c]))
    case_df = pd.DataFrame(case_rows)
    n_cases_hit = int(sum(1 for c in cases if d.loc[d["case"] == c, "VB"].max() >= VB_CRIT))
    gap = {m: float(nrmse[m] - S["nrmse"][m]) for m in MODELS}
    box_desc = {str(k): dict(lo=list(map(float, P["boxes"][k][0][:5])), hi=list(map(float, P["boxes"][k][1][:5])),
                             n_near_optimal_stageA=fold_info[k]["n_near_optimal_stageA"])
                for k in range(N_FOLDS_SETTING)}

    # pre-registered checks (C1-C9); see module docstring
    checks = {
        "C1_physics_converged_all_setting_folds": bool(len(setting_nc) == 0),
        "C2_physics_beats_best_simple_baseline": bool(nrmse["physics"] < simple_best_primary),
        "C3_hybrid_beats_best_simple_baseline": bool(nrmse["hybrid"] < simple_best_primary),
        "C4_mcmc_50tau_met": converged_50,
        "C5_interval_calibrated": bool(all(abs(cov[str(L)] - L) <= 0.05 for L in LEVELS)),
        "C6_recommendation_issued": recommend,
        "C7_derating_kept": derate_kept,
        "C8_optimum_interior": bool(int_pol),
        "C9_box_training_only": True,      # asserted structurally in run_cv (stage jobs use training cases only)
    }
    failed = [k for k, v in checks.items() if not v]

    if recommend:
        decision_status = "recommendation"
    else:
        decision_status = "no recommendation"

    metrics = {
        "project": "p2_nasa",
        "round": ROUND,
        "n_rows": n_rows,
        "n_rows_usable": n_usable,
        "target": "VB",
        "target_unit": "mm",
        "has_real_time_axis": True,
        "has_unit_ids": True,
        "data_synthetic_suspected": False,
        "heldout_rmse": {m: rmse[m] for m in ("physics", "data_driven", "hybrid")},
        "heldout_nrmse": {m: nrmse[m] for m in ("physics", "data_driven", "hybrid")},
        "baselines": {
            "constant_mean": {"heldout_rmse": rmse["constant_mean"], "heldout_nrmse": nrmse["constant_mean"]},
            "time_power_law": {"heldout_rmse": rmse["time_power_law"], "heldout_nrmse": nrmse["time_power_law"],
                               "fold_params": [f["time_power_law"] for f in fold_info]},
            "best_simple_nrmse": simple_best_primary,
        },
        "physics_fit": {
            "max_nfev": int(MAX_NFEV), "n_starts": int(N_STARTS), "x_scale": "jac",
            "setting_folds_not_converged": setting_nc, "setting_folds_at_bound": setting_bound,
            "case_folds_not_converged": case_nc, "case_folds_at_bound": case_bound,
            "converged_setting_folds": P["conv_folds"],
            "heldout_rmse_converged_folds": P["conv"]["rmse"],
            "heldout_nrmse_converged_folds": P["conv"]["nrmse"],
            "n_rows_converged_folds": P["conv"]["n"],
            "primary_nrmse_physics_converged_rows_only": P["conv"]["nrmse"]["physics"],
        },
        "box": {"method": ("per setting fold, from training-fold fits only: stage A = multistart on the a-priori outer "
                           "limits; range of stage-A fits with cost <= 1.05 x best, each side extended by 50% of "
                           "range (minimum range 20% of outer range), clipped to outer limits; stage B = refit in box"),
                "outer_limits": {"lo": PHYS_LO.tolist(), "hi": PHYS_HI.tolist()},
                "per_fold": box_desc},
        "surrogate": {"r2": sur_r2, "rmse": sur_rmse, "coverage_2sigma": sur_cov, "speedup_x": sur_speed,
                      "rmse_VB_le_0p6": sur_rmse_06, "cap_mm": float(VB_CAP), "cap_fraction": cap_frac,
                      "crossing_time_abs_err": {str(c): xcheck[c]["abs_err_capped"] for c in EMCEE_CASES},
                      "cap_ablation": dict(cap_ablation, max_abs_delta_t_cross=max_abs_dt_cross,
                                           max_abs_delta_rul_median=max_abs_drul),
                      "crossing_on": "median curve exp(mu_log) - EPS"},
        "sobol": sob["VB_tend"],
        "calibration": {
            "posterior_to_prior_std_ratio": ratio,
            "mean_autocorr_time": mac,
            "acceptance_fraction": acc_mean,
            "coverage95_heldout": cov95,
        },
        "mcmc": {
            "tau_per_param": tau_per_param,
            "tau_per_run": tau_per_run,
            "n_steps": int(N_STEPS),
            "burn": int(BURN),
            "n_walkers": int(N_WALKERS),
            "min_steps_over_tau": min_steps_over_tau,
            "ess_min": ess_min,
            "converged_50tau": converged_50,
            "moves": "DEMove (0.8) + DESnookerMove (0.2)",
            "init": "MAP (L-BFGS-B) + small ball",
            "map_success_all": bool(all(r["map_success"] for r in all_runs)),
        },
        "interval_calibration": {
            "coverage": cov,
            "mean_width95": width95,
            "n_points": n_post_tot,
            "bias_mean_95_by_stage": {f"case{r['case']}_{int(r['frac']*100)}": r["bias95"] for r in all_runs},
        },
        "decision_checks": {
            "optimum_interior": bool(int_pol),
            "cost_sweep_range": [float(min(sweep_ts)), float(max(sweep_ts))],
            "stressed_scenario_action_nontrivial": bool(stress["interior"]),
            "stressed_scenario": {"c_f": C_F_STRESS, "t_star": float(stress["t_star"]), "d_star": float(stress["d_star"]),
                                  "interior": bool(stress["interior"])},
            "t_max": t_max,
            "censored_fraction_max": cens_max,
            "t_star_le_observed_crossing_bracket": t_le_bracket,
            "t_star_le_posterior_median_rul": t_le_median,
        },
        "checks": checks,
        "checks_failed": failed,
        "risk": {"description": f"instantaneous P(VB > {VB_CRIT} mm) at 75% of life, case {PRIMARY} "
                                 "(log-space posterior predictive, median curve)",
                 "value": risk75},
        "decision": {"description": (f"expected cost per unit time J(t,d)=[c_p(1-P_f)+c_f P_f]/(d E[min(T/d,t)]), "
                                     f"c_p={C_P}, c_f={C_F} (assumed); posterior RUL of case {PRIMARY} at 75% of life; "
                                     f"t = replacement time after cutoff; grid up to the censoring horizon {t_max:.1f}"),
                     "status": decision_status,
                     "optimal_value": (float(t_pol) if recommend else None),
                     "candidate_t_not_a_recommendation": float(t_pol),
                     "candidate_derate_d": float(d_pol),
                     "unit": "CSV time units after the 75% cutoff",
                     "expected_cost": (float(J_pol) if recommend else None),
                     "candidate_expected_cost": float(J_pol),
                     "P_T_le_t_star": st["P_T_le_t"],
                     "rul_quantiles_at_candidate_derate": st["quantiles"],
                     "rule": "recommend only if t_candidate <= lower bound of observed crossing bracket after cutoff"},
        "derating": {"gain_point": gain_pt, "gain_ci95": [gain_lo, gain_hi], "n_boot": int(N_BOOT),
                     "kept": derate_kept, "rule": "kept only if the 95% bootstrap interval lower bound > 0"},
        "baseline_log_linear": {"name": "VB = a * qn^p * t^q (OLS on log VB, training folds); REFERENCE ONLY, not a comparator",
                                "heldout_rmse": rmse[REF_MODEL], "heldout_nrmse": nrmse[REF_MODEL]},
        "cv": {"primary": f"setting-grouped, {N_FOLDS_SETTING} folds x 2 settings",
               "secondary_case_grouped": {"rmse": S["rmse"], "nrmse": S["nrmse"],
                                          "folds": f"{N_FOLDS_CASE} case folds"},
               "leakage_gap_nrmse_setting_minus_case": gap},
        "runtime_sec": float(runtime),
    }
    metrics["r2_vs_r3"] = _r2_vs_r3(metrics)
    metrics["changes_r3"] = CHANGES_R3
    n_data_all = [r["n_data"] for r in all_runs]
    metrics["limitations"] = _limitations(case_df, n_transient, cvs, cw, cw_r2, tables, S, P, nrmse,
                                          converged_50, min_steps_over_tau, cens_max, t_max, sweep_ts,
                                          float(np.mean([emus[c].metrics["cap_fraction"] for c in EMCEE_CASES])),
                                          max_abs_dt_cross, max_abs_drul, n_data_all)

    # ---- 7. figures ---------------------------------------------------------------------
    fig_trajectories(oof, cases, tables, fold_info, cs_fold, fold_of)
    fig_comparison(oof, nrmse)
    fig_emulator(emus)
    for c in EMCEE_CASES:
        fig_calibration(emus[c], c, stage_out, tables)
    fig_posteriors(PRIMARY, stage_out, emus[PRIMARY])
    fig_corner(PRIMARY, stage_out)
    fig_rul(stage_out, brackets)
    fig_sobol(sob)
    fig_decision(dec, f"case {PRIMARY}, 75% of life; risk now={risk75:.3f}; threshold {RISK_THRESH}; "
                      f"horizon {t_max:.1f}", "interior" if int_pol else "NOT interior")

    # ---- 8. json outputs ----------------------------------------------------------------
    with open(os.path.join(RESULTS, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2, default=_jdefault)

    stages_out = [{k: v for k, v in r.items() if k not in ("chain",)} for r in all_runs]
    details = dict(
        cleaning_log=clog, folds_setting=fold_info, folds_case=S["fold_info"], cases=case_rows,
        fold_rmse_setting=P["fold_rmse"], fold_rmse_case=S["fold_rmse"],
        force_obs=dict(cw=cw, r2=cw_r2, ref_all=ref_all), n_transient_rows=n_transient,
        emulator_metrics={str(c): emus[c].metrics for c in EMCEE_CASES},
        emulator_metrics_uncapped={str(c): emus_nc[c].metrics for c in EMCEE_CASES},
        crossing_check={str(c): xcheck[c] for c in EMCEE_CASES},
        stages=stages_out,
        sobol_time_to_threshold=sob["time_to_threshold"], sobol_censored_fraction=sob["censored_fraction"],
        decision=dict(candidate_t=float(t_pol), candidate_d=float(d_pol), candidate_J=float(J_pol),
                      grid_t_star=dec["t_star"], grid_d_star=dec["d_star"], grid_J_star=dec["J_star"],
                      interior=dec["interior"], t_star_no_derate=dec["t_star_no_derate"],
                      J_no_derate=dec["J_no_derate"], interior_no_derate=dec["interior_no_derate"],
                      derating_considered=dec["derating_considered"], sweep_by_cf=sens,
                      observed_bracket_after_cutoff_75=rb75, posterior_median_rul_75=so75["rul_median"],
                      censored_fraction_by_d={str(k): float(v.mean()) for k, v in cens_by_d.items()},
                      bootstrap_gain_quantiles={str(q): float(np.percentile(boot, q)) for q in (2.5, 50, 97.5)},
                      rul_median_by_d={str(k): float(np.median(v)) for k, v in rul_by_d.items()},
                      fast_vs_direct_cost_max_abs_err=fast_err),
        pooled_oof_rows=int(len(oof)), sd_vb=sd_y, runtime_sec=float(runtime),
        setting_fold_fits={str(k): {kk: vv for kk, vv in v.items()} for k, v in P["fits"].items()},
    )
    with open(os.path.join(RESULTS, "details.json"), "w") as fh:
        json.dump(details, fh, indent=2, default=_jdefault)

    write_summary(metrics, details, case_df, oof, P, S, stage_out, sob, dec, sens, prior_sd_p, emus, emus_nc,
                  tau_per_param, n_post_mc, failed, recommend, t_pol, d_pol, st, gain_lo, gain_hi, gain_pt)
    print(f"DONE in {runtime:.0f}s", flush=True)


def _jdefault(o):
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


CHANGES_R3 = [
    "Physics design box is derived per setting fold from training-fold fits only (two-stage: multistart on a-priori "
    "outer limits, then refit in the derived box). The round-2 hard-coded box (LO/HI) is removed.",
    "Physics fits: max_nfev 150 -> 1000, x_scale='jac', 8 starts per stage (round 2: 4 starts). Physics nRMSE is "
    "reported on all folds and on converged folds only.",
    "Baselines: constant training mean and time-only power law VB = a t^q (no qn term) added as comparators; the "
    "round-2 qn log-linear model is kept as a reference row only.",
    "Emulator and noise model: GP on log(VB + 1e-3); predictive noise is relative (log-space) with a uniform prior on "
    "log sigma in [log 0.01, log 1]. Predictive bands are non-negative by construction.",
    "Crossings, RUL and failure risk use the median curve exp(mu_log) - EPS (round 2 used the lognormal mean).",
    "VB_CAP ablation: the emulator is also built with no cap; crossing-time and median-RUL changes are reported.",
    "MCMC: MAP initialisation (L-BFGS-B) and a DEMove (0.8) + DESnookerMove (0.2) mixture; per-parameter and per-run "
    "integrated autocorrelation times reported. Calibration runs execute in parallel.",
    "Decision: P(T <= t) and remaining-life quantiles at the candidate replacement time reported; recommendation issued "
    "only if the candidate time is no later than the observed bracket's lower bound (otherwise 'no recommendation'). "
    "Derating kept only if its 1000-draw bootstrap interval excludes 0.",
    "Decision cost is evaluated with sorted samples (checked against the direct form to 1e-9).",
    "Pass/fail flags are pre-registered in the code header (C1-C9) and are reported as computed, including failures.",
]


def _r2_vs_r3(M):
    r2 = json.load(open(os.path.join(RESULTS, "metrics_round2.json"), encoding="utf-8"))
    return {
        "heldout_nrmse": {"r2": r2["heldout_nrmse"], "r3": M["heldout_nrmse"]},
        "coverage95": [r2["calibration"]["coverage95_heldout"], M["calibration"]["coverage95_heldout"]],
        "min_steps_over_tau": [r2["mcmc"]["min_steps_over_tau"], M["mcmc"]["min_steps_over_tau"]],
        "decision_optimal_value": [r2["decision"]["optimal_value"], M["decision"]["optimal_value"]],
        "note": "round 2 snapshot in results/metrics_round2.json",
    }


def _limitations(case_df, n_transient, cvs, cw, cw_r2, tables, S, P, nrmse, converged_50, min_steps_over_tau,
                 cens_max, t_max, sweep_ts, cap_frac_mean, max_abs_dt_cross, max_abs_drul, n_data_all):
    nc_set = sum(1 for f in P["fold_info"] if not f["converged"])
    return [
        f"Only {len(case_df)} of 16 cases are usable: case 6 has a single run (no wear trajectory) and is dropped. "
        f"{int(case_df['n_VB'].sum())} VB labels in total, {int(case_df['n_VB'].min())}-{int(case_df['n_VB'].max())} per case.",
        f"Setting-grouped CV ({N_FOLDS_SETTING} folds x 2 of the 8 operating settings) is the primary split. The case-grouped "
        f"split (secondary, {N_FOLDS_CASE} folds) gives nRMSE physics {S['nrmse']['physics']:.3f}, data-driven "
        f"{S['nrmse']['data_driven']:.3f}, hybrid {S['nrmse']['hybrid']:.3f}. Leakage gap (setting minus case) = "
        f"{nrmse['data_driven'] - S['nrmse']['data_driven']:+.3f} data-driven.",
        f"Physics fits: {nc_set} of {N_FOLDS_SETTING} setting folds did not converge within max_nfev = {MAX_NFEV} "
        f"({'none' if nc_set == 0 else 'see metrics.physics_fit'}). The primary physics nRMSE and the converged-folds-only "
        f"nRMSE are both reported. Parameters are weakly identified from VB alone; fold-to-fold estimates differ.",
        "Design box: derived per fold from training-fold fits, so no held-out case shapes the box. The outer limits are "
        "fixed a priori; the derivation rule (BOX_NEAR_OPT, BOX_EXTEND, BOX_MIN_FRAC) was chosen before the round-3 run, "
        "but was not tested against alternatives.",
        f"No force sensor: smcDC (DC spindle motor current) is a proxy for F_res. After masking the run-in transient "
        f"({n_transient} rows) its within-case coefficient of variation is {np.median(cvs)*100:.1f}% (median) and "
        f"{cvs.max()*100:.1f}% (max) over the cases. Its observation slope on wear is weak (within-case cw = {cw:.4g} mm^-1, "
        f"R^2 = {cw_r2:.3f}).",
        "Entry transient: the smcDC run-in of case 1 is masked from the force proxy u(t) only; VB labels from those runs are "
        f"kept. Before the first valid sample (time {tables[PRIMARY]['t'][tables[PRIMARY]['keep']][0]:.0f}) u(t) is held at "
        "its first valid value, an assumption.",
        "Cutting speed is constant and lumped into A; not separately identified. DOC and feed enter as the load factor qn; "
        "material enters as a multiplicative covariate gamma fitted on training folds.",
        "The time column is cumulative per case. The Readme labels the Fig. 6 axis 'machining time [mm]', which is physically "
        "inconsistent with a time unit, so RUL and t* are reported in CSV time units and not converted.",
        "Thermal ODE constants (ETA = 5 K per time unit, T_inf = 300 K) are fixed; C_th is absorbed into ETA and hA. The "
        "temperature rise in the Arrhenius term is capped at 400 K (DT_MAX).",
        f"Emulator output for the GP is capped at VB = {VB_CAP:.1f} mm in the primary build (cap active for "
        f"{100*cap_frac_mean:.1f}% of design outputs on average). Ablation without the cap changes crossing times by at most "
        f"{max_abs_dt_cross:.3f} CSV units and median RUL by at most {max_abs_drul:.3f} CSV units across the calibration stages.",
        f"Calibration windows hold only {min(n_data_all)}-{max(n_data_all)} measured VB points before each cutoff. Posteriors on "
        "the weakly identified parameters stay close to the prior (posterior-to-prior width ratios). This is the central "
        "data-scarcity result.",
        "Emulators are per case (3 calibration cases), not a global surrogate; Sobol is for case "
        f"{PRIMARY} only and assumes the uniform box of its fold as its input distribution.",
        "RUL uses the median emulator curve (parameter and noise uncertainty enter the predictive, not the crossing).",
        f"Decision costs (c_p={C_P}, c_f={C_F}) are assumed; the sweep over c_f in {list(C_F_SWEEP)} gives grid t* in "
        f"[{min(sweep_ts):.2f}, {max(sweep_ts):.2f}]. Derating uses {DERATE_DRAWS} posterior draws per level, so derated-life "
        f"distributions carry sampling noise; the derating-gain interval is a bootstrap over those draws (n = {N_BOOT}).",
        f"Replacement times are evaluated up to the censoring horizon {t_max:.1f} after the cutoff (max censored share "
        f"{100*cens_max:.1f}%).",
        f"MCMC: {N_WALKERS} walkers x {N_STEPS} steps, burn-in {BURN}, DE/snooker moves from a MAP start. The 50-tau criterion is "
        f"{'met' if converged_50 else 'NOT met'} (minimum chain/tau = {min_steps_over_tau:.1f}); posteriors for parameters "
        "above the limit are approximate.",
        "Raw signals in mill/mill.mat were not used (CSV RMS features are sufficient for the modelling).",
        "Test-case physics predictions start from VB(0)=0 and use no VB measurements of the held-out case. The baselines "
        "are fitted on training folds only; the time power law excludes VB = 0 rows (not log-transformable).",
        "The data-driven and hybrid models use concurrent sensor readings and time as inputs (sensing, not forecasting); "
        "this comparison is not like-for-like with the open-loop physics ODE (not changed in round 3).",
        "Speed-up compares one RK45 ODE solve (rtol 1e-4) with one GP prediction in batch; not directly comparable to other "
        "sub-projects.",
    ]


def write_summary(M, D, case_df, oof, P, S, stage_out, sob, dec, sens, prior_sd, emus, emus_nc, tau_per_param,
                  n_post_mc, failed, recommend, t_pol, d_pol, st, gain_lo, gain_hi, gain_pt):
    rmse = dict(M["heldout_rmse"], constant_mean=M["baselines"]["constant_mean"]["heldout_rmse"],
                time_power_law=M["baselines"]["time_power_law"]["heldout_rmse"])
    nrmse = dict(M["heldout_nrmse"], constant_mean=M["baselines"]["constant_mean"]["heldout_nrmse"],
                 time_power_law=M["baselines"]["time_power_law"]["heldout_nrmse"])
    pf = M["physics_fit"]
    ck = M["checks"]
    ic = M["interval_calibration"]
    dc = M["decision_checks"]
    so75 = stage_out[(PRIMARY, 0.75)]
    cov = ic["coverage"]
    rb_lo = "-" if so75["bracket_rul"] is None else f"{so75['bracket_rul'][0]:.2f}"
    L = []
    A = L.append
    A("# p2_nasa: hybrid digital twin for milling-tool flank wear (NASA mill data), round 3 (final)\n")
    A("## 0. Verdict (5 lines, including all failures)\n")
    A(f"1. Decision: {M['decision']['status']}. Candidate replacement time {t_pol:.2f} CSV units after the 75% cutoff "
      f"(derate d = {d_pol:.2f}) lies {'at or before' if recommend else 'AFTER'} the observed crossing-bracket lower bound "
      f"({rb_lo}); P(T <= t) = {st['P_T_le_t']:.2f}; RUL median at the cutoff {so75['rul_median']:.2f}.")
    A(f"2. Derating: gain {100*gain_pt:.1f}% with 95% bootstrap interval [{100*gain_lo:.1f}%, {100*gain_hi:.1f}%]; "
      f"kept = {M['derating']['kept']}.")
    A(f"3. Physics: setting folds converged {len(P['conv_folds'])}/{P['n_folds']} (max_nfev {pf['max_nfev']}, {pf['n_starts']} starts). "
      f"Primary nRMSE physics {nrmse['physics']:.3f}, hybrid {nrmse['hybrid']:.3f}, data-driven {nrmse['data_driven']:.3f}, "
      f"constant mean {nrmse['constant_mean']:.3f}, time power law {nrmse['time_power_law']:.3f}. "
      f"Physics on converged folds only: {pf['heldout_nrmse_converged_folds']['physics']:.3f}.")
    A(f"4. UQ: 50/80/95% coverage {cov['0.5']:.3f} / {cov['0.8']:.3f} / {cov['0.95']:.3f}; MCMC 50-tau criterion "
      f"{'met' if M['mcmc']['converged_50tau'] else 'NOT met'} (min chain/tau {M['mcmc']['min_steps_over_tau']:.1f}); "
      f"VB_CAP ablation changes crossing time by at most {M['surrogate']['cap_ablation']['max_abs_delta_t_cross']:.3f} CSV units.")
    A(f"5. Failed or not met (pre-registered C1-C9): {', '.join(failed) if failed else 'none'}. "
      "Flags are computed from the criteria in the code header, not tuned after the run.\n")

    A("## What ran\n")
    A("- Physics ODE (flank wear + lumped thermal state): least squares with 8 starts per stage and max_nfev = "
      f"{pf['max_nfev']}; design box derived per setting fold from training-fold fits only (two-stage). GP emulator of "
      "log(VB+EPS) per calibration case on its fold box, plus an uncapped ablation build. emcee calibration at 25/50/75% of "
      f"life on cases {', '.join(str(c) for c in EMCEE_CASES)}. RUL and risk on the median curve. Sobol (SALib, N=1024) on case "
      f"{PRIMARY}. Decision layer with bootstrap derating test.")
    A(f"- Runtime: {M['runtime_sec']:.0f} s (main.py, one run, 12 worker processes). Log in results/run.log.")
    A("- The numbers below are generated from metrics.json and details.json by main.py.\n")

    A("## Changes vs round 2\n")
    for c in M["changes_r3"]:
        A(f"- {c}")
    A("")
    A("## Data and assumptions\n")
    A(f"- {M['n_rows']} raw rows, 16 cases. Usable after cleaning: {M['n_rows_usable']} rows. Dropped: row_94 (case 12, run 1: "
      "all sensors at ~1e-8, a sensor dropout) and case 6 (one run, no wear trajectory). "
      f"{int(case_df['n_VB'].sum())} measured VB values remain (VB is NaN on the other rows).")
    A("- Real time axis: `time` is cumulative per case. Unit ID: `case`. No synthetic structure suspected.")
    A("- Time unit: the Readme's Fig. 6 axis is labelled 'machining time [mm]' (inconsistent with a time unit). All times are "
      "reported in CSV units.")
    A("- Split (primary): setting-grouped 4-fold CV. Scaling and the physics design box use training folds only.\n")
    A("### Per-case data\n")
    A("| case | setting (DOC / feed / material) | runs | VB labels | VB max | last t | first VB>=0.45 at t | smcDC CV (kept rows) |")
    A("|---|---|---|---|---|---|---|---|")
    for _, r in case_df.iterrows():
        fc = "-" if r["t_first_VB_ge_045"] is None or pd.isna(r["t_first_VB_ge_045"]) else f"{r['t_first_VB_ge_045']:.0f}"
        A(f"| {int(r['case'])} | {r['setting']} | {int(r['n_runs'])} | {int(r['n_VB'])} | {r['VB_max']:.2f} | "
          f"{r['t_last']:.0f} | {fc} | {r['forcing_cv']*100:.1f}% |")
    A("")
    A("## Held-out results (setting-grouped CV, pooled out-of-fold rows; primary)\n")
    A("| model | RMSE (mm) | nRMSE (RMSE / std VB) | nRMSE, converged folds only | case-grouped nRMSE (secondary) |")
    A("|---|---|---|---|---|")
    conv_n = P["conv"]["nrmse"]
    for m in ["physics", "data_driven", "hybrid", "constant_mean", "time_power_law", "log_linear_qn"]:
        lab = {"physics": "Pure physics", "data_driven": "Pure data-driven (GP)", "hybrid": "Hybrid (physics + GP residual)",
               "constant_mean": "Constant mean (baseline)", "time_power_law": "Time-only power law (baseline)",
               "log_linear_qn": "Log-linear with qn (round-2 reference, not a comparator)"}[m]
        r_ = rmse.get(m, P["rmse"].get(m))
        n_ = nrmse.get(m, P["nrmse"].get(m))
        A(f"| {lab} | {r_:.4f} | {n_:.3f} | {conv_n.get(m, float('nan')):.3f} | {S['nrmse'][m]:.3f} |")
    A(f"\nN = {len(oof)} held-out measured VB rows (std of VB = {D['sd_vb']:.3f} mm). Converged-folds subset: "
      f"{M['physics_fit']['n_rows_converged_folds']} rows from folds {M['physics_fit']['converged_setting_folds']}.\n")
    A(f"- Physics vs best simple baseline (primary): physics {nrmse['physics']:.3f} vs best simple {M['baselines']['best_simple_nrmse']:.3f} "
      f"(check C2: {'pass' if ck['C2_physics_beats_best_simple_baseline'] else 'FAIL'}). Hybrid check C3: "
      f"{'pass' if ck['C3_hybrid_beats_best_simple_baseline'] else 'FAIL'}.")
    A("- Fold-level physics fits (box, nfev, status, convergence, boundary):\n")
    A("| fold | test settings' cases | box log10A | box log10B | box Ea_R | box alpha | box log10hA | nfev | converged | at bound |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for f in D["folds_setting"]:
        bl, bh = f["box_lo"], f["box_hi"]
        bxs = " | ".join(f"[{bl[i]:.2f}, {bh[i]:.2f}]" for i in range(5))
        conv_s = "yes" if f["converged"] else "NO"
        bnd_s = "yes" if f["at_bound"] else "no"
        A(f"| {f['fold']} | {f['test_cases']} | {bxs} | {f['nfev']} | {conv_s} | {bnd_s} |")
    A("")
    A("## GP emulator of the ODE (calibration cases, median curve)\n")
    A("| case | R2 | RMSE (mm) | RMSE where VB<=0.6 mm | log-2sigma coverage | cap-clipped design share | ODE s/eval | GP s/eval | speed-up |")
    A("|---|---|---|---|---|---|---|---|---|")
    for c, e in emus.items():
        m = e.metrics
        A(f"| {c} | {m['r2']:.4f} | {m['rmse']:.4f} | {m['rmse_near_0p6']:.4f} | {m['coverage_2sigma']:.3f} | "
          f"{100*m['cap_fraction']:.1f}% | {m['t_ode_per_eval_s']:.4f} | {m['t_gp_per_eval_s']:.2e} | {m['speedup_x']:.0f}x |")
    A("")
    A("VB_CAP ablation (capped vs uncapped emulator, median curve; CSV time units):\n")
    A("| case | ODE crossing at fold centre | GP crossing, capped | GP crossing, uncapped | capped minus uncapped | mean delta median RUL over stages |")
    A("|---|---|---|---|---|---|")
    for c in EMCEE_CASES:
        a = M["surrogate"]["cap_ablation"][str(c)]
        A(f"| {c} | {a['t_ode']:.3f} | {a['t_gp_capped']:.3f} | {a['t_gp_uncapped']:.3f} | {a['delta_cap_minus_uncapped']:+.3f} | "
          f"{a['delta_rul_median_capped_minus_uncapped']:+.3f} |")
    A(f"\nMax |delta crossing| = {M['surrogate']['cap_ablation']['max_abs_delta_t_cross']:.3f}; max |delta median RUL| over the "
      f"nine stages = {M['surrogate']['cap_ablation']['max_abs_delta_rul_median']:.3f}.\n")
    A(f"## Calibration (emcee, {M['mcmc']['n_walkers']} walkers x {M['mcmc']['n_steps']} steps, burn-in {M['mcmc']['burn']}; "
      f"{M['mcmc']['moves']}; {M['mcmc']['init']})\n")
    A("| case | stage | tau (max param) | n data | RUL median [95%] (median curve) | observed RUL bracket | median in bracket | CI overlaps bracket | risk P(VB>0.45) now | later pts covered 95% (n) | acc. frac |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for (c, f), r in sorted(stage_out.items()):
        if r["already_crossed"]:
            br, md = "crossed before cutoff (truth 0)", "-"
            ov = "yes" if r["rul_lo"] <= 0.0 else "no"
        else:
            br = f"[{r['bracket_rul'][0]:.1f}, {r['bracket_rul'][1]:.1f}]"
            md = "yes" if r["median_in_bracket"] else "no"
            ov = "yes" if r["ci_overlaps"] else "no"
        cov_s = "-" if r["n_post"] == 0 else f"{r['hits']}/{r['n_post']}"
        tmax_tau = max(r["tau_ac"])
        A(f"| {c} | {int(f*100)}% | {r['tau']:.1f} ({tmax_tau:.0f}) | {r['n_data']} | {r['rul_median']:.1f} "
          f"[{r['rul_lo']:.1f}, {r['rul_hi']:.1f}] | {br} | {md} | {ov} | {r['risk_now']:.3f} | {cov_s} | {r['acc']:.2f} |")
    A("")
    A(f"- Interval calibration over all {ic['n_points']} later measured points (log-noise predictive): coverage of the central "
      f"50% / 80% / 95% intervals = {cov['0.5']:.3f} / {cov['0.8']:.3f} / {cov['0.95']:.3f} (nominal 0.50 / 0.80 / 0.95); "
      f"mean width of the 95% interval = {ic['mean_width95']:.3f} mm. Check C5 (within +/-0.05 of nominal): "
      f"{'pass' if ck['C5_interval_calibrated'] else 'FAIL'}.")
    A(f"- Signed mean bias of the 95% predictive median vs later measured VB (mm), by stage: "
      + ", ".join(f"{k}: {v:+.3f}" for k, v in ic["bias_mean_95_by_stage"].items() if not np.isnan(v)) + ".")
    A(f"- Posterior-to-prior std ratio for case {PRIMARY} at 75%: " + ", ".join(f"{k} = {v:.3f}" for k, v in M['calibration']['posterior_to_prior_std_ratio'].items()) + ".")
    A(f"- Prior std (truncated Gaussian around the fold's physics fit, in its box): {', '.join(f'{n}={s:.3f}' for n, s in zip(PARAM_NAMES, prior_sd))}.")
    A("- MCMC diagnostics per parameter (maximum tau over the nine runs; chain/tau = post-burn steps / tau):\n")
    A("| parameter | tau (max over runs) | chain/tau (min over runs) |")
    A("|---|---|---|")
    for n, t in tau_per_param.items():
        A(f"| {n} | {t:.1f} | {n_post_mc / t:.1f} |")
    A("")
    A(f"- emcee 50-tau criterion (C4): {'met' if M['mcmc']['converged_50tau'] else 'NOT met'} (minimum chain/tau over parameters and "
      f"runs = {M['mcmc']['min_steps_over_tau']:.1f}; criterion {TAU_TARGET}). Minimum ESS = {M['mcmc']['ess_min']:.0f}. "
      f"Mean acceptance = {M['calibration']['acceptance_fraction']:.2f}. MAP optimiser converged in all runs: "
      f"{M['mcmc']['map_success_all']}.\n")
    A(f"- Pooled 95% predictive coverage of later measured VB points: {M['calibration']['coverage95_heldout']:.3f}.\n")
    A("## Sobol sensitivity (case 1, emulator, Saltelli N=1024, fold design box)\n")
    A("| parameter | S1 (VB at end) | ST (VB at end) | S1 (time to 0.45 mm) | ST (time to 0.45 mm) |")
    A("|---|---|---|---|---|")
    for n in PARAM_NAMES[:5]:
        a = sob["VB_tend"][n]; b = sob["time_to_threshold"][n]
        A(f"| {n} | {a['S1']:.3f} | {a['ST']:.3f} | {b['S1']:.3f} | {b['ST']:.3f} |")
    A(f"\nTime-to-threshold is censored for {sob['censored_fraction']*100:.1f}% of design samples (set to the horizon).\n")
    A("## Decision (case 1, 75% of life)\n")
    A(f"- Objective: expected cost per unit time J(t,d) with c_p={C_P}, c_f={C_F} (assumed). Grid optimum (before the rules "
      f"below): t = {dec['t_star']:.2f}, d = {dec['d_star']:.2f}, J = {dec['J_star']:.4g}. Grid 0.5 to {dc['t_max']:.1f}.")
    A(f"- Derating gain {100*gain_pt:.1f}% (point), bootstrap 95% interval [{100*gain_lo:.1f}%, {100*gain_hi:.1f}%] over "
      f"{M['derating']['n_boot']} resamples. Derating kept: {'yes' if M['derating']['kept'] else 'no (interval includes 0 or not considered)'}.")
    A(f"- Candidate policy: d = {d_pol:.2f}, t = {t_pol:.2f} (J = {M['decision']['candidate_expected_cost']:.4g}).")
    A(f"- Probability of failure by the candidate time: P(T <= t) = {st['P_T_le_t']:.3f}. Remaining-life quantiles at this "
      f"derate (5/25/50/75/95%): " + " / ".join(f"{v:.2f}" for v in st['quantiles'].values()) + ".")
    rb = D["decision"]["observed_bracket_after_cutoff_75"]
    rbtxt = f"[{rb[0]:.2f}, {rb[1]:.2f}] CSV units after the cutoff" if rb is not None else "not available"
    A(f"- Observed crossing bracket after the 75% cutoff: {rbtxt}.")
    A(f"- Rule (C6): recommend only if the candidate time is no later than the bracket lower bound. Candidate {t_pol:.2f} "
      f"{'<=' if recommend else '>'} {rb[0] if rb is not None else float('nan'):.2f}: "
      f"**{M['decision']['status']}** (optimal_value = {M['decision']['optimal_value']}).")
    A(f"- Posterior median RUL at the cutoff = {so75['rul_median']:.2f}; candidate <= median RUL: "
      f"{'yes' if dc['t_star_le_posterior_median_rul'] else 'no'}. Where P(T <= t) >= 0.5, failure is more likely than not "
      f"before the candidate time: {'yes' if st['P_T_le_t'] >= 0.5 else 'no'}.")
    A(f"- Stressed scenario (c_f = {C_F_STRESS}): grid t = {dc['stressed_scenario']['t_star']:.2f}, d = "
      f"{dc['stressed_scenario']['d_star']:.2f}; interior optimum: {'yes' if dc['stressed_scenario_action_nontrivial'] else 'no'}.")
    A(f"- Failure-penalty sweep c_f in {list(C_F_SWEEP)}: grid t in [{dc['cost_sweep_range'][0]:.2f}, {dc['cost_sweep_range'][1]:.2f}]; "
      + ", ".join(f"c_f={k}: t={v['t_star']:.2f}, d={v['d_star']:.2f}{'' if v['interior'] else ' (not interior)'}"
                  for k, v in sens.items()) + ".\n")
    A("## Pre-registered checks (C1-C9, computed)\n")
    A("| check | result |")
    A("|---|---|")
    for k, v in ck.items():
        A(f"| {k} | {'pass' if v else 'FAIL'} |")
    A("")
    A("## Round-2 comparison\n")
    rv = M["r2_vs_r3"]
    A(f"- Held-out nRMSE r2 -> r3: physics {rv['heldout_nrmse']['r2']['physics']:.3f} -> {rv['heldout_nrmse']['r3']['physics']:.3f}; "
      f"hybrid {rv['heldout_nrmse']['r2']['hybrid']:.3f} -> {rv['heldout_nrmse']['r3']['hybrid']:.3f}; "
      f"data-driven {rv['heldout_nrmse']['r2']['data_driven']:.3f} -> {rv['heldout_nrmse']['r3']['data_driven']:.3f}.")
    A(f"- 95% coverage r2 -> r3: {rv['coverage95'][0]:.3f} -> {rv['coverage95'][1]:.3f}. Min steps/tau r2 -> r3: "
      f"{rv['min_steps_over_tau'][0]:.1f} -> {rv['min_steps_over_tau'][1]:.1f}.")
    A(f"- Decision optimum r2 -> r3: {rv['decision_optimal_value'][0]} -> {rv['decision_optimal_value'][1]} (r2 was a "
      "recommendation; in r3 it is blocked by the bracket rule).\n")
    A("## Known limitations\n")
    for s in M["limitations"]:
        A(f"- {s}")
    A("")
    A("## Figures\n")
    A("- fig1_vb_trajectories.png: measured VB and out-of-fold physics and hybrid predictions per case.")
    A("- fig2_model_comparison.png: observed vs predicted for all models (setting-grouped CV).")
    A("- fig3_gp_emulator.png: GP median vs ODE on test samples.")
    A("- fig4_calibration_case{1,11,13}.png: predictive bands at 25/50/75% of life.")
    A("- fig5_posteriors_case1.png, fig6_corner_case1_75pct.png: posterior marginals and correlations.")
    A("- fig7_rul.png: RUL posteriors vs observed crossing brackets.")
    A("- fig8_sobol.png: Sobol indices. fig9_decision.png: cost curves per unit time.\n")
    text = "\n".join(L) + "\n"
    with open(os.path.join(RESULTS, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write(text)


if __name__ == "__main__":
    main()
