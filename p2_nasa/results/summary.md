# p2_nasa: hybrid digital twin for milling-tool flank wear (NASA mill data), round 3 (final)

## 0. Verdict (5 lines, including all failures)

1. Decision: no recommendation. Candidate replacement time 16.26 CSV units after the 75% cutoff (derate d = 1.00) lies AFTER the observed crossing-bracket lower bound (2.00); P(T <= t) = 0.82; RUL median at the cutoff 6.85.
2. Derating: gain 2.4% with 95% bootstrap interval [0.0%, 27.1%]; kept = False.
3. Physics: setting folds converged 4/4 (max_nfev 1000, 8 starts). Primary nRMSE physics 0.561, hybrid 0.504, data-driven 0.874, constant mean 1.031, time power law 0.949. Physics on converged folds only: 0.561.
4. UQ: 50/80/95% coverage 0.757 / 0.871 / 0.943; MCMC 50-tau criterion met (min chain/tau 111.6); VB_CAP ablation changes crossing time by at most 0.017 CSV units.
5. Failed or not met (pre-registered C1-C9): C5_interval_calibrated, C6_recommendation_issued, C7_derating_kept. Flags are computed from the criteria in the code header, not tuned after the run.

## What ran

- Physics ODE (flank wear + lumped thermal state): least squares with 8 starts per stage and max_nfev = 1000; design box derived per setting fold from training-fold fits only (two-stage). GP emulator of log(VB+EPS) per calibration case on its fold box, plus an uncapped ablation build. emcee calibration at 25/50/75% of life on cases 1, 11, 13. RUL and risk on the median curve. Sobol (SALib, N=1024) on case 1. Decision layer with bootstrap derating test.
- Runtime: 293 s (main.py, one run, 12 worker processes). Log in results/run.log.
- The numbers below are generated from metrics.json and details.json by main.py.

## Changes vs round 2

- Physics design box is derived per setting fold from training-fold fits only (two-stage: multistart on a-priori outer limits, then refit in the derived box). The round-2 hard-coded box (LO/HI) is removed.
- Physics fits: max_nfev 150 -> 1000, x_scale='jac', 8 starts per stage (round 2: 4 starts). Physics nRMSE is reported on all folds and on converged folds only.
- Baselines: constant training mean and time-only power law VB = a t^q (no qn term) added as comparators; the round-2 qn log-linear model is kept as a reference row only.
- Emulator and noise model: GP on log(VB + 1e-3); predictive noise is relative (log-space) with a uniform prior on log sigma in [log 0.01, log 1]. Predictive bands are non-negative by construction.
- Crossings, RUL and failure risk use the median curve exp(mu_log) - EPS (round 2 used the lognormal mean).
- VB_CAP ablation: the emulator is also built with no cap; crossing-time and median-RUL changes are reported.
- MCMC: MAP initialisation (L-BFGS-B) and a DEMove (0.8) + DESnookerMove (0.2) mixture; per-parameter and per-run integrated autocorrelation times reported. Calibration runs execute in parallel.
- Decision: P(T <= t) and remaining-life quantiles at the candidate replacement time reported; recommendation issued only if the candidate time is no later than the observed bracket's lower bound (otherwise 'no recommendation'). Derating kept only if its 1000-draw bootstrap interval excludes 0.
- Decision cost is evaluated with sorted samples (checked against the direct form to 1e-9).
- Pass/fail flags are pre-registered in the code header (C1-C9) and are reported as computed, including failures.

## Data and assumptions

- 167 raw rows, 16 cases. Usable after cleaning: 165 rows. Dropped: row_94 (case 12, run 1: all sensors at ~1e-8, a sensor dropout) and case 6 (one run, no wear trajectory). 145 measured VB values remain (VB is NaN on the other rows).
- Real time axis: `time` is cumulative per case. Unit ID: `case`. No synthetic structure suspected.
- Time unit: the Readme's Fig. 6 axis is labelled 'machining time [mm]' (inconsistent with a time unit). All times are reported in CSV units.
- Split (primary): setting-grouped 4-fold CV. Scaling and the physics design box use training folds only.

### Per-case data

| case | setting (DOC / feed / material) | runs | VB labels | VB max | last t | first VB>=0.45 at t | smcDC CV (kept rows) |
|---|---|---|---|---|---|---|---|
| 1 | DOC 1.5 / feed 0.5 / cast iron | 17 | 13 | 0.50 | 48 | 41 | 2.2% |
| 2 | DOC 0.75 / feed 0.5 / cast iron | 14 | 13 | 0.55 | 72 | 66 | 1.6% |
| 3 | DOC 0.75 / feed 0.25 / cast iron | 14 | 14 | 0.55 | 81 | 81 | 2.8% |
| 4 | DOC 1.5 / feed 0.25 / cast iron | 7 | 7 | 0.49 | 39 | 39 | 1.5% |
| 5 | DOC 1.5 / feed 0.5 / steel | 6 | 6 | 0.74 | 15 | 12 | 2.1% |
| 7 | DOC 0.75 / feed 0.25 / steel | 8 | 7 | 0.46 | 21 | 19 | 2.2% |
| 8 | DOC 0.75 / feed 0.5 / steel | 6 | 5 | 0.62 | 12 | 12 | 1.1% |
| 9 | DOC 1.5 / feed 0.5 / cast iron | 9 | 9 | 0.81 | 46 | 34 | 2.6% |
| 10 | DOC 1.5 / feed 0.25 / cast iron | 10 | 10 | 0.70 | 57 | 39 | 1.6% |
| 11 | DOC 0.75 / feed 0.25 / cast iron | 23 | 20 | 0.76 | 105 | 80 | 2.0% |
| 12 | DOC 0.75 / feed 0.5 / cast iron | 14 | 12 | 0.65 | 74 | 55 | 1.8% |
| 13 | DOC 0.75 / feed 0.25 / steel | 15 | 13 | 1.53 | 45 | 22 | 2.1% |
| 14 | DOC 0.75 / feed 0.5 / steel | 9 | 7 | 1.14 | 24 | 18 | 2.5% |
| 15 | DOC 1.5 / feed 0.25 / steel | 7 | 6 | 0.70 | 19 | 13 | 1.3% |
| 16 | DOC 1.5 / feed 0.5 / steel | 6 | 3 | 0.62 | 9 | 9 | 2.5% |

## Held-out results (setting-grouped CV, pooled out-of-fold rows; primary)

| model | RMSE (mm) | nRMSE (RMSE / std VB) | nRMSE, converged folds only | case-grouped nRMSE (secondary) |
|---|---|---|---|---|
| Pure physics | 0.1454 | 0.561 | 0.561 | 0.519 |
| Pure data-driven (GP) | 0.2263 | 0.874 | 0.874 | 0.545 |
| Hybrid (physics + GP residual) | 0.1306 | 0.504 | 0.504 | 0.473 |
| Constant mean (baseline) | 0.2671 | 1.031 | 1.031 | 1.012 |
| Time-only power law (baseline) | 0.2459 | 0.949 | 0.949 | 0.871 |
| Log-linear with qn (round-2 reference, not a comparator) | 0.3356 | 1.296 | 1.296 | 0.950 |

N = 145 held-out measured VB rows (std of VB = 0.259 mm). Converged-folds subset: 145 rows from folds [0, 1, 2, 3].

- Physics vs best simple baseline (primary): physics 0.561 vs best simple 0.949 (check C2: pass). Hybrid check C3: pass.
- Fold-level physics fits (box, nfev, status, convergence, boundary):

| fold | test settings' cases | box log10A | box log10B | box Ea_R | box alpha | box log10hA | nfev | converged | at bound |
|---|---|---|---|---|---|---|---|---|---|
| 0 | [3, 4, 10, 11] | [-2.98, -1.92] | [-2.92, -1.66] | [0.00, 223.37] | [0.00, 5.00] | [-4.00, 0.00] | 14 | yes | yes |
| 1 | [7, 13, 15] | [-2.88, -1.86] | [-2.98, -1.77] | [0.00, 229.99] | [0.00, 1.32] | [-3.59, 0.00] | 15 | yes | yes |
| 2 | [1, 2, 9, 12] | [-2.95, -1.76] | [-2.99, -1.68] | [0.00, 342.98] | [0.00, 5.00] | [-4.00, 0.00] | 13 | yes | yes |
| 3 | [5, 8, 14, 16] | [-2.93, -1.90] | [-3.01, -1.80] | [0.00, 403.53] | [0.19, 5.00] | [-4.00, -3.16] | 10 | yes | yes |

## GP emulator of the ODE (calibration cases, median curve)

| case | R2 | RMSE (mm) | RMSE where VB<=0.6 mm | log-2sigma coverage | cap-clipped design share | ODE s/eval | GP s/eval | speed-up |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.9957 | 0.0486 | 0.0017 | 0.958 | 2.0% | 0.0062 | 8.24e-06 | 753x |
| 11 | 0.9989 | 0.0214 | 0.0009 | 0.970 | 1.2% | 0.0094 | 8.56e-06 | 1093x |
| 13 | 0.9981 | 0.0327 | 0.0006 | 0.963 | 2.1% | 0.0069 | 9.89e-06 | 696x |

VB_CAP ablation (capped vs uncapped emulator, median curve; CSV time units):

| case | ODE crossing at fold centre | GP crossing, capped | GP crossing, uncapped | capped minus uncapped | mean delta median RUL over stages |
|---|---|---|---|---|---|
| 1 | 28.470 | 28.178 | 28.161 | +0.017 | +0.038 |
| 11 | 62.857 | 62.989 | 62.999 | -0.010 | -0.004 |
| 13 | 23.534 | 23.528 | 23.522 | +0.005 | +0.001 |

Max |delta crossing| = 0.017; max |delta median RUL| over the nine stages = 0.075.

## Calibration (emcee, 32 walkers x 12000 steps, burn-in 2000; DEMove (0.8) + DESnookerMove (0.2); MAP (L-BFGS-B) + small ball)

| case | stage | tau (max param) | n data | RUL median [95%] (median curve) | observed RUL bracket | median in bracket | CI overlaps bracket | risk P(VB>0.45) now | later pts covered 95% (n) | acc. frac |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 25% | 12.0 (57) | 2 | 36.7 [10.6, 64.8] | [26.0, 29.0] | no | yes | 0.081 | 11/11 | 0.15 |
| 1 | 50% | 24.0 (50) | 5 | 18.5 [0.0, 51.8] | [14.0, 17.0] | no | yes | 0.263 | 8/8 | 0.15 |
| 1 | 75% | 36.0 (48) | 9 | 6.9 [0.0, 29.8] | [2.0, 5.0] | no | yes | 0.420 | 4/4 | 0.16 |
| 11 | 25% | 26.2 (39) | 7 | 53.2 [22.8, 100.1] | [45.8, 53.8] | yes | yes | 0.092 | 13/13 | 0.19 |
| 11 | 50% | 52.5 (38) | 12 | 28.9 [5.0, 62.4] | [19.5, 27.5] | no | yes | 0.248 | 8/8 | 0.19 |
| 11 | 75% | 78.8 (34) | 15 | 2.5 [0.0, 30.1] | [0.0, 1.2] | no | yes | 0.480 | 5/5 | 0.19 |
| 13 | 25% | 11.2 (45) | 3 | 11.8 [4.8, 22.0] | [7.8, 10.8] | no | yes | 0.025 | 10/10 | 0.16 |
| 13 | 50% | 22.5 (67) | 6 | 0.0 [0.0, 3.6] | crossed before cutoff (truth 0) | - | yes | 0.484 | 5/7 | 0.15 |
| 13 | 75% | 33.8 (90) | 9 | 0.0 [0.0, 0.0] | crossed before cutoff (truth 0) | - | yes | 0.991 | 2/4 | 0.14 |

- Interval calibration over all 70 later measured points (log-noise predictive): coverage of the central 50% / 80% / 95% intervals = 0.757 / 0.871 / 0.943 (nominal 0.50 / 0.80 / 0.95); mean width of the 95% interval = 1.659 mm. Check C5 (within +/-0.05 of nominal): FAIL.
- Signed mean bias of the 95% predictive median vs later measured VB (mm), by stage: case1_25: +0.062, case1_50: +0.016, case1_75: -0.005, case11_25: +0.034, case11_50: +0.089, case11_75: +0.117, case13_25: +0.230, case13_50: +0.283, case13_75: +0.360.
- Posterior-to-prior std ratio for case 1 at 75%: log10A = 0.622, log10B = 0.772, Ea_R = 0.999, alpha = 1.000, log10hA = 0.982.
- Prior std (truncated Gaussian around the fold's physics fit, in its box): log10A=0.259, log10B=0.284, Ea_R=57.030, alpha=0.757, log10hA=0.601.
- MCMC diagnostics per parameter (maximum tau over the nine runs; chain/tau = post-burn steps / tau):

| parameter | tau (max over runs) | chain/tau (min over runs) |
|---|---|---|
| log10A | 89.6 | 111.6 |
| log10B | 70.3 | 142.2 |
| Ea_R | 49.9 | 200.5 |
| alpha | 52.8 | 189.4 |
| log10hA | 50.0 | 200.0 |
| log_sigma | 56.8 | 176.0 |

- emcee 50-tau criterion (C4): met (minimum chain/tau over parameters and runs = 111.6; criterion 50). Minimum ESS = 3570. Mean acceptance = 0.16. MAP optimiser converged in all runs: True.

- Pooled 95% predictive coverage of later measured VB points: 0.943.

## Sobol sensitivity (case 1, emulator, Saltelli N=1024, fold design box)

| parameter | S1 (VB at end) | ST (VB at end) | S1 (time to 0.45 mm) | ST (time to 0.45 mm) |
|---|---|---|---|---|
| log10A | 0.773 | 0.789 | 0.595 | 0.741 |
| log10B | 0.177 | 0.177 | 0.239 | 0.391 |
| Ea_R | 0.022 | 0.042 | 0.007 | 0.015 |
| alpha | -0.002 | 0.002 | 0.001 | 0.000 |
| log10hA | 0.003 | 0.012 | 0.007 | 0.008 |

Time-to-threshold is censored for 4.1% of design samples (set to the horizon).

## Decision (case 1, 75% of life)

- Objective: expected cost per unit time J(t,d) with c_p=1.0, c_f=5.0 (assumed). Grid optimum (before the rules below): t = 19.49, d = 0.85, J = 0.58. Grid 0.5 to 40.8.
- Derating gain 2.4% (point), bootstrap 95% interval [0.0%, 27.1%] over 1000 resamples. Derating kept: no (interval includes 0 or not considered).
- Candidate policy: d = 1.00, t = 16.26 (J = 0.5946).
- Probability of failure by the candidate time: P(T <= t) = 0.820. Remaining-life quantiles at this derate (5/25/50/75/95%): 0.00 / 0.00 / 6.20 / 13.30 / 21.37.
- Observed crossing bracket after the 75% cutoff: [2.00, 5.00] CSV units after the cutoff.
- Rule (C6): recommend only if the candidate time is no later than the bracket lower bound. Candidate 16.26 > 2.00: **no recommendation** (optimal_value = None).
- Posterior median RUL at the cutoff = 6.85; candidate <= median RUL: no. Where P(T <= t) >= 0.5, failure is more likely than not before the candidate time: yes.
- Stressed scenario (c_f = 10.0): grid t = 19.49, d = 0.85; interior optimum: yes.
- Failure-penalty sweep c_f in [2.0, 5.0, 10.0, 20.0]: grid t in [15.45, 40.70]; c_f=2.0: t=40.70, d=0.85, c_f=5.0: t=19.49, d=0.85, c_f=10.0: t=19.49, d=0.85, c_f=20.0: t=15.45, d=0.85.

## Pre-registered checks (C1-C9, computed)

| check | result |
|---|---|
| C1_physics_converged_all_setting_folds | pass |
| C2_physics_beats_best_simple_baseline | pass |
| C3_hybrid_beats_best_simple_baseline | pass |
| C4_mcmc_50tau_met | pass |
| C5_interval_calibrated | FAIL |
| C6_recommendation_issued | FAIL |
| C7_derating_kept | FAIL |
| C8_optimum_interior | pass |
| C9_box_training_only | pass |

## Round-2 comparison

- Held-out nRMSE r2 -> r3: physics 0.540 -> 0.561; hybrid 0.483 -> 0.504; data-driven 0.874 -> 0.874.
- 95% coverage r2 -> r3: 0.843 -> 0.943. Min steps/tau r2 -> r3: 17.5 -> 111.6.
- Decision optimum r2 -> r3: 4.641102756892232 -> None (r2 was a recommendation; in r3 it is blocked by the bracket rule).

## Known limitations

- Only 15 of 16 cases are usable: case 6 has a single run (no wear trajectory) and is dropped. 145 VB labels in total, 3-20 per case.
- Setting-grouped CV (4 folds x 2 of the 8 operating settings) is the primary split. The case-grouped split (secondary, 5 folds) gives nRMSE physics 0.519, data-driven 0.545, hybrid 0.473. Leakage gap (setting minus case) = +0.329 data-driven.
- Physics fits: 0 of 4 setting folds did not converge within max_nfev = 1000 (none). The primary physics nRMSE and the converged-folds-only nRMSE are both reported. Parameters are weakly identified from VB alone; fold-to-fold estimates differ.
- Design box: derived per fold from training-fold fits, so no held-out case shapes the box. The outer limits are fixed a priori; the derivation rule (BOX_NEAR_OPT, BOX_EXTEND, BOX_MIN_FRAC) was chosen before the round-3 run, but was not tested against alternatives.
- No force sensor: smcDC (DC spindle motor current) is a proxy for F_res. After masking the run-in transient (6 rows) its within-case coefficient of variation is 2.1% (median) and 2.8% (max) over the cases. Its observation slope on wear is weak (within-case cw = 0.01795 mm^-1, R^2 = 0.042).
- Entry transient: the smcDC run-in of case 1 is masked from the force proxy u(t) only; VB labels from those runs are kept. Before the first valid sample (time 19) u(t) is held at its first valid value, an assumption.
- Cutting speed is constant and lumped into A; not separately identified. DOC and feed enter as the load factor qn; material enters as a multiplicative covariate gamma fitted on training folds.
- The time column is cumulative per case. The Readme labels the Fig. 6 axis 'machining time [mm]', which is physically inconsistent with a time unit, so RUL and t* are reported in CSV time units and not converted.
- Thermal ODE constants (ETA = 5 K per time unit, T_inf = 300 K) are fixed; C_th is absorbed into ETA and hA. The temperature rise in the Arrhenius term is capped at 400 K (DT_MAX).
- Emulator output for the GP is capped at VB = 3.0 mm in the primary build (cap active for 1.8% of design outputs on average). Ablation without the cap changes crossing times by at most 0.017 CSV units and median RUL by at most 0.075 CSV units across the calibration stages.
- Calibration windows hold only 2-15 measured VB points before each cutoff. Posteriors on the weakly identified parameters stay close to the prior (posterior-to-prior width ratios). This is the central data-scarcity result.
- Emulators are per case (3 calibration cases), not a global surrogate; Sobol is for case 1 only and assumes the uniform box of its fold as its input distribution.
- RUL uses the median emulator curve (parameter and noise uncertainty enter the predictive, not the crossing).
- Decision costs (c_p=1.0, c_f=5.0) are assumed; the sweep over c_f in [2.0, 5.0, 10.0, 20.0] gives grid t* in [15.45, 40.70]. Derating uses 200 posterior draws per level, so derated-life distributions carry sampling noise; the derating-gain interval is a bootstrap over those draws (n = 1000).
- Replacement times are evaluated up to the censoring horizon 40.8 after the cutoff (max censored share 4.5%).
- MCMC: 32 walkers x 12000 steps, burn-in 2000, DE/snooker moves from a MAP start. The 50-tau criterion is met (minimum chain/tau = 111.6); posteriors for parameters above the limit are approximate.
- Raw signals in mill/mill.mat were not used (CSV RMS features are sufficient for the modelling).
- Test-case physics predictions start from VB(0)=0 and use no VB measurements of the held-out case. The baselines are fitted on training folds only; the time power law excludes VB = 0 rows (not log-transformable).
- The data-driven and hybrid models use concurrent sensor readings and time as inputs (sensing, not forecasting); this comparison is not like-for-like with the open-loop physics ODE (not changed in round 3).
- Speed-up compares one RK45 ODE solve (rtol 1e-4) with one GP prediction in batch; not directly comparable to other sub-projects.

## Figures

- fig1_vb_trajectories.png: measured VB and out-of-fold physics and hybrid predictions per case.
- fig2_model_comparison.png: observed vs predicted for all models (setting-grouped CV).
- fig3_gp_emulator.png: GP median vs ODE on test samples.
- fig4_calibration_case{1,11,13}.png: predictive bands at 25/50/75% of life.
- fig5_posteriors_case1.png, fig6_corner_case1_75pct.png: posterior marginals and correlations.
- fig7_rul.png: RUL posteriors vs observed crossing brackets.
- fig8_sobol.png: Sobol indices. fig9_decision.png: cost curves per unit time.

