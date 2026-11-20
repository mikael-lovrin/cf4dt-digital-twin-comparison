# p2_nasa: hybrid digital twin for milling-tool flank wear (NASA mill data), round 2

## What ran

- Physics ODE (flank wear + lumped cutting-zone thermal state), fitted by least squares per fold; GP emulator of the ODE per calibration case; data-driven GP; hybrid (physics + GP residual); empirical log-linear baseline; emcee calibration at 25/50/75% of life on cases 1, 11, 13; RUL to VB = 0.45 mm; Sobol (SALib, N=1024) on case 1; decision layer.
- Runtime: 453 s (main.py, one run). Full log in results/run.log.

## Changes vs round 1

- CV is grouped by operating setting (DOC, feed, material; 8 settings, 4 folds of 2). Replicates of a setting never straddle train and test. The round-1 case-grouped split is kept as a secondary result (nRMSE physics 0.513, data-driven 0.545, hybrid 0.468).
- Added the log-linear baseline VB = a*qn^p*t^q (fitted on training folds): held-out nRMSE 1.296.
- Run-in transient of case 1 (smcDC < 1.0; 6 rows) masked from the force proxy u(t).
- MCMC: quiet=True removed; per-parameter autocorrelation times and chain/tau reported; chains lengthened to 12000 steps (burn-in 2000). Check: NOT met.
- Predictive intervals reported at 50/80/95% (coverage and mean width).
- Decision: replacement grid extended to the censoring horizon; cost renamed per unit time (its denominator is time); interior-optimum test, cost sweep, stressed scenario and consistency check against the observed crossing added.
- Emulator: RMSE for VB <= 0.6 mm, cap fraction and crossing-time error reported. Physics fit convergence flags added.
- summary.md numbers are generated from metrics.json in code.

## Data and assumptions

- 167 rows, 16 cases. Dropped: row_94 (case 12, run 1: all sensors at ~1e-8, a sensor dropout) and case 6 (one run, no wear trajectory). 15 cases, 145 measured VB values remain (VB is NaN on the other rows).
- Real time axis: `time` is cumulative per case (restarts per case). Unit ID: `case`. No synthetic structure suspected.
- Time unit: the Readme's Fig. 6 axis is labelled 'machining time [mm]' (inconsistent with a time unit). All times are reported in CSV units.
- Force proxy: smcDC (DC spindle motor current), transient rows masked. Operating inputs: DOC and feed (via qn = DOC*feed normalised). Material is a covariate (gamma). Cutting speed is constant and lumped into A.
- Split (primary): setting-grouped 4-fold CV. Scaling uses training folds only. Physics initial condition VB(0)=0.

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

| model | RMSE (mm) | nRMSE (RMSE / std(VB)) | case-grouped nRMSE (secondary) |
|---|---|---|---|
| Pure physics | 0.1398 | 0.540 | 0.513 |
| Pure data-driven (GP) | 0.2263 | 0.874 | 0.545 |
| Hybrid (physics + GP residual) | 0.1251 | 0.483 | 0.468 |
| Log-linear baseline | 0.3356 | 1.296 | 0.950 |

N = 145 held-out measured VB rows (std of VB = 0.259 mm).

- Ranking by setting-grouped nRMSE: best = Hybrid (physics + GP residual). Spread across the four models = 0.812 in nRMSE; no significance test was run, and fold-level RMSE is in details.json. With 145 rows in 4 folds the ranking should not be over-read.

## GP emulator of the ODE (calibration cases)

| case | R2 | RMSE (mm) | RMSE where VB<=0.6 mm | 2-sigma coverage | cap-clipped design share | ODE s/eval | GP s/eval | speed-up |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.9596 | 0.2568 | 0.0221 | 0.961 | 27.0% | 0.0055 | 1.09e-05 | 510x |
| 11 | 0.9431 | 0.2624 | 0.0101 | 0.950 | 12.8% | 0.0086 | 1.13e-05 | 759x |
| 13 | 0.9619 | 0.2459 | 0.0163 | 0.954 | 23.8% | 0.0081 | 9.47e-06 | 853x |

Pooled over the three emulators: R2 = 0.9566, RMSE = 0.2554 mm, RMSE for VB <= 0.6 mm = 0.0162 mm, 2-sigma coverage = 0.955. Emulated VB is capped at 3.0 mm. Crossing-time error at the prior centre (GP vs ODE, CSV units): case 1: 2.064, case 11: 6.381, case 13: 1.204.

## Calibration (emcee, 32 walkers x 12000 steps, burn-in 2000)

| case | stage | tau | n data | RUL median [95%] | observed RUL bracket | median in bracket | CI overlaps bracket | risk P(VB>0.45) now | later pts covered 95% (n) | acc. frac |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 25% | 12.0 | 2 | 16.4 [2.6, 64.8] | [26.0, 29.0] | no | yes | 0.014 | 11/11 | 0.40 |
| 1 | 50% | 24.0 | 5 | 6.9 [1.2, 15.3] | [14.0, 17.0] | no | yes | 0.028 | 7/8 | 0.36 |
| 1 | 75% | 36.0 | 9 | 1.2 [0.0, 7.3] | [2.0, 5.0] | no | yes | 0.377 | 3/4 | 0.31 |
| 11 | 25% | 26.2 | 7 | 45.8 [30.7, 64.2] | [45.8, 53.8] | yes | yes | 0.000 | 13/13 | 0.31 |
| 11 | 50% | 52.5 | 12 | 28.1 [19.0, 37.7] | [19.5, 27.5] | no | yes | 0.000 | 8/8 | 0.30 |
| 11 | 75% | 78.8 | 15 | 0.4 [0.0, 7.8] | [0.0, 1.2] | yes | yes | 0.489 | 3/5 | 0.26 |
| 13 | 25% | 11.2 | 3 | 12.8 [6.1, 39.8] | [7.8, 10.8] | no | yes | 0.004 | 8/10 | 0.38 |
| 13 | 50% | 22.5 | 6 | 0.0 [0.0, 2.2] | crossed before cutoff (truth 0) | - | yes | 0.592 | 4/7 | 0.35 |
| 13 | 75% | 33.8 | 9 | 0.0 [0.0, 0.0] | crossed before cutoff (truth 0) | - | yes | 1.000 | 2/4 | 0.35 |

- Interval calibration over all 70 later measured points: coverage of the central 50% / 80% / 95% predictive intervals = 0.343 / 0.586 / 0.843 (nominal 0.50 / 0.80 / 0.95); mean width of the 95% interval = 0.542 mm.
- RUL vs observed bracket: CI overlaps the bracket in 7/7 stages where the crossing is still ahead of the cutoff; the median lies inside the bracket in 2/7. 2 stages have the crossing already observed before the cutoff (truth 0).
- Stages whose posterior median RUL is 0 while the crossing is still ahead of the cutoff (model crosses early): none. Stages where the crossing was already observed before the cutoff are correct at 0 by construction.
- Posterior-to-prior std ratio for case 1 at 75%: log10A = 0.413, log10B = 0.939, Ea_R = 0.573, alpha = 0.892, log10hA = 0.984.
- Prior std (truncated Gaussian around the least-squares centre): log10A=0.657, log10B=0.667, Ea_R=304.174, alpha=0.659, log10hA=0.220.
- MCMC diagnostics (integrated autocorrelation time, steps, per parameter; maximum over the nine runs):

| parameter | tau (max over runs) | chain/tau (min over runs) |
|---|---|---|
| log10A | 523.0 | 19.1 |
| log10B | 571.6 | 17.5 |
| Ea_R | 301.4 | 33.2 |
| alpha | 214.8 | 46.6 |
| log10hA | 224.8 | 44.5 |
| log_sigma | 212.8 | 47.0 |

- emcee 50-tau criterion: NOT met (minimum chain/tau over parameters and runs = 17.5; criterion 50). Minimum ESS = 560. Mean integrated tau over the five physics parameters = 233.9. Mean acceptance = 0.33.

- Pooled 95% predictive coverage of later measured VB points: 0.843. Small counts per stage: treat by-stage numbers as indicative only.

## Sobol sensitivity (case 1, emulator, Saltelli N=1024, uniform design box)

| parameter | S1 (VB at end) | ST (VB at end) | S1 (time to 0.45 mm) | ST (time to 0.45 mm) |
|---|---|---|---|---|
| log10A | 0.826 | 0.928 | 0.861 | 0.926 |
| log10B | 0.000 | 0.020 | 0.013 | 0.029 |
| Ea_R | 0.052 | 0.149 | 0.034 | 0.101 |
| alpha | -0.001 | 0.016 | 0.003 | 0.002 |
| log10hA | 0.007 | 0.037 | -0.000 | 0.024 |

Time-to-threshold is censored for 25.1% of design samples (set to the horizon).

## Decision (case 1, 75% of life)

- Expected cost per unit time J(t,d) with c_p=1.0 and c_f=5.0 (assumed). Replacement time after cutoff: t* = 4.64 CSV time units, derate d* = 0.85, J* = 2.006. Grid: 0.5 to 40.8 (censoring horizon after cutoff). Optimum interior: yes.
- Without derating: t* = 3.73, J = 2.1 (interior: yes). Derating considered: True (instantaneous risk 0.377 vs threshold 0.1).
- Optimum as a range over the failure-penalty sweep c_f in [2.0, 5.0, 10.0, 20.0]: t* in [2.72, 11.51]; c_f=2.0: t*=11.51, d*=0.80, c_f=5.0: t*=4.64, d*=0.85, c_f=10.0: t*=2.72, d*=0.85, c_f=20.0: t*=2.72, d*=0.85.
- Stressed scenario (c_f = 10.0): t* = 2.72, d* = 0.85; action non-trivial (interior optimum): yes.
- Consistency with observed crossings: the observed VB first reaches 0.45 mm at a time whose bracket after the 75% cutoff is [2.0, 5.0] CSV units after the cutoff. t* <= the lower bracket bound: NO (conflict). Posterior median RUL at the cutoff = 1.20; t* <= median RUL: NO (conflict).
- Derating gain: 4.4% of J (from J=2.100 to 2.006).
- Posterior at 75% of life: median RUL 1.20, P(VB > 0.45 at cutoff) = 0.38. Read together with the consistency check above; the decision is not a scheduling recommendation unless the consistency check passes.

## Known limitations

- Only 15 of 16 cases are usable: case 6 has a single run (no wear trajectory) and is dropped. 145 VB labels in total, 3-20 per case.
- Setting-grouped CV (4 folds x 2 of the 8 operating settings) is the primary split, so replicate inserts of a setting never sit in both train and test. The case-grouped split (secondary, 5 folds) gives nRMSE physics 0.513, data-driven 0.545, hybrid 0.468; the leakage gap (setting minus case nRMSE) is +0.329 for data-driven and +0.015 for hybrid.
- 15 of 15 usable cases have a measured VB >= 0.45 mm, so the crossing is observed for all of them. Readme: runs continued 'up to a wear limit (and sometimes beyond)', so the crossings lie near the end of each experiment. RUL forecasts from early stages are model extrapolations, not observed outcomes.
- No force sensor: smcDC (DC spindle motor current) is a proxy for F_res. After masking the run-in transient (6 rows of case 1 with smcDC < 1.0), its within-case coefficient of variation is 2.1% (median) and 2.8% (max) over the cases. Its observation slope on wear is weak (within-case cw = 0.01795 mm^-1, R^2 = 0.042).
- Entry transient: the smcDC run-in of case 1 (runs 1-6, smcDC 0.13-0.91 against a steady 1.3-1.45 elsewhere) is masked from the force proxy u(t) only; VB labels from those runs are kept. Before the first valid sample (time 19) u(t) is held at its first valid value, an assumption.
- Cutting speed is constant (200 m/min per Readme) and is lumped into A; it is not separately identified. DOC and feed enter as the load factor qn; material enters as a multiplicative covariate gamma fitted on training folds.
- The time column is cumulative per case. The Readme labels the Fig. 6 axis 'machining time [mm]', which is physically inconsistent with a time unit, so RUL and t* are reported in CSV time units and not converted.
- Thermal ODE constants (ETA = 5 K per time unit, T_inf = 300 K) are fixed, not calibrated; C_th is absorbed into ETA and hA. The temperature rise in the Arrhenius term is capped at 400 K (DT_MAX): without the cap the wear-heat feedback runs away at extreme parameters. Disclosed: the design box (LO/HI in physics_model.py) was narrowed in round 1 to the region the least-squares fits occupy, using fits from all folds; it has not been re-derived from training folds only, so the box choice can leak a little information about the held-out folds.
- Emulator output is capped at VB = 3.0 mm for the GP fit. The cap is active for 21.2% of the emulator design outputs (mean over the three calibration cases). Crossings at 0.45 mm lie far below the cap, so the cap does not move them (crossing-time error at the prior centre: 6.38 CSV units at most), but emulator-based Sobol variances are affected by it.
- Calibration windows hold only 2-15 measured VB points (VB measured at t > 0 and t <= cutoff). Posteriors on the weakly identified parameters stay close to the prior, and the posterior-to-prior width ratios and the held-out coverage quantify this. This is the central data-scarcity result.
- Physics least-squares fits: 2 of 4 setting folds reached the iteration limit (max_nfev = 150) without converging and 3 have a parameter at its bound; the case-grouped fits show 1 and 4. The ODE parameters are therefore weakly identified from VB alone and the prior centres are not well-determined estimates.
- Emulators are per case (3 calibration cases), not a global surrogate over DOC/feed/material; Sobol is for case 1 only and assumes the uniform design box as its input distribution.
- RUL uses GP-mean curves (parameter uncertainty only, emulator and sensor noise not added to the crossing). Failure risk includes sensor noise and emulator variance.
- Decision costs (c_p=1.0, c_f=5.0) are assumed; the sweep over c_f in [2.0, 5.0, 10.0, 20.0] gives t* in [2.72, 11.51]. Derating is evaluated by re-simulating the ODE with the load scaled after the cutoff, using 200 posterior draws per derate level (not the full posterior), so the derated-life distributions carry sampling noise. Replacement times are evaluated only up to the censoring horizon 40.8 after the cutoff (max censored share 0.0%).
- MCMC: 32 walkers x 12000 steps, burn-in 2000. The 50-tau criterion is NOT met (minimum chain/tau = 17); the posteriors for parameters with tau above the limit should be read as approximate.
- Raw signals in mill/mill.mat were not used (CSV RMS features are sufficient for the modelling, and the 72 MB file was not cheap to load).
- Test-case physics predictions start from VB(0)=0, so they use no VB measurements of the held-out case. The log-linear baseline is fitted on training folds only and excludes rows with VB = 0 (not log-transformable).
- Speed-up compares one RK45 ODE solve (rtol 1e-4, solve_ivp) with one GP prediction in batch; it is not a comparison against a different reference solver and is not directly comparable to speed-ups quoted by other sub-projects.

## Figures

- fig1_vb_trajectories.png: measured VB and out-of-fold physics and hybrid predictions per case.
- fig2_model_comparison.png: observed vs predicted for the four models (setting-grouped CV).
- fig3_gp_emulator.png: GP emulator vs ODE on test samples.
- fig4_calibration_case{1,11,13}.png: posterior predictive bands at 25/50/75% of life.
- fig5_posteriors_case1.png, fig6_corner_case1_75pct.png: posterior marginals and correlations.
- fig7_rul.png: RUL posteriors vs observed crossing brackets.
- fig8_sobol.png: Sobol indices. fig9_decision.png: cost curves per unit time.

