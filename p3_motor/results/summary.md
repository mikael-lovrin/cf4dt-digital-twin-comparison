# p3_motor: LPTN digital twin of PMSM magnet temperature (round 3)

Runtime 462 s (budget 600 s: PASS). Target: pm (magnet) temperature, deg C. Nominal limit 100 degC. Every number below is written by main.py from computed values (metrics.json holds the same numbers).

## 0. Verdict (5 lines, failures included)
1. Held-out pm NRMSE: physics 0.594, data-driven 0.357, hybrid with out-of-fold residual RF 0.693. The hybrid does NOT beat the data-driven model; the three held-out sessions give weak evidence for any ranking.
2. Predictive band (session 68, 3 stages pooled, bias-free AR(1) model error): coverage 50/80/95 = 0.064 / 0.280 / 0.663 vs nominal 0.50 / 0.80 / 0.95 (FAIL; tolerance 0.05). Mean posterior-median bias +10.01 degC is outside the band; interval score 95 = 39.20.
3. Pre-registered stress (limit 100 degC, currents x1.2, coolant +10 K, 50% stage): session 68 P(exceed | no action) = 0.343, optimum derate d=0.75 from +15 min, informative = True; replicate session 12 P = 0.000, optimum no action, informative = False. Break-even dP at d=0.8, tau=0 = 0.125 (stress dP = 0.343); the review's ~0.017 does not reproduce from the constants.
4. MCMC (AR(1) GLS, MAP start, DE-move mixture): converged (>= 50 tau) in 3 of 3 stages; the 75% stage has min steps/tau = 54.0. Runtime is within the 600 s budget. Where the 50-tau rule fails, the posterior is a short-chain approximation and is reported as such.
5. Surrogate 2-sigma coverage 0.980 after emulator-variance inflation (scale x3.74 on the emulator sd; 0.516 before). The inflation is not a calibration improvement and not model discrepancy. Forecasts use the measured future inputs (perfect foresight), and session 68 is both the calibration and the forecast session; session 12 is the only transfer check.

## Changes vs round 2
- Stress replaced: the limit is no longer the posterior median (T_STRESS removed). The pre-registered stress is limit 100 degC, measured currents x1.2 and coolant +10 K, applied only over the 30-min decision window, on session 68 and replicated on session 12.
- Headline decision block (metrics decision) is the NOMINAL scenario at the 50% stage; the stress is in decision_stress_prereg.
- Informativeness: the break-even dP is computed from the code constants. At d=0.8, tau=0 it is 0.125, not the review's ~0.017 (the review value does not reproduce from C_fail=1000, C_thr=20, C_act=5, H=30); the computed value is used.
- MCMC: MAP start (L-BFGS-B), emcee DE-move 0.8 / DE-snooker 0.2 mixture, burn-in 500, blocks of 500, post-burn cap 3000. converged_50tau is reported truthfully per stage; the 6000-step cap of round 2 is gone.
- Likelihood: the sqrt(integrated-time) variance inflation (round 2) is replaced by an AR(1) GLS likelihood on the three stator nodes (rho and bias-free sd from training residuals at 60 s). The emulator variance is a constant term in sigma^2.
- GP term renamed emulator-variance inflation; its scale factor and 50/80/95 emulator coverage are reported. It is not described as model discrepancy and not as a calibration improvement.
- Predictive band: bias-free AR(1) model error (sd, rho on training residuals only) instead of i.i.d. RMS noise. Coverage at 50/80/95, interval score 95, parameter-only and prior-predictive baselines reported. The 'calibration adds' claim is removed.
- Hybrid residual RF trained on out-of-fold residuals: two session-grouped folds of the 8 training sessions, each LPTN fitted on the other fold (both physics features and residual target are out of fold). Test-time physics is the nominal model.
- Baselines added to the held-out table: constant mean and time-only (training mean curve).
- Forecast initialised at the decision instant with the measured stator temperatures (magnet from the model); the same state is used by the policy table and the predictive band.
- Runtime: GP restarts 2 -> 1, GP used as mean only inside the MCMC (its variance is a constant term).
- Limitations added: perfect-foresight of future inputs; session 68 used for calibration and forecast; session 12 transfer check.

## 1. Data inspection
- File: `_data/p3_motor/measures_v2.csv` (outside the project folder, read-only), 1330816 rows x 13 columns, 0 NaN, 0 duplicate rows.
- Columns: u_q, coolant, stator_winding, u_d, stator_tooth, motor_speed, i_d, i_q, pm, stator_yoke, ambient, torque, profile_id.
- Sessions: 69 `profile_id`, 69 contiguous runs (rows of a session are contiguous). Raw length min/median/max: [2176, 17142.0, 43971] samples.
- Ranges (min, max): i_d [-278, 0.0519]; i_q [-293.4, 301.7]; motor_speed [-275.5, 6000]; torque [-246.5, 261]; coolant [10.62, 101.6]; ambient [8.783, 30.71]; stator_winding [18.59, 141.4]; pm [20.86, 113.6].
- Units check: the brief said torque/speed/currents are normalised. They are not: speed spans -275.5 to 6000 rpm, temperatures are plausible degC (ambient 8.783 to 30.71, pm max 113.6). No standardisation was applied.
- Time axis: no timestamp column. Row order is chronological (pm lag-1 autocorrelation 1.0000, 99th percentile of |pm step| per raw sample 0.3789 degC). A 2 Hz sampling rate is assumed from the dataset documentation, not verified. `has_real_time_axis` is set to false.
- Unit IDs: `profile_id` is a measurement session. Splits are by session (no leakage).
- High-frequency sensor noise estimate (MAD of second differences, degC): stator_winding 0.01002, stator_tooth 0.005255, stator_yoke 0.004942, pm 0.007237.
- Synthetic structure: none suspected, `data_synthetic_suspected` = false.
- Decimation: every 10th raw sample (5 s step), 133114 decimated rows in total.

## 2. Split
- Training (8 sessions): [7, 27, 36, 54, 61, 66, 67, 78].
- Held-out (3 sessions): [2, 12, 68]. Random choice with seed 2026. Replicate session for the stress: 12.
- Calibration session for emulator, MCMC and decision: held-out session 68 (longest, 2334 decimated steps).

## 3. Physics model (LPTN)
- Nodes: winding W, stator tooth T, stator yoke Y, magnet M. Boundaries: coolant (to W and Y), ambient (to Y).
- Copper loss k_cu d^2 (i_d^2+i_q^2)(1+0.0039(T_W-20)); iron loss k_fe1|n|+k_fe2 n^2 (50/50 T/Y); magnet loss k_m n^2.
- Integration: exact matrix-exponential step on the 5 s grid, inputs averaged per step, winding feedback lagged by one step.
- Check against `scipy.integrate.solve_ivp` (LSODA, rtol 1e-6): max abs. deviation on the first training session = 0.01221 degC.
- Nominal parameters: least_squares on log-parameters, free-run residuals of all four nodes on the 8 training sessions (status 2, 32 function evaluations). Out-of-fold fits (4 sessions each, status 2 and 2) are used only for the hybrid residual target.

| parameter | nominal value |
|---|---|
| C_W | 4.782e+05 |
| C_T | 6.308e+04 |
| C_Y | 8.759e+05 |
| C_M | 6.695e+05 |
| G_WT | 7560 |
| G_TY | 1e+04 |
| G_TM | 572.9 |
| G_WC | 0.1 |
| G_YC | 6523 |
| G_YA | 1743 |
| k_cu | 2.803 |
| k_fe1 | 8863 |
| k_fe2 | 4000 |
| k_m | 462.4 |

- Training in-sample free-run RMSE per node (winding, tooth, yoke, magnet) [degC]: 4.275, 2.971, 2.794, 7.067.
- Out-of-fold free-run RMSE per node on the training sessions [degC]: 6.005, 3.823, 2.468, 8.776.

## 4. Held-out comparison (pm, free run from session start, 3 sessions pooled)
| model | RMSE [degC] | NRMSE (RMSE/std) | bias [degC] | per-session RMSE |
|---|---|---|---|---|
| physics | 10.31 | 0.594 | +2.72 | 16.77, 5.323, 5.868 |
| data_driven | 6.189 | 0.3566 | +3.64 | 5.144, 5.653, 7.355 |
| hybrid | 12.03 | 0.693 | +4.82 | 21.76, 2.328, 1.728 |
| hybrid_regularised_rf (extra) | 11.78 | 0.6788 | +4.65 | 21.24, 2.234, 2.36 |
| physics_feature_rf (extra) | 10.17 | 0.5857 | +6.31 | 17.65, 4.215, 3.347 |
| baseline_constant_mean (extra) | 20.22 | 1.165 | +10.38 | 26.51, 18.67, 14.9 |
| baseline_time_only (extra) | 18.57 | 1.07 | +6.19 | 22.39, 14.85, 18.21 |

- Data-driven: random forest on measured inputs and exponentially weighted averages of the losses and coolant/ambient (no temperatures, no initial state).
- Hybrid: physics free run (nominal, fitted on all 8 training sessions) plus RF residual trained on OUT-OF-FOLD residuals of the physics model (2 session-grouped folds). Regularised hybrid: same target, min_samples_leaf 200, max_depth 4. Physics-feature RF: RF on the same OOF features predicting pm.
- Baselines: constant mean (training mean of pm) and time-only (mean training pm trajectory against time since session start).
- The physics model receives the measured initial state of all nodes at session start; the other models do not.

**Hybrid (NRMSE 0.693) vs RF (0.3566) and physics (0.594).** The residual RF now learns OOF residuals. The hybrid does NOT beat the data-driven model. The dominant error is the free-run drift of the unobserved magnet node (held-out physics bias +2.72 degC), which a residual correction on the current features does not remove. Regularised hybrid NRMSE 0.6788, physics-feature RF NRMSE 0.5857.

## 5. GP emulator and emulator-variance inflation
- Design: 300-point LHS over 6 uncertain log-multipliers (C_W, C_M, G_WT, G_TM, G_YC, k_cu; +-0.5 in log) and the derating scale d in [0.5, 1].
- Output: trajectories of all four nodes at 120 time points, PCA to 12 components (0.9994 of variance). Kernel: Constant x Matern(2.5, ARD) + White, normalised outputs, one restart.
- Emulator-variance inflation: a per-output variance (mean squared emulator error minus mean GP variance on a separate 100-point design) added to the GP predictive variance. Scale factor of the emulator sd on that design: x3.741. It measures the emulator's own error against the LPTN ODE, not model-data discrepancy.
- Validation on a third set of 100 LHS points: R2 = 0.9999, RMSE = 0.1198 degC, speed-up vs solve_ivp per run = 1.972e+04x (solve_ivp 3.966 s/run, GP 0.0002012 s/run).
- Coverage on the validation design, before inflation: 2-sigma 0.5157. After inflation: 2-sigma 0.9803, 50 / 80 / 95 % = 0.6501 / 0.9026 / 0.9786 (nominal 0.50 / 0.80 / 0.95).
- Status of the 2-sigma coverage: FAIL (over/under-coverage outside 0.03 of nominal) (outside 0.03 of nominal = over-coverage, the emulator sd is too wide after inflation; the inflation does not change the posterior, see section 7).
- Inflation variance relative to the observation variance on the stator nodes: 0.0049, 0.0022, 0.0011.

## 6. Sobol sensitivity (GP emulator, operating scale d=1)
| parameter | S1 peak pm | ST peak pm | S1 time-to-exceed | ST time-to-exceed |
|---|---|---|---|---|
| C_W | 0.001656 | 0.001713 | -0.002259 | 0.02246 |
| C_M | 0.1522 | 0.1618 | 0.05551 | 0.2737 |
| G_WT | 0.0004694 | 0.000639 | 0.006806 | 0.01123 |
| G_TM | 0.0009309 | 0.009976 | -0.0008339 | 0.04957 |
| G_YC | 0.07302 | 0.08325 | 0.06201 | 0.2377 |
| k_cu | 0.7491 | 0.7604 | 0.5784 | 0.8278 |
- Output 1 is the peak pm over the session; output 2 is the first time pm exceeds the limit (censored at session end). N=1024 Saltelli base samples.

## 7. Calibration, predictive band and risk (AR(1) GLS, 32 walkers) on held-out session 68
Observations: stator winding, tooth, yoke every 1 min up to the stage instant; pm is not observed in the fit. Likelihood: AR(1) GLS per stator node, sigma (bias-free residual sd) = 3.507, 2.744, 2.569 degC, rho at 60 s = 0.921, 0.884, 0.927, emulator inflation included as a constant. Start: MAP (L-BFGS-B). Sampler: DE-move 0.8 / DE-snooker 0.2, burn-in 500, blocks of 500, cap 3000 post-burn steps.

| stage | obs | post-burn steps | min steps / tau | converged (>=50 tau) | acceptance | tau max | ESS min |
|---|---|---|---|---|---|---|---|
| 25% | 48 | 1500 | 59.65 | True | 0.3132 | 25.15 | 1909 |
| 50% | 97 | 1500 | 63.5 | True | 0.3294 | 23.62 | 2032 |
| 75% | 145 | 1500 | 53.98 | True | 0.3219 | 27.79 | 1727 |

| stage | bias pm [degC] | 95% band width [degC] | cov50 | cov80 | cov95 (predictive) | cov95 (parameter only) | IS95 | P(pm>100 in 30 min, no action) | P(pm>100 before session end, no action) |
|---|---|---|---|---|---|---|---|---|---|
| 25% | +9.06 | 25.07 | 0.1507 | 0.4726 | 0.7123 | 0.2192 | 39.54 | 0 | 0.27 |
| 50% | +9.52 | 23.73 | 0.04082 | 0.3673 | 0.7041 | 0.07143 | 36.45 | 0 | 0.01667 |
| 75% | +11.46 | 24.02 | 0 | 0 | 0.5714 | 0 | 41.62 | 0.006667 | 0.006667 |

- Pooled over stages (session 68): predictive coverage 0.5/0.8/0.95 = 0.06383 / 0.28 / 0.6626 (nominal 0.5/0.8/0.95); mean 95% width 24.27 degC; interval score 95 39.2.
- Parameter-only coverage (no model error): 0.5/0.8/0.95 = 0.01826 / 0.04795 / 0.09687.
- Prior-predictive baseline (parameters from the prior box, same error model): 0.5/0.8/0.95 = 0.2789 / 0.9534 / 1, mean 95% width 32.82 degC.
- Transfer to held-out session 12 (posterior of session 68 at 50%, not calibrated on session 12): coverage 0.5/0.8/0.95 = 0.837 / 1 / 1, bias -0.60 degC.
- Bias: posterior median of the no-action forecast minus measured pm is +9.06 degC at 25 % (not in the band).
- Posterior-to-prior std ratio (prior uniform on [-0.5,0.5], std 0.289), 75 % stage: C_W 0.168, C_M 0.6517, G_WT 0.2521, G_TM 0.5316, G_YC 0.19, k_cu 0.1152. Parameters with ratio >= 0.8 are called prior-like: 0 of 6.
- Per-stage ratios (25/50/75 %):
  - C_W: 0.2761, 0.2292, 0.168
  - C_M: 0.8114, 0.7914, 0.6517
  - G_WT: 0.3208, 0.3386, 0.2521
  - G_TM: 0.8167, 0.5858, 0.5316
  - G_YC: 0.2653, 0.2313, 0.19
  - k_cu: 0.1886, 0.1659, 0.1152
- Per-parameter tau at 75% stage: C_W 24.06, C_M 27.79, G_WT 21.79, G_TM 27.2, G_YC 21.79, k_cu 22.94.

Posterior correlation matrix (75% stage):

| | C_W | C_M | G_WT | G_TM | G_YC | k_cu |
|---|---|---|---|---|---|---|
| C_W | +1.00 | +0.25 | +0.66 | -0.15 | +0.60 | +0.90 |
| C_M | +0.25 | +1.00 | +0.35 | -0.27 | +0.02 | +0.18 |
| G_WT | +0.66 | +0.35 | +1.00 | -0.26 | +0.31 | +0.63 |
| G_TM | -0.15 | -0.27 | -0.26 | +1.00 | -0.03 | -0.02 |
| G_YC | +0.60 | +0.02 | +0.31 | -0.03 | +1.00 | +0.64 |
| k_cu | +0.90 | +0.18 | +0.63 | -0.02 | +0.64 | +1.00 |

Ridge (|rho| > 0.7): C_W-k_cu (+0.90). k_cu is the parameter with the clearest posterior reduction (ratio 0.1152); its partners on the ridge are the G and C terms named above.

## 8. Decision
- Policy: start derating d (fraction of current scale) at tau minutes after the calibration instant, for a 30-min horizon. Expected cost per unit time = [C_act 1[d<1] + C_thr (1-d)(H-tau) + C_fail P(pm>limit)] / H, C_fail=1000, C_thr=20, C_act=5 (illustrative).
- Forecast: initialised at the decision instant with the measured stator temperatures and the model magnet temperature; measured future inputs (perfect foresight).
- **Headline (nominal, limit 100 degC, session 68, 50% stage):** P(exceed in window | no action) = 0. Recommendation: no action (no derating policy beats no action).
- **Pre-registered stress (limit 100 degC, currents x1.2, coolant +10 K over the window):** session 68 P(exceed | no action) = 0.3433, replicate session 12 P(exceed | no action) = 0. Break-even dP(d=0.8, tau=0) = 0.125; stress dP(d=0.8, tau=0) on session 68 = 0.3433.

| scenario | session | stage | optimum d* | tau* [min] | expected cost [per min] | no-action cost [per min] | action pays |
|---|---|---|---|---|---|---|---|
| nominal 100 degC | 68 | 25% | 1.00 | none | 0 | 0 | no |
| nominal 100 degC | 68 | 50% | 1.00 | none | 0 | 0 | no |
| nominal 100 degC | 68 | 75% | 1.00 | none | 0.2222 | 0.2222 | no |
| stress (pre-reg.) 100 degC | 68 | 50% | 0.75 | 15 | 2.889 | 11.44 | yes |
| stress (pre-reg.) 100 degC | 12 | 50% | 1.00 | none | 0 | 0 | no |

**Decision checks:** stress optimum interior (session 68) = True; stress action non-trivial = True; informative (session 68 / 12) = True / False; C_thr sweep range of the optimal stress cost = [0.9167, 7]; risk (no action, nominal) equals the policy table value = True.

Every policy's expected cost per minute, nominal limit 100 degC, session 68, 50% stage:

| start tau [min] \ derating d | 0.50 | 0.55 | 0.60 | 0.65 | 0.70 | 0.75 | 0.80 | 0.85 | 0.90 | 0.95 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 10.167 | 9.167 | 8.167 | 7.167 | 6.167 | 5.167 | 4.167 | 3.167 | 2.167 | 1.167 |
| 5 | 8.500 | 7.667 | 6.833 | 6.000 | 5.167 | 4.333 | 3.500 | 2.667 | 1.833 | 1.000 |
| 10 | 6.833 | 6.167 | 5.500 | 4.833 | 4.167 | 3.500 | 2.833 | 2.167 | 1.500 | 0.833 |
| 15 | 5.167 | 4.667 | 4.167 | 3.667 | 3.167 | 2.667 | 2.167 | 1.667 | 1.167 | 0.667 |
| 20 | 3.500 | 3.167 | 2.833 | 2.500 | 2.167 | 1.833 | 1.500 | 1.167 | 0.833 | 0.500 |
| 25 | 1.833 | 1.667 | 1.500 | 1.333 | 1.167 | 1.000 | 0.833 | 0.667 | 0.500 | 0.333 |

No-action cost (nominal) = 0 per min. Lowest policy cost = 0.

Every policy's expected cost per minute, pre-registered stress (limit 100 degC), session 68, 50% stage:

| start tau [min] \ derating d | 0.50 | 0.55 | 0.60 | 0.65 | 0.70 | 0.75 | 0.80 | 0.85 | 0.90 | 0.95 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 10.167 | 9.167 | 8.167 | 7.167 | 6.167 | 5.167 | 4.167 | 3.389 | 4.056 | 8.056 |
| 5 | 8.500 | 7.667 | 6.833 | 6.000 | 5.167 | 4.333 | 3.611 | 3.444 | 4.833 | 8.444 |
| 10 | 6.833 | 6.167 | 5.500 | 4.833 | 4.167 | 3.500 | 3.056 | 3.389 | 5.167 | 8.278 |
| 15 | 5.167 | 4.667 | 4.167 | 3.667 | 3.167 | 2.889 | 3.056 | 4.000 | 6.389 | 8.667 |
| 20 | 9.167 | 9.167 | 9.389 | 9.389 | 9.611 | 9.611 | 9.722 | 10.278 | 10.611 | 11.278 |
| 25 | 11.167 | 11.000 | 11.278 | 11.667 | 11.833 | 11.667 | 11.611 | 11.444 | 11.389 | 11.444 |

No-action cost (stress) = 11.44 per min. Lowest policy cost = 2.889.

Every policy's expected cost per minute, pre-registered stress (limit 100 degC), replicate session 12, 50% stage:

| start tau [min] \ derating d | 0.50 | 0.55 | 0.60 | 0.65 | 0.70 | 0.75 | 0.80 | 0.85 | 0.90 | 0.95 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 10.167 | 9.167 | 8.167 | 7.167 | 6.167 | 5.167 | 4.167 | 3.167 | 2.167 | 1.167 |
| 5 | 8.500 | 7.667 | 6.833 | 6.000 | 5.167 | 4.333 | 3.500 | 2.667 | 1.833 | 1.000 |
| 10 | 6.833 | 6.167 | 5.500 | 4.833 | 4.167 | 3.500 | 2.833 | 2.167 | 1.500 | 0.833 |
| 15 | 5.167 | 4.667 | 4.167 | 3.667 | 3.167 | 2.667 | 2.167 | 1.667 | 1.167 | 0.667 |
| 20 | 3.500 | 3.167 | 2.833 | 2.500 | 2.167 | 1.833 | 1.500 | 1.167 | 0.833 | 0.500 |
| 25 | 1.833 | 1.667 | 1.500 | 1.333 | 1.167 | 1.000 | 0.833 | 0.667 | 0.500 | 0.333 |

No-action cost (stress, session 12) = 0 per min. Lowest policy cost = 0.

- Sensitivity to C_thr (5 / 10 / 20 / 40 / 60), 50 % stage, session 68:
  - nominal, C_thr=5.0: d*=1.00, tau*=none, cost 0
  - nominal, C_thr=10.0: d*=1.00, tau*=none, cost 0
  - nominal, C_thr=20.0: d*=1.00, tau*=none, cost 0
  - nominal, C_thr=40.0: d*=1.00, tau*=none, cost 0
  - nominal, C_thr=60.0: d*=1.00, tau*=none, cost 0
  - stress, C_thr=5.0: d*=0.70, tau*=15, cost 0.9167
  - stress, C_thr=10.0: d*=0.75, tau*=15, cost 1.639
  - stress, C_thr=20.0: d*=0.75, tau*=15, cost 2.889
  - stress, C_thr=40.0: d*=0.80, tau*=15, cost 5.056
  - stress, C_thr=60.0: d*=0.85, tau*=15, cost 7

## 9. Figures
- `results/figures/fig1_data_overview.png`
- `results/figures/fig2_physics_fit_train.png`
- `results/figures/fig3_heldout_predictions.png`
- `results/figures/fig4_surrogate_validation.png`
- `results/figures/fig5_sobol.png`
- `results/figures/fig6_posterior.png`
- `results/figures/fig7_calibration_predictive.png`
- `results/figures/fig8_decision.png`

## 10. Known limitations
- Data: one public test-bench dataset (69 sessions, 2 Hz nominal, no timestamp column). Time axis is row order at an assumed 2 Hz; not verified against a clock.
- Units: values are NOT standardised/normalised as the brief suggested (speed in rpm, currents in A, temperatures in degC); no rescaling was applied beyond the train-only statistics for sigma.
- Split: 8 training and 3 held-out sessions (2, 12, 68; lengths >= 2.36 h); with so few sessions the physics fit, the out-of-fold physics fits (4 sessions each) and the residual/data-driven models are weakly constrained.
- Perfect foresight of inputs: the forecasts and the decision use the MEASURED future torque, speed, currents, coolant and ambient of the session. The decision is therefore conditional on known future load, which a real controller does not have; the derating benefit is an upper bound on what a forecast-based policy could achieve.
- Session 68 is used for calibration (MCMC, emulator, Sobol) AND as the forecast session for the coverage and the decision; it is not an independent test set. Session 12 (not used for calibration) is the transfer check: it applies the posterior of session 68 to a different session and gives a single additional forecast.
- Forecasts are initialised at the decision instant with the MEASURED stator temperatures at that instant (the magnet temperature is not measured and comes from the model). The calibration itself is a free run from the session start; the calibration likelihood does not use the stator observations to re-initialise the state.
- Decimation to 5 s: fine for the thermal time constants used (dt/tau < 1 for the winding), inputs averaged per step; the matrix-exponential step is stable but first-order accurate in the input averaging.
- LPTN is a 4-node simplification (no separate rotor/air-gap/end-winding nodes); no magnet heat from eddy currents beyond k_m n^2; iron loss is a speed polynomial without flux dependence.
- Copper temperature coefficient alpha_cu fixed at 0.0039 1/K; winding feedback uses the previous step (explicit).
- Initial state of every node is taken from the measurement at session start, including the magnet temperature (an assumption: in practice the magnet temperature is not measured; this favours the physics model).
- Likelihood: AR(1) GLS on the three stator nodes with independent channels; rho is estimated from training residuals at 60 s and sigma is the bias-free training residual sd. The cross-node correlation is ignored. The GLS posterior is conditional on this error model; if the residuals are not AR(1), the posterior is overconfident (see coverage).
- Emulator-variance inflation: per-output variance added to the emulator's own predictive variance, estimated on a separate 100-point design at one calibration session; it is an emulator-error scale, not a model-discrepancy term, and it is a constant in the likelihood (its magnitude is far below the observation variance).
- Predictive pm band: bias-free AR(1) model error (sd and rho from training residuals only). The posterior-median bias of the no-action forecast is NOT included in the band, so coverage is limited by that bias.
- Interval coverage is pooled over the three calibration stages of session 68, which share the same session and are not independent; the session-12 transfer coverage is a single forecast per level.
- Stress decision: the limit is 100 degC (chosen operating limit, not a manufacturer rating); the stress (currents x1.2, coolant +10 K over the window) is pre-registered and not calibrated to make the action non-trivial. Cost constants are illustrative; no economic data was provided.
- Break-even rule: the review's ~0.017 break-even at d=0.8, tau=0 does not reproduce from the code constants; the value computed from them is reported and used.
- Derating acts on the current scale; the torque and copper-loss reduction are modelled, the speed-dependent iron loss is unchanged.
- Held-out evaluation uses only 3 sessions; the RF, regularised hybrid, physics-feature RF and baseline comparisons carry large uncertainty.
- MCMC: MAP start, DE-move / DE-snooker mixture, burn-in 500 and a post-burn cap of 3000 steps for every stage (runtime budget of 600 s). The 50-tau criterion is reported per stage; where it is false, the chain is too short for it.
- Predictive coverage of the 50/80/95 bands differs from nominal by more than 0.05 (see section 7); this is reported as a failure, not corrected.
