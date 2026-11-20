# p4_rotor: physics + data hybrid digital twin of rotating-machinery unbalance vibration (round 2)

Adaptation of the common brief to the user's bench dataset. The COBEM 2025 bench context (triaxial accelerometer 1 kHz,
0.25 HP motor, two disks A/B) is user-supplied and is NOT in the data; it is not used by any computation.
No wear or time axis exists, so the wear ODE is replaced by a Jeffcott rotor unbalance response and the RUL/time
decision by a speed-and-rebalance decision.

Runtime: 431.0 s. Seed 0; CV seeds [0, 1, 2, 3, 4]. Every number below is written by `main.py` from values computed in
this run (round-1 values are read from `results/metrics_r1.json`).

## 0. Headline (round 2)
- Calibrated range: 300-2400 RPM. Headline risk P(RMS_Z > 0.1028 g) at 2400 RPM, no action, 75% posterior = **1.000**.
- Expected-cost optimum (baseline costs, 75% posterior) = **2100 RPM** (rebalance); over the
  81-combination cost sweep the optimum lies in **1500-2400 RPM**.
- Pooled 95% interval coverage on unused in-range points = **0.43** (n = 7; nominal 0.95).
  Coverage at 50/80/95%: 0.00 / 0.29 / 0.43.
- Hybrid vs data-driven, 5 RF seeds: pooled RMSE diff -0.0117 g (sd 0.0007); hybrid better in
  14/18 groups (sign test p = 0.031). The hybrid is significantly better than data-driven at the 5% level by the sign test.
- m and e are not separately identifiable (section 7); the calibration runs in (k/m, c/m, e/m).

## 1. Data inspection
- 5980 rows, 52 columns, NaN = 0. 42 feature columns (3 axes x 14).
- Speeds: 300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300 RPM. Positions 1-3. 38 experimental conditions, 19 (Misalignment, Unbalance) groups.
- Every (condition, Position, RPM) cell holds 5 repeats (1196 cells). Repeats stay in one fold.
- Partial conditions (not 165 rows): {'M0.5_U13.6_A0B1': 70, 'M0.5_U20.4_A0B1': 65, 'M0.5_U27.2_A0B1': 70}.
- Healthy reference (Normal = 1): 165 rows.
- No time column, no serial number. The experimental condition is the unit ID for splits.
- RMS and Standard Deviation are the same signal: max |RMS - STD| over the three axes = 0.0077 g, so RMS is the only target used.
- Healthy background (Z RMS, mean over Normal rows at that RPM) = 0.0209 g at 300 RPM and 0.0701 g at 3300 RPM.
  (Round 1 called this "Z mean"; it is Z RMS.)

## 2. Target axis choice
Criterion: Pearson correlation between log(excess RMS) and log(U_eff * RPM^2) on Misalignment=0, Unbalance>0 rows, where
excess = sqrt(RMS^2 - baseline(RPM)^2). The raw correlation is also reported; it is inflated by the speed trend shared with the baseline.

| axis | corr raw | corr excess (criterion) | clipped rows |
|---|---|---|---|
| X | 0.863 | 0.162 | 700 |
| Y | 0.804 | 0.263 | 706 |
| Z | 0.750 | 0.689 | 413 |

Chosen target: **Z Root Mean Squared**. Leak check (round-1 review item 5): the axis is recomputed on the training rows of each of the
18 LOGO folds; Z wins in 18/18 folds, so the held-out groups do not change the choice.
Limit = P99 of healthy Z RMS = 0.1028 g.

## 3. Pure Physics model (Jeffcott rotor)
Steady-state unbalance response, RMS acceleration = w^2 * m_u e w^2 / sqrt((k - m w^2)^2 + (c w)^2) / sqrt(2) / g0, with
m_u = U_eff * 1e-3 kg, plus the healthy floor in quadrature. Least squares on log RMS with m fixed at 3.0 kg (8 random starts).

Nominal fit on all Mis=0 unbalance rows: k = 1.971e+05 N/m, c = 1.467e+02 N s/m, e = 2.300e-03 m
(at m = 3.0 kg: k/m = 6.569e+04 1/s^2, c/m = 4.891e+01 1/s, e/m = 7.667e-04 m/kg), R2(log) = 0.728.
Natural speed sqrt(k/m) = 2447 RPM.

Per-fold fits (held-out group in the first column):

| held-out group | k [N/m] | c [N s/m] | e [m] | R2(log) on train | Z chosen on training rows |
|---|---|---|---|---|---|
| M0.0_U13.6 | 1.98e+05 | 1.50e+02 | 3.28e-03 | 0.900 | Z |
| M0.0_U20.4 | 2.09e+05 | 1.03e+02 | 1.65e-03 | 0.626 | Z |
| M0.0_U6.8 | 1.91e+05 | 1.54e+02 | 2.13e-03 | 0.706 | Z |
| M0.5_U0.0 | 1.97e+05 | 1.47e+02 | 2.30e-03 | 0.728 | Z |
| M0.5_U13.6 | 1.97e+05 | 1.47e+02 | 2.30e-03 | 0.728 | Z |
| M0.5_U20.4 | 1.97e+05 | 1.47e+02 | 2.30e-03 | 0.728 | Z |
| M0.5_U27.2 | 1.97e+05 | 1.47e+02 | 2.30e-03 | 0.728 | Z |
| M0.5_U6.8 | 1.97e+05 | 1.47e+02 | 2.30e-03 | 0.728 | Z |
| M1.0_U0.0 | 1.97e+05 | 1.47e+02 | 2.30e-03 | 0.728 | Z |
| M1.0_U13.6 | 1.97e+05 | 1.47e+02 | 2.30e-03 | 0.728 | Z |
| M1.0_U20.4 | 1.97e+05 | 1.47e+02 | 2.30e-03 | 0.728 | Z |
| M1.0_U27.2 | 1.97e+05 | 1.47e+02 | 2.30e-03 | 0.728 | Z |
| M1.0_U6.8 | 1.97e+05 | 1.47e+02 | 2.30e-03 | 0.728 | Z |
| M1.5_U0.0 | 1.97e+05 | 1.47e+02 | 2.30e-03 | 0.728 | Z |
| M1.5_U13.6 | 1.97e+05 | 1.47e+02 | 2.30e-03 | 0.728 | Z |
| M1.5_U20.4 | 1.97e+05 | 1.47e+02 | 2.30e-03 | 0.728 | Z |
| M1.5_U27.2 | 1.97e+05 | 1.47e+02 | 2.30e-03 | 0.728 | Z |
| M1.5_U6.8 | 1.97e+05 | 1.47e+02 | 2.30e-03 | 0.728 | Z |

## 4. Held-out comparison (leave-one-(Unbalance, Misalignment)-group-out, 18 groups, 5 RF seeds)
Normal group always in training. Pooled over all held-out rows. RF uses 100 trees (round 1: 200).

| model | RMSE (g), mean | sd over 5 seeds | NRMSE (RMSE / std(y)) |
|---|---|---|---|
| Pure Physics | 0.1255 | 0 (deterministic) | 0.779 |
| Pure Data, condition inputs only (RPM, Position, Mis, Unb, TypeA, TypeB) | 0.0694 | 0.0003 | 0.430 |
| Pure Data, condition + other-axis features | 0.0673 | 0.0003 | 0.418 |
| Hybrid (physics + RandomForest residual on the same inputs as data-driven) | 0.0557 | 0.0008 | 0.345 |

std(y) of the evaluated rows = 0.1612 g.

Paired per-group comparison (RMSE per group, averaged over the 5 seeds; difference = hybrid - pure data (cond.+feat.)):

| group | Pure Physics | Data (cond.) | Data (cond.+feat.) | Hybrid | sd over seeds (hybrid) | hybrid - data(full) |
|---|---|---|---|---|---|---|
| M0.0_U13.6 | 0.2186 | 0.1086 | 0.1223 | 0.0813 | 0.0010 | -0.0410 |
| M0.0_U20.4 | 0.2219 | 0.1980 | 0.1761 | 0.1435 | 0.0067 | -0.0326 |
| M0.0_U6.8 | 0.0805 | 0.0567 | 0.0587 | 0.0621 | 0.0005 | 0.0034 |
| M0.5_U0.0 | 0.0095 | 0.0117 | 0.0165 | 0.0092 | 0.0002 | -0.0073 |
| M0.5_U13.6 | 0.1459 | 0.0311 | 0.0287 | 0.0219 | 0.0010 | -0.0068 |
| M0.5_U20.4 | 0.0716 | 0.1057 | 0.0506 | 0.0552 | 0.0034 | 0.0046 |
| M0.5_U27.2 | 0.1209 | 0.0529 | 0.0411 | 0.0569 | 0.0061 | 0.0157 |
| M0.5_U6.8 | 0.0415 | 0.0408 | 0.0244 | 0.0215 | 0.0005 | -0.0029 |
| M1.0_U0.0 | 0.0090 | 0.0114 | 0.0092 | 0.0090 | 0.0002 | -0.0002 |
| M1.0_U13.6 | 0.1380 | 0.0245 | 0.0405 | 0.0325 | 0.0015 | -0.0080 |
| M1.0_U20.4 | 0.0658 | 0.0442 | 0.0372 | 0.0332 | 0.0010 | -0.0041 |
| M1.0_U27.2 | 0.1365 | 0.0625 | 0.0681 | 0.0620 | 0.0020 | -0.0060 |
| M1.0_U6.8 | 0.0376 | 0.0139 | 0.0269 | 0.0172 | 0.0006 | -0.0096 |
| M1.5_U0.0 | 0.0150 | 0.0141 | 0.0113 | 0.0101 | 0.0002 | -0.0011 |
| M1.5_U13.6 | 0.1374 | 0.0121 | 0.0317 | 0.0216 | 0.0003 | -0.0102 |
| M1.5_U20.4 | 0.0801 | 0.0350 | 0.0432 | 0.0485 | 0.0007 | 0.0053 |
| M1.5_U27.2 | 0.1369 | 0.0552 | 0.0675 | 0.0631 | 0.0039 | -0.0044 |
| M1.5_U6.8 | 0.0452 | 0.0172 | 0.0322 | 0.0262 | 0.0010 | -0.0060 |

- Hybrid vs data-driven: pooled difference -0.0117 g (sd over seeds 0.0007; hybrid better in
  5/5 seeds). Groups where the hybrid has the lower seed-mean RMSE: 14/18
  (two-sided sign test p = 0.031). Bootstrap 95% CI of the mean group difference over groups:
  [-0.0125, -0.0010] g.
- Hybrid vs pure physics: pooled difference -0.0699 g; hybrid better in 18/18 groups
  (sign p = 0.000).
- Pure physics group RMSE ranges from 0.009 to 0.222 g (seed mean).

Interpretation: the physics explains part of the Mis=0 unbalance response (nominal log-R2 = 0.73, so the
single-DOF shape is only approximate). Pure data uses the same inputs as the hybrid, minus the physics term. The
other-axis features change pooled RMSE only from 0.0694 to 0.0673 g (condition-only vs condition+features), so the data models mostly learn the RPM/condition structure.
The hybrid's advantage over data-driven is a small but consistent effect in this design; the physics term includes the healthy-floor baseline, which is measured from data.

## 5. GP surrogate of the physics
Inputs (log k/m, log c/m, log e/m, RPM) (round 1: 5 inputs with log m); output log unit response (per 1 g of effective unbalance);
Matern(nu=2.5) ARD + White kernel; 800 LHS training points, 400 LHS test points. The inputs are the identifiable combinations only.
- R2(log) = 0.9988; RMSE = 4.822e-02 g per g-unit; 2-sigma coverage = 0.963 (ideal 0.95).
- Fitted ARD length scales (standardised inputs): log(k/m) = 2.81, log(c/m) = 10.96, log(e/m) = 100.00, RPM = 9.93. Computed, not assumed: the length scale of at the upper optimiser bound (log(e/m)), which is the expected signature of the exact linearity of log response in log(e/m).
  Round 1 asserted this claim without computing it; it is now read from the fitted kernel.
- Speedup of the GP mean (threaded matmul path) vs the closed form on 20k points = 0.0059x. The GP is slower than the closed form (the physics here is analytic); the surrogate is kept because the brief requires it for Sobol and MCMC.
- Fast mean path vs sklearn reference: max abs difference 8.7e-10 (log units).

## 6. Sobol sensitivity (SALib, Saltelli N=1024, 6144 evaluations, GP surrogate)
Output: log of the peak unbalance response over 300-3300 RPM, for a 1 g unit unbalance.
The peak is taken over the extrapolated range as well, because the prior box covers it; this is a sensitivity of the
surrogate, not of the calibrated range.

Physical parametrisation (k, c, m, e) with log-uniform prior boxes (kept for comparability with round 1):

| parameter | S1 | ST |
|---|---|---|
| k | 0.193 | 0.251 |
| c | 0.096 | 0.145 |
| m | 0.014 | 0.041 |
| e | 0.634 | 0.634 |

Identifiable parametrisation (k/m, c/m, e/m), which is what the calibration uses:

| parameter | S1 | ST |
|---|---|---|
| k_over_m | 0.181 | 0.219 |
| c_over_m | 0.101 | 0.139 |
| e_over_m | 0.679 | 0.678 |

Caveats (round-1 review item 6): (i) all indices are conditional on the prior boxes; (ii) the response is exactly linear in
log(e/m), so S_e = ST_e follows from model structure and is not a finding by itself; (iii) m enters only through the ratios,
so its index measures the width of the m prior, not a physical effect.

## 7. Sequential Bayesian calibration (emcee, 32 walkers, 12000 steps, burn-in 1500)
Held-out experiment M0.0_U20.4_A1B0 (group M0.0_U20.4 held out of the physics fit). Observations: mean Z RMS over repeats and positions per RPM,
11 RPM points (300-3300 RPM); the first 25/50/75% of the sweep are used: 3 (up to 900 RPM), 6 (up to 1800 RPM), 8 (up to 2400 RPM) RPM points for the three stages.

**Identifiability.** m and e are not separately identifiable: the RMS response is a function of k/m, c/m and e/m only. The
largest change of the physics response when (k, c, m, e) are jointly rescaled with the mass changed is 5.4e-16
(relative), i.e. machine precision. The calibration therefore uses (log k/m, log c/m, log e/m) with the prior box inherited
from the round-1 box for (k, c, m, e). The posterior over m is not computed and is not reported.

**Likelihood (round 2).**
- Per-RPM GP variance: the GP predictive std is evaluated once per stage at the posterior-median theta from a short pilot chain
  (2000 steps, burn 500), one value per RPM, and kept fixed in the final chain. Round 1 used one global value
  (0.2617 log, 132% of sigma_obs). Per-RPM values at the 75% pilot median (and their ratio to the 75% median sigma_obs):

| RPM | GP sd (log) at posterior median | sd / sigma_obs (median) |
|---|---|---|
| 300 | 0.2318 | 7.84 |
| 600 | 0.1891 | 6.39 |
| 900 | 0.1824 | 6.17 |
| 1200 | 0.1799 | 6.09 |
| 1500 | 0.1819 | 6.15 |
| 1800 | 0.1848 | 6.25 |
| 2100 | 0.1892 | 6.40 |
| 2400 | 0.1870 | 6.32 |
| 2700 | 0.1766 | 5.97 |
| 3000 | 0.1858 | 6.28 |
| 3300 | 0.2351 | 7.95 |

- Model discrepancy: an additive GP term in log RMS over RPM, amplitude a_disc (prior 1e-3 to 1), fixed correlation length
  500 RPM. Its posterior is conditioned on the calibration residuals, which widens the predictive where the data do not constrain it.
- Predictive for unseen RPM: GP-mean physics + conditional discrepancy + sigma_obs + GP delta-method variance.

| stage | RPM pts | acceptance | max tau | post-burn steps / max tau | k/m (median) | c/m (median) | e/m (median) | sigma_obs (median, log units) | a_disc (median, log) | GP sd change pilot->final (max rel.) |
|---|---|---|---|---|---|---|---|---|---|---|
| 25% | 3 | 0.348 | 307.3 | 34.2 | 3.754e+05 | 2.449e+02 | 1.390e-04 | 0.0241 | 0.1315 | 0.079 |
| 50% | 6 | 0.272 | 344.8 | 30.5 | 1.358e+05 | 3.670e+01 | 5.656e-03 | 0.0333 | 0.1728 | 0.065 |
| 75% | 8 | 0.434 | 455.9 | 23.0 | 8.811e+04 | 4.872e+00 | 2.329e-03 | 0.0296 | 0.1715 | 0.001 |

**MCMC diagnostics** (tau per parameter, post-burn steps / max tau, 50-tau check for each stage):

| stage | tau k/m | tau c/m | tau e/m | tau sigma | tau a_disc | steps / max tau | 50 tau satisfied |
|---|---|---|---|---|---|---|---|
| 25% | 135.2 | 150.3 | 151.8 | 208.5 | 307.3 | 34.2 | no |
| 50% | 329.5 | 293.8 | 300.1 | 270.0 | 344.8 | 30.5 | no |
| 75% | 455.9 | 228.4 | 348.1 | 152.4 | 268.8 | 23.0 | no |

The 50-tau rule is NOT satisfied for every stage; see the table
(minimum steps/max-tau = 23.0). The largest tau is 455.9 at the
75% stage. Effective sample size: minimum over stages and parameters = 737.

**Posterior / prior std ratio at 75%** (1.0 = no information): k_over_m = 0.265, c_over_m = 0.479, e_over_m = 0.302, sigma_obs = 0.659, a_disc = 0.515.
Natural speed sqrt(k/m) at 75%: median 2835 RPM, 95% interval [1101, 12670].
Correlation matrix at 75% (order k/m, c/m, e/m, sigma, a_disc):

```
[[ 1.   -0.23  0.43 -0.03 -0.01]
 [-0.23  1.    0.21 -0.04  0.15]
 [ 0.43  0.21  1.   -0.11  0.26]
 [-0.03 -0.04 -0.11  1.   -0.51]
 [-0.01  0.15  0.26 -0.51  1.  ]]
```

The largest absolute posterior correlation among (log k/m, log c/m, log e/m) at 75% is 0.43; correlation of log(k/m) with log(e/m) = 0.43, of log(c/m) with log(e/m) = 0.21.

**Coverage of the posterior predictive on unused RPM points** (intervals at 50/80/95%; n = number of unused points, in-range
(<= 2400 RPM) and extrapolated (2700-3300 RPM) reported separately):

| stage | 50% in-range | 80% in-range | 95% in-range | 95% extrapolated |
|---|---|---|---|---|
| 25% | 0.00 (5) | 0.40 (5) | 0.60 (5) | 1.00 (3) |
| 50% | 0.00 (2) | 0.00 (2) | 0.00 (2) | 1.00 (3) |
| 75% | n/a (0) | n/a (0) | n/a (0) | 0.33 (3) |

Pooled in-range coverage (25% and 50% stages; the same RPM points appear in both, so the evaluations are not independent):
0.00 / 0.29 / 0.43 at 50 / 80 / 95% (n = 7).
Extrapolated 95% coverage = 0.78 (n = 9). Mean 95% width on 300-2400 RPM at 75% = 0.1196 g.
Round-1 reference (metrics_r1.json, 25% stage only, including extrapolated points): 0.625.

**Sensitivity to the GP variance treatment** (75% stage):

| variant | sigma_obs (median) | P(exceed at 2400) no action | optimum (baseline costs) | mean 95% width 300-2400 (g) | steps / max tau |
|---|---|---|---|---|---|
| per-RPM at posterior median (round 2) | 0.0296 | 1.000 | 2100 (rebalance) | 0.1196 | 23.0 |
| constant sd=0.05 (6000 steps, not 50-tau converged) | 0.1275 | 1.000 | 1950 (rebalance) | 0.1090 | 15.7 |
| constant sd=0.26 (6000 steps, not 50-tau converged) | 0.0302 | 1.000 | 1950 (rebalance) | 0.1509 | 25.5 |

Constant-sd rows use 6000 steps and are not 50-tau converged; they are sensitivity checks only.

## 8. Risk and decision (calibrated range 300-2400 RPM)
- Limit: P99 of healthy Z RMS, 0.1028 g (percentile assumption, not an ISO limit).
- Risk: P(RMS > limit) from the posterior predictive (sigma_obs, GP delta-method variance, conditional discrepancy).
  Headline: 2400 RPM (top of calibrated range), no action, 75% posterior = **1.000**.
- Extrapolation, NOT part of the headline (flagged): P(exceed) at 2550 RPM: 1.000, 2700 RPM: 1.000, 2850 RPM: 1.000, 3000 RPM: 1.000, 3150 RPM: 0.999, 3300 RPM: 0.998 (75% posterior, no action). The 3300 RPM value of
  round 1 (0.9998) came from the global-sd likelihood and from speeds that were never observed.
- Decision: minimise J(r, a) = V_full (1 - r/r_max) + C_fail h_fail P_exceed(r, a) + a C_rb / H_amort over setpoints
  300-2400 RPM (150 step, r_max = 2400, derating = r/r_max) and rebalance (a = 1, residual = resid_frac x U).
  Baseline costs: V_full = 10, C_fail = 500, h_fail = 0.1, C_rb = 200, H_amort = 200, resid_frac = 0.1. All are ASSUMPTIONS.

| calibration stage | max safe RPM (no action, P<=5%) | optimum RPM | action | J* (cost/h) | derate factor | P(exceed) at 2400 RPM |
|---|---|---|---|---|---|---|
| 25% | 1950.0 | 2400 | rebalance | 1.54 | 1.00 | 0.074 |
| 50% | 1800.0 | 1650 | continue | 3.21 | 0.69 | 0.837 |
| 75% | 1650.0 | 2100 | rebalance | 2.89 | 0.88 | 1.000 |

One-at-a-time sensitivity (75% stage; other costs at baseline):

| parameter | value | optimum RPM | action | J* | |
|---|---|---|---|---|---|
| V_full | 5.0 | 1650 | continue | 1.84 |  |
| V_full | 10.0 | 2100 | rebalance | 2.89 | baseline |
| V_full | 20.0 | 2100 | rebalance | 4.14 |  |
| h_fail | 0.05 | 2100 | rebalance | 2.57 |  |
| h_fail | 0.1 | 2100 | rebalance | 2.89 | baseline |
| h_fail | 0.2 | 1950 | rebalance | 3.12 |  |
| resid_frac | 0.1 | 2100 | rebalance | 2.89 | baseline |
| resid_frac | 0.5 | 1650 | continue | 3.40 |  |
| resid_frac | 1.0 | 1650 | continue | 3.40 |  |
| C_fail | 100.0 | 2250 | rebalance | 2.07 |  |
| C_fail | 500.0 | 2100 | rebalance | 2.89 | baseline |
| C_fail | 2000.0 | 1950 | rebalance | 3.37 |  |

Factorial sweep over V_full {5, 10, 20} x h_fail {0.05, 0.1, 0.2} x resid_frac {0.1, 0.5, 1.0} x C_fail {100, 500, 2000}
(81 combinations, 75% posterior): optimum RPM range **1500-2400 RPM**;
96% of combinations have an interior optimum (300 < r* < 2400).
resid_frac = 1.0 (rebalance does not help) is included; there the rebalance option is never preferred.

Decision checks:
- optimum_interior (baseline optimum strictly inside 300-2400): **True** (baseline optimum 2100 RPM).
- Stressed scenario (C_fail = 2000, h_fail = 0.2): optimum 1800 RPM, action rebalance;
  action non-trivial (rebalance or derate below 2400 RPM): **True**.

## 9. Changes vs round 1
Round-1 values are read from `results/metrics_r1.json`; round-2 values are from this run.

| quantity | round 1 | round 2 |
|---|---|---|
| GP sd in likelihood (log) | 0.2617 (one global value) | per-RPM 0.177-0.235 |
| Calibration chain (steps / burn) | 6000 / 1500 | 12000 / 1500 |
| Mean autocorr time (75%, over params) | 151.9 | 290.7 |
| Post-burn steps / max tau (worst stage) | 22.3 (50% stage) | 23.0 (all stages, min) |
| 95% coverage (calibration) | 0.625 (25% stage, incl. extrapolated) | 0.429 (pooled, in-range, n=7) |
| Headline risk | 0.9998 at 3300 RPM | 1.0000 at 2400 RPM |
| Decision optimum (RPM) | 2250 (grid 300-3300) | 2100 (grid 300-2400); range 1500-2400 |
| Decision J* (cost/h) | 4.915 | 2.892 |
| Hybrid NRMSE | 0.345 (seed 0) | 0.345 (mean of 5 seeds) |
| Data-driven NRMSE | 0.419 (seed 0) | 0.418 (mean of 5 seeds) |
| Sobol S_e (4-param) | 0.650 | 0.634 (S_m = 0.014) |
| Surrogate R2 (log) | 0.9973 | 0.9988 |
| Runtime (s) | 194.6 | 431.0 |

Changes made, mapped to the round-1 review:
1. Headline risk and decision restricted to 300-2400 RPM; 2700-3300 RPM only as a flagged extrapolation (review item 1).
2. Single global GP sd replaced by per-RPM sd at the posterior median; model-discrepancy GP added; coverage re-measured (item 2).
3. CAL_NSTEPS 6000 -> 12000 with burn 1500; per-stage 50-tau check written to metrics.json (item 3).
4. Decision sweep over V_full, h_fail, resid_frac and C_fail; optimum reported as a range; interior and stressed-scenario checks added (item 4).
5. RF CV repeated over 5 seeds with paired per-group differences and a sign test (item 5, first part).
6. Summary numbers and claims generated in code: Z RMS vs mean wording fixed, GP length-scale claim computed, user-supplied context labelled
   and not used in any computation, sigma labels corrected to log units (item 5, second part).
7. m and e reparametrised to (k/m, c/m, e/m) for the GP and the calibration; identifiability checked numerically (group item).
8. Axis choice re-checked on training rows of every LOGO fold (review item 5, leakage).
9. Fast matmul GP mean path (threaded) to fit 12000-step chains in the runtime budget; agreement with sklearn checked.

## 10. Known limitations
- Target is the Z-axis RMS (chosen by a documented criterion); the axis choice is recomputed on the training rows of every LOGO fold (18/18 folds pick Z), so the held-out groups do not drive it. The Normal reference is always in training.
- The data have no time axis: no wear/degradation trajectory, so there is no RUL; the decision is a speed/rebalance choice at a fixed unbalance state.
- Misalignment is not modelled physically; the Jeffcott physics only describes the unbalance-induced response. Misalignment-only groups are predicted by the healthy floor in Pure Physics.
- m and e are NOT separately identifiable from RMS vs RPM: the response depends only on k/m, c/m and e/m (identity checked numerically, see section 4). The calibration therefore runs in (k/m, c/m, e/m); m is not a calibrated quantity and the absolute values of k, c, e are only conditional on the fixed M_FIX = 3 kg.
- The sensor gain and the physical unbalance radius are absorbed into e/m; RMS is assumed to be in g. No sensor calibration was available.
- Both-disc configurations (TypeA=TypeB=1) are modelled as twice the unbalance mass in phase; phase between the two discs is not modelled.
- The healthy-machine background floor (per-RPM mean of the Normal rows) is added in quadrature. It is always kept in training, so the Normal group is not a held-out test set.
- Calibration uses one held-out experiment (group M0.0_U20.4, TypeA only). Only 7 in-range unused point evaluations (pooled over the 25% and 50% stages, which share the same points) and 9 extrapolated ones are available, so the coverage estimate is coarse.
- The model-discrepancy GP has a fixed correlation length of 500 RPM (assumption) and is added in log space; the same discrepancy is applied to the rebalanced scenario, which is an approximation.
- Severity limit = P99 of healthy Z-RMS (0.1028 g; a percentile assumption, no ISO limit for raw g units). Decision costs are assumptions, not data; they are swept (V_full, h_fail, resid_frac, C_fail) and the optimum is reported as a range.
- The GP surrogate is slower than the closed form (speedup_x = 0.0059); the physics is analytic. The surrogate is kept because the brief requires it for Sobol and MCMC; no speedup claim is made.
- The 5 repeats per (condition, Position, RPM) cell are treated as repeated segments and stay in the same fold; some conditions miss positions for some RPMs (see partial_conditions), so position averaging is unbalanced.
- Dataset is recorded bench data according to the user (COBEM 2025 setup, not in the data); features are not checked against a second acquisition chain.
- Round 2 uses 100 RF trees per fit (round 1: 200) to keep 5 seeds within the runtime budget.
- Calibration: 3-8 RPM points, one held-out experiment; the discrepancy amplitude and sigma_obs are only weakly separated by the data.
- The discrepancy correction is applied additively in log space to the rebalanced scenario as well (approximation).
- Sobol indices depend on the prior boxes (section 6).

## 11. Files
- Code: `p4_rotor/main.py`, `p4_rotor/src/data_loader.py`, `physics_model.py`, `surrogate_model.py`, `uq_calibration.py`, `decision.py`
- Results: `p4_rotor/results/metrics.json`, `p4_rotor/results/summary.md`, `p4_rotor/results/run.log`, `p4_rotor/results/metrics_r1.json` (round-1 copy), `p4_rotor/results/figures/fig1..fig6*.png`
- Raw data was read only: `_data/p4_rotor/features.csv`
