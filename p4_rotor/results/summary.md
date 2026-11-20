# p4_rotor: physics + data digital twin of rotating-machinery unbalance vibration (round 3)

Adaptation of the common brief to the user's 2-disc bench dataset. The COBEM 2025 bench context (triaxial accelerometer 1 kHz,
0.25 HP motor, two disks A/B) is user-supplied and is not in the data; it is not used by any computation. No wear or time axis
exists, so the time-to-action decision is replaced by a speed-and-rebalance decision at a fixed unbalance state.

Runtime and seeds: see section 11 and `run.log`. Seed 0; CV seeds [0, 1, 2, 3, 4]. Every number below is written by `main.py`
from values computed in this run; round-1 and round-2 values are read from `metrics_r1.json` and `metrics_r2.json`.

## 0. Verdict (round 3, honest)
- **Physics.** The 2-DOF model (per-disc unbalance with phase, per-Position weights) gives a pooled held-out RMSE of 0.1090 g against 0.1504 g for the 1-DOF ablation (criterion C1: PASS). On the A1B1 configuration the 2-DOF RMSE is 0.1202 g against 0.3382 g (C2: PASS). The pure 2-DOF physics is still worse than the pure data model (0.0673 g pooled, 0.0222 g on A1B1) and the hybrid (0.0563 g pooled, 0.0140 g on A1B1). It does not reach the measurement noise: training log-RMSE 0.336 against within-cell noise 0.008 (C3: FAIL); the largest per-RPM residuals of the fit are at 2100 and 2700 RPM.
- **Calibration.** MCMC convergence at 50 tau: NOT met (min steps/tau = 8.2 against 23.0 in round 2; C4: FAIL); runs below 50 tau: 25% stage, 50% stage, 75% stage, replication M0.0_U13.6_A1B0 50%, sensitivity a_disc=0 (75%). Out-of-sample per-point coverage on the fair 50% stage (unused 2100-3300 RPM, all Positions) is 50% 0.40, 80% 0.93, 95% 1.00 (C5: FAIL - boundary case: the value sits exactly on 0.95 + 0.05, and the floating-point inclusive test evaluates it as a miss by a rounding margin; recorded as computed). The replication on M0.0_U13.6_A1B0 gives 95% 1.00.
- **Identifiability.** 7 of 11 calibrated physics parameters have a posterior/prior std ratio below 0.5 at the 75% stage (C6: PASS); the others are not separated from their prior at this threshold.
- **Risk and decision.** P(any Position > 0.1028 g) at 2400 RPM without action is 1.000 at 75% (in-sample resonance) and 0.976 at 50% (fair). At 75% it is at or above 0.99 on 27% of the calibrated grid, so the risk is saturated only on part of the grid. The optimum is not robust: the cost sweep gives 1350-2400 RPM. The rebalance-vs-continue boundary moves with resid_frac (summary section 8).
- **Round-3 verdict.** 5 of 8 pre-registered criteria pass; failed: C3, C4, C5. The 2-DOF physics improves on the single-DOF ablation on both physics criteria, but it does not describe the bench to the measurement noise. Round-2 headline values that the review showed to be non-comparable (pooled coverage 0.429, single optimum 2100 RPM) are replaced by the like-for-like values in sections 7 and 8.

Pre-registered criteria (defined in main.py before the run): 5 of 8 pass. Failed: C3, C4, C5.

## 1. Data inspection
- 5980 rows, 55 columns, NaN = 0. 42 feature columns (3 axes x 14).
- Speeds: 300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300 RPM. Positions 1-3 (sensor index; the axial location is not documented).
  38 experimental conditions, 19 (Misalignment, Unbalance) groups. 165 Normal rows.
- Every (condition, Position, RPM) cell holds 5 repeats (1196 cells). Repeats stay in one fold.
- Partial conditions (not 165 rows): {'M0.5_U13.6_A0B1': 70, 'M0.5_U20.4_A0B1': 65, 'M0.5_U27.2_A0B1': 70}. The Misalignment=0.5 Unbalance=13.6/20.4/27.2 partial runs are used as they are.
- No time column, no serial number. The experimental condition is the unit ID for splits.
- RMS and Standard Deviation are the same signal: max |RMS - STD| = 0.0077 g, so RMS is the target.
- Healthy background per Position (Z RMS, Normal rows) at 300 RPM: P1 0.0202 g, P2 0.0150 g, P3 0.0275 g.
  The Position-3 value is above every unbalance run at low speed (0.011-0.018 g), so a fixed floor cannot match Position 3 at low speed;
  a per-Position floor scale zeta_p is therefore fitted.
- Within-cell repeat noise: median log sd = 0.0084 (5 repeats). This is the floor any model can reach on these cells.

## 2. Target axis choice
Round-2 criterion, unchanged: Pearson correlation between log(excess RMS) and log(U_eff RPM^2) on Mis=0, Unbalance>0 rows.

| axis | corr raw | corr excess (criterion) | clipped rows |
|---|---|---|---|
| X | 0.863 | 0.162 | 700 |
| Y | 0.804 | 0.263 | 706 |
| Z | 0.750 | 0.689 | 413 |

Chosen target: **Z Root Mean Squared**. Leak check: Z wins in 18/18 LOGO folds when the axis is recomputed on the training rows.
Limit = P99 of healthy Z RMS = 0.1028 g.

## 3. Primary physics: two-DOF lumped rotor (closed form)
Two disc coordinates on bearing springs k_A, k_B, coupled by a shaft spring k_s, viscous bearing damping c_A, c_B (per unit mass,
disc masses normalised to 1). Unbalance forcing e0*U_A*w^2 on disc A and kappa_B*e0*U_B*w^2*exp(i*phi) on disc B. Steady state
X = Z^-1 f with the closed 2x2 inverse. Output at Position p: w^2 |G_p (X_A + rho_p X_B)| / sqrt(2) / g0 (RMS, g), plus the
healthy floor zeta_p * floor_p(RPM) in quadrature. G_1 = 1 (scale absorbed by e0). 16 parameters.

Full fit on all Mis=0 cells (incl. A1B1), 80 random starts: training log-RMSE 0.3362
(within-cell noise 0.0084). Parameters (log values where named log_):

| parameter | value |
|---|---|
| log_kA | 11.97 |
| log_kB | 9.21 |
| log_ks | 11.04 |
| log_cA | 6.803 |
| log_cB | -2.303 |
| log_e0 | -12.72 |
| log_kappaB | -1.587 |
| phi | 1.97 |
| rho_p1 | -1.527 |
| rho_p2 | -5.109 |
| rho_p3 | 6 |
| log_G_p2 | -1.29 |
| log_G_p3 | -1.137 |
| log_zeta_p1 | 0.001143 |
| log_zeta_p2 | -0.005474 |
| log_zeta_p3 | -0.5986 |

Undamped natural speeds of the fitted coupled system (eigenvalues of the stiffness matrix): 2125 and 4703 RPM.
Position weights rho_p and G_p act through the two disc coordinates; the model makes no axial-position claim.

Leave-one-group-out physics: only the three Mis=0 unbalance groups change the fit; each refit excludes the held-out group.

| held-out group | physics fit | 2-DOF training log-RMSE | 1-DOF ablation training log-RMSE | axis on training rows |
|---|---|---|---|---|
| M0.0_U13.6 | refit | 0.339 | 0.387 | Z |
| M0.0_U20.4 | refit | 0.309 | 0.410 | Z |
| M0.0_U6.8 | refit | 0.347 | 0.415 | Z |
| M0.5_U0.0 | full fit (no Mis=0 cell held out) | 0.336 | 0.405 | Z |
| M0.5_U13.6 | full fit (no Mis=0 cell held out) | 0.336 | 0.405 | Z |
| M0.5_U20.4 | full fit (no Mis=0 cell held out) | 0.336 | 0.405 | Z |
| M0.5_U27.2 | full fit (no Mis=0 cell held out) | 0.336 | 0.405 | Z |
| M0.5_U6.8 | full fit (no Mis=0 cell held out) | 0.336 | 0.405 | Z |
| M1.0_U0.0 | full fit (no Mis=0 cell held out) | 0.336 | 0.405 | Z |
| M1.0_U13.6 | full fit (no Mis=0 cell held out) | 0.336 | 0.405 | Z |
| M1.0_U20.4 | full fit (no Mis=0 cell held out) | 0.336 | 0.405 | Z |
| M1.0_U27.2 | full fit (no Mis=0 cell held out) | 0.336 | 0.405 | Z |
| M1.0_U6.8 | full fit (no Mis=0 cell held out) | 0.336 | 0.405 | Z |
| M1.5_U0.0 | full fit (no Mis=0 cell held out) | 0.336 | 0.405 | Z |
| M1.5_U13.6 | full fit (no Mis=0 cell held out) | 0.336 | 0.405 | Z |
| M1.5_U20.4 | full fit (no Mis=0 cell held out) | 0.336 | 0.405 | Z |
| M1.5_U27.2 | full fit (no Mis=0 cell held out) | 0.336 | 0.405 | Z |
| M1.5_U6.8 | full fit (no Mis=0 cell held out) | 0.336 | 0.405 | Z |

Interpretation: the 2-DOF physics fits the Mis=0 cells with a log-RMSE of 0.336, which is 40 times the
within-cell repeat noise. It is a partial description: the largest per-RPM residuals are at 2100 and 2700 RPM, and the
per-Position log-RMSE is 0.277 (Position 1), 0.316 (Position 2) and 0.403 (Position 3). The Position-3
healthy floor is above the unbalance runs at 300 RPM (floor 0.0275 g against 0.0110-0.0265 g), which is
why zeta_3 is fitted. The 1-DOF ablation (A1B1 excluded from its fit) is listed for comparison.

## 4. Held-out comparison (leave-one-(Unbalance, Misalignment)-group-out, 18 groups, 5 RF seeds)
Normal group always in training. Pooled over all held-out rows. RF uses 100 trees. Physics is deterministic.

| model | RMSE (g), mean | sd over seeds | NRMSE (RMSE / std(y)) |
|---|---|---|---|
| Pure Physics, 2-DOF (primary) | 0.1090 | 0 | 0.676 |
| Ablation: 1-DOF Jeffcott (A1B1 excluded from fit) | 0.1504 | 0 | 0.933 |
| Pure Data, condition inputs | 0.0694 | 0.0003 | 0.430 |
| Pure Data, condition + other-axis features | 0.0673 | 0.0003 | 0.418 |
| Hybrid (2-DOF + RandomForest residual) | 0.0563 | 0.0006 | 0.349 |

std(y) of the evaluated rows = 0.1612 g. Criterion C1 (2-DOF vs 1-DOF pooled): PASS.

Breakdown by Position and by configuration (RMSE in g, mean over seeds; A1B1 = both discs, all Misalignment and Mis=0 only):

| model | P1 | P2 | P3 | A1B1 (all Mis) | A1B1 (Mis=0) |
|---|---|---|---|---|---|
| Pure Physics, 2-DOF (primary) | 0.1060 | 0.0889 | 0.1296 | 0.1202 | 0.2381 |
| Ablation: 1-DOF Jeffcott (A1B1 excluded from fit) | 0.1421 | 0.1490 | 0.1597 | 0.3382 | 0.3362 |
| Pure Data, condition inputs | 0.0583 | 0.0589 | 0.0875 | 0.0162 | 0.0214 |
| Pure Data, condition + other-axis features | 0.0658 | 0.0526 | 0.0815 | 0.0222 | 0.0256 |
| Hybrid (2-DOF + RandomForest residual) | 0.0531 | 0.0427 | 0.0704 | 0.0140 | 0.0205 |

Criterion C2 (A1B1, 2-DOF vs 1-DOF): PASS (0.1202 vs 0.3382 g).
Note for the fold M0.0_U13.6: A1B1 at Mis=0 exists only at 13.6 g, so that fold has no A1B1 training cells and phi keeps the full-data
value (flagged, see section 10).

Paired per-group comparison (RMSE per group, averaged over seeds):

| group | 2-DOF | 1-DOF (ablation) | Data (cond.) | Data (cond.+feat.) | Hybrid | hybrid - data(full) |
|---|---|---|---|---|---|---|
| M0.0_U13.6 | 0.1683 | 0.2186 | 0.1086 | 0.1223 | 0.0821 | -0.0402 |
| M0.0_U20.4 | 0.1812 | 0.1843 | 0.1980 | 0.1761 | 0.1562 | -0.0199 |
| M0.0_U6.8 | 0.0642 | 0.0649 | 0.0567 | 0.0587 | 0.0580 | -0.0007 |
| M0.5_U0.0 | 0.0152 | 0.0095 | 0.0117 | 0.0165 | 0.0094 | -0.0071 |
| M0.5_U13.6 | 0.0434 | 0.2180 | 0.0311 | 0.0287 | 0.0202 | -0.0085 |
| M0.5_U20.4 | 0.1094 | 0.1022 | 0.1057 | 0.0506 | 0.0468 | -0.0038 |
| M0.5_U27.2 | 0.1700 | 0.1594 | 0.0529 | 0.0411 | 0.0782 | 0.0371 |
| M0.5_U6.8 | 0.0337 | 0.0335 | 0.0408 | 0.0244 | 0.0201 | -0.0043 |
| M1.0_U0.0 | 0.0174 | 0.0090 | 0.0114 | 0.0092 | 0.0106 | 0.0014 |
| M1.0_U13.6 | 0.0488 | 0.2014 | 0.0245 | 0.0405 | 0.0229 | -0.0176 |
| M1.0_U20.4 | 0.0954 | 0.0879 | 0.0442 | 0.0372 | 0.0317 | -0.0056 |
| M1.0_U27.2 | 0.1766 | 0.1698 | 0.0625 | 0.0681 | 0.0615 | -0.0066 |
| M1.0_U6.8 | 0.0314 | 0.0313 | 0.0139 | 0.0269 | 0.0177 | -0.0091 |
| M1.5_U0.0 | 0.0203 | 0.0150 | 0.0141 | 0.0113 | 0.0113 | 0.0001 |
| M1.5_U13.6 | 0.0542 | 0.2037 | 0.0121 | 0.0317 | 0.0232 | -0.0085 |
| M1.5_U20.4 | 0.1151 | 0.1080 | 0.0350 | 0.0432 | 0.0407 | -0.0025 |
| M1.5_U27.2 | 0.1819 | 0.1752 | 0.0552 | 0.0675 | 0.0456 | -0.0219 |
| M1.5_U6.8 | 0.0387 | 0.0388 | 0.0172 | 0.0322 | 0.0222 | -0.0100 |

- Hybrid vs data-driven: pooled difference -0.0111 g (hybrid better in 15/18 groups,
  two-sided sign test p = 0.008; bootstrap 95% CI of the mean group difference [-0.0137, -0.0003] g).
  The row-weighted pooled difference is -0.0111 g; the unweighted mean of the per-group differences is -0.0071 g. Both are reported because they can differ.
- Hybrid vs 2-DOF physics: pooled difference -0.0527 g; hybrid better in 18/18 groups (sign p = 0.000).
- 2-DOF vs 1-DOF ablation: pooled difference -0.0414 g; 2-DOF better in 7/18 groups (sign p = 0.481).

## 5. Closed-form verification (replaces the round-2 GP surrogate)
The round-2 GP emulated the single-DOF response; it was slower than the closed form (speedup 0.0059x measured in round 2) and its
emulator variance inflated the calibration likelihood. Round 3 removes it. MCMC and Sobol call the closed form directly. To check the
closed form, 300 random parameter sets, speeds, positions and disc loadings were compared with an independent real 4x4 linear
solve of the same equations: R2(log)=1.000000000000, RMSE=6.57e-17 g, max relative error=2.70e-15; time ratio (4x4 loop / closed-form loop over the same single points, dominated by Python overhead) = 0.97. No speed claim is made from this ratio; the batch evaluations used by MCMC and Sobol are the speed-relevant paths. The Sobol study and the calibration use the same closed form, so this check covers them.

## 6. Sobol sensitivity (SALib, Saltelli N=1024, 9216 evaluations, 2-DOF closed form)
Output: log of the peak unbalance response (A1B1, 20.4 g per disc) over 3 Positions and 300-3300 RPM, for the 7 parameters that
define the response shape. Prior boxes are the log-uniform boxes of the physical parameters.

| parameter | S1 | ST |
|---|---|---|
| log_kA | 0.074 | 0.240 |
| log_kB | 0.172 | 0.320 |
| log_ks | 0.030 | 0.130 |
| log_cA | 0.011 | 0.072 |
| log_cB | 0.040 | 0.086 |
| log_kappaB | 0.422 | 0.534 |
| phi | 0.005 | 0.005 |

Caveats: (i) indices are conditional on the prior boxes; (ii) kappa_B and phi enter only through disc B, so their indices measure
the second disc's effect on the peak; (iii) the peak is taken over the extrapolated range too, because the prior box covers it.

## 7. Sequential Bayesian calibration (emcee, 32 walkers, 22000 steps, burn-in 3000)
Held-out experiment M0.0_U20.4_A1B0 (group M0.0_U20.4 excluded from the centre). Observations: linear mean over the 5 repeats for each
(Position, RPM), so 3 x n data points per stage. Likelihood: per Position, z ~ N(log model, sigma^2 I + a^2 K), K the RPM
discrepancy kernel (ELL = 500 RPM, assumption). The closed-form model has no emulator variance.

Calibrated: the 11 parameters that single-disc A1B0 data can inform (log k_A, log k_B, log k_s, log c_A, log c_B, log e0, rho_p1..3,
log G_p2, log G_p3) plus log sigma_obs and log a_disc. kappa_B, phi and the floor scales are fixed at the leave-group-out centre
(M0.0_U20.4 excluded). Priors are uniform in the log boxes. Initialisation: MAP by L-BFGS-B from the centre and 3 random starts.
Moves: DEMove 0.8, DESnookerMove 0.2.

**Which stages contain the resonance.** Per-Position peak speeds of the A1B0 U20.4 sweep: Position 1 at 2400 RPM, Position 2 at 2400 RPM, Position 3 at 2100 RPM.
The per-Position peak speeds of the calibration sweep (argmax of the cell means) are 2400 RPM (Position 1),
2400 RPM (Position 2) and 2100 RPM (Position 3).
- 25% stage (300-900 RPM, 3 points): the resonance is **not** in the calibration data. The unused RPM points (8 unique) are the round-1 point set; they contain the peak out of sample.
- 50% stage (300-1800 RPM, 6 points): the resonance peak is **not** in the calibration data. The unused RPM points (5 unique, 2100, 2400, 2700, 3000, 3300 RPM) include the peaks: this is the **fair out-of-sample resonance test**.
- 75% stage (300-2400 RPM, 8 points): the resonance is **in-sample**. The unused RPM points (3 unique, 2700, 3000, 3300 RPM) are extrapolation only.
Round 2 pooled the 25% and 50% stages and counted 2100 and 2400 RPM twice (n = 7 for 5 unique points). Round 3 reports unique
RPM points per stage and does not pool.

Chain diagnostics. Autocorrelation times are per parameter; "nan" means a flat or stuck direction (counted as not converged).

| stage | RPM pts | obs (pts x Positions) | acceptance | max tau | non-finite tau params | steps / max tau | MAP log posterior | sigma_obs (median) | a_disc (median) |
|---|---|---|---|---|---|---|---|---|---|
| 25% | 3 | 9 | 0.060 | 602.4 | 0 | 31.5 | 7.57 | 0.0537 | 0.1950 |
| 50% | 6 | 18 | 0.021 | 1525.3 | 0 | 12.5 | 15.12 | 0.0535 | 0.1799 |
| 75% | 8 | 24 | 0.022 | 2304.4 | 0 | 8.2 | 10.87 | 0.0810 | 0.1638 |

Replication on M0.0_U13.6_A1B0 at 50%: steps/max-tau = 15.1; sensitivity with a_disc = 0 at 75%: steps/max-tau = 8.6.
50-tau rule (C4): NOT met for at least one run; minimum over all runs = 8.2.
Effective sample size, minimum over runs = 264.

| stage | tau per parameter (11 physics, sigma, a_disc) | steps/max-tau | 50 tau met |
|---|---|---|---|
| 25% | 260 | 243 | 271 | 252 | 243 | 292 | 237 | 267 | 277 | 220 | 322 | 479 | 602 | 31.5 | no |
| 50% | 1383 | 1525 | 1130 | 1147 | 1054 | 1238 | 1180 | 1183 | 1264 | 1309 | 1226 | 1157 | 1015 | 12.5 | no |
| 75% | 1482 | 2056 | 1489 | 1350 | 1319 | 1800 | 1982 | 1588 | 1977 | 1577 | 1870 | 2304 | 1752 | 8.2 | no |

Posterior / prior std ratio at 75% (1.0 = no information; ratio < 0.5 counted as identified, C6):

| parameter | ratio |
|---|---|
| log_kA | 0.035 |
| log_kB | 0.361 |
| log_ks | 0.233 |
| log_cA | 0.621 |
| log_cB | 0.572 |
| log_e0 | 0.225 |
| rho_p1 | 1.153 |
| rho_p2 | 0.486 |
| rho_p3 | 0.724 |
| log_G_p2 | 0.254 |
| log_G_p3 | 0.347 |
| log_sigma_obs | 0.300 |
| log_a_disc | 0.693 |

7 of 11 physics parameters are identified by this definition. Correlation matrix at 75% (order as in the
parameter list above, sigma and a_disc last):

```
[[ 1.    0.74 -0.3   0.06  0.33  0.48 -0.12  0.08 -0.01 -0.2  -0.29  0.39
  -0.25]
 [ 0.74  1.    0.01 -0.01  0.37  0.3   0.01  0.25 -0.18 -0.08 -0.12  0.49
  -0.28]
 [-0.3   0.01  1.    0.01  0.14 -0.11  0.11  0.5  -0.15  0.24  0.19  0.16
  -0.15]
 [ 0.06 -0.01  0.01  1.    0.09  0.01 -0.02  0.25  0.25 -0.02 -0.03 -0.07
  -0.03]
 [ 0.33  0.37  0.14  0.09  1.    0.41 -0.29  0.23  0.24 -0.25 -0.39  0.2
  -0.2 ]
 [ 0.48  0.3  -0.11  0.01  0.41  1.   -0.78  0.04  0.38 -0.81 -0.92  0.28
  -0.38]
 [-0.12  0.01  0.11 -0.02 -0.29 -0.78  1.   -0.   -0.62  0.77  0.87  0.12
   0.1 ]
 [ 0.08  0.25  0.5   0.25  0.23  0.04 -0.    1.    0.21 -0.05 -0.02  0.15
  -0.17]
 [-0.01 -0.18 -0.15  0.25  0.24  0.38 -0.62  0.21  1.   -0.48 -0.6  -0.45
   0.18]
 [-0.2  -0.08  0.24 -0.02 -0.25 -0.81  0.77 -0.05 -0.48  1.    0.85 -0.01
   0.14]
 [-0.29 -0.12  0.19 -0.03 -0.39 -0.92  0.87 -0.02 -0.6   0.85  1.   -0.05
   0.22]
 [ 0.39  0.49  0.16 -0.07  0.2   0.28  0.12  0.15 -0.45 -0.01 -0.05  1.
  -0.68]
 [-0.25 -0.28 -0.15 -0.03 -0.2  -0.38  0.1  -0.17  0.18  0.14  0.22 -0.68
   1.  ]]
```

**Out-of-sample per-point coverage** (unique unused RPM; every Position is one evaluation; intervals at 50/80/95%):

| stage | unused RPM points | 50% (all) | 80% (all) | 95% (all) | 95% position-averaged (r1 definition) | 95% in-range | 95% extrapolated |
|---|---|---|---|---|---|---|---|
| main 25% | 8 (5 in-range, 3 extrapolated) | 0.12 | 0.83 | 1.00 | 1.00 | 1.00 | 1.00 |
| main 50% | 5 (2 in-range, 3 extrapolated) | 0.40 | 0.93 | 1.00 | 1.00 | 1.00 | 1.00 |
| main 75% | 3 (0 in-range, 3 extrapolated) | 0.22 | 0.56 | 0.78 | 1.00 | n/a | 0.78 |
| replication M0.0_U13.6_A1B0 50% | 5 (2 in-range, 3 extrapolated) | 0.47 | 0.87 | 1.00 | 1.00 | 1.00 | 1.00 |

Like-for-like with round 1 and round 2: the round-1 definition (95%, 25% stage, the 8-point set 1200-3300 RPM, observation averaged over
Positions) gives **1.000** in round 3. The round-2 value for the same 25% stage, computed like-for-like from run_r2.log
(3 of 5 in-range plus 3 of 3 extrapolated), is **0.750**. The round-2 pooled value of 0.429 is not comparable and is not used.
The fair 50% stage gives **1.00** per point at 95% for the in-range (2100-2400) and extrapolated points together; the in-range
part is shown in the table.

Per-point table (every out-of-sample evaluation):

| stage | RPM | range | Position | observed (g) | 95% interval (g) | 50% hit | 80% hit | 95% hit |
|---|---|---|---|---|---|---|---|---|
| main 25% | 1200 | in | P1 | 0.0298 | [0.0230, 0.0568] | yes | yes | yes |
| main 25% | 1500 | in | P1 | 0.0359 | [0.0243, 0.1613] | yes | yes | yes |
| main 25% | 1800 | in | P1 | 0.0540 | [0.0257, 0.3479] | no | yes | yes |
| main 25% | 2100 | in | P1 | 0.1129 | [0.0238, 0.7234] | no | yes | yes |
| main 25% | 2400 | in | P1 | 0.8112 | [0.0277, 1.2014] | no | no | yes |
| main 25% | 2700 | extrap | P1 | 0.5694 | [0.0351, 2.1154] | no | no | yes |
| main 25% | 3000 | extrap | P1 | 0.3861 | [0.0381, 3.5154] | no | yes | yes |
| main 25% | 3300 | extrap | P1 | 0.5628 | [0.0350, 6.0108] | no | yes | yes |
| main 25% | 1200 | in | P2 | 0.0287 | [0.0188, 0.0464] | no | yes | yes |
| main 25% | 1500 | in | P2 | 0.0356 | [0.0188, 0.1364] | no | yes | yes |
| main 25% | 1800 | in | P2 | 0.0545 | [0.0207, 0.3070] | no | yes | yes |
| main 25% | 2100 | in | P2 | 0.1114 | [0.0206, 0.6398] | no | yes | yes |
| main 25% | 2400 | in | P2 | 0.6495 | [0.0272, 1.0830] | no | no | yes |
| main 25% | 2700 | extrap | P2 | 0.3983 | [0.0357, 1.7220] | no | yes | yes |
| main 25% | 3000 | extrap | P2 | 0.1969 | [0.0418, 2.9502] | no | yes | yes |
| main 25% | 3300 | extrap | P2 | 0.5407 | [0.0646, 5.1316] | no | yes | yes |
| main 25% | 1200 | in | P3 | 0.0242 | [0.0151, 0.0509] | yes | yes | yes |
| main 25% | 1500 | in | P3 | 0.0451 | [0.0155, 0.1542] | no | yes | yes |
| main 25% | 1800 | in | P3 | 0.1333 | [0.0156, 0.3503] | no | yes | yes |
| main 25% | 2100 | in | P3 | 0.7532 | [0.0148, 0.8222] | no | no | yes |
| main 25% | 2400 | in | P3 | 0.5436 | [0.0151, 1.2309] | no | yes | yes |
| main 25% | 2700 | extrap | P3 | 0.4120 | [0.0186, 2.5931] | no | yes | yes |
| main 25% | 3000 | extrap | P3 | 0.3727 | [0.0198, 4.1139] | no | yes | yes |
| main 25% | 3300 | extrap | P3 | 0.4057 | [0.0192, 6.6368] | no | yes | yes |
| main 50% | 2100 | in | P1 | 0.1129 | [0.0451, 0.5874] | yes | yes | yes |
| main 50% | 2400 | in | P1 | 0.8112 | [0.0438, 2.1815] | no | no | yes |
| main 50% | 2700 | extrap | P1 | 0.5694 | [0.0488, 1.7921] | no | yes | yes |
| main 50% | 3000 | extrap | P1 | 0.3861 | [0.0507, 1.7556] | no | yes | yes |
| main 50% | 3300 | extrap | P1 | 0.5628 | [0.0500, 2.7844] | no | yes | yes |
| main 50% | 2100 | in | P2 | 0.1114 | [0.0457, 0.5318] | yes | yes | yes |
| main 50% | 2400 | in | P2 | 0.6495 | [0.0471, 2.3585] | no | yes | yes |
| main 50% | 2700 | extrap | P2 | 0.3983 | [0.0509, 2.0815] | no | yes | yes |
| main 50% | 3000 | extrap | P2 | 0.1969 | [0.0500, 2.6588] | yes | yes | yes |
| main 50% | 3300 | extrap | P2 | 0.5407 | [0.0769, 2.8656] | no | yes | yes |
| main 50% | 2100 | in | P3 | 0.7532 | [0.1160, 2.1381] | no | yes | yes |
| main 50% | 2400 | in | P3 | 0.5436 | [0.0953, 9.5178] | yes | yes | yes |
| main 50% | 2700 | extrap | P3 | 0.4120 | [0.0889, 10.8596] | yes | yes | yes |
| main 50% | 3000 | extrap | P3 | 0.3727 | [0.0940, 10.6492] | no | yes | yes |
| main 50% | 3300 | extrap | P3 | 0.4057 | [0.0735, 12.4849] | yes | yes | yes |
| main 75% | 2700 | extrap | P1 | 0.5694 | [0.0469, 0.7435] | no | no | yes |
| main 75% | 3000 | extrap | P1 | 0.3861 | [0.0531, 0.4751] | no | no | yes |
| main 75% | 3300 | extrap | P1 | 0.5628 | [0.0447, 0.4292] | no | no | no |
| main 75% | 2700 | extrap | P2 | 0.3983 | [0.0960, 1.5139] | no | yes | yes |
| main 75% | 3000 | extrap | P2 | 0.1969 | [0.0817, 0.5587] | yes | yes | yes |
| main 75% | 3300 | extrap | P2 | 0.5407 | [0.0889, 0.4513] | no | no | no |
| main 75% | 2700 | extrap | P3 | 0.4120 | [0.2859, 3.8308] | no | yes | yes |
| main 75% | 3000 | extrap | P3 | 0.3727 | [0.2539, 2.1495] | no | yes | yes |
| main 75% | 3300 | extrap | P3 | 0.4057 | [0.2625, 1.6027] | yes | yes | yes |
| replication M0.0_U13.6_A1B0 50% | 2100 | in | P1 | 0.0759 | [0.0359, 0.2033] | no | yes | yes |
| replication M0.0_U13.6_A1B0 50% | 2400 | in | P1 | 0.2096 | [0.0383, 1.0157] | no | yes | yes |
| replication M0.0_U13.6_A1B0 50% | 2700 | extrap | P1 | 0.5846 | [0.0457, 1.2261] | no | no | yes |
| replication M0.0_U13.6_A1B0 50% | 3000 | extrap | P1 | 0.3689 | [0.0495, 1.0944] | no | yes | yes |
| replication M0.0_U13.6_A1B0 50% | 3300 | extrap | P1 | 0.2615 | [0.0459, 1.4463] | no | yes | yes |
| replication M0.0_U13.6_A1B0 50% | 2100 | in | P2 | 0.0849 | [0.0372, 0.3629] | yes | yes | yes |
| replication M0.0_U13.6_A1B0 50% | 2400 | in | P2 | 0.3366 | [0.0455, 1.7189] | no | yes | yes |
| replication M0.0_U13.6_A1B0 50% | 2700 | extrap | P2 | 0.3502 | [0.0553, 2.2659] | yes | yes | yes |
| replication M0.0_U13.6_A1B0 50% | 3000 | extrap | P2 | 0.1430 | [0.0608, 2.4511] | yes | yes | yes |
| replication M0.0_U13.6_A1B0 50% | 3300 | extrap | P2 | 0.8199 | [0.0941, 3.6149] | no | yes | yes |
| replication M0.0_U13.6_A1B0 50% | 2100 | in | P3 | 0.6488 | [0.0452, 0.8109] | no | no | yes |
| replication M0.0_U13.6_A1B0 50% | 2400 | in | P3 | 0.4322 | [0.0411, 4.6685] | yes | yes | yes |
| replication M0.0_U13.6_A1B0 50% | 2700 | extrap | P3 | 0.3183 | [0.0462, 6.2639] | yes | yes | yes |
| replication M0.0_U13.6_A1B0 50% | 3000 | extrap | P3 | 0.2756 | [0.0458, 6.8142] | yes | yes | yes |
| replication M0.0_U13.6_A1B0 50% | 3300 | extrap | P3 | 0.2863 | [0.0499, 8.9760] | yes | yes | yes |

Replication (M0.0_U13.6_A1B0, 50%): 95% per-point coverage 1.00 over 5 unique RPM points.

Mean 95% interval width on 300-2400 RPM, 75% stage (all Positions): 0.8929 g.

Sensitivity to the discrepancy term (75% stage):

| variant | sigma_obs (median) | P(any > limit) at 2400 | optimum RPM (action) | steps / max tau |
|---|---|---|---|---|
| discrepancy a_disc free (baseline) | 0.0810 | 1.000 | 2100 (rebalance) | 8.2 |
| a_disc = 0 (no discrepancy term) | 0.1631 | 1.000 | 1950 (rebalance) | 8.6 |

## 8. Risk and decision (calibrated range 300-2400 RPM only)
- Limit: P99 of healthy Z RMS, 0.1028 g (percentile assumption, not an ISO limit).
- Risk: P(any Position RMS > limit) from the posterior predictive (sigma_obs, conditional discrepancy); positions treated as
  conditionally independent given theta.
- Headline: 2400 RPM (top of the calibrated range), no action, 75% stage = **1.000**; at the fair 50% stage = 0.976.
  The 75% value is in-sample (the resonance is in the data). Above 2400 RPM the curve is an extrapolation and is not used (fig. 6).
- Decision: minimise J(r, a) = V_full (1 - r/r_max) + C_fail h_fail P(r, a) + a C_rb / H_amort over setpoints 300-2400 RPM and
  rebalance (a = 1, residual = resid_frac x U). Baseline costs (assumptions): V_full = 10, C_fail = 500, h_fail = 0.1, C_rb = 200,
  H_amort = 200, resid_frac = 0.1.

| stage | max safe RPM (no action, P<=5%) | optimum RPM | action | J* (cost/h) | P(exceed) at 2400, no action | share of grid with P >= 0.99 |
|---|---|---|---|---|---|---|
| 25% | 1350.0 | 1950 | rebalance | 4.04 | 0.385 | 0% |
| 50% | 1650.0 | 1950 | rebalance | 2.98 | 0.976 | 0% |
| 75% | 1650.0 | 2100 | rebalance | 2.49 | 1.000 | 27% |

**Saturation.** At 75%, P >= 0.99 on 27% of the calibrated grid
from 1950 RPM. Over most of the grid the risk is not saturated, so the optimum is informed by the posterior.

**Rebalance-vs-continue boundary as a function of resid_frac (75%, baseline costs):** rebalance is optimal for resid_frac up to 0.4 and continue from 0.45 on (scan 0.05-1.0, 20 values).

| resid_frac | action | optimum RPM | J* (cost/h) |
|---|---|---|---|
| 0.050 | rebalance | 2400 | 1.40 |
| 0.100 | rebalance | 2100 | 2.49 |
| 0.150 | rebalance | 1950 | 2.89 |
| 0.200 | rebalance | 1950 | 2.93 |
| 0.250 | rebalance | 1950 | 3.25 |
| 0.300 | rebalance | 1800 | 3.53 |
| 0.350 | rebalance | 1800 | 3.56 |
| 0.400 | rebalance | 1800 | 3.64 |
| 0.450 | continue | 1500 | 3.82 |
| 0.500 | continue | 1500 | 3.82 |
| 0.550 | continue | 1500 | 3.82 |
| 0.600 | continue | 1500 | 3.82 |
| 0.650 | continue | 1500 | 3.82 |
| 0.700 | continue | 1500 | 3.82 |
| 0.750 | continue | 1500 | 3.82 |
| 0.800 | continue | 1500 | 3.82 |
| 0.850 | continue | 1500 | 3.82 |
| 0.900 | continue | 1500 | 3.82 |
| 0.950 | continue | 1500 | 3.82 |
| 1.000 | continue | 1500 | 3.82 |

Same boundary on the fair 50% stage (rebalance is optimal for resid_frac up to 0.55 and continue from 0.6 on):

| resid_frac | action | optimum RPM | J* (cost/h) |
|---|---|---|---|
| 0.050 | rebalance | 1950 | 2.93 |
| 0.100 | rebalance | 1950 | 2.98 |
| 0.150 | rebalance | 1950 | 3.15 |
| 0.200 | rebalance | 1950 | 3.38 |
| 0.250 | rebalance | 1800 | 3.50 |
| 0.300 | rebalance | 1800 | 3.50 |
| 0.350 | rebalance | 1800 | 3.50 |
| 0.400 | rebalance | 1800 | 3.50 |
| 0.450 | rebalance | 1800 | 3.51 |
| 0.500 | rebalance | 1800 | 3.52 |
| 0.550 | rebalance | 1800 | 3.57 |
| 0.600 | continue | 1500 | 3.75 |
| 0.650 | continue | 1500 | 3.75 |
| 0.700 | continue | 1500 | 3.75 |
| 0.750 | continue | 1500 | 3.75 |
| 0.800 | continue | 1500 | 3.75 |
| 0.850 | continue | 1500 | 3.75 |
| 0.900 | continue | 1500 | 3.75 |
| 0.950 | continue | 1500 | 3.75 |
| 1.000 | continue | 1500 | 3.75 |

Factorial sweep over V_full {5, 10, 20} x h_fail {0.05, 0.1, 0.2} x resid_frac {0.1, 0.5, 1.0} x C_fail {100, 500, 2000}
(81 combinations, 75%): optimum RPM range **1350-2400 RPM**; rebalance optimal in 35 of 81.
The single-optimum headline is therefore not robust; the boundary above is the decision output.

One-at-a-time sensitivity (75%; other costs at baseline):

| parameter | value | optimum RPM | action | J* |  |
|---|---|---|---|---|---|
| V_full | 5.0 | 2100 | rebalance | 1.86 |  |
| V_full | 10.0 | 2100 | rebalance | 2.49 | baseline |
| V_full | 20.0 | 2100 | rebalance | 3.74 |  |
| h_fail | 0.05 | 2100 | rebalance | 2.37 |  |
| h_fail | 0.1 | 2100 | rebalance | 2.49 | baseline |
| h_fail | 0.2 | 2100 | rebalance | 2.73 |  |
| resid_frac | 0.1 | 2100 | rebalance | 2.49 | baseline |
| resid_frac | 0.5 | 1500 | continue | 3.82 |  |
| resid_frac | 1.0 | 1500 | continue | 3.82 |  |
| C_fail | 100.0 | 2100 | rebalance | 2.30 |  |
| C_fail | 500.0 | 2100 | rebalance | 2.49 | baseline |
| C_fail | 2000.0 | 1950 | rebalance | 2.91 |  |

Decision checks:
- optimum_interior (75% baseline optimum strictly inside 300-2400, C7): **True** (baseline 2100 RPM).
- Stressed scenario (C_fail = 2000, h_fail = 0.2; fixed in round 2, not tuned): optimum 1950 RPM, action rebalance;
  action non-trivial (C8): **True**.

## 9. Changes vs round 2 and round 1
Round-1 and round-2 values are read from `metrics_r1.json` and `metrics_r2.json`; round-3 values are from this run.

| quantity | round 1 / round 2 | round 3 |
|---|---|---|
| Physics model | 1-DOF Jeffcott, Ueff = U*(A+B) | 2-DOF lumped rotor, per-disc unbalance + phase, per-Position weights |
| Pooled held-out RMSE, physics (g) | 0.1255 (r1) | 0.1090 (2-DOF) |
| A1B1 held-out RMSE, physics (g) | not separated in r2 | 0.1202 (2-DOF) vs 0.3382 (1-DOF) |
| Hybrid NRMSE | 0.345 (r2) | 0.349 (r3, mean of seeds) |
| Data-driven NRMSE | 0.418 (r2) | 0.418 (r3) |
| Min steps / max tau (all calibration runs) | 23.0 (r2) | 8.2 (r3) |
| Headline risk (2400 RPM, no action) | 1.0000 (r2) | 1.0000 (r3, 75%) |
| Decision optimum, 75% (RPM) | 2100 (r2) | 2100 (r3; sweep 1350-2400) |
| Coverage 95%, r1 point set (25% stage) | 0.625 (r1) | 1.000 (r3, position-averaged) |
| Runtime (s) | 194.6 (r1), 431.0 (r2) | see run.log |

Changes made (round-2 review items mapped):
1. Physics replaced by a 2-DOF lumped rotor with per-disc unbalance and relative phase (review problem 1, Q1-Q2); single-DOF kept as ablation and fitted without A1B1.
2. Per-Position calibration and CV; Position-level breakdowns and per-Position floor scales (review problem 2).
3. Coverage as a per-point table; round-1 definition kept under its own key; stages not pooled; like-for-like comparison (review problem 3).
4. Closed-form likelihood (no GP emulator variance; review problem 5); the emulator is retired; a discrepancy-off sensitivity added.
5. MCMC: MAP init, DEMove/DESnookerMove mixture, 22000 steps; convergence reported honestly (review problem 4).
6. Decision: calibrated range only; rebalance-vs-continue boundary over resid_frac; P >= 0.99 saturation stated (review problem 6).
7. Summary: section 0 states failures; section 5 sentence repaired; the typed 132% replaced by a computed ratio; every number generated (review problem 7).
8. Pre-registered criteria C1-C8 added before the run; results evaluated against them only (group item).

## 10. Known limitations
- Target is the Z-axis RMS (round-2 criterion, rechecked on the training rows of each LOGO fold: 18/18 folds). The axis criterion still uses the in-phase drive U*(TypeA+TypeB) as a regressor; it is a selection heuristic only.
- No time axis: no wear trajectory, no RUL. The decision is a speed/rebalance choice at a fixed unbalance state.
- Misalignment is not modelled physically. Misalignment groups are predicted by the unbalance response plus floor.
- The 2-DOF masses are normalised (equal disc masses, mass ratio fixed to 1). Bearing and coupling stiffnesses and damping are per unit mass; e0 absorbs eccentricity and unit scale.
- Position is not documented. It is treated as a sensor index with free output weights (rho_p, G_p) and floor scales (zeta_p); the model does not infer an axial location.
- Fit quality: the 2-DOF training log-RMSE is 0.336 against a within-cell repeat noise of 0.008. The structural model error is therefore about 40 times the measurement noise; the 2-DOF is a partial description (see summary section 3).
- Position-3 healthy floor (Normal reference) is higher than the unbalance runs at 300 RPM (floor 0.0275 g against 0.0110-0.0265 g); the per-Position floor scale zeta_3 = 0.550 is fitted, which is a calibration of the background, not a measurement.
- A1B1 exists at Mis=0 only for U=13.6. In the LOGO fold for M0.0_U13.6 the disc phase phi is not identified from the training cells, so that fold keeps the full-data phi (flagged; leakage of one parameter).
- Calibration uses one held-out experiment per stage (M0.0_U20.4_A1B0, plus a 50% replication on M0.0_U13.6_A1B0). kappa_B and phi are not informed by single-disc data and are fixed at the leave-group-out centre.
- The discrepancy GP has a fixed correlation length of 500 RPM (assumption); the rebalanced scenario applies the same discrepancy (approximation).
- Severity limit = P99 of healthy Z RMS (0.1028 g; percentile assumption). Decision costs are assumptions; sweeps are reported, not optimised.
- P(any Position) assumes positions are conditionally independent given theta and their discrepancy terms.
- Sobol indices are prior-box dependent (log-uniform boxes of the physical parameters).
- Round 3 uses 100 RF trees and 5 CV seeds; the physics is deterministic.

## 11. Files and runtime
- Code: `p4_rotor/main.py`, `p4_rotor/src/data_loader.py`, `physics_model.py`, `surrogate_model.py`, `uq_calibration.py`, `decision.py`.
- Results: `p4_rotor/results/metrics.json`, `p4_rotor/results/summary.md`, `p4_rotor/results/run.log`, figures `fig1..fig7` in `results/figures/`.
- Round-1 and round-2 copies: `metrics_r1.json`, `metrics_r2.json`, `run_r2.log`, `summary_r2.md`.
- Runtime of this run: 428.0 s (budget 600 s; measured at the end of the run).
- Raw data was read only: `_data/p4_rotor/features.csv`.
