# p1_cnc: CNC flank-wear digital twin (pseudo-time reconstruction)

## 1. Data inspection (what the CSV actually contains)
- Shape 2000 x 17, missing values: 0, duplicate rows: 0.
- Columns: CF_Feature_1..5, Vib_Feature_1..5, AE_Feature_1..5 (floats), VB_mm (mm), Wear_Class.
- No tool ID, run ID, unit ID or timestamp column exists. Rows are unordered snapshots.
- `Wear_Class` counts: {'Moderate': 854, 'Healthy': 580, 'Worn': 566}. Its VB ranges do not overlap: {'Healthy': [-0.015304371426405589, 0.09999469345015355], 'Moderate': [0.10021791445083011, 0.24989371514020467], 'Worn': [0.25059898208521536, 0.3758505903863404]}.
  It is therefore a deterministic binning of VB_mm, not an independent label.
- VB_mm: min -0.0153, max 0.3759, mean 0.1749; **20 negative values**
  (physically impossible for flank wear).
- All raw features are uniform on their range (KS p-values in table below), a strong sign of **synthetic data**.
  Earlier orchestrator check (first row equals numpy seed-42 rand*100) is consistent with this.

## 2. Pseudo-time reconstruction (assumption, documented in `src/data_loader.py`)
- Rows are sorted on VB_mm ascending, so **VB is monotone in pseudo-time by construction**.
- Pseudo-time unit: **PTU** (1 PTU = one sorted sample index, 0..1999). **All RUL values below are in pseudo-time units (PTU), not hours or cycles.**
- Blocked split: 20 contiguous segments of 100 PTU; segments 4, 9, 14, 19 (20 %) are held out.
- F_res = sqrt(CF_Feature_1^2 + CF_Feature_2^2 + CF_Feature_3^2). These are **feature columns, not measured resultant force Fx,Fy,Fz**.
- Vibration/AE: one snapshot value per row, so the spec's RMS/std of ax, ay, az cannot be computed; raw values are used.

## 3. Do the features explain VB_mm?
- Max |Pearson r| with VB_mm over all 16 inputs: **0.059**.
- Random forest (300 trees) CV R2: **-0.022** (shuffled 5-fold), **-0.120** (blocked 5-fold by segment).
- Interpretation: the features carry no measurable information about wear in this file. Any "hybrid gain" would be noise.

| feature | Pearson r | Spearman rho | KS p (uniform) |
|---|---|---|---|
| CF_Feature_1 | -0.022 | -0.022 | 0.316 |
| CF_Feature_2 | -0.016 | -0.016 | 0.678 |
| CF_Feature_3 | +0.004 | +0.004 | 0.401 |
| CF_Feature_4 | -0.006 | -0.006 | 0.473 |
| CF_Feature_5 | -0.021 | -0.021 | 0.366 |
| Vib_Feature_1 | +0.043 | +0.043 | 0.244 |
| Vib_Feature_2 | +0.018 | +0.017 | 0.365 |
| Vib_Feature_3 | -0.027 | -0.027 | 0.084 |
| Vib_Feature_4 | +0.003 | +0.003 | 0.167 |
| Vib_Feature_5 | +0.020 | +0.020 | 0.765 |
| AE_Feature_1 | -0.047 | -0.047 | 0.315 |
| AE_Feature_2 | -0.026 | -0.025 | 0.258 |
| AE_Feature_3 | +0.007 | +0.006 | 0.611 |
| AE_Feature_4 | -0.059 | -0.060 | 0.035 |
| AE_Feature_5 | -0.035 | -0.035 | 0.801 |

## 4. Models and held-out comparison (held-out = 400 rows, 4 blocks)
- **Physics**: coupled wear ODE + lumped thermal ODE, RK45. Least-squares fit on training rows only.
  Fitted: k1=0.01277, k2=6.952e-07, alpha=0.5055, hA=0.02612, VB0=9.856e-11; vc=1 fixed.
  **Caveat:** time is the target-derived sorted index, so this is a trend fit, not an independent physical prediction.
- **Data-driven**: random forest on 16 features (standardised with train statistics), no time input.
- **Hybrid**: physics + random forest on the training residual VB_obs - VB_phys.

| model | held-out RMSE (mm) | NRMSE = RMSE/std(target) |
|---|---|---|
| Pure physics | 0.0039 | 0.039 |
| Pure data-driven | 0.1114 | 1.120 |
| Hybrid DT | 0.0036 | 0.036 |

std(held-out VB_mm) = 0.0994. Data-driven held-out R2 = -0.255.

## 5. GP surrogate of the ODE
- Input u = (log k1, log k2, alpha, log hA, vc), LHS box around the LS fit (+/-1 log for k1, hA; +/-2 log for k2; alpha in [0,2]; vc in [0.8,1.2]).
- 400 training ODE runs, 100 LHS test runs, output = VB on 61-point grid to 3000 PTU. Kernel: Constant x Matern(2.5, ARD) + White.
- Test R2 (flattened over time and runs) = **0.9965**; R2 at end-of-record VB = 0.9966; RMSE = 0.3239 mm;
  fraction within +/-2 sigma = 0.946; speed-up (batched GP vs per-run ODE) = **521x**.
- The flattened R2 is inflated by the shared time trend; the end-of-record R2 is the stricter number.

## 6. Sobol sensitivity (N=1024 Saltelli, 7168 GP evaluations, parameters in prior box)
RUL is censored at the 3000 PTU horizon in 15.4 % of samples.

| parameter | S1 (RUL) | ST (RUL) | S1 (VB_end) | ST (VB_end) |
|---|---|---|---|---|
| log_k1 | 0.134 | 0.219 | 0.163 | 0.578 |
| log_k2 | 0.348 | 0.493 | 0.008 | 0.017 |
| alpha | -0.006 | 0.051 | 0.035 | 0.318 |
| log_hA | 0.286 | 0.429 | 0.217 | 0.597 |
| vc | 0.026 | 0.097 | 0.020 | 0.169 |

## 7. Bayesian calibration (emcee, 32 walkers, 600 burn-in + 1000 steps)
- Likelihood: Gaussian on inspected VB (every 25th training row, sensor noise sigma_obs + GP variance) plus Gaussian on force-like observations
  F_res = F0 + c_w VB (every 10th training row, sigma_F = sd(F_res), F0 = mean(F_res), both from training rows).
- Held-out rows are never used in calibration. Vc fixed at 1. Prior is uniform in the box above (log space for k1, k2, hA, sigma_obs).
- Sequential stages: 25 %, 50 %, 75 % of the 2000-PTU record (500, 1000, 1500 PTU).

Posterior-to-prior standard-deviation ratio (1.0 = no information gained):

| parameter | posterior mean (75 %) | posterior sd (75 %) | ratio 75 % | ratio 25 % | ratio 50 % |
|---|---|---|---|---|---|
| k1 | -4.301 | 0.3122 | 0.541 | 0.838 | 0.597 |
| k2 | -14.42 | 0.7893 | 0.684 | 0.479 | 0.528 |
| alpha | 0.9345 | 0.2787 | 0.483 | 0.698 | 0.545 |
| hA | -3.485 | 0.2211 | 0.383 | 0.560 | 0.426 |
| sigma_obs | -6.175 | 1.146 | 0.574 | 0.726 | 0.606 |
| c_w | 0.2791 | 33.09 | 0.057 | 0.397 | 0.090 |

Mean integrated autocorrelation time over the 3 stages: 55.2 steps; mean acceptance fraction: 0.10.
Convergence rule of thumb: steps/tau >~ 50. Per-stage tau and acceptance are in `metrics`-adjacent run log; see the stage table note in section 8 for RUL.

## 8. RUL, risk and replacement decision (RUL in PTU)
VB_crit = 0.30 mm. True RUL = first sorted index with VB >= 0.30 minus stage time (pseudo-time truth, circular by construction).
Costs c_p = 1.0, c_f = 5.0 (assumed). Derating d = max(0.7, 1 - 0.5 * P(fail within T*)) (assumed law).

| stage | t (PTU) | VB / force obs | RUL pred median [95 %] | RUL true | truth in 95 %? | T* (PTU) | cost rate | P(fail by T*) | derating |
|---|---|---|---|---|---|---|---|---|---|
| 25% | 500 | 17 / 41 | 1054 [343, 2500] | 1215 | yes | 727 | 0.0021 | 0.115 | 0.943 |
| 50% | 1000 | 33 / 81 | 652 [301, 1930] | 715 | yes | 390 | 0.0036 | 0.090 | 0.955 |
| 75% | 1500 | 49 / 121 | 189 [0, 563] | 215 | yes | 380 | 0.0169 | 0.652 | 0.700 |

- Headline decision (50 % life): replace after **390 PTU**, expected cost rate 0.0036 c_p per PTU,
  failure risk within T* = 0.090.
- 95 % coverage of held-out VB_mm by the 75 %-life posterior predictive: **1.000** (nominal 0.95).

## 9. Limitations (honest list)
- Data appear SYNTHETIC: uniform features, no tool/time/unit ID, VB_mm has 20 negative values; Wear_Class is a deterministic binning of VB_mm (Healthy<0.10<=Moderate<0.25<=Worn).
- Pseudo-time is reconstructed by sorting on VB_mm, so VB is monotone in pseudo-time by construction; physics predictions use a target-derived time index and are NOT an independent predictive test.
- Features explain almost nothing: max|Pearson r|=0.059, RF CV R2 random=-0.022, blocked=-0.120. Hybrid gain over physics is nil.
- RUL is in pseudo-time units (PTU, 1 PTU = one sorted sample index), not hours or cycles.
- F_res = sqrt(CF_1^2+CF_2^2+CF_3^2) is a combination of feature columns, not measured Fx,Fy,Fz.
- Vibration/AE are single snapshot values per row; no ax,ay,az time series, so no windowed RMS/std.
- Nominal constants (sigma_n, Ea/R, C_th, T_inf, eta, T rise) are fixed, not fitted; only k1,k2,alpha,hA,vc vary.
- Costs c_p=1, c_f=5 and derating law (gamma=0.5, d_min=0.7) are assumptions, not data-derived.
- Calibration uses one pseudo-trajectory (no unit replicates); posterior is conditional on the GP surrogate and on the assumed noise/force model; c_w is essentially unidentified.
- Posterior RUL uses GP mean trajectories (GP variance enters the likelihood and the coverage band, not the RUL draws).
- Surrogate horizon 3000 PTU exceeds the 2000-row record; F_res is wrapped periodically beyond 2000 PTU.

## 10. Figures
`results/figures/fig01` to `fig09`: pseudo-trajectory, feature diagnostics, physics fit and model comparison,
GP parity, Sobol, posterior narrowing, RUL distributions, cost rate, held-out predictive band.

Runtime: 69 s.
