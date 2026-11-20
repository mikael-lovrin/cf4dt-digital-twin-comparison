# Round 1 - group evaluation (orchestrator)
Individual scores /40: P4 rotor 26, P3 motor 23, P1 CNC(synthetic) 22, P2 NASA 20. Data suitability (reviewers, /5): P3 4, P2 3, P4 3, P1 2.

## Headline table (held-out NRMSE = RMSE/std, lower is better; calibration coverage95 should be ~0.95)
| proj | physics | data | hybrid | surrogate R2 / cov2sigma | coverage95 | N/tau | decision |
| P1 | 0.039 (circular) | 1.12 | 0.036 | .996 / .946 | 1.00 (too wide) | ~11 | trivial/circular |
| P2 | 0.523 | 0.545 | 0.486 | .955 / .953 | 0.71 (too narrow) | 27 | t* at grid edge |
| P3 | 0.594 | 0.357 | 0.682 (worst) | 1.0 / 0.52 | 0.98 (too wide) | ~19 | never derates |
| P4 | 0.779 | 0.419 | 0.345 (best) | .997 / .968 | 0.62 (too narrow) | ~22 | extrapolates 3300 RPM |

## Cross-project problems (fix in every project)
1. MCMC unconverged everywhere (N/tau < 50). Use tau-based run length, report per-parameter tau and ESS.
2. Coverage miscalibrated in both directions. Report coverage AND mean interval width at 50/80/95 %, use per-point GP variance + a model-discrepancy/obs-noise term fitted on training data, not a global constant.
3. Decision layers are weak: trivial (P3), edge-of-grid optimum (P2), extrapolation (P4), circular truth (P1). Require: interior optimum check, cost sensitivity sweep, a stressed scenario that makes the action non-trivial, optimum reported as a range.
4. Leakage/circularity: P1 physics uses target-sorted index; P2 grouped-CV leaks replicate settings (8 settings x 2 inserts).
5. Hybrid not guaranteed to help (P3 worst). Add strong simple baselines (log-linear / RF on same inputs) and per-group paired differences.
6. Speed-up claims not comparable across projects (RK4 vs solve_ivp, closed form vs GP). Report speed-up against the *same* reference solver or drop the claim.
7. summary.md text must be generated from metrics.json (several mismatches found).
