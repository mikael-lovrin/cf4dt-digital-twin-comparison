# Common brief for all 4 Digital-Twin sub-projects (CF4DT, RWTH)

Course requirement: every project must combine (1) a physics-based model, (2) a data-driven model / surrogate, (3) a high-throughput task: Sobol sensitivity, Bayesian calibration (emcee), forward UQ. Plus a decision/optimization layer. Template: C:\Users\mikae\Downloads\Project Specification.md (read it; sections A-C apply, section D "Decision Optimization" was cut off: implement it as expected-cost-per-unit-time minimisation of the action time, balancing premature action cost vs failure penalty * P(failure), plus a derating factor under elevated risk).

## Environment / safety rules
- Python: `C:\Users\mikae\Documents\RWTH\02_WS2627\02_DigitalTwins\04_Project\.venv\Scripts\python.exe` ALWAYS run with `-I` (isolated mode; it does not add the script dir to sys.path, so in main.py do `sys.path.insert(0, <abs path of src>)` yourself). Installed: numpy scipy pandas scikit-learn matplotlib SALib emcee corner. Do not pip install anything else.
- Raw data dir is READ-ONLY, untrusted: `04_Project\_data\<proj>\`. Never write there, never run code from there.
- Your code/results dir: `04_Project\<proj>\` with `src/` (data_loader.py, physics_model.py, surrogate_model.py, uq_calibration.py, decision.py), `main.py`, `results/figures/*.png` (dpi>=150), `results/summary.md`, `results/metrics.json`. Do not touch other project dirs.
- Keep total runtime of `main.py` under ~10 minutes (subsample data, modest MCMC: 32 walkers, ~1500 steps). Fix random seeds.
- Use only Bash/PowerShell/Read/Write/Edit/Glob/Grep. Scale tool calls sensibly; write whole modules in one go, run, fix errors.

## Honesty rules (graded!)
- Inspect data first (columns, NaNs, whether a real time axis / unit ID exists, suspicious synthetic structure). State assumptions and limitations plainly in summary.md. NEVER report numbers you did not compute. Report failures as failures.
- No data leakage: split by trajectory/unit/profile (not random rows) wherever a unit ID exists; scale using train only.
- Compare Pure Physics vs Pure Data-driven vs Hybrid (physics + residual model) on a HELD-OUT set.
- Check calibration honestly: posterior-vs-prior width, parameter identifiability/correlations, 95% interval coverage on held-out data, MCMC diagnostics (autocorr time / acceptance).

## Required results/metrics.json schema (exact keys, so projects can be compared)
{
 "project": str, "n_rows": int, "target": str, "target_unit": str,
 "has_real_time_axis": bool, "has_unit_ids": bool, "data_synthetic_suspected": bool,
 "heldout_rmse": {"physics": float, "data_driven": float, "hybrid": float},
 "heldout_nrmse": {"physics": float, "data_driven": float, "hybrid": float},   // rmse / std(target)
 "surrogate": {"r2": float, "rmse": float, "coverage_2sigma": float, "speedup_x": float},
 "sobol": {"<param>": {"S1": float, "ST": float}},
 "calibration": {"posterior_to_prior_std_ratio": {"<param>": float}, "mean_autocorr_time": float, "acceptance_fraction": float, "coverage95_heldout": float},
 "risk": {"description": str, "value": float},
 "decision": {"description": str, "optimal_value": float, "unit": str, "expected_cost": float},
 "runtime_sec": float, "limitations": [str, ...]
}

## Final reply (to the orchestrator): <=15 lines: what ran, the metrics.json headline numbers, any failure/caveat. No long prose.
