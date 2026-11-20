# Round 3 (FINAL improvement round) addendum - read BRIEF_COMMON.md, BRIEF_R2.md, reviews/r2_<proj>.md, reviews/r2_GROUP.md
Implement all Top-5 items from your r2 review plus the group items. Hard rules:
- Every pass/fail flag in metrics.json must be computed by an independent criterion defined BEFORE looking at results (write the criterion in code comments). No tuning a scenario/limit/threshold to make an action non-trivial. If something fails, report it as failed.
- Keep ALL round-2 metrics.json keys, update "round": 3, and add "changes_r3": [str,...] and "r2_vs_r3": {"heldout_nrmse": {...r2...}, "coverage95": [r2, r3], "min_steps_over_tau": [r2, r3]}.
- Runtime budget: main.py <= 600 s total, hard. Run it ONCE in the foreground (Bash timeout 600000). If you must re-run after a crash, say so. Never start background jobs or kill processes you didn't start.
- summary.md fully generated from metrics.json; section 0 = 5-line honest verdict INCLUDING all failures.
Reply <=12 lines: new headline numbers from metrics.json, what you could not fix.
