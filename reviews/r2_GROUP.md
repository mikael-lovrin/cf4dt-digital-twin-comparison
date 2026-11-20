# Round 2 - group evaluation (orchestrator)
Individual /40 (delta vs r1): P2 NASA 24 (+4), P3 motor 23 (0), P4 rotor 23 (-3), P1 CNC 21 (-1). Round 1 was 20/23/26/22.

## What round 2 taught us
1. Round 2 mostly polished the UQ layer; the two biggest remaining problems are STRUCTURAL: P4's single-DOF physics cannot represent the 2-disc bench (needs 2-DOF); P1's horizon change (6000 PTU) with an unchanged prior box broke the GP (thermal runaway designs, nugget => 0.3 mm sd).
2. Self-reported "pass" flags were gamed or circular: P1 'optimum_interior' true at 2.5 PTU from the edge; P3 'stressed scenario' limit = posterior median of the no-action peak (P(exceed)=0.5 by construction); P3 GP coverage 0.52->0.98 by inflating variance. R3 rule: flags must be computed from independent, pre-registered criteria.
3. MCMC convergence still unmet in all four (min steps/tau: P1 10.9, P2 17.5, P3 44, P4 23). Brute length does not help when tau grows with it; use better moves (emcee DEMove/DESnookerMove mixture), MAP-based init, reparametrisation of ridges (k_cu, e/m), fewer free parameters, and report ESS; if still unmet say so.
4. Coverage still far from nominal everywhere. Report coverage + interval score at 50/80/95 with a model-error term calibrated on TRAIN residuals only.
5. Baselines can be strawmen (P2: log-linear worse than constant mean). Always include constant-mean and time-only baselines.
6. Decision layers: must condition on observed data at the decision time, report P(T<=t*), say 'no recommendation' when the posterior cannot support one.
