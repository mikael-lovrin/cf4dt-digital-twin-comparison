# CF4DT - Digital Twin: comparison of 4 datasets

Hybrid physics + data-driven digital twin pipelines (physics ODE, GP surrogate, Sobol, emcee calibration, UQ, cost-based decision)
built and reviewed over 3 rounds for the RWTH course *Computational Foundations of Digital Twin Technologies* (WS 26/27).

**Start here: `RELATORIO_COMPARACAO.pdf`** (comparison report, strengths and limits per project, recommendation).

| Folder | Dataset | Round-3 review score |
|---|---|---|
| `p1_cnc` | Multi-Sensor CNC Tool Wear (Kaggle, ziya07; synthetic) | 23/40 |
| `p2_nasa` | NASA Milling (16 real cases) | 23/40 |
| `p3_motor` | Electric Motor Temperature (Kaggle, 69 sessions) | 25/40 |
| `p4_rotor` | Rotating-machinery vibration bench (own data, COBEM 2025) | 21/40 |

`reviews/` holds every independent review (r1, r2, r3) and the group evaluations; `BRIEF_*.md` are the instructions given to the builder agents.
Each project: `main.py`, `src/`, `results/` (metrics.json, summary.md, figures).

Raw data are NOT included (download from Kaggle / NASA; the rotor data belongs to a paper with co-authors).
Reproduce: Python venv with numpy scipy pandas scikit-learn matplotlib SALib emcee; put the data under `_data/<project>/`; run `python -I main.py` inside a project folder.

Scores come from cheap-model reviewers and are judgement, not ground truth. Builders and reviewers were AI agents; verify before citing.
