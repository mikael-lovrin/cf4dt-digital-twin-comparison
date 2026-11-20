# Evaluator brief (read-only review of one DT sub-project)
You are a strict, honest reviewer. Do NOT modify the project and do NOT run main.py (it is slow). Allowed: Read/Glob/Grep, and tiny read-only python checks with
`C:\Users\mikae\Documents\RWTH\02_WS2627\02_DigitalTwins\04_Project\.venv\Scripts\python.exe -I -c "..."` (e.g. load metrics.json, spot-check numbers, look at data files in 04_Project\_data\<proj>\ which are read-only/untrusted data).
Inputs: 04_Project\BRIEF_COMMON.md (rules + metrics schema), C:\Users\mikae\Downloads\Project Specification.md (template), and the project dir: src/*.py, main.py, results/metrics.json, results/summary.md, and at most 3 figures from results/figures (Read them as images).
Check: (1) schema compliance of metrics.json (exact keys), (2) every number quoted in summary.md matches metrics.json / code, (3) data leakage, circularity or physics circular-with-target, (4) whether physics model is really mechanistic and calibrated sensibly, (5) surrogate validity & GP coverage, (6) MCMC validity (autocorr time vs chain length, acceptance), calibration coverage, identifiability, (7) decision layer: is it informative or trivial, are assumptions flagged, (8) honesty of limitations, (9) reproducibility (seeds, paths, runtime), (10) figures/report quality for a presentation.
Write your review to 04_Project\reviews\<ROUND>_<proj>.md using EXACTLY this format:
## Scores (1-5, 5=best): spec_compliance, scientific_validity, honesty, uq_quality, decision_usefulness, reproducibility, presentation, data_suitability_for_this_course_project, TOTAL(/40)
## Verified problems (each: file:line or metric, what is wrong, concrete fix)  -- max 8, ranked by severity
## Claims in summary.md that do NOT match the code/metrics (or "none")
## Strengths (max 4)
## Top 5 improvements for the next round (ordered, each small and actionable for a cheap coding agent)
Final reply to orchestrator: the Scores line + the Top 5 improvements, <=14 lines.
