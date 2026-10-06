# Seed robustness

Generated from `model/seeds/` by `src/make_seed_report.py` — do not edit.

Protocol: the full 500-patient / 21-day pipeline run under each generator master seed in `DIGITAL_TWIN_SEED` (both generator stages draw from it; per-patient streams are still hashed from the identifier). The model, the 5-fold grouped-CV protocol and the estimator seeds are held fixed, so the spread below isolates generator randomness — this is the answer to “did one lucky data draw carry the headline?”.

| seed | headline AUC | PR-AUC | base rate | global perm | within-patient floor | observed (same split) | +Δ vs shortcut (p) | +Δ vs best non-fusion (p) | train (s) |
|---|---|---|---|---|---|---|---|---|---|
| canonical | 0.8756 ± 0.0197 | 0.248 | 1.82% | 0.495 | 0.802 | 0.861 | +0.0216 (< 0.001) | +0.0046 (0.184) | 612 |
| seed7 | 0.8483 ± 0.0414 | 0.220 | 2.03% | 0.503 | 0.799 | 0.860 | +0.0339 (< 0.001) | -0.0035 (0.576) | 574 |
| seed21 | 0.8463 ± 0.0146 | 0.188 | 1.77% | 0.497 | 0.708 | 0.793 | +0.0370 (< 0.001) | -0.0093 (0.151) | 578 |
| seed84 | 0.8278 ± 0.0167 | 0.183 | 1.64% | 0.498 | 0.663 | 0.793 | +0.0328 (< 0.001) | -0.0098 (0.119) | 649 |
| seed123 | 0.8154 ± 0.0391 | 0.162 | 1.67% | 0.502 | 0.654 | 0.768 | +0.0210 (0.007) | -0.0115 (0.040) | 676 |

**Headline: 0.8427 ± 0.0205 ROC-AUC, mean ± sd over 5 generator seeds.** This is the number quoted in the README. It is deliberately not a single seed: the sweep exists precisely because one draw is not a central estimate, and picking the seed that reads best would reintroduce the selection the sweep was run to remove. The shipped model artefact was fitted on the canonical cohort, so its own score is a per-draw number and is reported as such (0.8756) wherever an artefact-level figure is needed.

**Spread across seeds.** Headline AUC ranges 0.8154–0.8756 (Δ 0.0602); mean ± sd = 0.8427 ± 0.0205. **The shipped `.joblib` and dashboard were fitted on the canonical cohort, whose AUC is the maximum of this range.** It is listed because it describes the artefact that ships, not because it is the central estimate. Treat any single-seed AUC as one draw, not the law.

**Significance stability.** Fusion vs shortcut p < 0.05 in 5/5 seeds. The paired gain is a load-bearing, reproducible signal.

**Fusion over the best non-fusion subset is not supported.** The delta ranges -0.0115 to +0.0046; fusion is **below** the best non-fusion subset in 4/5 cohorts, and significantly below in 1/5. The shipped canonical run is in the minority where fusion is ahead, so its positive delta is a property of that cohort draw, not of the method.

**Base rate** ranges 1.64%–2.03%; the label and its controls stay in their designed operating regime across seeds.

The `metrics_*.json` snapshots are committed alongside this report, so every number above is auditable from `model/seeds/`.
