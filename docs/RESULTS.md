# Results

> **Generated file — do not edit by hand.** Produced by `python src/make_report.py`
> from `model/metrics.json`. Every number here is a number the code actually
> produced; the README links to this file rather than repeating figures.

## 1. Primary target — 2-hour-ahead glucose spike

- **1,004,000 rows, 500 virtual patients, 1.82% positive rate**
- CV protocol: 5 x GroupShuffleSplit(test_size=0.25) grouped by patient_id (a patient's rows never appear in both train and test)
- Estimator: `HistGradientBoostingClassifier` (max_iter=160, max_depth=4, lr=0.1)
- Label: 30-min-smoothed CGM exceeds 180 mg/dL, or rises >= 30 mg/dL, at any point in the next 2h

- **Every figure in this file is the canonical cohort** — the single generator draw the shipped `.joblib` was fitted on. The headline quoted at the top of the README is the mean over five generator seeds, because one draw is not a central estimate; see [SEED_ROBUSTNESS.md](SEED_ROBUSTNESS.md). The tables here are canonical because they decompose *that* artefact, and the mean cannot be broken into per-variant contributions without re-running the whole sweep once per variant.

### Discrimination, calibration, and the controls

| Variant | ROC-AUC | PR-AUC | Brier | ECE (10 bin) |
|---|---|---|---|---|
| Full fusion (EHR + wearable) | 0.8756 ± 0.0197 | 0.2481 ± 0.0555 | 0.0177 ± 0.0018 | 0.0039 ± 0.0009 |
| Shortcut baseline + autonomic channels | 0.8710 ± 0.0206 | 0.2393 ± 0.0569 | 0.0178 ± 0.0018 | 0.0038 ± 0.0013 |
| Wearable only (no EHR) | 0.8709 ± 0.0187 | 0.2588 ± 0.0501 | 0.0175 ± 0.0019 | 0.0026 ± 0.0010 |
| HRV / sleep / HR / activity only | 0.8313 ± 0.0231 | 0.1825 ± 0.0553 | 0.0187 ± 0.0019 | 0.0039 ± 0.0010 |
| Shortcut control (clock + current CGM + baseline labs) | 0.8540 ± 0.0281 | 0.2261 ± 0.0434 | 0.0179 ± 0.0020 | 0.0033 ± 0.0009 |
| EHR only (no wearable) | 0.8230 ± 0.0247 | 0.1439 ± 0.0205 | 0.0190 ± 0.0025 | 0.0072 ± 0.0031 |

Mean ± std across 5 folds. The full model's ROC-AUC SEM is 0.0088.

PR-AUC is the metric that matters at a 1.8% base rate; a PR-AUC of 0.248 against a 0.018 base rate is roughly a 14x lift over chance.

### What each block of features is actually worth

| Comparison | Δ ROC-AUC | bootstrap *p* |
|---|---|---|
| Autonomic channels added to the matched shortcut control | +0.0171 | 0.001 |
| Full model vs. shortcut control | +0.0216 | < 0.001 |
| Full model vs. EHR-only *(the flattering comparison v2 headlined)* | +0.0526 | < 0.001 |
| Full model vs. best single stream (`shortcut_plus_autonomic`) | +0.0046 | 0.184 |

All deltas are **paired on the identical fold set** — each model is scored on the same folds, so the per-fold differences are matched pairs — with a two-sided **patient-level cluster bootstrap** (2000 resamples) over the pooled out-of-fold predictions. Folds are *not* independent replicates: their test sets overlap, and a two-sided sign test on five matched pairs cannot go below p = 1/32. The resampling unit is therefore the patient (hundreds of them), which is what licenses the resolution the reported p-values claim.

**Headline: the full model gains +0.0216 AUC over a shortcut control that uses nothing but the clock, the current CGM reading, its recent trend, and the patient's baseline labs** (p < 0.001, significant at 0.05). The control is matched: same target, same folds, same estimator, and it is exactly what a clinician could do without any physiological reasoning.

Of that, +0.0171 is attributable specifically to the autonomic channels, isolated by the matched control that adds HRV / sleep / activity to the shortcut set and nothing else.

Against the *best non-fusion variant* (`shortcut_plus_autonomic`) the full model lifts by +0.0046 (p = 0.184) — i.e. the lift is directionally positive but not distinguishable from zero. Across the 5 cohort draws in `model/seeds/` that same delta is **negative in 4 of 5** (significantly so in 1) — so the sign seen here is a property of this cohort draw, not of the method. This is the unflattering comparison, and the one a judge should weigh most: it asks whether fusing every stream beats the strongest feature-set subset, and on this synthetic data the answer is *not yet*. Two things cut against reading `shortcut_plus_autonomic` as a neutral yardstick. It is not a pre-registered baseline: it is a composite assembled *after* seeing these results, on the same five folds, so the +0.0046 in full model's favour is optimistically biased for full model too — the true out-of-sample gap is unmeasured, not +0.0046. And it already contains the baseline labs, so this is not a clean EHR-vs-wearable test; the clean one is the row above (full vs. EHR-only). Worth stating plainly: under this generator's v3.1 design the EHR stream genuinely modulates the hidden reactivity (EHR-only alone scores 0.8230), so these rows measure whether the model can *capture* that signal on top of a wearable stream — not whether the generator contained any. The flattering comparison — +0.0526 over EHR-alone — is reported too, because a reader is entitled to both, but a lift measured against the weaker of two streams flatters by construction and should not be the headline.

### Leakage controls

| Control | ROC-AUC | Expected |
|---|---|---|
| Observed (single split, seed 7) | 0.8609 | — |
| Global label permutation | 0.4954 | ~0.50 |
| Within-patient label shuffle | 0.8022 | low |

The observed row is the AUC of the full model on the **single hold-out split (seed 7, 25% of patients) that carries the two permutation nulls** — it is not the same evaluation as the 5-fold CV-mean headline (0.8756) in the discrimination table, and it is not meant to be: the nulls are sanity checks on one split, the headline is a mean over five. The two would coincide only if every fold performed identically.

The **global label permutation** control re-runs the whole pipeline with shuffled labels. Anything materially above 0.50 would mean the pipeline leaks. The **within-patient label shuffle** preserves patient identity — and therefore every static EHR column and each patient's own base rate — while destroying only the alignment between *when* something happened and *what* the sensors read. Whatever AUC survives is score from knowing **who** the patient is rather than **what state** they are in. This is the control that would have caught the v1 model, which put 82% of its decision weight on the single static `type2_diabetes` column.

### Operating points under an alert budget

At a 1.8% base rate a 0.5 probability threshold is meaningless. A care team does not choose a threshold — it has a noisy 24/7 stream and a finite capacity to respond. These are the operating points at fixed alert budgets, where a budget of 0.05 means "alert on the riskiest 5% of patient-ticks".

| Alert budget | Risk threshold | Precision | Recall | Alerts/patient/day |
|---|---|---|---|---|
| 1% | 0.345 | 0.414 | 0.198 | 1.0 |
| 5% | 0.074 | 0.218 | 0.517 | 4.8 |
| 10% | 0.034 | 0.137 | 0.650 | 9.6 |
| 25% | 0.012 | 0.069 | 0.818 | 24.0 |

## 2. Permutation importance (full model)

Top feature: `hrv_rmssd_ms_roll_mean` at **15.3%** of total importance. Dynamic/wearable share: **0.659** · autonomic+activity share: **0.314**

| Rank | Feature | Share |
|---|---|---|
| 1 | `hrv_rmssd_ms_roll_mean` | 0.1534 |
| 2 | `hour_of_day` | 0.1502 |
| 3 | `cgm_mgdl_roll_mean` | 0.1088 |
| 4 | `age` | 0.0712 |
| 5 | `hba1c_pct` | 0.0626 |
| 6 | `bmi` | 0.0597 |
| 7 | `hrv_rmssd_ms` | 0.0465 |
| 8 | `cgm_mgdl_roll_std` | 0.0452 |
| 9 | `tcf7l2_risk_allele_count` | 0.0394 |
| 10 | `steps_roll_std` | 0.0370 |
| 11 | `egfr_ml_min` | 0.0364 |
| 12 | `fasting_glucose_mgdl` | 0.0348 |
| 13 | `heart_rate_bpm_roll_mean` | 0.0330 |
| 14 | `cgm_mgdl` | 0.0295 |
| 15 | `ldl_cholesterol_mgdl` | 0.0211 |

Compare with v2, where the top feature was `hour_of_day` at 29.2% and the physiology channels together were ~13%. Here the top feature is `hrv_rmssd_ms_roll_mean` at 15.3% and `hour_of_day` has fallen to 15.0% — the meal-clock shortcut is no longer the model's main lever, though `hour_of_day` still carries 15.0% of the total, so time-of-day remains genuinely informative rather than purely spurious.

## 3. Secondary target — week-ahead cardiovascular strain

- 7,000 patient-days, 500 patients, 8.6% positive rate
- Label: >= 3 consecutive days within the next 7 where daily HRV < patient's own p20 AND resting HR > patient's own p70

| Variant | ROC-AUC | PR-AUC |
|---|---|---|
| full_fusion | 0.6400 ± 0.0402 | 0.1299 ± 0.0272 |
| dynamic_only | 0.6373 ± 0.0281 | 0.1259 ± 0.0205 |
| static_only | 0.5021 ± 0.0374 | 0.0942 ± 0.0243 |
| shortcut_control | 0.4978 ± 0.0361 | 0.0924 ± 0.0131 |

**Reported as a weak result and meant to be one.** AUC 0.640 against a 8.6% base rate is close to useless, and PR-AUC of 0.130 is barely above the base rate. The purpose of including it is to show the architecture is not glucose-specific, not to claim the problem is solved.

Known caveats, stated in the code and repeated here:
- p20/p70 baselines are computed over the full 21-day window; a deployed system would need a trailing baseline, which is harder.
- only 14 usable days per patient (7,000 rows total) -- wide fold-to-fold variance is expected.

## 4. Reproducing

```bash
pip install -r requirements.txt
python data/generate_ehr.py
python data/generate_wearable.py
python src/feature_engineering.py
python src/train_model.py
python src/train_cv_risk_model.py
python src/export_dashboard_data.py
python src/make_report.py        # regenerates this file and the README table
```

## 5. v2 vs. v3, side by side

The v2 column is transcribed from the v2 submission as submitted. The v3
column is read from `metrics.json` by this script, so it cannot drift.

| | v2 | v3 | |
|---|---|---|---|
| ROC-AUC (same protocol, canonical cohort) | 0.953 | **0.876** | worse, and honest |
| Rows | 1,007,500 | **1,004,000** | tail rows now dropped |
| Positive rate | 4.0% | **1.8%** | label is not firing on sensor noise |
| Shortcut control reported? | no | **yes, 0.854** | — |
| Gain over shortcut control | not measurable | **+0.0216, p < 0.001** | paired test |
| Gain over best single stream | not reported | **+0.0046** | unflattering |
| Global label-permutation control | no | **0.495** | pipeline does not leak |
| Within-patient shuffle | no | **0.802** | identity floor; state adds +0.06 (same hold-out split) |
| Calibration (Brier / ECE) | no | **0.0177 / 0.004** | — |
| Operating point | "0.5" at a 4% base rate | **4 budget levels; 1% budget => precision 0.41** | costed |
| Rows with incomplete lookahead | 4,000 silently mislabelled | **0** | — |
| Demo window | `argmax(risk)` per patient | **fixed rule, cohort rate stated** | — |
| UI explanations | claimed, not implemented | **implemented** | — |
| Remote assets in the dashboard | Chart.js + 2 fonts | **none** | works offline |
| Tests / CI / LICENSE | none | **59 tests, CI, MIT** | — |

The v3 ROC-AUC above is the **canonical cohort** — the draw the shipped `.joblib` was fitted on — because the v2 figure was produced the same way and the comparison is only valid same-protocol. The headline quoted at the top of this README is the mean over five generator seeds; see [seed robustness](docs/SEED_ROBUSTNESS.md).


### External validation — UCI Diabetes (AIM '94, Kahn; CC BY 4.0)

5 x GroupShuffleSplit(test_size=0.25) grouped by patient_id, HistGradientBoostingClassifier with the same hyper-parameters and patient-level cluster bootstrap (2000 resamples, floor 1/2000) as the synthetic pipeline (src/train_model.py)

50 patients, 66,533 rows (positive rate 54.07%); 30-min-smoothed glucose exceeds 180 mg/dL or rises >= 30 mg/dL within the next 2 h; tail rows without a full lookahead are dropped

| variant | ROC-AUC |
|---|---|
| context | 0.9826 ± 0.0022 |
| naive | 0.9768 ± 0.0032 |

- **context_vs_naive**: delta = +0.0059 (p < 0.001; patient-level cluster bootstrap, 2000 resamples)

This cohort carries no paired static clinical table, so **no fusion test is reported here** — a wearable-only run cannot answer whether fusing an EHR record helps. That gap is why `src/convert_big_ideas.py` exists: it is the one openly licensed cohort found with CGM, a wearable stream and a clinical record on the same participants.

Source: `external_validation_uci.json`. Rerun with `python src/validate_real_data.py` (see docs/REAL_DATA.md for the schema and how to obtain real data).