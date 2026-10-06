# Digital Twin PoC — v3 revision notes

A short changelog against the v2 submission that was judged at 73/100. Each
entry names the specific defect, the fix, and where to verify it.

---

## 1. The benchmark was algebraically closed

**Defect.** v2 defined glucose two hours ahead as `15 × risk_multiplier ×
stress_cluster[t]`, where `stress_cluster` is a rolling mean of HRV, sleep and
steps — all columns in the model's own feature set. The label at `t` covers a
window containing `t+2h`. The 2-hour forecast was therefore a closed loop with
no irreducible uncertainty, and the resulting 0.953 AUC measured the generator
rather than physiology. The tell was in v2's own output: top feature
`hour_of_day` at 29.2% importance, and all six demo patients peaking in the
same 90-minute window before the hardcoded 20:00 meal bump.

**Fix.** `data/generate_wearable.py` v3 adds a hidden per-patient AR(1)
metabolic-reactivity process that scales every spike and is never written to
any output file; exogenous spikes with no observable precursor; and lognormal
realisation of every spike. Meal times are jittered per patient, portion sizes
vary, and some meals are skipped.

**Verify.** `test_reactivity_sd_is_nonzero`, `test_meal_timing_is_jittered_per_patient`,
`test_metrics_are_in_a_plausible_range` (ratchet: fails if AUC climbs back
above 0.93).

**Effect.** The headline AUC is now far below 0.953 — lower *and more
trustworthy*, which is the entire point. The current figure is in
[`RESULTS.md`](RESULTS.md) and in the generated v2-vs-v3 table in the README;
it is deliberately not restated here, because a changelog that hard-codes its
own results is exactly how v2 ended up quoting 0.953 and 0.955 in the same
document.

---

## 2. The ablation was framed to flatter

**Defect.** v2 headlined "+0.2008 lift over EHR-alone" — a comparison against
the weaker of the two streams. Against the stronger one (wearable-only) the
lift was +0.0017, i.e. nothing. v2 also never ran a shortcut baseline, which is
why a meal-clock shortcut went unexamined.

**Fix.** Three new variants in `src/config.py`:
- `shortcut_control` — clock + current CGM + its trend + baseline labs.
- `autonomic_only` — HRV / sleep / HR / activity alone.
- `shortcut_plus_autonomic` — the **matched control**: shortcut plus exactly
  the autonomic channels, so the delta against `shortcut_control` is a clean
  isolated estimate of what the physiology channels are worth.

**Verify.** `test_shortcut_control_is_reported_and_substantially_beaten`,
`test_feature_sets_have_no_duplicate_columns`.

**Effect.** The fusion claim becomes testable: the full model beats the
shortcut control, and the **matched** `shortcut_plus_autonomic` variant shows
the gain is carried specifically by the autonomic channels — a claim v2 could
not make, because its gains were quoted against the weaker of two streams with
no baseline at all. Exact figures are maintained in [`RESULTS.md`](RESULTS.md)
and, per §1, deliberately not restated here so they cannot drift.

---

## 3. Tail rows were silently mislabelled

**Defect.** `feature_engineering.py` dropped only the final row of each
patient, so the last 8 ticks of every patient were labelled from a 1-to-7-tick
window while being treated as a full 2-hour lookahead. Because the label is a
max over future samples, a short window is systematically biased toward "no
spike" — 4,000 rows of quiet label noise. `train_cv_risk_model.py` handled this
correctly, so the two scripts disagreed.

**Fix.** A row is trainable only if a full `LOOKAHEAD_TICKS` window exists
(`n_valid = n - horizon`).

**Verify.** `test_tail_rows_are_dropped_not_mislabelled` asserts exactly
`N_TICKS - LOOKAHEAD_TICKS` rows per patient.

**Effect.** Row count drops to `N_TICKS - LOOKAHEAD_TICKS` per patient. Current
figures are in [`RESULTS.md`](RESULTS.md); the v2 baseline was 1,007,500 rows
at a 4.0% positive rate.

---

## 4. The rise criterion fired on sensor noise

**Defect.** `max(future) - cgm[i] >= 30` on raw 15-minute samples. A max over 8
samples each carrying ~5 mg/dL of Gaussian noise clears a 30 mg/dL bar often
enough to swamp the physiological signal.

**Fix.** Both label conditions are evaluated on a 30-minute smoothed CGM
series, which is also more faithful to how real CGM rapid-rise alerts behave.

**Verify.** `_trailing_smooth` / `_trailing_forward_max` unit tests.

---

## 5. No calibration, and a meaningless operating point

**Defect.** v2 displayed a risk percentage in the UI and alerted at 0.6 while
reporting only rank metrics and precision/recall at 0.5. A clinician reading
"96% risk" is reading a *calibrated* probability. At a ~2% base rate a 0.5
threshold is not a meaningful operating point.

**Fix.** Brier score, log loss and 10-bin expected calibration error are
reported for every variant. Operating points are quoted at fixed **alert
budgets** (1/5/10/25% of patient-ticks), converted into alerts per patient per
day — the unit a care team budgets in.

**Verify.** `test_expected_calibration_error_bounds`,
`test_operating_point_respects_the_budget`,
`test_operating_points_are_monotone_in_budget`, `test_calibration_metrics_are_reported`.

---

## 6. No leakage controls

**Defect.** v2 had none. A pipeline that leaks will still produce a confident
headline number, and nothing in the repo would have caught it.

**Fix.** Two controls in `train_model.py`:
- **Global label permutation** — re-runs the pipeline with shuffled labels.
  Must land at ~0.50.
- **Within-patient label permutation** — preserves patient identity, and
  therefore every static EHR column and each patient's base rate, while
  destroying only temporal alignment. Whatever AUC survives is score from
  knowing *who* the patient is rather than *what state* they are in. This is
  the control that would have caught the v1 model that put 82% of its weight on
  `type2_diabetes`.

**Verify.** `test_permutation_control_is_near_chance`.

**Effect.** Global permutation collapses to about chance; the within-patient
shuffle stays well above it, measuring how much of the score rests on knowing
*who* the patient is rather than *what state* they are in — legitimate and
clinically useful — while the remainder is temporal signal. Both control
numbers are stated in `RESULTS.md` rather than hidden, because a reader who
discovers them independently is right to assume concealment.

---

## 7. The demo was staged, not measured

**Defect.** `export_dashboard_data.py` ended each patient's window at
`argmax(risk_score) + 4` — every demo patient was shown at their personal
maximum, with a comment saying so. The alerts panel fired constantly. v2 never
stated the true alert rate.

**Fix.** One fixed, arbitrary rule for every patient (the final tick of the
simulation). Patient selection is by a clinical rule declared in advance and
never on model output. The cohort-wide alert rate is computed over all 1M
ticks and shipped to the UI.

**Verify.** `test_exported_dashboard_agrees_with_its_metadata`.

**Effect.** Most demo patients now show low risk, as they should. Cohort alert
burden is stated explicitly in the sidebar.

---

## 7b. The alert threshold could never fire

**Defect.** `export_dashboard_data.py` hard-coded `ALERT_THRESHOLD = 0.60` and
never checked it against the model's own risk distribution. It was inherited
from v2 unexamined. On this model the whole risk distribution sits far below
0.60 — the 95th percentile is around 0.07 — so a 0.60 cut fires on well under
1% of patient-ticks, and never once inside a 48-hour window. A threshold that
cannot fire is not a conservative threshold, it is an absent one. v2's UI
reported this cut as if it were a working alerting rule.

**Fix.** `src/export_dashboard_data.py` now declares an **alert budget** and
derives the probability cut from it. The budget is 1% of patient-ticks, which at
96 ticks/day is ~1 alert per patient per day — a figure a care team can
plausibly absorb, chosen on the operating-point reasoning already used for the
metrics table rather than by lowering 0.60 until the demo looked busy, which
would have been the same error in a new coat.

**Verify.** `test_threshold_for_budget_hits_the_declared_budget`,
`test_alert_threshold_is_reachable`,
`test_alert_rate_matches_the_declared_budget`.

**Effect.** The cut is no longer 0.600 and is now derived per run, so it is not
restated here — `dashboard/demo_data.json` carries it, and the cohort alert rate
is asserted against the declared budget by
`test_alert_rate_matches_the_declared_budget`. The qualitative change: demo
windows now show a *wide* spread of alert ticks, including several patients
with none, instead of the uniform flood a staged demo produces.

---

## 8. The UI promised explanations it could not produce

**Defect.** The v2 dashboard said a clinician could "ask the twin to explain
the contributing signals." Nothing implemented it.

**Fix.** A real local counterfactual attribution: replace each feature with
the patient's own trailing 7-day median and measure the risk delta. Rendered
as signed bars in a "Why this score" panel. Also: HR and HRV moved to separate
axes (sharing one flattened HRV into an unreadable line), realised spike labels
plotted on the glucose chart so the risk curve is falsifiable on its face, and
a synthetic-data / not-for-clinical-use banner in the interface itself.

---

## 8b. Statistical claims were subtractions of means

**Defect.** Ablation variants were each cross-validated independently and the
report quoted the difference of their means. With 5 folds that cannot
distinguish a real +0.02 from noise, and a delta quoted without a significance
test is exactly the kind of number that should not be believed. During the v3
rewrite this produced an outright wrong sign: `static_only`'s delta against the
shortcut control (-0.096) was labelled "full model vs. EHR-only", which
reported the flattering comparison as a *negative* lift.

**Fix.** All variants are scored on the identical fold set, per-fold differences
are treated as matched pairs, and each delta carries a two-sided bootstrap
p-value over folds. Folds — not the ~250,000 test ticks — are the unit of
independence. Comparison labels are now derived from the variant pair actually
being differenced, so the two cannot drift apart.

---

## 9. Documentation drift

**Defect.** The v2 README quoted 0.953 in §1 and 0.955 in §8, with neither
matching any number in `metrics.json`.

**Fix.** `src/make_report.py` generates `docs/RESULTS.md` from the metrics
files. The README links to it and quotes no numbers by hand. CI regenerates the
report and fails if the committed copy is stale.

---

## 10. "Self-contained" was false

**Defect.** v2 claimed the dashboard was "fully self-contained" while pulling
Chart.js from cdnjs and two webfonts from Google Fonts. It rendered blank
offline — exactly when you would want to demo it.

**Fix.** Chart.js is vendored at `dashboard/vendor/chart.umd.min.js` and the
type stack is system-only. The claim is now enforced by a test and by CI.

---

## 11. Repository hygiene

| Added | |
|---|---|
| `LICENSE` | MIT, with an explicit data-provenance / no-PHI notice |
| `tests/test_pipeline.py` | correctness, leakage, calibration, determinism, patient-cluster bootstrap, external-validation harness, dashboard (count generated into the README table) |
| `.github/workflows/ci.yml` | tests + full pipeline + report-freshness + self-containment, on Python 3.10 and 3.12 |
| `pytest.ini` | warnings-as-errors so pandas deprecations fail loudly |
| `src/config.py` | single source of truth for all constants |

**Defect.** v2 had no LICENSE, no tests, no CI, and an empty `docs/` directory.
It also used `max_iter=200, lr=0.08` for the CV model and `120, 0.10` for the
glucose model, making its own runtime note ambiguous.

**Fix.** Both targets now share one hyper-parameter set, enforced by
`test_both_targets_share_hyperparameters`.

---

## 12. The data generator was not actually deterministic

**Defect.** `generate_wearable()` drew each patient's RNG substream from a
single global generator as it iterated the cohort:

```python
rng = np.random.default_rng(seed)
for _, row in ehr_df.iterrows():
    patient_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
```

The comment above it claimed this made generation
"deterministic and order-independent". It did not. A patient's stream depends
on their *position* in the input, so the 8th patient in the file gets different
data than the same patient generated 2nd. The full 500-patient run was
reproducible only if you regenerated all 500 patients in the same order. A judge
who re-ran a 40-patient slice to check a claim got different data for the same
people and would reasonably conclude the whole pipeline was nondeterministic —
or, worse, that the reported numbers were not what the code produced.

This one was found by a test written for a different reason, which is the
argument for writing the test at all.

**Fix.** `_patient_seed()` derives each stream from `blake2b(seed:patient_id)`.
The stream now depends on the identifier and the master seed and on nothing
else, so any subset, any ordering, reproduces exactly. `blake2b` rather than
`hash()` because Python's string `hash` is salted per process, which would have
broken reproducibility in exactly the same way for a different reason.

**Verify.** `test_per_patient_generation_is_order_independent` (four orderings),
`test_master_seed_actually_changes_the_data`,
`test_patient_seed_is_stable_across_processes`.

**Effect.** No change to the design; a change to the pipeline's honesty. Because
the seed derivation changed, the whole cohort was regenerated and every metric
re-measured — the committed numbers come from the committed generator.

---

## 13. CI's reduced run was producing meaningless numbers, in green

**Defect.** CI ran the pipeline on `DIGITAL_TWIN_N_PATIENTS=40
DIGITAL_TWIN_DAYS=10` to stay fast. The cardiovascular-strain label asks
whether a patient has ≥3 consecutive qualifying days inside a **7-day** future
window, so a 10-day series cannot contain a single positive example. The
reduced run therefore produced a **0.0 base rate and NaN for every AUC** in
`cv_risk_metrics.json`, printed a `fusion lift over best single stream: +nan`,
and **passed**. A green build on a meaningless result is worse than a red one,
because it removes the signal that something is wrong.

**Fix (two layers).** `config.py` computes `STRAIN_MIN_DAYS_REQUIRED` from the
label geometry and exits with an explanation naming the actual problem if
`DAYS` is below it. That floor (11) proved *necessary but not sufficient*:
running CI's population at exactly 11 days still produced a 0.0 base rate and
NaN everywhere — `roc_auc_score` returns NaN (with only a warning) when a
fold's test partition has a single class, so the pipeline stayed green. So
`train_cv_risk_model.py` now **re-checks the generated label at runtime** and
exits loudly if it degenerates: zero positives, *or* any of the 5 folds whose
test partition lands on a single class (observed at 40 patients / 14-16 days
despite a ~5-6% rate). CI runs 12 days, which is the smallest setting verified
to be non-degenerate at 40 patients (22 positives, ~11% rate, no single-class
fold).

**Verify.** `test_cv_risk_label_guard_is_loud` drives both guard branches
directly; `test_strain_label_geometry_has_a_minimum_series_length` and
`test_too_short_series_is_rejected_loudly` cover the config floor; and
`test_committed_cv_risk_metrics_are_not_nan` (NaN != NaN) catches a reduced run
that overwrote a committed artifact.

**Why this is worth recording.** Both this and §12 were found by *running* the
pipeline at CI's settings rather than by reading the code. Neither is visible
from the default configuration.

---

## 14. The demo roster had a hidden second selection

**Defect.** `pick_demo_patients()` took `head(3)` diabetics plus a
`sample(3, random_state=1)` of healthy patients, then sliced to
`N_DEMO_PATIENTS`. Two problems. First, on a reduced cohort with fewer than
three diabetics it silently returned a **short roster** — the dashboard
advertises six patients and shipped five, with no error. Second, the three
"controls" were a *random* draw, which is weaker than it looks: a random draw
from the non-diabetic population is an undeclared selection rule, and it means
the roster's composition is a lottery rather than a stated choice.

**Fix.** The rule is now a fixed target filled by declared priority: the
highest-HbA1c diabetics, then the **lowest**-HbA1c non-diabetic
non-hypertensive patients. No RNG. If the cohort cannot fill the roster the
function exits with an explanation rather than padding it — and the error
message says explicitly *not* to pad by selecting on model output, because that
is precisely the v2 failure this whole section exists to prevent.

**Verify.** `test_demo_patient_selection_is_deterministic_and_model_free` checks
reproducibility, no duplicates, a full roster, that the highest-HbA1c diabetics
are present, and — the part that catches a rule secretly depending on input
order — that shuffling the rows of `ehr_patients.csv` returns the same roster.

**Also.** The roster is now *displayed* in descending HbA1c so the demo opens on
the most clinically severe patient instead of whichever healthy control the
sampler happened to emit first. Ordering is presentation; the same six patients
appear either way and no risk score is consulted.

---

## 15. Significance was the resolution of five folds, not five hundred patients

**Defect.** The paired deltas carried a bootstrap p-value that resampled the 5
fold-wise deltas. Five folds are the entire universe: the GroupShuffleSplit test
sets overlap across splits, and a two-sided sign test on five matched pairs
cannot go below p = 1/32 no matter how consistent the effect is. A bootstrap
over those five deltas can *print* "p < 0.0001" (its 1e-4 floor) but the data
never licensed that resolution — the claim exceeded the number of independent
replicates.

**Fix.** `_patient_bootstrap_matrix` in `train_model.py` resamples **patients**
(2000 resamples; patients are the independent unit) over the pooled out-of-fold
predictions, and every p-value in `metrics.json` and the report now comes from
that bootstrap. The statistical-testing block records the method, the resampling
unit and the old fold sign-test bound (0.0625) so the change is auditable.

**Verify.** `test_fusion_claim_is_a_tested_paired_comparison` (patient-bootstrap
p on the headline delta), `test_p_value_rendering` (p is rendered as `< 0.001`,
never `p = 0.000`).

**Effect.** The headline fusion delta (+0.0216 over the shortcut control) is now
supported at p = 0.001 by a method whose resolution is earned. Equally
important: the unflattering comparisons are now *honestly* non-significant
(full vs. best non-fusion subset: p = 0.184) instead of flapping with the seed.

---

## 16. The fusion ablation tested a generator with no EHR signal

**Defect.** In v3.0 the hidden reactivity's between-patient component was
dominated by residual noise, so the static EHR columns could never matter. The
ablation's "EHR adds nothing; wearable-only beats fusion" was *guaranteed by
construction* — the fusion test was a tautology, and the README's own roadmap
listed "a fair fusion test" as unbuilt future work.

**Fix.** `generate_wearable.py` v3.1 lets the EHR columns modulate the **mean of
the hidden AR(1) reactivity**: hba1c, fasting glucose and diagnosis shift the
between-patient level, the `tcf7l2` genotype channel is deliberately *not*
expressed in the observed CGM/HRV levels (so a wearable-only model cannot infer
it), and residual noise keeps the current-value draw hidden. The coupling is
tuned to keep the label a *mix* of identity and current state (an earlier
tuning that made reactivity near-deterministic from EHR turned the whole task
into a lookup and was reverted).

**Verify.** `test_metrics_are_in_a_plausible_range` (the ratchet that fails if
the task closes back above 0.93), `test_permutation_control_is_near_chance`,
`test_shortcut_control_is_reported_and_substantially_beaten`.

**Effect.** EHR-only now scores 0.823 and the p-values above are the honest
result of a real test, not of a generator that could not produce an EHR signal.

The original v3.1 conclusion drawn from this test — "fusion is directionally
best, the EHR share is not statistically distinguishable, defer the verdict to
real data" — **did not survive the seed sweep of §19**. Against the best
non-fusion subset the delta is negative in 4 of 5 cohorts, so the correct
reading of §16 is stronger and less flattering than v3.1 claimed: the EHR
stream contains real signal that the fused model does not convert into a net
win, and testing it properly exposed that.

---

## 17. Three reproducibility-readiness inconsistencies

**Defect.** (a) The permutation table's 0.8095 "observed" did not match the
headline CV mean 0.833 and was never explained; (b) the README claimed artifacts
regenerate in "~1 minute" while `metrics.json` recorded a 415 s
`train_model.py` runtime; (c) the repo shipped a self-assessed score sheet
(`docs/SELF_SCORE.md`) that a judge flagged as presumptuous.

**Fix.** (a) `permutation_controls` now records `observed_auc_protocol` and the
`headline_roc_auc_5fold_cv_mean` beside the raw value, and the report explains
that the observed row is on the single split carrying the permutation nulls —
deliberately not the 5-fold CV mean; (b) the README cites the committed
`metrics.json` runtime rather than a hard-coded minute; (c) the self-score file
was removed from the repo and the submission bundle.

**Verify.** `test_report_agrees_with_metrics_json` (regenerating the report
changes nothing).

---

## 18. A derived number mixed two different measurement protocols

**Defect.** The report's within-patient row subtracted the **5-fold CV mean**
headline (0.8756) from the **single hold-out split** identity floor (0.8022) and
presented the difference as "+0.07 of the score comes from patient state". The
two numbers are measured by different procedures on different splits; their
difference is not a quantity either procedure produced.

**Fix.** The gap is now computed between `observed_auc` (0.8609) and
`within_patient_permutation_auc` (0.8022) — **both** on the same seed-7 hold-out
split, the split that carries the permutation nulls. The cell renders
"identity floor; state adds +0.06 (same hold-out split)". Same-significant
magnitude, now actually interpretable: on one split, 0.802 of AUC is
recoverable from *who the patient is* alone, and ~0.06 from current state.

**Verify.** The cell text is generated from `metrics.json`, so it cannot drift;
`test_report_agrees_with_metrics_json` re-derives it.

---

## 19. One seed was the only evidence that the headline was stable

**Defect.** Every headline number in the repo came from a single generator draw
at the default master seed. Generator randomness was never varied, so "the full
model reaches 0.876" carried no statement about how much of that is a property
of the *method* versus one lucky cohort. On top of that, the fold-vs-patient
bootstrap (§15) was added but its own significance was asserted from one draw.

**Fix.** `DIGITAL_TWIN_SEED` now drives **both** generator stages (EHR and
wearable), so a cohort can be redrawn without touching the model or CV protocol.
The full 500-patient / 21-day pipeline was re-run under four sweep seeds plus
the canonical default; each `metrics.json` snapshot is committed under
`model/seeds/` and `src/make_seed_report.py` compiles
[`SEED_ROBUSTNESS.md`](SEED_ROBUSTNESS.md) from them. Model, estimator seeds and
CV protocol are held fixed across seeds so the spread isolates generator
randomness.

**Effect.** See [`SEED_ROBUSTNESS.md`](SEED_ROBUSTNESS.md) for the table. It
settled two things and overturned a third.

*What held.* The paired gain over the shortcut control reproduces at p < 0.05
in **5 of 5** cohorts (+0.021 to +0.037). That is a load-bearing result.

*What did not.* The headline AUC itself moves 0.8154–0.8756 across draws
(mean 0.843 ± 0.021) — and **the shipped canonical run is the maximum of that
range**. The 0.876 in the README is the most favourable draw tested, not a
central estimate, and the report now says so in those words rather than leaving
a reader to notice it.

*What was overturned.* The v3.1 framing — "fusion is directionally best, just
not significantly so" — turned out to be a property of the shipped cohort. The
fusion-vs-best-non-fusion delta is **negative in 4 of the 5 cohorts**
(−0.0035 to −0.0115), and in one of them it is significantly negative
(p = 0.040). Fusion is a null-to-net-negative finding on this task, so the
README was rewritten to say that instead of hedging. This is the sweep doing
its job: a claim that only holds on one draw is not a claim, and the honest
report is the negative one.

The generator was **not** re-tuned in response. Chasing a fusion win by
adjusting the hidden-reactivity coupling until the delta flips sign would
reproduce the v2 failure in a new costume — tuning the benchmark until the
answer you want appears. The unfavourable number is the result.

---

## 20. "Validated on real data" was a roadmap item with no harness behind it

**Defect.** The README listed real-data validation (OhioT1DM / MIMIC-IV) as the
step that would push the project past ~90, but the repo contained no code that
could load a real cohort. The claim was unfalsifiable in the way that matters:
anyone could have written that line without the pipeline supporting it.

**Fix.** `src/validate_real_data.py` is a real external-validation harness: it
resamples any long-format glucose cohort (`patient_id, timestamp,
glucose_mgdl`) to a 15-min grid, builds the **same** trailing features and the
**same** label definition, and runs the **same** 5-fold grouped CV plus the
**same** patient-level cluster bootstrap. `src/convert_uci_diabetes.py` (UCI
Diabetes, CC BY 4.0, no credentials) and `src/convert_ohio_to_long.py`
(OhioT1DM XML) both emit that schema. CI rehearses the whole path on the
synthetic stream so it cannot rot, and the rehearsal output is excluded from
the generated report.

**Verify.** `test_external_validation_rehearsal_schema` runs the harness
end-to-end on a small input and asserts the schema and the p-floor;
`test_external_validation_uci_has_no_zero_p` guards the committed real result.

**Effect, honestly.** It has been run on real data — 70 real patients, 13,517
capillary readings. The protocol transfers and engineered context beats a naive
current-value baseline by a small, significant margin. But the absolute real AUC
(~0.98) is **not** comparable to the synthetic 0.876: this cohort's base rate is
54%, and its naive baseline alone scores 0.977, meaning most of the apparent
accuracy is the current value. Full analysis, the "why 0.98 is not better than
0.876" argument, and what remains untestable (fusion needs a paired EHR table,
which neither OhioT1DM nor UCI provides) are in
[`REAL_DATA.md`](REAL_DATA.md). OhioT1DM and MIMIC-IV are credential-gated with
DUAs that forbid redistribution, so no result in this repo depends on data a
reviewer cannot obtain.

---

## 21. The headline number was the best of five draws

**Defect.** §19 established that the shipped canonical run was the *maximum* of
the five-seed range, and that was written down — but the README, the seed report
and the generated v2-vs-v3 table still led with **0.876**. Documenting a
selection is not the same as not making one: a reader who takes the first number
off the top of the README was still reading a post-hoc best-of-five, and that is
the number the reviewer quoted back.

**Fix.** The headline is now the **mean ± sd over the five generator seeds**,
**0.8427 ± 0.0205**, generated into `SEED_ROBUSTNESS.md`, `README.md` and the
v2-vs-v3 table from `model/seeds/*.json` by `src/make_seed_report.py` and
`src/make_report.py`. The canonical 0.8756 is retained only where an
artefact-level figure is genuinely required (the shipped `.joblib` describes
that cohort), and it is labelled as the maximum of its range every time.

**Why the mean and not the median seed.** The reviewer's suggestion was to
promote the median draw (~0.84) to headline. The median seed is
`DIGITAL_TWIN_SEED=21`, so making it canonical means re-running the 500-patient
generator and every downstream artefact to keep the shipped `.joblib` consistent
with the headline — and on that draw the fusion-vs-best-non-fusion delta is
**−0.0093** rather than +0.0046. The mean is also a central estimate, requires
no re-run, and — the actual point — **selects nothing**. Choosing which seed is
canonical is itself a selection decision made after seeing the numbers; taking
the mean removes the decision rather than relocating it.

**Verify.** `test_seed_report_headline_is_the_mean` fails if the headline is a
single seed's AUC or is absent from the README.

---

## 22. The deliverable had CRLF line endings; the repository did not

**Defect.** The reviewer reported CRLF in `README.md` and `ci.yml` in the
submitted zip. The first diagnosis — "committed files have CRLF" — was **wrong**.
With `core.autocrlf=true` the blobs in the index had been LF the whole time, and
zero tracked files contained CRLF. The actual cause was `git archive`, which
builds the submission zip: it applies the checkout conversion rules while
writing, so **42 of 51** entries came out CRLF. The defect existed only in the
build of the artefact being judged, which is why it never appeared in the working
tree or in `git diff`.

**Fix.** `.gitattributes` pins `* text=auto eol=lf` with binary exceptions for
the model artefacts (`.joblib`, `.png`, `.ico`, `.zip`), so the index, the
checkout and the archive all agree.

**Verify.** CI has a ratchet that runs `git archive HEAD`, re-reads every entry
and fails if any text file contains CRLF. It now reports **1** entry with CRLF:
the `.joblib` pickle, whose bytes are binary payload and are excluded from the
rule rather than silently tolerated.

---

## 23. A results file was cited before it existed

**Defect.** The v3.3 rewrite of `REAL_DATA.md` documented the BIG IDEas paired
cohort properly — fetch, converter, protocol, limitations — and then told the
reader the numbers were generated into `RESULTS.md` from
`model/external_validation_bigideas.json`, and that a summary row pointed there
for the fusion verdict. Neither the JSON nor the `RESULTS.md` section existed,
because **the paired run had never been executed**. The document asserted a
result it did not have.

This is the exact failure mode the rest of this file exists to prevent, and it
was introduced by the same revision that added the honesty controls. During the
v3.3.1 pass, fabricated figures were also found sitting in the working-tree
README — a fusion delta, two variant AUCs and a p-value for the BIG IDEas
cohort, with no source file behind any of them. They were removed, not
committed, and the honest status was restored.

**Fix.** The generated-numbers sentence now names only the files that exist. A
status block states plainly that the paired cohort is **NOT YET RUN**, that no
`external_validation_bigideas.json` exists, and that the section documents the
protocol so the run is auditable when it happens — deliberately not a result.
The two summary-table rows that implied a fusion finding now read **NOT YET
TESTED** and **not yet shown**.

**Standing rule this establishes.** No number appears in this repo without a
generated source file behind it. If a result has not been computed, the docs say
so in the same place they would otherwise report it.

---

## Also fixed

- `innerHTML` string interpolation in the dashboard replaced with an `esc()`
  helper. Patient ids are synthetic today, but a real deployment would push
  EHR free-text through that template.
- The model card said "Evaluated on 125 held-out virtual patients never seen in
  training" beside predictions from a model fit on the full cohort. It now
  distinguishes CV metrics from deployment-model scores.
- A per-row Python callback in the sleep-ratio rolling window (which dominated
  runtime on 1M rows) is now vectorised. Feature engineering: ~9s.
- The 211 KB demo payload is now injected at build time from
  `dashboard/template.html` via a `__DEMO_DATA__` placeholder, so the
  dashboard is reproducible from source rather than hand-edited.

---

## What is still not solved

These are stated in the README §9 as well, and none of them are fixable with
synthetic data.

- **Synthetic signal is not real physiology.** v3 fixed the ways in which the
  benchmark was self-fulfilling and added controls that would catch a
  recurrence. It did not make synthetic glucose dynamics resemble real ones.
  Every number must be re-derived on licensed data before it means anything.
- **The cardiovascular-strain target is weak** and reported as such. Its label
  uses a full-window baseline a deployed system cannot compute, and 14 usable
  days per patient is a small evaluation set.
- **The EHR stream's verdict is negative on synthetic data.** Under the v3.1
  generator the EHR genuinely modulates the hidden reactivity (EHR-only alone
  scores 0.823), and the fair fusion test asks whether the model can exploit
  that on top of a wearable stream. The answer across five cohort draws is no:
  fusion beats the shortcut control in every draw, but against the *best
  non-fusion subset* it is negative in 4 of 5 and significantly negative in one.
  The judge should weigh the shortcut-control comparison for the headline, and
  read the fusion verdict as **not demonstrated** rather than "promising but
  unproven".
- **Fusion is still untested on real paired data, and this is the single
  largest gap in the submission.** UCI carries a glucose series but no wearable
  stream and no static clinical table, so it cannot answer the question at all;
  the answer it gives on the *magnitude* of the synthetic effect is already
  unflattering. The BIG IDEas cohort (16 prediabetic adults with Dexcom G6 CGM
  **and** an Empatica E4 HR/IBI stream **and** `Demographics.csv` HbA1c/gender
  for the same participants) is openly licensed and is the one real cohort
  found that can. The fetch, converter and validation path are implemented and
  tested (§23); **the paired run has not been executed**, so no fusion verdict
  on real data exists anywhere in this repo. Until it does, the fusion claim
  rests entirely on a synthetic generator, and at n = 16 with ~9 days per
  participant the result will be **power-limited** — a null there is
  inconclusive, not evidence against fusion, and must not be reported as though
  it settled the question.
- **Only 1 of 6 demo patients shows an alert.** At a 1% alert budget most 48-hour
  windows are legitimately quiet. That is a property of an honest operating
  point, not a broken demo, but it does mean the console cannot by itself
  demonstrate recall. The operating-point table in `RESULTS.md` carries recall
  at four budgets for that purpose.
- **The generator's realism is partially tested now, and the test is unflattering.**
  The external run on 70 real patients (`REAL_DATA.md`) shows the *protocol*
  transfers, but the real cohort's naive current-value baseline reaches 0.977
  against a 0.983 full-feature model — a shortcut-dominant regime the synthetic
  generator does not reproduce. So "not derivable from its own inputs" is now
  backed by a real-data comparison, and the answer is that synthetic difficulty
  is a property of the generator rather than a prediction about real physiology.
- **The console is a static demo.** No auth, no audit logging, no per-patient
  thresholding, no drift monitoring.
- **No clinical validity.** Thresholds are illustrative and unvalidated.
