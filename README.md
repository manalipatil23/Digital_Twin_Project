# Digital Twin Project — Predictive Glucose-Spike Alert

## Unstop Digital Twin Challenge 2026

### Team Details
- **Team Name:** TechThonX
- **Participation:** Individual Participant
- **Participant:** Manali Patil

### College / Incubator Information
- **College:** Government College of Engineering, Chhatrapati Sambhajinagar
- **Incubator:** Not Applicable

### Project Title
**Digital Twin for Predictive Glucose-Spike Alert**

### Healthcare Use Case
A healthcare-focused digital twin prototype that combines static EHR information with wearable/IoT data to represent a virtual patient state and predict a potential glucose spike up to 2 hours in advance.

## Technical Stack

- **Programming Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Model Persistence:** Joblib
- **Testing:** Pytest
- **Dashboard:** HTML, CSS, JavaScript
- **Visualization:** Chart.js

## AI/ML Model Details

- **Task:** Binary classification of a glucose spike occurring within the next 2 hours.
- **Model:** `HistGradientBoostingClassifier` from scikit-learn.
- **Validation:** 5 × `GroupShuffleSplit`, grouped by `patient_id`.
- **Grouping:** A patient's records never appear in both training and test sets.
- **Evaluation:** ROC-AUC, Brier score, log loss, ECE, precision at alert budgets, and patient-level cluster-bootstrap significance testing.
- **Input Data:** Static EHR information combined with dynamic wearable/IoT and CGM-derived features.

---

A Phase-1 prototype that fuses a **static EHR** record and a **wearable/IoT
stream** to predict an adverse health event before it happens — a glucose spike
within the next 2 hours. Built for the Unstop Digital Twin Challenge 2026, organized by Happiest Health.
Every figure in this repo is generated from `model/metrics.json`; none is
hand-copied.

## In plain language

**What it does.** It watches a patient's continuous glucose stream alongside
their clinic record and tries to warn them roughly two hours before glucose
spikes. The clock starts when the system is handed a new reading; it then
decides whether to raise an alert.

**How well it works, on synthetic data.**A ROC-AUC of 0.843 means the model ranks a randomly selected positive case above a randomly selected negative case about 84% of the time. — a useful signal, and better than comparing
patients on their clinic history alone. Averaged over five independent runs of
the data generator, the headline figure is **0.843**, with a spread of ±0.021.
At this alert rate a patient would get **under two warnings a day**, and roughly
4 in 10 of those would be real.

**The honest catch.** Two of the project's own tests argue against its headline
claim. Against a deliberately unfair rival — a baseline assembled *after* seeing
the results, on the same data — combining the wearable and clinic streams is
**not** better, and this README says so in those words. And the headline is the
*average* of five draws, not the best one: the model that ships happens to be
the luckiest of the five, and this README says that too.

**On real patients.** The external cohort contains 70 real patients with diabetes; 50 had sufficient contiguous context to produce a valid scored prediction window.
The same code ran unchanged, and the improvement over a naive guess was real
but roughly **four times smaller** than on synthetic data. That gap is reported,
not smoothed over. The shortcut baseline scored 0.977 on that cohort (base rate
54%), which shows the generator's difficulty profile does not match real
capillary data.

**What is not finished.** The strongest claim — that combining clinic records
with wearable data genuinely helps — has **not** been proven on real data. A
second cohort that could test it (BIG IDEas, 16 prediabetic adults with CGM +
wearable + a static clinical table for the *same* people) has been located,
downloaded and wired up, but the paired test **has not been run**. No number
exists for it, and none is quoted anywhere in this repo.
[docs/REAL_DATA.md](docs/REAL_DATA.md) documents the cohort and the protocol so
the run is auditable when it happens.

## The current result

1. **Score.** ROC-AUC **0.843 ± 0.021** for a 2-hour-ahead spike at a ~1.8%
   base rate. **That is the mean ± sd over five generator seeds, not one
   draw** — naming a single seed as the headline is a selection decision, and
   the sweep exists to remove exactly that. Per-seed the AUC ranges
   **0.815–0.876**. The shipped `.joblib` was fitted on the canonical cohort
   and scores 0.876 on it; that is a per-draw figure, not the headline, and is
   quoted only where an artefact-level number is needed. See
   [seed robustness](docs/SEED_ROBUSTNESS.md).
2. **Beats the shortcut, and that one holds up.** A matched control (clock,
   current CGM, trend, baseline labs) scores 0.854; the full model gains
   **+0.0216** (p < 0.001, patient-level cluster bootstrap), carried by the
   autonomic channels (+0.0171). The gain reproduces at p < 0.05 in **5 of 5**
   seeds.
3. **Fusion does not pay for itself.** Be precise about which comparison that
   means. Fusion beats **EHR-only by +0.053** and the pre-specified shortcut
   control by **+0.022** (both p < 0.001) — so both streams carry signal and the
   model captures it. The single comparison it loses is against the *best
   non-fusion subset*, worth +0.0046 (p = 0.184) here, and **negative in 4 of the
   5 cohort seeds**, significantly so in one. That reference is not neutral
   ground either: it is a composite assembled after seeing these results, on the
   same folds, so +0.0046 flatters fusion and the true out-of-sample gap is
   unmeasured. Verdict on this generator: **null-to-net-negative against a
   tuned composite, positive against every pre-specified baseline** — reported
   as a negative finding, not hedged as "directionally best".

## Key findings

- **The 0.65–0.93 AUC band is a tuned design target, not a finding.** Above
  ~0.93 the task is algebraically closed again; below ~0.65 the features carry
  no signal. A ratchet test fails the build if the loop ever closes.
- **Significance is patient-level, not fold-level.** Every p-value is a
  2000-resample cluster bootstrap over *patients*. The 5 folds' test sets
  overlap, and a two-sided sign test on five matched pairs cannot go below
  p = 1/32 — no claim here is licensed by resampling folds.
- **Robustness was measured, and it moved the verdict.** The full pipeline was
  re-run under five generator seeds ([seed robustness](docs/SEED_ROBUSTNESS.md)).
  The AUC moved 0.815–0.876, which is why the headline is the mean (0.843 ±
  0.021) rather than any one seed. The shortcut-control gain survived all five.
  The fusion-vs-best-single delta did not: it is negative in 4 of 5.
- **Real data: the protocol runs, but performance does not directly transfer.** The external
  harness has been run end-to-end on real patients (UCI Diabetes, CC BY; 70
  available, 50 with enough contiguous context to score). Engineered glucose
  context beats a naive current-value baseline by +0.006 (p < 0.001) — same
  direction as synthetic, ~4× smaller margin. The absolute AUC (0.98) is **not**
  comparable to the synthetic headline: this cohort's base rate is 54% and its
  naive baseline already scores 0.977. See [docs/REAL_DATA.md](docs/REAL_DATA.md).

## How to run it

```bash
pip install -r requirements.txt
python data/generate_ehr.py && python data/generate_wearable.py   # 1. data
python src/feature_engineering.py                                 # 2. fusion + labels
python src/train_model.py                                         # 3. train, CV, controls
python src/export_dashboard_data.py                               # 4. dashboard bundle
python src/make_report.py                                         # 5. RESULTS.md + README table
```

`python src/train_cv_risk_model.py` additionally trains the optional
cardiovascular-strain target. Open `dashboard/index.html` in a browser — it is
self-contained (no network; CI greps for remote resources). Tests:
`python -m pytest tests/`.

On real data (same protocol, external cohort):

```bash
python src/validate_real_data.py --cgm data/real/uci_cgm.csv --out uci   # docs/REAL_DATA.md
```

CI runs the same pipeline at 40 patients / 12 days via the
`DIGITAL_TWIN_N_PATIENTS` / `DIGITAL_TWIN_DAYS` overrides. The small cohort is
kept non-degenerate by a `config.py` floor, and the training scripts exit
loudly if a label ever degenerates. `DIGITAL_TWIN_SEED` redraws the whole
cohort under a different master seed (both generator stages), which is what
the seed-robustness report uses.

## The evidence

Every claim below is paired with the control that could falsify it. Full tables
in [`docs/RESULTS.md`](docs/RESULTS.md).

- **Shortcut baseline** (clock + CGM + trend + baseline labs) — does the
  wearable stream earn its place? 0.854.
- **Permutation controls** — global 0.495 (must be ~0.50) and within-patient
  0.802 (what *who the patient is* alone buys; reported, not hidden).
- **Paired deltas** — identical folds; each delta carries a two-sided
  patient-level cluster-bootstrap p-value (rendered `p < 0.001`, never
  `p = 0.000`).
- **Calibration + operating points** — Brier / log loss / ECE; alerts quoted at
  1 / 5 / 10 / 25% budgets (= alerts per patient per day).

The v2-vs-v3 comparison (v2 column transcribed from the prior submission; the
v3 column is generated and cannot drift):

<!-- BEGIN GENERATED: v2-vs-v3 (src/make_report.py) -->
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
<!-- END GENERATED: v2-vs-v3 -->

**Label.** `glucose_spike_2h` = 1 if, within 2 h, 30-min-smoothed CGM exceeds
**180 mg/dL** or rises **≥ 30 mg/dL** (smoothed, not raw — v2's rise criterion
fired on sensor noise).

**Model.** `HistGradientBoostingClassifier`, **5 × GroupShuffleSplit** grouped
by patient_id (a patient's rows never span train and test), hyper-parameters
shared between both targets in `src/config.py`.

## What would push this past a prototype

- **Clinical validation on CGM.** Done on real (capillary) data — see above.
  The remaining gap is CGM-resolution data plus a *paired* EHR table, which is
  what an external fusion verdict needs; OhioT1DM and MIMIC-IV are both
  credential-gated.
- **Operations.** Auth, audit logging, alert-fatigue management, per-patient
  thresholds, drift/calibration monitoring — none present, all mapped onto the
  production components below.

## Limitations (what is still not solved)

- **Synthetic signal is not real physiology.** "Not derivable from its own
  inputs" is not "realistic". The real-data run above tests the protocol, not
  the generator: on 70 real patients the shortcut baseline reaches 0.977,
  which says the synthetic generator's difficulty is not reproduced by real
  capillary data.
- **Fusion is null-to-net-negative, not a win.** Against the best non-fusion
  subset: +0.0046 on the shipped cohort (p = 0.184), and **negative in 4 of 5
  cohorts** (significantly so in one). The EHR stream carries genuine signal in
  this generator (EHR-only: 0.823) — the fused model simply does not convert it
  into a net win. Only a real cohort with a *paired* EHR table can say whether
  it ever will.
- **A large share of the score rests on *who* the patient is, not their current
  state.** The within-patient identity floor is 0.802; on the same hold-out
  split the state component is +0.06. The state share is real but the smaller
  one on this data.
- **The cardiovascular-strain target is weak** (AUC 0.64) and reported as such.
- **The console is a static, precomputed demo** — no auth, audit logging, or
  drift loop; no clinical validity is claimed.

## The clinician console

`dashboard/index.html` is a conceptual UI over precomputed output for six
virtual patients: roster with live risk pills, streams, CGM with realised
spikes, an alerts feed, and a per-feature attribution panel. Un-staged by
construction: one fixed 48-h window per patient, a clinical roster rule, an
alert threshold from a 1%-of-ticks budget (not a hard-coded 0.60), the cohort
alert rate stated, and a synthetic-data / not-for-clinical-use banner.

## Repository structure

```
digital-twin-poc/
├── .github/workflows/ci.yml   # tests + reduced pipeline + freshness checks
├── data/                      # synthetic generators (DIGITAL_TWIN_SEED override)
├── src/                       # config, features, train_*, export, make_report
│                               # validate_real_data.py = external-validation harness
├── model/                     # predictors, metrics.json, seeds/, external_validation_*.json
├── dashboard/                 # template.html (source) + index.html (GENERATED)
├── tests/                     # correctness + leakage + determinism tests
└── docs/                      # RESULTS.md (GENERATED), SEED_ROBUSTNESS.md (GENERATED)
                               # REAL_DATA.md, REVISION_NOTES.md
```

~50 MB of generated artifacts are gitignored and regenerate deterministically:
full-scale `train_model.py` takes ~9 min (recorded in `metrics.json`), the CI
reduced run under 2 min. A 2,000-row sample is committed so reviewers can
inspect the schema without running anything.

## Data, privacy & production

The headline model is fit **only on synthetic data**, so no real patient records are included in the shipped training dataset or model artifact. One external run did use real patient
records — the public, CC BY 4.0 UCI Diabetes cohort, processed entirely
locally; **no raw real data is committed** (`data/real/` is gitignored, and the
credential-gated datasets whose DUAs forbid redistribution are never fetched or
bundled). Only the derived validation metrics are committed.

The schemas are drop-in targets for **MIMIC-IV** / Synthea (EHR) and
**OhioT1DM** / PhysioNet (wearables), and the external harness is already wired
to them (`src/convert_ohio_to_long.py`, `docs/REAL_DATA.md`). Each stage maps
to a production component: `data/` + `feature_engineering.py` play the sources
and feature store, the two `train_*.py` are the inference services, and the
dashboard is the clinician console.

## Provenance: how not to fool yourself

Two earlier versions produced high AUCs that measured the generator, not
physiology:

- **v1** — the diagnosis flag did everything: AUC 0.986, 82% weight on
  `type2_diabetes` (a plain EHR lookup in costume). Caught by the
  *within-patient label permutation*.
- **v2** — algebraically closed: glucose 2 h ahead was an algebraic function
  of its own features, worth 0.953 via the meal-clock shortcut
  (`hour_of_day`, 29.2%). Caught by the *shortcut baseline*.

Every defect and its fix, including two found only by CI's reduced settings, is
in [`docs/REVISION_NOTES.md`](docs/REVISION_NOTES.md).
