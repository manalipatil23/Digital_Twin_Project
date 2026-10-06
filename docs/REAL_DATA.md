# Real-data validation

Two real cohorts, doing two different jobs. Neither is a demo of the harness
running; both are attempts to break a specific claim.

| cohort | n | what it can test | what it cannot |
|---|---|---|---|
| **UCI Diabetes (AIM '94)** | 70 real, 50 scoring | that the protocol runs unchanged on real patients; that engineered glucose context beats a naive current-value baseline | anything about EHR fusion — **no static clinical table** |
| **BIG IDEAs Lab Glycemic Variability and Wearable Device Data** | 16 prediabetic adults | **fusion**, because it pairs CGM + wearable + a static clinical table for the *same* people | population generalisation (16 people); CGM-grade scale |

Numbers are generated into `docs/RESULTS.md` from
`model/external_validation_uci.json`; none are hand-copied. Raw patient data is
never committed (`data/real/` is gitignored); the metrics JSON is.

> **Status of the paired cohort: NOT YET RUN.** The BIG IDEas fetch, converter
> and validation path are implemented and tested, but the download covers 6 of
> 16 participants and `--temporal`/fusion scoring has not been executed on the
> full cohort. There is **no** `model/external_validation_bigideas.json` and no
> BIG IDEas section in `docs/RESULTS.md`. Nothing below reports a fusion result,
> because none has been computed. This section documents the cohort, the
> protocol and the known limitations so the run is auditable when it happens —
> it is deliberately not a result.

## Read this before quoting any number here

| | |
|---|---|
| **It does show** | the whole protocol (resample → features → label → grouped 5-fold CV → patient-level cluster bootstrap) runs unchanged on real patient data. |
| **It does NOT show** | that the synthetic headline transfers. Absolute AUCs on both cohorts are **not** comparable to it — see "Why the absolute numbers are misleading". |
| **It cannot show** | that EHR fusion helps, *if the paired run comes back null*. 16 participants with 2 static features is a very low-power test; a null there is uninformative, not evidence against fusion. See "The power limit on the fusion test". |

---

## Cohort 1 — UCI Diabetes (AIM '94): the protocol rehearsal

70 real type-1 patients, 1991–1993, 13,517 capillary glucose readings, CC BY
4.0, fetched by `src/convert_uci_diabetes.py`. After resampling, **50 of the
70** retained enough contiguous context to score; the rest were dropped by the
strict 2-hour gap tolerance rather than silently filled.

| variant | features | ROC-AUC |
|---|---|---|
| `context` | glucose level, Δ15m, Δ30m, Δ1h, mean/σ 3h, max 6h, hour sin/cos | **0.9826** ± 0.0022 |
| `naive` | glucose level, Δ15m, hour sin/cos only | 0.9768 ± 0.0032 |

**context − naive = +0.0059, p < 0.001** (patient-level cluster bootstrap).

Direction matches synthetic (engineered context > shortcut) and is significant,
but the *margin is small on real data* — 4× smaller than the synthetic +0.0216
over its shortcut control. That is the honest external signal: multi-hour
glucose context adds real information over current-value-plus-trend, and not
very much of it.

**This cohort contributed one thing the rest of the repo could not: proof the
protocol runs on real patients.** Its limitation — no static table — is exactly
what cohort 2 fixes.

---

## Cohort 2 — BIG IDEas Lab: the paired fusion test

This is the cohort the fusion claim needed. Until v3.2 the fusion verdict rested
on synthetic data alone, because every CGM-grade cohort that pairs glucose with
a clinical record (OhioT1DM, MIMIC-IV) is credential-gated and
non-redistributable.

**BIG IDEAs Lab Glycemic Variability and Wearable Device Data**, PhysioNet
v1.1.3, **Open Access under ODC-By v1.0**. The licence is the point: it permits
redistribution and derivative works, so unlike the OhioT1DM/MIMIC DUA it does
not forbid sharing the derived tables. Fetched by
`src/fetch_big_ideas.py` (stdlib only, resumable via HTTP Range, SHA256-verified
against the published `SHA256SUMS.txt`), which writes `provenance.json`
recording DOI, licence and version.

**16 prediabetic adults**, ~9–14 days each, carrying three genuinely paired
streams for the same person on the same clock:

| stream | source | resolution |
|---|---|---|
| Glucose | Dexcom G6 CGM (Clarity export) | 5 min |
| Heart rate + HRV | Empatica E4 (`HR_*.csv`, `IBI_*.csv`) → RMSSD, SDNN, mean IBI | beat-level → 5-min bins |
| Static clinical | `Demographics.csv` → HbA1c, sex | one row per participant |
| Treatment | `Food_Log_*.csv` → logged carbs, calories | per logged meal |

### What was deliberately left out, and why

- **Raw BVP (1.37 GB/participant) and ACC (879 MB/participant).** ~4 GB per
  participant, ~64 GB for the cohort. The E4's derived HR/IBI streams are the
  analysis-ready form of the same physiology; re-deriving RMSSD from raw BVP
  would have spent the entire compute budget reproducing a number this cohort
  already ships.
- **EDA and TEMP.** Not exposed as separate CSV streams in the release.
- **`glucose_roc`, `insulin_u`, `carbs_g` on Dexcom EGV rows.** These are
  **entirely null** — Dexcom's export populates them only on discrete
  Alert/treatment rows, and this cohort has no insulin pump (participants are
  prediabetic and diet-managed). Emitting them would create an all-NaN column,
  and the harness drops any row with a NaN feature, so they would silently
  delete the whole cohort. Carbohydrate comes from the food log instead.
- **Age, from DOB.** The de-identified DOB implies age 30 for every
  participant, which contradicts the published 35–65 inclusion criterion. It is
  therefore both untrustworthy and quasi-identifying, and is excluded.
- **Sex is used, and it contradicts the abstract.** The dataset abstract reads
  as a female-only study; `Demographics.csv` records **7 male / 9 female**. The
  file wins.

### A conversion detail that was a bug, not a feature

`carbs_g` and `calories` are **event streams**, not sampled signals: one row per
logged meal, ~4% of ticks. Left as NaN, the harness's 3-hour rolling mean
(`min_periods=3`) finds no logged meal in almost any window, returns NaN, and
the "drop rows with a NaN feature" rule then deletes ~78% of the cohort. That
is a data-source artefact masquerading as a modelling result. The converter now
writes dense zeros, which is the definitional choice rather than an imputation —
but it changes the feature's meaning from *carbs eaten* to **carbs logged**, so
under-logging is absorbed into the zero. It is a treatment-adherence signal,
not an intake measure.

### HRV coverage is partial, and that biases what survives

The E4 was not worn continuously. HRV is present on 51–87% of ticks depending
on participant, and because the harness requires every feature to be present,
**the surviving rows are biased toward good-wear periods** — the model is
scored on when the device worked, not on a random sample of time. This is
published as `scorable_coverage` in the results JSON, because the alert budget
is denominated in *scorable* patient-days, not wear-days:

```
scorable_coverage: 60.7% of the glucose window (alert rate is per scorable patient-day)
```

A reader who sees "1.4 alerts per patient per day" without that line would
correctly assume it applies around the clock. It does not.

### The label is a rise label on this cohort, not a threshold label

The pre-registered label is "smoothed glucose > 180 mg/dL **or** rises ≥ 30
mg/dL within 2 h". On the synthetic generator both arms fire. On real
prediabetic volunteers the absolute arm is largely inert — participant 001
ranges 46–155 mg/dL (median 105) and never approaches 180 — so essentially
every positive comes from the **rise** arm (16.2% of scored ticks).

**The thresholds were not retuned to suit the cohort.** Doing so would break
comparability with the synthetic run and would shade toward whichever threshold
flatters the result. The inert arm is reported as a finding instead.

---

## The fusion decomposition

With a static table present the harness scores four variants and reports
**three paired comparisons, in both directions**, so a reader cannot infer a
sign from which row was listed first:

| variant | features |
|---|---|
| `context` | CGM-derived only (level, Δ15m/30m/1h, mean/σ 3h, max 6h, hour sin/cos) |
| `static_only` | HbA1c + sex |
| `full` | `context` + `static_only` |
| `naive` | glucose level + Δ15m + hour sin/cos (shortcut control) |

| comparison | question it answers |
|---|---|
| `full_vs_context` | does the static EHR record add anything to the stream? |
| `full_vs_static_only` | does the wearable/CGM stream add anything to the EHR record? |
| `static_only_vs_full` | the same, reversed — reported so the sign cannot be inferred from row order |

A model that wins only one of the first two has fused, not integrated.

### The power limit on the fusion test

This is the most important caveat in the document.

`static_only` is **constant within a patient** — HbA1c and sex do not change
hour to hour. Under patient-grouped CV it therefore cannot learn anything about
within-patient risk at all; at best it learns a between-patient trend from 16
people using 2 features, and is then scored on held-out patients.

So this cohort **can** rule out a large fusion effect. It **cannot** resolve a
small one, and a null result must be reported as *inconclusive*, never as
evidence against fusion. Any claim stronger than "no large effect detected" from
n = 16 would be over-reading the design.

---

## Temporal validation

`--temporal` re-runs the comparison forward in time instead of across random
patients. Each patient is split at a quantile of their **own** observation span
(cuts at 0.5 / 0.6 / 0.7 — quantiles rather than calendar dates because these
cohorts are date-shifted per participant and share no calendar origin); the
earlier part trains, the later part scores.

| | |
|---|---|
| **excludes temporal leakage** | every scored row postdates every training row for that patient |
| **does NOT exclude patient leakage** | each patient is on both sides *by construction* |

Expect the temporal AUC to **exceed** the cross-patient one — the same mechanism
as the within-patient identity floor in the synthetic results. **A high number
here is not evidence of population generalisation.** Only the
`GroupShuffleSplit` result speaks to that, and it is the one to quote.

---

## Why the absolute numbers are misleading

An AUC of ~0.98 on the UCI cohort is *not* a better model than the synthetic
headline. Both differences are properties of the **cohort**, not the model:

1. **Base rate is 54%, not ~1.8%.** Capillary glucose a few times daily,
   interpolated to a 15-min grid, means "spike within 2 h" is usually just the
   next reading after a meal. A near-balanced binary problem is simply easier.
2. **The naive baseline already scores 0.977.** Current glucose + one 15-min
   delta + time-of-day lands within 0.006 of the full feature set. Most of the
   apparent accuracy *is* the current value — exactly the shortcut this project
   has spent three versions trying to detect and control for.

This is the project's own thesis applied to real data: **a high AUC is not
evidence of a good model until a shortcut baseline is run against it.**

---

## Degeneracy guards

Both guards exist because this harness *did* publish a wrong number without
them. They are documented here because a silent NaN is worse than a crash.

| guard | catches |
|---|---|
| `MIN_CLASS_ROWS = 50` | a positive rate inside (0,1) that is still too rare. At 0.01% most folds are single-class, `roc_auc_score` returns NaN, and the bootstrap still prints a confident p-value. |
| `MIN_FOLD_PATIENTS = 2` | **cohort-level balance is not enough.** A cohort can hold 1,424 rows at a 16% positive rate — passing the guard above — and still be unscorable, because `GroupShuffleSplit` moves *whole patients*: with 3 participants each test fold inherits one and comes back single-class. Every AUC is NaN and the bootstrap reports "p < 0.001" beside "delta +nan". Found by running the harness on the real paired cohort at 3 participants. |

Both refuse to write an output file. The second names `--temporal` as the
fallback, since the temporal protocol partitions *within* patients and does not
have this failure mode.

---

## How to run it

```bash
# --- Cohort 1: UCI (public, CC BY 4.0, no credentials) ---
curl -L -o /tmp/diabetes.zip https://archive.ics.uci.edu/static/public/34/diabetes.zip
unzip -q /tmp/diabetes.zip -d /tmp/diabetes
tar -xf /tmp/diabetes/diabetes-data.tar.Z      # old .tar.Z — bsdtar/7-Zip handles LZW
python src/convert_uci_diabetes.py /tmp/diabetes/Diabetes-Data --out data/real/uci_cgm.csv
python src/validate_real_data.py --cgm data/real/uci_cgm.csv --out uci \
    --data-source "UCI Diabetes (AIM '94, Kahn; CC BY 4.0)"

# --- Cohort 2: BIG IDEas Lab (PhysioNet, Open Access, ODC-By v1.0) ---
python src/fetch_big_ideas.py --out data/real/bigideas    # resumable, checksum-verified
python src/convert_big_ideas.py --root data/real/bigideas
python src/validate_real_data.py \
    --cgm    data/real/bigideas_long.csv \
    --static data/real/bigideas_static.csv \
    --temporal --out bigideas \
    --data-source "BIG IDEas Lab Glycemic Variability and Wearable Device Data (PhysioNet v1.1.3; ODC-By v1.0)"
```

`--static` is what enables the `full` and `static_only` variants, and therefore
the fusion comparisons. `--temporal` adds the forward-in-time protocol. Neither
is required, and the UCI path above is unchanged from v3.2.

Output goes to `model/external_validation_<name>.json`, which
`src/make_report.py` renders into `docs/RESULTS.md` automatically.

---

## The long-format schema

One CSV, one row per glucose sample:

| column | required | meaning |
|---|---|---|
| `patient_id` | yes | groups the CV and the cluster bootstrap |
| `timestamp` | yes | any pandas-parseable datetime; normalised to `datetime64[ns]` |
| `glucose_mgdl` (or `cgm_mgdl`, `glucose`, `value`) | yes | glucose in mg/dL |
| `hrv_rmssd_ms`, `sdnn_ms`, `heart_rate_bpm`, `mean_ibi_ms`, `steps`, `sleep_stage` | no | extra dynamic channels; picked up if present |
| `carbs_g`, `calories` | no | **event streams** — sparse by nature; the converter writes dense zeros |

Optionally `--static <csv>` (one row per patient, `patient_id` + any EHR
columns) to enable the fusion variants.

Preprocessing is fixed and stated in the JSON: per-patient 15-min grid, linear
interpolation across gaps ≤ 2 h, shorter-context rows dropped, tail rows
without a full 2-h lookahead dropped. The tolerance is deliberately strict so
long offline stretches are never silently filled.

---

## OhioT1DM and MIMIC-IV

Now secondary. Both would still be the stronger validation — more patients,
institutional CGM, and MIMIC-IV's lab/diagnosis tables — and both remain
unreachable from a sandbox:

- **OhioT1DM** — institutional-email request + signed DUA, or PhysioNet
  credentialed access. The DUA additionally forbids redistributing the raw
  files, so nothing could be committed even with access.
- **MIMIC-IV** — PhysioNet credentialing (CITI course + institutional and
  project approval).

`src/convert_ohio_to_long.py` already parses OhioT1DM's per-patient-day XML
(`<glucoseLevel ts=... value=.../>`) into the same long format, after which the
command is identical. No reported result depends on data nobody can obtain.

---

## Honest summary

| claim | external status |
|---|---|
| protocol transfers to real data | **shown** — two cohorts, end to end |
| engineered context > naive current-value | **shown, small** — +0.0059, p < 0.001 (UCI) |
| absolute AUC ≈ 0.98 transfers | **refuted** — sampling regime and 54% base rate make it incomparable |
| **EHR + wearable fusion helps** | **NOT YET TESTED.** Path is implemented and tested; the paired run has not been executed on the full 16-participant cohort. No number exists. |
| CGM-resolution behaviour | **not yet shown** — BIG IDEas provides genuine 5-min CGM (verifiable from the raw files), but the paired validation has not been run |
| population generalisation | **untested** — the temporal protocol excludes temporal leakage only, never patient leakage |