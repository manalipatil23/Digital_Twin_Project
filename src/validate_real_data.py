"""
validate_real_data.py
---------------------
External-validation harness: run the *same* modelling protocol (grouped
5-fold CV, HGB with the pinned hyper-parameters, patient-level cluster
bootstrap) on REAL glucose time series supplied as one long-format CSV:

    patient_id, timestamp, glucose_mgdl

Optional extra dynamic columns (hrv_rmssd_ms, heart_rate_bpm, steps,
sleep_stage) are picked up if present. An optional static table
(patient_id + any EHR-ish columns, one row per patient) enables the fusion
and static-only variants.

The label and the patient-level significance engine are IDENTICAL to the
synthetic pipeline (30-min-smoothed CGM exceeds 180 mg/dL or rises >= 30
within the next 2 h; 2000-resample cluster bootstrap with a 1/2000 floor),
so an external result is directly comparable to the headline.

A static table (--static) enables the fusion and static-only variants. The
paired BIG IDEAs cohort (see src/convert_big_ideas.py) supplies one, which is
what makes a real-data fusion verdict possible at all; the UCI and OhioT1DM
paths have no static clinical table, so those runs cover the
wearable/glucose-derived stream only. That asymmetry is stated in
docs/REAL_DATA.md rather than quietly omitted.

Usage:
    python src/validate_real_data.py --cgm data/real/cgm.csv [--static data/real/static.csv]
                                       [--out uci] [--data-source "UCI Diabetes (AIM'94)"]
                                       [--temporal]

Rehearsal (CI smoke): point --cgm at any small synthetic long-format file and
--out rehearsal; the JSON is then asserted by tests to be schema-consistent.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

import config as C

TICK_MINUTES = 60 // C.TICKS_PER_HOUR          # 15
HORIZON_TICKS = C.LOOKAHEAD_TICKS              # 2h lookahead
SMOOTH_TICKS = max(1, 30 // TICK_MINUTES)      # 30-min smoothing
MAX_GAP_TICKS = 8                              # 2h of contiguous context required
MIN_ROWS_PER_PATIENT = 50
CV_RS = 4242

# A positive rate inside (0, 1) is not enough for a paired bootstrap to mean
# anything: at a 0.01% rate every fold's test partition can be single-class, and
# roc_auc_score then returns NaN while _bootstrap_pair_p still reports a
# confident-looking p-value off a degenerate label. The synthetic pipeline
# already exits loudly on a degenerate label; this harness must as well.
MIN_CLASS_ROWS = 50

# Minimum distinct patients a cross-patient fold must hold before its AUC is
# trusted. GroupShuffleSplit partitions by patient, so with few participants
# each fold can inherit a single patient and be single-class however many rows
# the cohort holds. Checked per fold in _evaluate, not on the cohort totals.
MIN_FOLD_PATIENTS = 2

# Optional dynamic channels, with the trailing window (15-min ticks) each is
# summarised over. Declared in one place so the converter and this harness
# cannot disagree about what a real cohort is allowed to contribute.
OPTIONAL_DYNAMIC = (
    ("hrv_rmssd_ms", 12),
    ("heart_rate_bpm", 12),
    ("steps", 12),
    # Paired BIG IDEAs streams (Empatica E4 HRV detail + treatment record).
    ("sdnn_ms", 12),
    ("mean_ibi_ms", 12),
    ("carbs_g", 12),
    ("calories", 12),
    ("glucose_roc", 4),
)

# Temporal validation: fraction of each patient's own observation window used
# for fitting. The remainder is scored. Cut points are quantiles of each
# patient's span, not wall-clock dates, because these cohorts are date-shifted.
TEMPORAL_CUTS = (0.5, 0.6, 0.7)


# --------------------------------------------------------------------------
# Loading / resampling
# --------------------------------------------------------------------------
_GLUCOSE_ALIASES = ("glucose_mgdl", "cgm_mgdl", "glucose", "value")


def _glucose_col(df: pd.DataFrame) -> str:
    for name in _GLUCOSE_ALIASES:
        if name in df.columns:
            return name
    raise SystemExit(
        "cgm table needs a glucose column; got "
        + ", ".join(list(df.columns)))


def load_long_format(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"patient_id", "timestamp"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(
            f"{path} is missing required columns {sorted(missing)}; see "
            "docs/REAL_DATA.md for the long-format schema")
    gcol = _glucose_col(df)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)
    df["glucose_mgdl"] = pd.to_numeric(df[gcol], errors="coerce")
    return df[df["glucose_mgdl"].notna()].reset_index(drop=True)


def _resample_to_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Per-patient 15-min grid with linear interpolation across short gaps.

    Gaps longer than MAX_GAP_TICKS * 15 min stay NaN and are dropped (their
    surrounding rows are then treated as separate segments). Long offline
    stretches must not be silently filled.
    """
    chunks = []
    # Normalise the timestamp unit up front. pandas >= 2 defaults to
    # datetime64[us] while pd.date_range yields [ns]; reindexing across the
    # two resolutions matches nothing and silently yields all-NaN columns,
    # which then zeroes the whole row-keep mask. One unit, everywhere.
    df = df.assign(timestamp=df["timestamp"].astype("datetime64[ns]"))
    for pid, grp in df.groupby("patient_id"):
        grp = grp.set_index("timestamp")
        grid = grp[["glucose_mgdl"]].resample(f"{TICK_MINUTES}min").asfreq()
        interp = grid.interpolate(method="linear", limit=MAX_GAP_TICKS,
                                  limit_area="inside")
        interp = interp[interp["glucose_mgdl"].notna()].reset_index()
        extras = [c for c in df.columns
                  if c not in ("patient_id", "timestamp", "glucose_mgdl")]
        if extras:
            key = pd.DatetimeIndex(interp["timestamp"])
            extra_vals = (grp[extras]
                          .reindex(grp.index.union(key))
                          .ffill(limit=MAX_GAP_TICKS)
                          .reindex(key))
            interp[extras] = extra_vals.to_numpy()
        if len(interp) >= MIN_ROWS_PER_PATIENT:
            interp["patient_id"] = pid
            chunks.append(interp)
    if not chunks:
        raise SystemExit(
            "No patient survived resampling with >= 96 rows of contiguous "
            "context. Check the timestamp column and the gap tolerances.")
    return pd.concat(chunks, ignore_index=True)


# --------------------------------------------------------------------------
# Features + label (trailing windows only; mirror of feature_engineering.py)
# --------------------------------------------------------------------------
def _build_table(df: pd.DataFrame, static: pd.DataFrame | None) -> tuple[
        pd.DataFrame, np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    """Return (X, y, group_codes, feature_names, meta).

    ``meta`` carries the kept rows' ``patient_id`` and ``timestamp``. The
    temporal protocol needs to know where each patient's own observation window
    ends, and the alert budget needs real observation time to convert a
    threshold into alerts/patient/day. Neither is a model feature, so both stay
    out of X.
    """
    g = df["glucose_mgdl"].to_numpy(dtype=float)

    def roll(vals, n, agg):
        s = pd.Series(vals)
        return (s.rolling(n, min_periods=3).mean() if agg == "mean"
                else s.rolling(n, min_periods=3).std()).to_numpy()

    hour = df["timestamp"].dt.hour.to_numpy()
    X = {
        "cgm_level": g,
        "cgm_delta_15m": np.concatenate([[np.nan], np.diff(g)]),
        "cgm_delta_30m": np.concatenate([[np.nan, np.nan], np.diff(g, 2)]),
        "cgm_change_1h": np.concatenate([[np.nan] * 4, np.diff(g, 4)]),
        "cgm_mean_3h": roll(g, 12, "mean"),
        "cgm_sd_3h": roll(g, 12, "std"),
        "cgm_max_6h": pd.Series(g).rolling(24, min_periods=3).max().to_numpy(),
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
    }
    for col, win in OPTIONAL_DYNAMIC:
        if col in df.columns:
            X[f"{col}_mean_3h"] = roll(df[col].to_numpy(dtype=float), win,
                                       "mean")
    if "sleep_stage" in df.columns:
        X["sleep_stage"] = (df["sleep_stage"].ffill(limit=MAX_GAP_TICKS)
                            .fillna(0).astype(float).to_numpy())

    # Label: same definition as feature_engineering.build_dynamic_features.
    smoothed = pd.Series(g).rolling(SMOOTH_TICKS, min_periods=1).mean()
    sm = smoothed.to_numpy()
    # Trailing forward max within (i, i+horizon]:
    padded = np.concatenate([sm, np.full(HORIZON_TICKS, sm[-1])])
    fmax = np.array([
        padded[i + 1:i + 1 + HORIZON_TICKS].max() for i in range(len(sm))
    ])
    y = ((fmax > C.SPIKE_ABS_THRESHOLD_MGDL)
         | ((fmax - sm) >= C.SPIKE_RISE_THRESHOLD_MGDL)).astype(np.int8)
    valid = np.ones(len(sm), dtype=bool)
    valid[-HORIZON_TICKS:] = False           # no lookahead at the tail
    valid &= np.isfinite(sm)

    if static is not None:
        static_cols = [c for c in static.columns if c != "patient_id"]
        st = static.set_index("patient_id")
        for c in static_cols:
            X[c] = df["patient_id"].map(st[c]).to_numpy(dtype=float)

    X = pd.DataFrame(X)
    feature_names = [c for c in X.columns]
    keep = valid & X.notna().all(axis=1).to_numpy()
    X = X[keep].reset_index(drop=True)
    y = y[keep]
    groups = pd.factorize(df["patient_id"][keep].to_numpy())[0]
    stamps = df["timestamp"][keep].reset_index(drop=True)
    meta = pd.DataFrame({
        "patient_id": df["patient_id"][keep].to_numpy(),
        "timestamp": stamps.to_numpy(),
    })
    return X, y, groups, feature_names, meta


# --------------------------------------------------------------------------
# Evaluation (same protocol as train_model.py)
# --------------------------------------------------------------------------
def _variant_names(static: pd.DataFrame | None) -> dict[str, list[str]]:
    context = _context_names
    naive = ["cgm_level", "cgm_delta_15m", "hour_sin", "hour_cos"]
    out = {"context": context, "naive": naive}
    if static is not None:
        sc = [c for c in static.columns if c != "patient_id"]
        out["full"] = context + sc
        out["static_only"] = sc
    return out


_context_names: list[str] = []   # set by run_validation before _variant_names


def _evaluate(X, y, groups, variants: dict[str, list[str]],
              static_cols: list[str] | None) -> dict:
    from train_model import _bootstrap_pair_p, _patient_bootstrap_matrix

    gss = GroupShuffleSplit(n_splits=C.N_SPLITS, test_size=C.TEST_SIZE,
                            random_state=CV_RS)
    splits = list(gss.split(X, y, groups=groups))
    # Fail loudly if a fold cannot produce an AUC. Checking the cohort-level
    # class balance (MIN_CLASS_ROWS) is not enough: a cohort can hold thousands
    # of rows and hundreds of positives and STILL produce single-class test
    # folds, because GroupShuffleSplit moves whole patients and a handful of
    # patients means each fold inherits only one of them. When that happens
    # roc_auc_score returns NaN, the mean is NaN, and the paired bootstrap goes
    # on to print a confident p-value next to "delta nan" -- which reads as a
    # result. Catch the degenerate fold here, where the cause is still visible.
    n_test_groups = [len(np.unique(groups[te])) for _, te in splits]
    n_pat = int(groups.max()) + 1
    thin = [i for i, n in enumerate(n_test_groups) if n < MIN_FOLD_PATIENTS]
    single = [i for i, (_, te) in enumerate(splits) if len(np.unique(y[te])) < 2]
    if thin or single:
        worst = single[0] if single else thin[0]
        _, te = splits[worst]
        raise SystemExit(
            f"Cross-patient CV cannot score this cohort: fold {worst} of "
            f"{len(splits)} holds {n_test_groups[worst]} patient(s) and "
            f"{int((y[te] == 1).sum())} positive row(s) out of {len(te)}. "
            f"{'Every ' if len(single) == len(splits) else ''}test partition "
            "is degenerate. GroupShuffleSplit moves whole patients, so a cohort "
            "with too few of them yields single-class folds: roc_auc_score "
            "returns NaN, the fold mean becomes NaN, and the paired bootstrap "
            "then reports a confident p-value against a NaN delta. That is a "
            "silently wrong result, not a weak one. "
            f"test_size={C.TEST_SIZE} x n_splits={C.N_SPLITS} needs at least "
            f"~{MIN_FOLD_PATIENTS * len(splits)} patients for every fold to "
            f"clear MIN_FOLD_PATIENTS={MIN_FOLD_PATIENTS}; this cohort has "
            f"{n_pat}. Use more participants, or score with --temporal, which "
            "partitions within patients and does not have this failure mode.")
    results = {name: [] for name in variants}
    oof = {name: np.zeros(len(X)) for name in variants}
    for tr, te in splits:
        for name, cols in variants.items():
            pipe = _pipeline()
            pipe.fit(X.iloc[tr][cols], y[tr])
            pr = pipe.predict_proba(X.iloc[te][cols])[:, 1]
            results[name].append(float(_roc_auc(y[te], pr)))
            oof[name][te] = pr
    summary = {}
    for name in variants:
        arr = np.array(results[name])
        summary[name] = {
            "roc_auc_mean": float(arr.mean()),
            "roc_auc_std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "roc_auc_per_fold": [float(v) for v in arr],
        }
    deltas = {}
    pairs = [("context", "naive")]
    if static_cols:
        # The two comparisons a fusion claim actually rests on, reported
        # explicitly and in both directions so a reader cannot infer a sign:
        #   full - context    : does the static EHR record add anything?
        #   full - static_only: does the wearable stream add anything?
        # A model that only wins one of these has fused, not integrated.
        pairs += [("full", "context"), ("full", "static_only"),
                  ("static_only", "full")]
    codes = groups
    yf = y.astype(float)
    for a, b in pairs:
        if a not in variants or b not in variants:
            continue
        boot = _patient_bootstrap_matrix(
            codes, yf, {a: oof[a], b: oof[b]}, n_resamples=2000, seed=7)
        p = _bootstrap_pair_p(boot[a], boot[b])
        deltas[f"{a}_vs_{b}"] = {
            "delta_roc_auc": float(float(summary[a]["roc_auc_mean"])
                                   - float(summary[b]["roc_auc_mean"])),
            "bootstrap_p": p,
            "p_rendered": ("p < 0.001" if p < 0.001 else f"p = {p:.3f}")
            if p > 0 else "p < 0.001",
            "n_resamples": 2000,
        }
    return {"variants": summary, "paired_deltas": deltas, "oof": oof}


def _pipeline():
    from train_model import make_pipeline
    return make_pipeline()


# --------------------------------------------------------------------------
# Temporal validation
# --------------------------------------------------------------------------
def _evaluate_temporal(X, y, groups, stamps: pd.Series,
                       variants: dict[str, list[str]]) -> dict:
    """Score each variant forward in time instead of across random patients.

    GroupShuffleSplit answers "does this generalise to OTHER patients". It does
    not answer the question a deployed alert system faces, which is "having
    seen this patient's own history, does the model still predict their next
    few days". Here every patient is split at a quantile of their OWN
    observation span: the earlier part trains, the later part scores.

    Read the result accordingly, because the two protocols are not
    interchangeable:

      * This protocol EXCLUDES TEMPORAL LEAKAGE. Every scored row postdates
        every training row for that patient, so nothing is predicted from its
        own future.
      * This protocol DOES NOT EXCLUDE PATIENT LEAKAGE. Each patient appears
        on both sides of the split by construction, so the model has seen that
        individual's baseline. Expect this AUC to EXCEED the cross-patient one
        -- the same mechanism as the within-patient identity floor in the
        synthetic results. A high number here is therefore not evidence of
        population generalisation; only the GroupShuffleSplit result is.

    Cut points are quantiles rather than calendar dates because these cohorts
    are date-shifted per participant and share no common calendar origin.
    """
    from train_model import _bootstrap_pair_p, _patient_bootstrap_matrix

    per_cut: dict[str, list[float]] = {name: [] for name in variants}
    oof: dict[str, list[float]] = {name: [] for name in variants}
    y_true: list[float] = []
    codes_all: list[int] = []
    n_patients_scored = 0

    for cut in TEMPORAL_CUTS:
        tr, te = [], []
        for pid in np.unique(groups):
            idx = np.flatnonzero(groups == pid)
            t = stamps.iloc[idx].sort_values()
            split_at = t.quantile(cut)
            tr.append(idx[t.values <= split_at])
            te.append(idx[t.values > split_at])
        tr, te = np.concatenate(tr), np.concatenate(te)
        # A cut that leaves one side without both classes cannot be scored.
        if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
            continue
        n_patients_scored += len(np.unique(groups[te]))
        for name, cols in variants.items():
            pipe = _pipeline()
            pipe.fit(X.iloc[tr][cols], y[tr])
            pr = pipe.predict_proba(X.iloc[te][cols])[:, 1]
            per_cut[name].append(float(_roc_auc(y[te], pr)))
            oof[name].append(pr)
        y_true.append(y[te])
        codes_all.append(groups[te])

    if not y_true:
        return {"status": "skipped",
                "reason": ("no temporal cut left both classes present in the "
                           "train and score partitions")}

    summary = {name: {
        "roc_auc_mean": float(np.mean(v)),
        "roc_auc_std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
        "roc_auc_per_cut": [float(x) for x in v],
    } for name, v in per_cut.items() if v}

    y_cat = np.concatenate(y_true)
    codes_cat = np.concatenate(codes_all)
    deltas = {}
    for a, b in (("context", "naive"), ("full", "context"),
                 ("full", "static_only"), ("static_only", "full")):
        if a not in summary or b not in summary:
            continue
        # Compare the two variants on the *same* scored rows. They were fitted
        # on identical training partitions, so a paired test is the correct one.
        boot = _patient_bootstrap_matrix(
            codes_cat, y_cat.astype(float),
            {a: np.concatenate(oof[a]), b: np.concatenate(oof[b])},
            n_resamples=2000, seed=7)
        p = _bootstrap_pair_p(boot[a], boot[b])
        deltas[f"{a}_vs_{b}"] = {
            "delta_roc_auc": float(summary[a]["roc_auc_mean"]
                                   - summary[b]["roc_auc_mean"]),
            "bootstrap_p": p,
            "p_rendered": "p < 0.001" if p < 0.001 else f"p = {p:.3f}",
            "n_resamples": 2000,
        }
    return {
        "status": "ok",
        "protocol": (f"per-patient temporal split at quantiles {TEMPORAL_CUTS} "
                     "of each patient's own observation span; every scored row "
                     "postdates every training row for that patient, but each "
                     "patient appears on BOTH sides by construction, so this "
                     "measures forward prediction for a known individual and "
                     "must not be read as population generalisation"),
        "excludes_temporal_leakage": True,
        "excludes_patient_leakage": False,
        "n_cuts_scored": len(y_true),
        "n_patient_slices_scored": n_patients_scored,
        "n_scored_rows": int(len(y_cat)),
        "variants": summary,
        "paired_deltas": deltas,
    }


def _operating_points(y, pr, groups, stamps: pd.DataFrame,
                      budgets=C.ALERT_BUDGETS) -> dict:
    """Cost the alert: threshold, precision, recall, alerts per patient per day.

    A threshold quoted without a denominator is not actionable, so every budget
    is converted into alerts/patient/day using the real observation time in the
    scored window rather than an assumed row rate.
    """
    y = np.asarray(y)
    pr = np.asarray(pr)
    days = {}
    for pid, grp in stamps.groupby(groups):
        span_h = (grp["timestamp"].max() - grp["timestamp"].min()).total_seconds() / 3600.0
        days[pid] = max(span_h / 24.0, 1e-9)
    total_days = sum(days.values())
    out = {}
    for b in budgets:
        k = int(round(b * len(pr)))
        if k < 1:
            continue
        thr = float(np.sort(pr)[::-1][k - 1])
        fired = pr >= thr
        tp = int(((y == 1) & fired).sum())
        out[f"budget_{b}"] = {
            "budget": b,
            "risk_threshold": round(thr, 4),
            "precision": round(tp / max(int(fired.sum()), 1), 4),
            "recall": round(tp / max(int((y == 1).sum()), 1), 4),
            "alerts_per_patient_day": round(int(fired.sum()) / total_days, 3),
            "observation_patient_days": round(total_days, 2),
        }
    return out


def _roc_auc(y, pr):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, pr)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def run_validation(cgm: pd.DataFrame, static: pd.DataFrame | None,
                   out_name: str, data_source: str,
                   feature_set: str = "auto", temporal: bool = False,
                   temporal_only: bool = False) -> str:
    t0 = time.time()
    df = _resample_to_grid(cgm)
    static_cols = None
    if static is not None:
        static_cols = [c for c in static.columns if c != "patient_id"]
    X, y, groups, _names, meta = _build_table(df, static)

    global _context_names
    _context_names = [c for c in X.columns
                      if not (static_cols and c in static_cols)]
    variants = _variant_names(static)
    # Prune variants to columns that actually exist on this table.
    variants = {k: [c for c in cols if c in X.columns]
                for k, cols in variants.items()}
    variants = {k: cols for k, cols in variants.items() if cols}

    pos_rate = float(y.mean())
    if not (0.0 < pos_rate < 1.0):
        raise SystemExit(
            f"Degenerate positive rate {pos_rate:.3f} after resampling + "
            "labelling. Raise the max-gap tolerance or check the glucose "
            "units.")
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if min(n_pos, n_neg) < MIN_CLASS_ROWS:
        raise SystemExit(
            f"Label is degenerate: {n_pos} positive / {n_neg} negative rows "
            f"(positive rate {pos_rate:.4%}), need >= {MIN_CLASS_ROWS} of each. "
            "A paired cluster bootstrap over so few positives produces NaN AUCs "
            "in most folds while still reporting a p-value, so this stops here "
            "rather than publishing one. On a cohort that never approaches the "
            f"absolute threshold ({C.SPIKE_ABS_THRESHOLD_MGDL:.0f} mg/dL), the "
            "rapid-rise arm is the only criterion that can fire; check the "
            "cohort's glucose distribution before relaxing this.")

    ev = _evaluate(X, y, groups, variants, static_cols)

    # Cost the alert on the pooled out-of-fold predictions, for the variant a
    # clinician would actually run. On a paired cohort that is the fused model;
    # otherwise the best single stream.
    headline_variant = ("full" if "full" in ev["variants"]
                        else max(ev["variants"], key=lambda k: ev["variants"][k]["roc_auc_mean"]))
    ops = _operating_points(y, ev["oof"][headline_variant], groups, meta)

    # The alert denominator is *scorable* time, not wear time: the harness drops
    # any row missing a feature, so on this cohort the budget is conditional on
    # the E4 being worn and producing HRV. Publish the coverage alongside the
    # rate, or "alerts per patient per day" reads as if it applied around the
    # clock. On the UCI rehearsal table coverage is 1.0 and this is inert.
    full_rows = df.groupby("patient_id").size()
    kept_rows = meta.groupby("patient_id").size()
    coverage = {
        "definition": ("fraction of each patient's resampled glucose window on "
                       "which every model feature was available; the alert "
                       "budget denominator is this window, not total wear time"),
        "overall": round(float(kept_rows.sum() / max(int(full_rows.sum()), 1)), 4),
        "per_patient": {str(k): round(float(kept_rows.get(k, 0) / int(v)), 4)
                        for k, v in full_rows.items()},
    }

    temporal_ev = _evaluate_temporal(X, y, groups, meta["timestamp"], variants) if temporal else None

    out = {
        "generated_by": "src/validate_real_data.py",
        "data_source": data_source,
        "out_name": out_name,
        "protocol": (f"{C.N_SPLITS} x GroupShuffleSplit(test_size="
                     f"{C.TEST_SIZE}) grouped by patient_id, "
                     "HistGradientBoostingClassifier with the same "
                     "hyper-parameters and patient-level cluster bootstrap "
                     "(2000 resamples, floor 1/2000) as the synthetic "
                     "pipeline (src/train_model.py)"),
        "tick_minutes": TICK_MINUTES,
        "label_definition": ("30-min-smoothed glucose exceeds 180 mg/dL or "
                             "rises >= 30 mg/dL within the next 2 h; tail "
                             "rows without a full lookahead are dropped"),
        "resampling": ("per-patient 15-min grid, linear interpolation "
                       f"across gaps <= {MAX_GAP_TICKS * TICK_MINUTES} min"),
        "n_patients": int(groups.max()) + 1,
        "n_rows": int(len(y)),
        "positive_rate": round(pos_rate, 4),
        "feature_columns": _context_names,
        "feature_set": feature_set,
        "variants": ev["variants"],
        "paired_deltas": ev["paired_deltas"],
        "alert_operating_points": ops,
        "alert_operating_points_variant": headline_variant,
        "scorable_coverage": coverage,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    if temporal_ev is not None:
        out["temporal_validation"] = temporal_ev
    out_path = f"{C.MODEL_DIR}/external_validation_{out_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {out_path}")
    print(f"  source        : {data_source}")
    print(f"  patients/rows : {out['n_patients']} / {out['n_rows']:,}  "
          f"(positive rate {pos_rate:.2%})")
    if coverage["overall"] < 0.999:
        print(f"  scorable coverage: {coverage['overall']:.1%} of the glucose "
              f"window (alert rate is per scorable patient-day)")
    for name, s in ev["variants"].items():
        print(f"  {name:<28} AUC {s['roc_auc_mean']:.4f} "
              f"+- {s['roc_auc_std']:.4f}")
    for k, d in ev["paired_deltas"].items():
        print(f"  {k:<28} delta {d['delta_roc_auc']:+.4f}  "
              f"({d['p_rendered']})")
    if ops:
        op = ops.get("alert_budget_0.01") or next(iter(ops.values()))
        print(f"  operating point @1% budget : precision {op['precision']:.3f} "
              f"recall {op['recall']:.3f} "
              f"{op['alerts_per_patient_day']:.2f} alerts/patient/day")
    if temporal_ev and temporal_ev.get("status") == "ok":
        print(f"  temporal ({temporal_ev['n_cuts_scored']} cuts, "
              f"{temporal_ev['n_scored_rows']:,} scored rows):")
        for name, s in temporal_ev["variants"].items():
            print(f"    {name:<26} AUC {s['roc_auc_mean']:.4f} "
                  f"+- {s['roc_auc_std']:.4f}")
        for k, d in temporal_ev["paired_deltas"].items():
            print(f"    {k:<24} delta {d['delta_roc_auc']:+.4f} "
                  f"({d['p_rendered']})")
    elif temporal_ev:
        print(f"  temporal: {temporal_ev.get('reason', 'skipped')}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cgm", default=os.environ.get(
        "DIGITAL_TWIN_REAL_DIR") + "/cgm.csv"
        if os.environ.get("DIGITAL_TWIN_REAL_DIR") else "data/real/cgm.csv")
    ap.add_argument("--static", default=None)
    ap.add_argument("--out", default="external_validation")
    ap.add_argument("--data-source", default="real glucose time series")
    ap.add_argument("--temporal", action="store_true",
                    help="also score forward in time within each patient")
    args = ap.parse_args()

    cgm = load_long_format(args.cgm)
    static = None
    if args.static and os.path.exists(args.static):
        static = pd.read_csv(args.static)
    run_validation(cgm, static, args.out, args.data_source,
                   temporal=args.temporal)


if __name__ == "__main__":
    main()
