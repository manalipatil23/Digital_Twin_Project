"""
train_model.py
---------------
Trains the Digital Twin's predictive core: a classifier that fuses STATIC (EHR)
+ DYNAMIC (wearable) features to predict a glucose spike within the next 2 hours.

Five things this script does that v2 did not, each added because its absence
would have made a reported number misleading:

1.  SHORTCUT CONTROL. `SHORTCUT_FEATURE_COLS` is a deliberately weak reference
    model built only from columns available with no wearable stream and no
    HRV/sleep/activity reasoning: the clock, the current CGM reading, its recent
    trend, and the patient's baseline labs. A CGM spike alert fundamentally is
    "where is this patient's glucose right now, relative to where it usually
    is, and is a meal due". If the 28-feature fusion model cannot beat that by a
    real margin, the other 22 features are decoration. In v2 the single most
    important feature was `hour_of_day` and the README never mentioned this
    control existed.

2.  LIFT OVER THE *BETTER* SINGLE STREAM. v2 headlined "+0.20 lift over
    EHR-alone" -- a comparison against the weaker of the two streams, which
    flatters by construction. The meaningful number is full_fusion minus
    max(static_only, dynamic_only, shortcut), and it is now the headline.

3.  CALIBRATION. v2 displayed a risk percentage in the UI and alerted on a 0.6
    threshold while reporting only rank metrics and precision/recall at 0.5. A
    clinician reading "96% risk" is reading a calibrated probability, so Brier
    score, log loss and expected calibration error are now reported.

4.  ALERT-BUDGET OPERATING POINTS. At a 2% base rate a 0.5 threshold is
    meaningless. Clinicians do not get to pick a threshold; they get a noisy
    24/7 stream and a budget of alerts they can absorb. Metrics are reported at
    fixed alert budgets (top 1/5/10/25% of patient-ticks) and converted to
    alerts-per-patient-day.

5.  TWO PERMUTATION CONTROLS. These are leakage tests, not model selection:
      - global label permutation      -> should be ~0.50. Catches pipeline bugs.
      - within-patient label shuffle  -> measures how much of the score comes
        from *patient identity* rather than *current physiology*. v2 put 82% of
        its decision weight on the static `type2_diabetes` column; this control
        is what would have surfaced that as a number.

Outputs:
  model/spike_predictor.joblib     trained pipeline (full_fusion, fit on all data)
  model/metrics.json               cross-validated metrics, all variants + controls
  model/feature_importance.csv     permutation importance of the full_fusion model
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config as C


FEATURE_VARIANTS = {
    "full_fusion":  C.DYNAMIC_FEATURE_COLS + C.STATIC_FEATURE_COLS,
    "dynamic_only": C.DYNAMIC_FEATURE_COLS,
    "static_only":  C.STATIC_FEATURE_COLS,
    # Matched control: shortcut + autonomic channels, nothing else. The delta
    # against `shortcut_control` is a clean isolated estimate of what the
    # wearable physiology channels are worth.
    "shortcut_plus_autonomic": C.SHORTCUT_PLUS_AUTONOMIC_FEATURE_COLS,
    "autonomic_only": C.AUTONOMIC_FEATURE_COLS,
    "shortcut_control": C.SHORTCUT_FEATURE_COLS,
}

# Variants that represent "one stream only" -- used for the honest lift number.
SINGLE_STREAM_VARIANTS = ("dynamic_only", "static_only", "shortcut_control",
                          "autonomic_only", "shortcut_plus_autonomic")

# Patient-level cluster bootstrap resamples. 2000 gives a two-sided resolution
# of 1/2000; CI and quick inspections can override to make iteration faster.
BOOTSTRAP_N_RESAMPLES = int(os.environ.get("DIGITAL_TWIN_BOOTSTRAP_N") or 2000)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Weighted mean gap between confidence and empirical accuracy."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi if hi < 1.0 else y_prob <= hi)
        if not mask.any():
            continue
        ece += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def operating_points(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """PPV / recall at fixed *alert budgets* rather than at an arbitrary 0.5 cut.

    A budget of 0.05 means: alert on the riskiest 5% of patient-ticks. We also
    convert that into alerts-per-patient-day (96 ticks/day), which is the unit a
    care team actually budgets in.
    """
    ticks_per_day = 24 * C.TICKS_PER_HOUR
    out = {}
    order = np.argsort(-y_prob)
    y_sorted = y_true[order]
    cum_pos = np.cumsum(y_sorted)
    for budget in C.ALERT_BUDGETS:
        k = max(1, int(round(budget * len(y_prob))))
        tp = int(cum_pos[k - 1])
        pos_total = int(y_true.sum())
        threshold = float(y_prob[order[k - 1]])
        out[f"alert_budget_{budget:.2f}"] = {
            "budget": budget,
            "risk_threshold": round(threshold, 4),
            "precision": round(tp / k, 4),
            "recall": round(tp / pos_total, 4) if pos_total else 0.0,
            "alerts_per_patient_day": round(budget * ticks_per_day, 1),
        }
    return out


def make_pipeline():
    """Identical hyper-parameters for both predictive targets (see config)."""
    return Pipeline(steps=[
        ("scale", StandardScaler()),
        ("clf", HistGradientBoostingClassifier(
            max_iter=C.MODEL_MAX_ITER,
            max_depth=C.MODEL_MAX_DEPTH,
            learning_rate=C.MODEL_LEARNING_RATE,
            random_state=C.MODEL_RANDOM_STATE,
        )),
    ])


def score_fold(y_test: np.ndarray, proba: np.ndarray) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "average_precision": float(average_precision_score(y_test, proba)),
        "brier": float(brier_score_loss(y_test, proba)),
        "log_loss": float(log_loss(y_test, proba, labels=[0, 1])),
        "ece_10bin": expected_calibration_error(y_test, proba),
    }


def _patient_bootstrap_matrix(codes: np.ndarray, y: np.ndarray,
                              probas: dict[str, np.ndarray],
                              n_resamples: int = 2000, seed: int = 123,
                              max_rows_per_resample: int = 40_000,
                              ) -> dict[str, np.ndarray]:
    """Cluster-bootstrap AUCs, resampling *patients* — not folds, not ticks.

    Legacy criticism (correct): the old bootstrap resampled the 5 fold-wise
    deltas. Five folds are the entire universe, its test sets overlap
    (GroupShuffleSplit per split), and a two-sided sign test on five matched
    pairs cannot go below p = 1/32 no matter how consistent the effect is.
    Claiming "p < 0.001" on that basis was resolution the data did not have.

    Here the resampling unit is the patient (hundreds of them): each resample
    draws `n_patients` patients with replacement, pools *all* of their
    out-of-fold rows, subsamples at most `max_rows_per_resample` of those rows
    for speed, and re-evaluates every variant's AUC on that block. All
    variants share the same out-of-fold rows, so the per-variant AUCs are
    comparable and the differences are paired by construction.

    `codes` must be dense integer patient codes (0 .. n_patients - 1).
    Returns {variant_name: ndarray[n_resamples]} of AUCs.
    """
    codes = np.asarray(codes, dtype=np.int64)
    y = np.asarray(y, dtype=float)
    n_pat = int(codes.max()) + 1 if len(codes) else 0
    order = np.argsort(codes, kind="stable")
    counts = np.bincount(codes, minlength=n_pat)
    bounds = np.cumsum(counts)
    starts = np.zeros(n_pat, dtype=np.int64)
    starts[1:] = bounds[:-1]
    names = list(probas)
    aucs = {name: np.empty(n_resamples) for name in names}
    prev = {name: 0.5 for name in names}
    rng = np.random.default_rng(seed)
    for s in range(n_resamples):
        sampled = np.unique(rng.choice(n_pat, size=n_pat, replace=True))
        if len(sampled):
            idx = np.concatenate([order[starts[p]:bounds[p]] for p in sampled])
        else:
            idx = np.array([], dtype=np.int64)
        if idx.size == 0:
            for name in names:
                aucs[name][s] = prev[name]
            continue
        if idx.size > max_rows_per_resample:
            idx = idx[rng.choice(idx.size, size=max_rows_per_resample, replace=False)]
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            # A resample that happened to draw only negative rows cannot yield
            # an AUC; carry the previous draw rather than fabricate a number.
            for name in names:
                aucs[name][s] = prev[name]
            continue
        for name in names:
            v = roc_auc_score(yy, np.asarray(probas[name])[idx])
            aucs[name][s] = v
            prev[name] = v
    return aucs


def _bootstrap_pair_p(auc_a: np.ndarray, auc_b: np.ndarray) -> float:
    """Two-sided cluster-bootstrap p for H0: mean(AUC_a - AUC_b) == 0.

    The resampling unit is the patient (see `_patient_bootstrap_matrix`), so
    the p-value inherits that bootstrap's resolution (~1/2000 two-sided floor).
    It is explicitly not finer than what the resamples can detect.
    """
    delta = np.asarray(auc_a, dtype=float) - np.asarray(auc_b, dtype=float)
    n = len(delta)
    if n == 0:
        return 1.0
    raw = 2.0 * min((delta <= 0).mean(), (delta >= 0).mean())
    return float(max(min(raw, 1.0), 1.0 / n))


def evaluate_variant(df: pd.DataFrame, feature_cols: list[str], y: np.ndarray,
                     groups: np.ndarray,
                     baseline_aucs: list[float] | None = None) -> tuple[dict, dict]:
    """Cross-validate one feature-set variant on the shared fold set.

    If `baseline_aucs` is supplied (per-fold AUCs for a reference variant), the
    same folds are reused so the per-fold deltas are paired. The per-variant
    significance test is *not* computed here: it needs every variant's
    out-of-fold predictions, which only exist after the whole ablation loop has
    run, and it resamples *patients*, not the 5 folds.
    """
    per_fold, ops, aucs = [], [], []
    oof_ids, oof_y, oof_proba = [], [], []
    for split_i in range(C.N_SPLITS):
        splitter = GroupShuffleSplit(n_splits=1, test_size=C.TEST_SIZE,
                                    random_state=C.BASE_SEED + split_i)
        tr, te = next(splitter.split(df, y, groups=groups))
        pipe = make_pipeline()
        pipe.fit(df.iloc[tr][feature_cols], y[tr])
        proba = pipe.predict_proba(df.iloc[te][feature_cols])[:, 1]
        y_test = y[te]
        per_fold.append(score_fold(y_test, proba))
        aucs.append(per_fold[-1]["roc_auc"])
        ops.append(operating_points(y_test, proba))
        oof_ids.append(groups[te])
        oof_y.append(y_test)
        oof_proba.append(proba)

    summary = {f"{k}_mean": float(np.mean([f[k] for f in per_fold])) for k in per_fold[0]}
    summary.update({f"{k}_std": float(np.std([f[k] for f in per_fold], ddof=1)) for k in per_fold[0]})
    summary["n_folds"] = C.N_SPLITS
    summary["roc_auc_sem"] = float(np.std(aucs, ddof=1) / np.sqrt(C.N_SPLITS))
    summary["roc_auc_per_fold"] = [round(a, 4) for a in aucs]

    if baseline_aucs is not None and len(baseline_aucs) == len(aucs):
        delta = np.array(aucs) - np.array(baseline_aucs)
        summary["delta_roc_auc_vs_reference"] = round(float(delta.mean()), 4)
        # delta_bootstrap_p is filled later, from the patient-level bootstrap.

    summary["operating_points_mean"] = {
        k: {m: round(float(np.mean([o[k][m] for o in ops])), 4) for m in v}
        for k, v in ops[0].items()
    }
    oof = {
        "ids": np.concatenate(oof_ids),
        "y": np.concatenate(oof_y),
        "proba": np.concatenate(oof_proba),
    }
    return summary, oof


# --------------------------------------------------------------------------
# Leakage controls
# --------------------------------------------------------------------------
def _single_fold_auc(df, cols, y, groups, seed):
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=C.TEST_SIZE,
                                    random_state=seed).split(df, y, groups=groups))
    pipe = make_pipeline()
    pipe.fit(df.iloc[tr][cols], y[tr])
    return float(roc_auc_score(y[te], pipe.predict_proba(df.iloc[te][cols])[:, 1]))


def permutation_controls(df: pd.DataFrame, feature_cols: list[str], y: np.ndarray,
                         groups: np.ndarray, headline_cv_mean_auc: float) -> dict:
    """Two leakage tests. Single fold each -- these are sanity checks, not metrics.

    The 'observed' reference is on ONE hold-out split (below), which is
    deliberately not the same evaluation as the 5-fold CV-mean headline of this
    file. A single-split observed AUC that differs from the CV mean is expected;
    recording the protocol here is what stops it being misread as a
    discrepancy between two measurements of the same thing.
    """
    out = {}
    observed = _single_fold_auc(df, feature_cols, y, groups, seed=7)
    out["observed_auc"] = round(observed, 4)
    out["observed_auc_protocol"] = (
        "single GroupShuffleSplit hold-out (seed=7, 25% of patients) -- the "
        "reference split that carries the permutation nulls, and therefore not "
        "the same evaluation as the 5-fold CV mean"
    )
    out["headline_roc_auc_5fold_cv_mean"] = round(float(headline_cv_mean_auc), 4)

    # (a) Global label permutation. Any residual signal here means the pipeline
    #     itself is leaking, independent of anything physiological.
    rng = np.random.default_rng(0)
    y_glob = y.copy()
    rng.shuffle(y_glob)
    out["global_label_permutation_auc"] = round(
        _single_fold_auc(df, feature_cols, y_glob, groups, seed=7), 4)

    # (b) Within-patient label shuffle. Patient identity -- and therefore every
    #     static EHR column, and each patient's own mean base rate -- is left
    #     intact; only the alignment between "when" and "did a spike happen" is
    #     destroyed. Whatever AUC survives is score from knowing WHO the patient
    #     is, not WHAT STATE they are in right now.
    y_wide = pd.DataFrame({"pid": groups, "y": y})
    shuffled = (y_wide.groupby("pid", sort=False)["y"]
                .transform(lambda s: s.sample(frac=1.0, random_state=1).to_numpy()))
    out["within_patient_permutation_auc"] = round(
        _single_fold_auc(df, feature_cols, shuffled.to_numpy(), groups, seed=7), 4)

    out["note"] = (
        "global_label_permutation_auc should be ~0.50. "
        "within_patient_permutation_auc isolates patient-identity memorisation: "
        "a high value means the model is keying on who the patient is rather than "
        "on their current physiological state."
    )
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    os.makedirs(C.MODEL_DIR, exist_ok=True)
    df = pd.read_parquet(f"{C.DATA_DIR}/fused_training_table.parquet")
    y = df["glucose_spike_2h"].values
    groups = df["patient_id"].to_numpy()
    n_patients = pd.Series(groups).nunique()

    print(f"Fused table: {len(df):,} rows, {n_patients} patients, "
          f"positive rate {y.mean():.4f}")
    print(f"CV protocol: {C.N_SPLITS}x GroupShuffleSplit(test_size={C.TEST_SIZE}), "
          f"grouped by patient_id\n")

    # ---- 1) Ablation + controls, all on the identical fold set ----
    # The shortcut control is evaluated first so every other variant can be
    # reported as a PAIRED delta against it on the same folds. Unpaired per-fold
    # std cannot distinguish a real +0.02 from noise at n=5, and a delta quoted
    # without a significance test is exactly the kind of number that should not
    # be believed. The significance test itself resamples PATIENTS (below), not
    # the 5 folds.
    ablation = {}
    oofs = {}
    reference_name = "shortcut_control"
    order = [reference_name] + [k for k in FEATURE_VARIANTS if k != reference_name]

    for name in order:
        cols = FEATURE_VARIANTS[name]
        ref = ablation[reference_name]["roc_auc_per_fold"] if reference_name in ablation else None
        print(f"  {name:23s} ({len(cols):2d} features) ...", flush=True)
        ablation[name], oofs[name] = evaluate_variant(df, cols, y, groups, baseline_aucs=ref)
        row = ablation[name]
        print(f"  {'':23s}    ROC-AUC {row['roc_auc_mean']:.4f} "
              f"+/- {row['roc_auc_std']:.4f}   "
              f"AP {row['average_precision_mean']:.4f}   "
              f"Brier {row['brier_mean']:.4f}   "
              f"ECE {row['ece_10bin_mean']:.4f}")
        if "delta_roc_auc_vs_reference" in row:
            print(f"  {'':23s}    d_vs_shortcut {row['delta_roc_auc_vs_reference']:+.4f}")

    # ---- Patient-level cluster bootstrap over pooled out-of-fold predictions ----
    # The old fold-level bootstrap had at most five independent units, and a
    # two-sided sign test on five matched pairs cannot go below p = 1/32. In
    # other words "p < 0.001" was never supported by resampling five folds,
    # however consistent the deltas. Resampling patients (hundreds of them)
    # earns the reported resolution.
    cluster_codes = pd.factorize(oofs[reference_name]["ids"])[0]
    cluster_y = np.asarray(oofs[reference_name]["y"], dtype=float)
    probas = {name: oofs[name]["proba"] for name in oofs}
    print(f"\n  patient-level cluster bootstrap (resampling patients, not folds, "
          f"from pooled out-of-fold predictions) ...", flush=True)
    boot_auc = _patient_bootstrap_matrix(
        cluster_codes, cluster_y, probas, n_resamples=BOOTSTRAP_N_RESAMPLES)

    boot_n = len(next(iter(boot_auc.values())))
    print(f"  {boot_n} resamples over {len(np.unique(cluster_codes))} patients")
    for name in ablation:
        if "delta_roc_auc_vs_reference" in ablation[name]:
            p = _bootstrap_pair_p(boot_auc[name], boot_auc[reference_name])
            ablation[name]["delta_bootstrap_p"] = round(p, 4)
            print(f"  p {name:23s} vs {reference_name}: "
                  f"{p:.4g} (two-sided, patient-cluster)")

    fusion_auc = ablation["full_fusion"]["roc_auc_mean"]
    best_single_name = max(SINGLE_STREAM_VARIANTS, key=lambda n: ablation[n]["roc_auc_mean"])
    best_single_auc = ablation[best_single_name]["roc_auc_mean"]

    # The deltas that matter, in descending order of usefulness. Every
    # comparison is PAIRED on the shared fold set; the p-value comes from the
    # patient-level cluster bootstrap above, so the resolution it claims is real.
    #
    # Note the reference for each comparison: `static_only` is paired against
    # `shortcut_control` (it is a control, not a competitor), while the
    # fusion-vs-EHR-only figure is a direct pairing of the two models. Reading
    # static_only's delta-vs-shortcut as "fusion over EHR-only" was a labelling
    # error that inverted the sign of the headline flattering comparison.
    def _paired(variant: str, reference: str) -> tuple[float, float]:
        a = np.array(ablation[variant]["roc_auc_per_fold"], dtype=float)
        b = np.array(ablation[reference]["roc_auc_per_fold"], dtype=float)
        d = a - b
        p = _bootstrap_pair_p(boot_auc[variant], boot_auc[reference])
        return float(d.mean()), p

    gain_from_autonomic, p_autonomic = _paired("shortcut_plus_autonomic", "shortcut_control")
    gain_over_shortcut, p_shortcut = _paired("full_fusion", "shortcut_control")
    gain_over_static, p_static = _paired("full_fusion", "static_only")
    gain_over_best_single, p_best_single = _paired("full_fusion", best_single_name)

    print("\n  --- paired feature-value decomposition (identical folds, "
          "patient-level bootstrap p) ---")
    print(f"  {gain_from_autonomic:+.4f} (p={p_autonomic:.3f})  autonomic channels added to the "
          f"matched shortcut control")
    print(f"  {gain_over_shortcut:+.4f} (p={p_shortcut:.3f})  full 28-feature model vs. "
          f"shortcut control  <- the fusion claim")
    print(f"  {gain_over_best_single:+.4f} (p={p_best_single:.3f})  full model vs. best single "
          f"stream ({best_single_name})  <- the unflattering one")
    print(f"  {gain_over_static:+.4f} (p={p_static:.3f})  full model vs. EHR-only  "
          f"(the flattering comparison v2 headlined)")
    significant = p_shortcut < 0.05
    print(f"\n  Fusion claim over the shortcut control is "
          f"{'STATISTICALLY SUPPORTED' if significant else 'NOT statistically supported'} "
          f"at p<0.05 (p={p_shortcut:.3f}).")

    # ---- 2) Leakage controls ----
    print("\n  permutation controls ...", flush=True)
    controls = permutation_controls(df, FEATURE_VARIANTS["full_fusion"], y, groups,
                                    fusion_auc)
    print(f"    observed (single fold, seed 7)  : {controls['observed_auc']:.4f}")
    print(f"      (headline 5-fold CV mean      : {controls['headline_roc_auc_5fold_cv_mean']:.4f})")
    print(f"    global label permutation        : "
          f"{controls['global_label_permutation_auc']:.4f}  (expect ~0.50)")
    print(f"    within-patient label shuffle    : "
          f"{controls['within_patient_permutation_auc']:.4f}  "
          f"(patient-identity memorisation floor)")

    # ---- 3) Deployment model: full_fusion fit on all data ----
    feature_cols = FEATURE_VARIANTS["full_fusion"]
    final_pipeline = make_pipeline()
    final_pipeline.fit(df[feature_cols], y)

    # ---- 4) Permutation importance on a held-out patient sample ----
    _, imp_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.15,
                                        random_state=99).split(df, y, groups=groups))
    imp_sample = df.iloc[imp_idx]
    if len(imp_sample) > C.PERMUTATION_MAX_ROWS:
        imp_sample = imp_sample.sample(C.PERMUTATION_MAX_ROWS, random_state=0)
    perm = permutation_importance(
        final_pipeline, imp_sample[feature_cols], imp_sample["glucose_spike_2h"].values,
        n_repeats=C.PERMUTATION_N_REPEATS, random_state=0, scoring="roc_auc", n_jobs=-1,
    )
    imp_df = pd.DataFrame({"feature": feature_cols, "importance": perm.importances_mean})
    imp_df["importance"] = imp_df["importance"].clip(lower=0)
    total = imp_df["importance"].sum()
    if total > 0:
        imp_df["importance"] = imp_df["importance"] / total
    imp_df = imp_df.sort_values("importance", ascending=False)
    imp_df.to_csv(f"{C.MODEL_DIR}/feature_importance.csv", index=False)

    dyn_share = float(imp_df[imp_df["feature"].isin(C.DYNAMIC_FEATURE_COLS)]["importance"].sum())
    physiology_share = float(imp_df[imp_df["feature"].isin(C.AUTONOMIC_FEATURE_COLS)]
                             ["importance"].sum())
    top_feature = imp_df.iloc[0]
    top_share = float(top_feature["importance"])

    metrics = {
        "generated_by": "src/train_model.py",
        "seed": C.BASE_SEED,
        "n_rows": int(len(df)),
        "n_patients": int(n_patients),
        "positive_rate": float(y.mean()),
        "n_cv_folds": C.N_SPLITS,
        "cv_protocol": f"{C.N_SPLITS} x GroupShuffleSplit(test_size={C.TEST_SIZE}) grouped by patient_id",
        "model": {
            "estimator": "HistGradientBoostingClassifier",
            "max_iter": C.MODEL_MAX_ITER, "max_depth": C.MODEL_MAX_DEPTH,
            "learning_rate": C.MODEL_LEARNING_RATE,
        },
        "label_definition": (
            f"30-min-smoothed CGM exceeds {C.SPIKE_ABS_THRESHOLD_MGDL:.0f} mg/dL, or rises "
            f">= {C.SPIKE_RISE_THRESHOLD_MGDL:.0f} mg/dL, at any point in the next "
            f"{C.LOOKAHEAD_HOURS}h"
        ),
        "lookahead_hours": C.LOOKAHEAD_HOURS,
        "resolution_minutes": 15,
        "ablation": ablation,
        "headline": {
            "roc_auc": fusion_auc,
            "average_precision": ablation["full_fusion"]["average_precision_mean"],
            "brier": ablation["full_fusion"]["brier_mean"],
            "ece_10bin": ablation["full_fusion"]["ece_10bin_mean"],
        },
        "feature_value_decomposition": {
            "_note": ("All deltas are PAIRED against the named reference on the "
                      "identical fold set (a patient's rows never span train and "
                      "test). p-values come from a patient-level cluster "
                      "bootstrap over the pooled out-of-fold predictions (2000 "
                      "resamples, patients are the resampling unit): folds are "
                      "not independent replicates, so resampling five of them "
                      "could never licence a p below 1/32."),
            "autonomic_channels_vs_matched_shortcut": {
                "delta_roc_auc": gain_from_autonomic, "bootstrap_p": p_autonomic},
            "full_model_vs_shortcut_control": {
                "delta_roc_auc": gain_over_shortcut, "bootstrap_p": p_shortcut},
            "full_model_vs_ehr_only": {
                "delta_roc_auc": gain_over_static, "bootstrap_p": p_static},
            "full_model_vs_best_single_stream": {
                "delta_roc_auc": gain_over_best_single,
                "variant": best_single_name, "bootstrap_p": p_best_single,
            },
        },
        "statistical_testing": {
            "paired_delta": "5-fold matched deltas (identical folds per variant)",
            "p_value_method": "patient-level cluster bootstrap over pooled out-of-fold predictions",
            "n_resamples": int(boot_n),
            "resampling_unit": "patient",
            "why_not_fold_bootstrap": (
                "the previous bootstrap resampled the 5 folds; their test sets "
                "overlap (GroupShuffleSplit per split) and a two-sided sign test "
                "on five matched pairs cannot go below p=1/32, so 'p < 0.001' "
                "claimed resolution the folds could not provide"
            ),
            "fold_sign_test_two_sided_min_p": 0.0625,
        },
        "fusion_lift_over_static_only_auc": float(gain_over_static),
        "fusion_lift_over_shortcut_control_auc": float(gain_over_shortcut),
        "fusion_lift_over_best_single_stream_auc": float(gain_over_best_single),
        "fusion_lift_shortcut_bootstrap_p": p_shortcut,
        "fusion_claim_significant_at_05": bool(significant),
        "gain_from_autonomic_channels_auc": float(gain_from_autonomic),
        "best_single_stream_variant": best_single_name,
        "permutation_controls": controls,
        "dynamic_feature_importance_share": dyn_share,
        "physiology_feature_importance_share": physiology_share,
        "top_feature": str(top_feature["feature"]),
        "top_feature_importance_share": round(top_share, 4),
        "n_train_patients": int(round(n_patients * (1 - C.TEST_SIZE))),
        "n_test_patients": int(round(n_patients * C.TEST_SIZE)),
        "runtime_seconds": round(time.time() - t0, 1),
    }
    with open(f"{C.MODEL_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    joblib.dump({"pipeline": final_pipeline, "feature_cols": feature_cols},
                f"{C.MODEL_DIR}/spike_predictor.joblib")

    print("\n=== TOP 10 FEATURES (permutation importance) ===")
    print(imp_df.head(10).to_string(index=False))
    print(f"\n  top feature '{top_feature['feature']}' = {top_share:.1%} of importance")
    print(f"  dynamic share {dyn_share:.3f} | autonomic+activity share {physiology_share:.3f}")
    print(f"  (v2's top feature, hour_of_day, held 29.2% -- the meal-clock shortcut)")
    print(f"  total runtime {metrics['runtime_seconds']}s")
    print(f"\nWrote {C.MODEL_DIR}/metrics.json, feature_importance.csv, spike_predictor.joblib")


if __name__ == "__main__":
    main()
