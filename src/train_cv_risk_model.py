"""
train_cv_risk_model.py
-------------------------
SECOND adverse-event target, present to show the pipeline is not
glucose-specific and generalises to a different horizon, modality and label.

Target: will this patient show a SUSTAINED AUTONOMIC-STRAIN EPISODE in the
next 7 days -- a run of 3+ days where daily-mean HRV falls below their own
21-day baseline (bottom 20th percentile) while resting heart rate stays
elevated (top 30th percentile)? Falling HRV with persistently raised resting
HR is a well-established leading indicator of cardiovascular strain, well
before a hard clinical event.

This model is reported honestly as WEAK. That is the point of including it: a
submission that only ever shows its best number is not demonstrating that an
architecture generalises, it is demonstrating that it can hit one target.
v2 quoted a "not a lookup" guarantee for this label; that claim is not
defensible and has been removed -- the label is a forward-looking function of
each patient's own HRV/HR trajectory, and the same rolling-window features that
predict it are autocorrelated with it, so there is a real risk of fitting the
autocorrelation rather than the physiology. The within-patient permutation
control below is run precisely to measure that risk.

Two caveats stated up front:
  * The HRV/HR baselines (p20 / p70) are computed over each patient's full
    21-day window. That is correct for a retrospective study but it is NOT
    computable online -- a deployed system would have to use a trailing
    baseline and re-estimate. This inflates apparent performance.
  * 14 usable days per patient x 500 patients is a small evaluation set; the
    fold-to-fold standard deviation is correspondingly wide and is reported.

Output:
  model/cv_risk_predictor.joblib
  model/cv_risk_metrics.json
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
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config as C

FEATURE_VARIANTS = {
    "full_fusion": C.CV_DYNAMIC_COLS + C.CV_STATIC_COLS,
    "dynamic_only": C.CV_DYNAMIC_COLS,
    "static_only": C.CV_STATIC_COLS,
}

# Label-invariant controls available in the EHR: age/BMI/diagnoses alone cannot
# know which days a patient's HRV dipped, so this should sit near chance.
SHORTCUT_COLS = ["age", "bmi", "hypertension", "prior_cardiac_event", "type2_diabetes"]


def build_daily_table(wear: pd.DataFrame) -> pd.DataFrame:
    wear = wear.copy()
    wear["date"] = wear["timestamp"].dt.date
    wear["hour"] = wear["timestamp"].dt.hour
    is_night = wear["hour"].between(23, 23) | wear["hour"].between(0, 6)

    resting_hr = (wear[is_night].groupby(["patient_id", "date"])["heart_rate_bpm"]
                  .mean().rename("resting_hr"))
    hrv_daily = (wear.groupby(["patient_id", "date"])["hrv_rmssd_ms"]
                 .mean().rename("hrv_daily_mean"))
    steps_daily = (wear.groupby(["patient_id", "date"])["steps"].sum().rename("steps_daily"))
    sleep_ratio = (wear.groupby(["patient_id", "date"])["sleep_stage"]
                   .apply(lambda s: float(np.mean(np.isin(s, [2, 3]))))
                   .rename("restorative_sleep_ratio"))

    daily = pd.concat([resting_hr, hrv_daily, steps_daily, sleep_ratio], axis=1)
    return daily.reset_index().sort_values(["patient_id", "date"]).reset_index(drop=True)


def add_trend_features_and_label(daily: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for _, g in daily.groupby("patient_id", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        n = len(g)

        g["hrv_3d_trend"] = g["hrv_daily_mean"].diff(3).fillna(0)
        g["hr_3d_trend"] = g["resting_hr"].diff(3).fillna(0)
        g["hrv_7d_mean"] = g["hrv_daily_mean"].rolling(7, min_periods=1).mean()
        g["hr_7d_mean"] = g["resting_hr"].rolling(7, min_periods=1).mean()

        # NOTE: retrospective full-window baseline -- see module docstring.
        hrv_p20 = np.nanpercentile(g["hrv_daily_mean"], C.HRV_BASELINE_PERCENTILE)
        hr_p70 = np.nanpercentile(g["resting_hr"], C.HR_ELEVATED_PERCENTILE)
        strain_day = ((g["hrv_daily_mean"] < hrv_p20) & (g["resting_hr"] > hr_p70)
                      ).astype(int).to_numpy()

        label = np.full(n, -1, dtype=int)
        need = C.STRAIN_LOOKAHEAD_DAYS
        for i in range(n - need):
            window = strain_day[i + 1:i + 1 + need]
            max_run = cur = 0
            for v in window:
                cur = cur + 1 if v else 0
                max_run = max(max_run, cur)
            label[i] = int(max_run >= C.STRAIN_MIN_CONSECUTIVE_DAYS)
        g["cv_strain_7d"] = label
        frames.append(g)

    return pd.concat(frames, ignore_index=True)


def make_pipeline():
    """Same estimator + hyper-parameters as the glucose model (config.py)."""
    return Pipeline(steps=[
        ("scale", StandardScaler()),
        ("clf", HistGradientBoostingClassifier(
            max_iter=C.MODEL_MAX_ITER, max_depth=C.MODEL_MAX_DEPTH,
            learning_rate=C.MODEL_LEARNING_RATE, random_state=C.MODEL_RANDOM_STATE)),
    ])


def _fold_splits(y, groups):
    """The exact train/test partitions used by every 5-fold evaluation.

    Defined once so the label-guard check and the evaluation loop cannot drift
    apart: if either ever changed the splits independently, the guard could
    bless a configuration whose folds actually produce NaN AUCs.
    """
    for split_i in range(C.N_SPLITS):
        splitter = GroupShuffleSplit(n_splits=1, test_size=C.TEST_SIZE,
                                    random_state=C.BASE_SEED + split_i)
        yield split_i, next(splitter.split(np.zeros(len(y)), y, groups=groups))


def _assert_label_is_estimable(y, groups, n_patients, n_days):
    """A degenerate label must fail loudly, never emit NaN AUCs.

    `roc_auc_score` returns NaN (with a warning) when a fold's test partition
    has a single class, so a short reduced series can produce a
    cv_risk_metrics.json full of NaN that the pipeline reports as green -- the
    exact failure this guard exists to prevent. Both problems below were
    observed in real reduced runs (40 patients / 11 days -> 0.0 base rate; and
    40 patients / 14-16 days -> single-class folds despite a ~5-6% rate).
    """
    if float(np.mean(y)) == 0.0:
        raise SystemExit(
            f"\ntrain_cv_risk_model: the cardiovascular-strain label has a 0.0 "
            f"positive rate at {n_patients} patients / {n_days} days "
            f"({len(y):,} label rows) -> every AUC would be NaN.\n"
            f"The geometric floor in config.py is only a lower bound; a small "
            f"reduced cohort needs a longer series or more patients for the "
            f"label to fire at all. Raise DIGITAL_TWIN_DAYS (12+ works at 40 "
            f"patients) and/or DIGITAL_TWIN_N_PATIENTS, then re-run.")
    for split_i, (_, te) in _fold_splits(y, groups):
        if y[te].min() == y[te].max():
            raise SystemExit(
                f"\ntrain_cv_risk_model: CV fold {split_i} has a test partition "
                f"with a single class ({int(y[te].sum())} positives in "
                f"{len(y[te])} rows) -> that fold's AUC is NaN and the reported "
                f"mean silently becomes NaN.\n"
                f"At {n_patients} patients / {n_days} days the label is too thin "
                f"for the {C.N_SPLITS}-fold protocol. Raise DIGITAL_TWIN_DAYS "
                f"and/or DIGITAL_TWIN_N_PATIENTS.")


def evaluate_variant(df, cols, y, groups) -> dict:
    folds = []
    for split_i, (tr, te) in _fold_splits(y, groups):
        pipe = make_pipeline()
        pipe.fit(df.iloc[tr][cols], y[tr])
        proba = pipe.predict_proba(df.iloc[te][cols])[:, 1]
        folds.append({
            "roc_auc": float(roc_auc_score(y[te], proba)),
            "average_precision": float(average_precision_score(y[te], proba)),
            "brier": float(brier_score_loss(y[te], proba)),
        })
    summary = {f"{k}_mean": float(np.mean([f[k] for f in folds])) for k in folds[0]}
    summary.update({f"{k}_std": float(np.std([f[k] for f in folds], ddof=1)) for k in folds[0]})
    summary["n_folds"] = C.N_SPLITS
    return summary


def within_patient_permutation_auc(df, cols, y, groups) -> float:
    wide = pd.DataFrame({"pid": groups, "y": y})
    shuffled = (wide.groupby("pid", sort=False)["y"]
                .transform(lambda s: s.sample(frac=1.0, random_state=1).to_numpy()))
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=C.TEST_SIZE,
                                    random_state=7).split(df, y, groups=groups))
    pipe = make_pipeline()
    pipe.fit(df.iloc[tr][cols], y[tr])
    return float(roc_auc_score(shuffled.to_numpy()[te],
                               pipe.predict_proba(df.iloc[te][cols])[:, 1]))


def main() -> None:
    t0 = time.time()
    ehr = pd.read_csv(f"{C.DATA_DIR}/ehr_patients.csv")
    wear = pd.read_csv(f"{C.DATA_DIR}/wearable_timeseries.csv", parse_dates=["timestamp"])

    daily = add_trend_features_and_label(build_daily_table(wear))
    daily = daily[daily["cv_strain_7d"] >= 0]
    fused = daily.merge(ehr, on="patient_id", how="left", validate="many_to_one")

    y = fused["cv_strain_7d"].values
    groups = fused["patient_id"].values
    print(f"CV-strain daily table: {len(fused):,} rows, {pd.Series(groups).nunique()} patients, "
          f"positive rate {y.mean():.3f}\n")
    _assert_label_is_estimable(y, groups,
                               int(pd.Series(groups).nunique()), C.DAYS)

    variants = dict(FEATURE_VARIANTS)
    variants["shortcut_control"] = SHORTCUT_COLS

    ablation = {}
    for name, cols in variants.items():
        ablation[name] = evaluate_variant(fused, cols, y, groups)
        print(f"  {name:17s} AUC {ablation[name]['roc_auc_mean']:.4f} "
              f"+/- {ablation[name]['roc_auc_std']:.4f}   "
              f"AP {ablation[name]['average_precision_mean']:.4f}")

    ctrl = within_patient_permutation_auc(fused, FEATURE_VARIANTS["full_fusion"], y, groups)
    print(f"\n  within-patient label shuffle: {ctrl:.4f} "
          f"(patient-identity floor; the gap above it is the temporal signal)")

    final = make_pipeline()
    final.fit(fused[FEATURE_VARIANTS["full_fusion"]], y)
    os.makedirs(C.MODEL_DIR, exist_ok=True)
    joblib.dump({"pipeline": final, "feature_cols": FEATURE_VARIANTS["full_fusion"]},
                f"{C.MODEL_DIR}/cv_risk_predictor.joblib")

    best_single = max(("dynamic_only", "static_only", "shortcut_control"),
                      key=lambda n: ablation[n]["roc_auc_mean"])
    metrics = {
        "generated_by": "src/train_cv_risk_model.py",
        "n_rows": int(len(fused)),
        "n_patients": int(pd.Series(groups).nunique()),
        "positive_rate": float(y.mean()),
        "n_cv_folds": C.N_SPLITS,
        "label_definition": (
            f">= {C.STRAIN_MIN_CONSECUTIVE_DAYS} consecutive days within the next "
            f"{C.STRAIN_LOOKAHEAD_DAYS} where daily HRV < patient's own "
            f"p{C.HRV_BASELINE_PERCENTILE} AND resting HR > patient's own "
            f"p{C.HR_ELEVATED_PERCENTILE}"
        ),
        "known_caveats": [
            f"p20/p70 baselines are computed over the full {C.DAYS}-day window; a "
            f"deployed system would need a trailing baseline, which is harder.",
            f"only {C.DAYS - C.STRAIN_LOOKAHEAD_DAYS} usable days per patient "
            f"({len(fused):,} rows total) -- wide fold-to-fold variance is expected.",
        ],
        "ablation": ablation,
        "best_single_stream_variant": best_single,
        "fusion_lift_over_best_single_stream_auc": float(
            ablation["full_fusion"]["roc_auc_mean"] - ablation[best_single]["roc_auc_mean"]),
        "within_patient_permutation_auc": round(ctrl, 4),
        "runtime_seconds": round(time.time() - t0, 1),
    }
    with open(f"{C.MODEL_DIR}/cv_risk_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  fusion lift over best single stream ({best_single}): "
          f"{metrics['fusion_lift_over_best_single_stream_auc']:+.4f}")
    print("Reported honestly as a weak result. The value of this model is that it "
          "exists and is measured, not that it performs well.")
    print(f"Wrote {C.MODEL_DIR}/cv_risk_metrics.json, cv_risk_predictor.joblib "
          f"({metrics['runtime_seconds']}s)")


if __name__ == "__main__":
    main()
