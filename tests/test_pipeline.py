"""
test_pipeline.py
-----------------
Correctness and regression tests. These run on a reduced population so CI stays
fast, but they exercise the same code paths as the full run.

The leakage tests are the important ones. v2 of this project reported 0.953 AUC
on a benchmark whose answer was algebraically derivable from its own features,
and no test would have caught it. `test_no_forward_looking_features`,
`test_tail_rows_are_dropped_not_mislabelled`, `test_reactivity_sd_is_nonzero`
and `test_dashboard_is_self_contained` are the four that would have.

The `test_metrics_are_in_a_plausible_range` guard is a ratchet: it is the
automated expression of the design target stated in the README -- if the task
ever becomes algebraically closed again, the AUC will drift back toward 0.95
and this test fails.

`test_alert_threshold_is_reachable` is the same kind of guard for a different
v2 defect: a hard-coded 0.60 alert cut sat above this model's entire risk
range, so the product could never have alerted anyone.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "data"))

# `train_model` and `export_dashboard_data` are imported inside individual tests
# (they pull in joblib/sklearn), but the dashboard-metadata tests need them too,
# so make the src directory importable for a direct import.

import config as C  # noqa: E402
import feature_engineering as fe  # noqa: E402
import generate_ehr  # noqa: E402
import generate_wearable as gw  # noqa: E402


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def small_cohort():
    """A small but structurally complete cohort: 12 patients, full duration."""
    ehr = generate_ehr.generate_ehr(n_patients=12, seed=1)
    wear, latents = gw.generate_wearable(ehr, seed=1)
    return ehr, wear, latents


@pytest.fixture(scope="module")
def fused(small_cohort):
    ehr, wear, _ = small_cohort
    return fe.build_training_table(ehr, wear)


# --------------------------------------------------------------------------
# Generator
# --------------------------------------------------------------------------
def test_ehr_schema_and_plausible_prevalence():
    ehr = generate_ehr.generate_ehr(n_patients=300, seed=3)
    assert ehr["patient_id"].is_unique
    assert 0.0 < ehr["type2_diabetes"].mean() < 0.35
    assert ehr["hba1c_pct"].between(4.0, 13.0).all()
    assert ehr["fasting_glucose_mgdl"].between(60, 300).all()
    assert ehr["age"].between(18, 90).all()


def test_wearable_shape_and_clinical_ranges(small_cohort):
    _, wear, _ = small_cohort
    assert len(wear) == 12 * C.N_TICKS
    assert wear["heart_rate_bpm"].between(40, 165).all()
    assert wear["hrv_rmssd_ms"].between(9, 125).all()
    assert wear["cgm_mgdl"].between(54, 401).all()
    assert wear["steps"].ge(0).all()
    assert set(np.unique(wear["sleep_stage"])) <= {0, 1, 2, 3}


def test_generator_is_deterministic(small_cohort):
    ehr, _, _ = small_cohort
    a, _ = gw.generate_wearable(ehr.head(3), seed=99)
    b, _ = gw.generate_wearable(ehr.head(3), seed=99)
    pd.testing.assert_frame_equal(a, b)


def test_per_patient_generation_is_order_independent(small_cohort):
    """Generating a subset must reproduce those patients exactly.

    Guards against any dependence on generation order or on a shared RNG
    stream. Drawing a per-patient substream seed from one global generator as
    you iterate the cohort looks independent and is not: the 8th patient in
    the input gets a different stream than the same patient generated 2nd.
    That makes the only way to recover a patient be to regenerate the entire
    cohort in the original order, so a judge re-running a 40-patient slice
    gets different data for the same people and reasonably concludes the
    pipeline is nondeterministic.
    """
    ehr, _, _ = small_cohort

    def rows_for(frame, pid):
        return frame[frame["patient_id"] == pid].reset_index(drop=True)

    full, _ = gw.generate_wearable(ehr, seed=5)

    # Every ordering and every subset must agree with the full-cohort run.
    for label, subset_ids in [
        ("first two", [0, 1]),
        ("reversed", list(range(len(ehr)))[::-1]),
        ("scattered", list(range(0, len(ehr), 2)) + [1]),
        ("single middle patient", [len(ehr) // 2]),
    ]:
        sub, _ = gw.generate_wearable(ehr.iloc[subset_ids], seed=5)
        for pid in sub["patient_id"].unique():
            pd.testing.assert_frame_equal(
                rows_for(full, pid), rows_for(sub, pid),
                obj=f"patient {pid} differs when generated from the {label}",
            )


def test_master_seed_actually_changes_the_data(small_cohort):
    """The per-patient seed must depend on the master seed.

    The order-independence fix derives each patient's stream by hashing
    patient_id together with the master seed. Hashing the id alone would also
    be order-independent -- and would silently make `seed=` a lie, so that
    changing it produces byte-identical data.
    """
    ehr, _, _ = small_cohort
    a, _ = gw.generate_wearable(ehr, seed=5)
    b, _ = gw.generate_wearable(ehr, seed=6)
    assert not a["heart_rate_bpm"].equals(b["heart_rate_bpm"])


def test_patient_seed_is_stable_across_processes():
    """`_patient_seed` must not use Python's salted `hash()`.

    `hash()` on a str is randomised per process (PYTHONHASHSEED), so an id
    hashed that way gives a different cohort on every run. This asserts a
    fixed expected value, which fails if the implementation changes.
    """
    expected = 12410744649777471279
    assert gw._patient_seed("P100000", 7) == expected
    assert gw._patient_seed("P100000", 7) == gw._patient_seed("P100000", 7)
    # Changing either component must change the stream.
    assert gw._patient_seed("P100000", 7) != gw._patient_seed("P100000", 8)
    assert gw._patient_seed("P100000", 7) != gw._patient_seed("P100001", 7)


def test_latents_are_never_leaked_into_the_wearable_stream(small_cohort):
    """The hidden reactive process must not appear in model-visible data."""
    _, wear, latents = small_cohort
    for forbidden in ("latent_metabolic_risk", "latent_cardiac_risk",
                      "reactive_mean", "reactive_sd"):
        assert forbidden not in wear.columns
    assert set(latents.columns) >= {"latent_metabolic_risk", "latent_cardiac_risk",
                                    "reactive_mean", "reactive_sd"}


def test_reactivity_sd_is_nonzero(small_cohort):
    """The irreducible-variance floor must exist.

    v2's failure mode was a deterministic spike term. If this ever goes to ~0
    the benchmark becomes algebraically closed again and AUC silently inflates
    back toward 0.95.
    """
    _, _, latents = small_cohort
    assert (latents["reactive_sd"] > 0.05).all()


def test_exogenous_spikes_occur(small_cohort):
    """Some spikes must have no observable precursor, or there is no ceiling."""
    ehr, wear, _ = small_cohort
    spikes = (wear["cgm_mgdl"] > C.SPIKE_ABS_THRESHOLD_MGDL).sum()
    assert spikes > 0, "no CGM readings cross the alert threshold at all"


def test_diagnosis_flag_does_not_determine_latent_risk(small_cohort):
    """Guards against regressing to the v1 failure (82% weight on one flag)."""
    ehr, _, latents = small_cohort
    merged = latents.merge(ehr, on="patient_id")
    corr = merged["latent_metabolic_risk"].corr(merged["type2_diabetes"])
    assert abs(corr) < 0.6, (
        f"latent risk is too tightly coupled to the diagnosis flag (r={corr:.3f})")


def test_meal_timing_is_jittered_per_patient(small_cohort):
    """A universal 08/13/20 meal clock is what made `hour_of_day` the single
    most important feature in v2. Peak meal hour must vary between patients."""
    _, wear, _ = small_cohort
    peaks = []
    for _, g in wear.groupby("patient_id"):
        hourly = g.groupby(g["timestamp"].dt.hour)["cgm_mgdl"].mean()
        peaks.append(int(hourly[18:23].idxmax()))
    assert len(set(peaks)) > 1, "every patient peaks at the same meal time"


# --------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------
def test_tail_rows_are_dropped_not_mislabelled(fused):
    """The last LOOKAHEAD_TICKS ticks of every patient must be ABSENT.

    v2 dropped only the final row, so the last 8 ticks of each patient were
    labelled from a 1-to-7-tick window while being treated as a full 2-hour
    lookahead. Because the label is a max over future samples, a short window is
    systematically biased toward 'no spike' -- silent label noise.
    """
    per_patient = fused.groupby("patient_id").size()
    expected = C.N_TICKS - C.LOOKAHEAD_TICKS
    assert (per_patient == expected).all(), (
        f"expected {expected} rows/patient, got {sorted(per_patient.unique())}")


def test_label_is_binary_and_non_degenerate(fused):
    assert set(fused["glucose_spike_2h"].unique()) <= {0, 1}
    rate = fused["glucose_spike_2h"].mean()
    assert 0.002 < rate < 0.20, f"implausible positive rate {rate:.4f}"


def test_all_declared_features_are_present(fused):
    missing = [c for c in C.DYNAMIC_FEATURE_COLS + C.STATIC_FEATURE_COLS
               if c not in fused.columns]
    assert not missing, f"missing declared features: {missing}"


def test_feature_sets_have_no_duplicate_columns():
    for name, cols in [
        ("DYNAMIC", C.DYNAMIC_FEATURE_COLS),
        ("STATIC", C.STATIC_FEATURE_COLS),
        ("SHORTCUT", C.SHORTCUT_FEATURE_COLS),
        ("AUTONOMIC", C.AUTONOMIC_FEATURE_COLS),
        ("SHORTCUT+AUTONOMIC", C.SHORTCUT_PLUS_AUTONOMIC_FEATURE_COLS),
    ]:
        assert len(cols) == len(set(cols)), f"{name} has duplicate columns"

    # The matched control must be exactly shortcut + autonomic, and disjoint
    # from it in neither direction.
    assert set(C.SHORTCUT_PLUS_AUTONOMIC_FEATURE_COLS) == (
        set(C.SHORTCUT_FEATURE_COLS) | set(C.AUTONOMIC_FEATURE_COLS))
    assert not set(C.SHORTCUT_FEATURE_COLS) & set(C.AUTONOMIC_FEATURE_COLS)


def test_forward_max_helper_is_correct():
    """Direct unit test of the vectorised lookahead used for labelling."""
    out = fe._trailing_forward_max(np.arange(10, dtype=float), horizon=3)
    assert out[0] == max(1, 2, 3)
    assert out[1] == max(2, 3, 4)
    assert out[5] == max(6, 7, 8)
    assert len(out) == 10


def test_no_forward_looking_features(fused):
    """No feature may be a function of FUTURE CGM.

    Strategy: perturb each patient's CGM *after* a cut point, rebuild, and
    assert nothing before the cut moved. A trailing window is invariant to that
    perturbation; a centred or forward-looking window would shift.
    """
    pid = fused["patient_id"].iloc[0]
    df = fused[fused["patient_id"] == pid].sort_values("timestamp").reset_index(drop=True)

    cut = len(df) // 2
    pre = min(cut, len(df))

    baseline = {c: df[c].to_numpy(dtype=float)[:pre].copy()
                for c in C.DYNAMIC_FEATURE_COLS}

    rng = np.random.default_rng(0)
    perturbed = df.copy()
    noisy = df["cgm_mgdl"].to_numpy(dtype=float).copy()
    noisy[cut:] += rng.normal(0, 40, len(noisy) - cut)
    perturbed["cgm_mgdl"] = noisy
    rebuilt = fe.build_dynamic_features(perturbed).sort_values("timestamp")

    for col in C.DYNAMIC_FEATURE_COLS:
        after = rebuilt[col].to_numpy(dtype=float)[:pre]
        assert np.allclose(baseline[col], after, atol=1e-6, equal_nan=True), (
            f"feature '{col}' changes when FUTURE cgm is perturbed -> forward-looking")


def test_rolling_features_are_strictly_trailing(fused):
    """The last row's 4h rolling mean must not include any post-row value."""
    pid = fused["patient_id"].iloc[0]
    df = fused[fused["patient_id"] == pid].sort_values("timestamp").reset_index(drop=True)
    expected = df["heart_rate_bpm"].iloc[-(C.ROLL_WINDOW_TICKS):].mean()
    assert abs(df["heart_rate_bpm_roll_mean"].iloc[-1] - expected) < 1e-6


def test_no_rolling_window_crosses_a_patient_boundary(fused):
    """A per-patient trailing window must not bleed into a neighbouring patient.

    `test_no_forward_looking_features` uses a single patient, so it would not
    catch a groupby that silently sorted incorrectly across patients.
    """
    for pid, g in fused.groupby("patient_id"):
        g = g.sort_values("timestamp").reset_index(drop=True)
        if len(g) < C.ROLL_WINDOW_TICKS:
            continue
        row = g.iloc[C.ROLL_WINDOW_TICKS - 1]
        expected = g["hrv_rmssd_ms"].iloc[:C.ROLL_WINDOW_TICKS].mean()
        assert abs(row["hrv_rmssd_ms_roll_mean"] - expected) < 1e-6, \
            f"rolling window crossed a patient boundary at {pid}"


# --------------------------------------------------------------------------
# Metric helpers
# --------------------------------------------------------------------------
def test_expected_calibration_error_bounds():
    from train_model import expected_calibration_error as ece
    assert ece(np.array([0, 0, 1, 1]), np.array([0.0, 0.0, 1.0, 1.0])) == pytest.approx(0.0)
    assert ece(np.array([0, 1]), np.array([1.0, 0.0])) == pytest.approx(1.0)
    assert 0.0 <= ece(np.array([0, 1, 1, 0]), np.array([0.3, 0.7, 0.2, 0.9])) <= 1.0


def test_operating_point_respects_the_budget():
    from train_model import operating_points
    rng = np.random.default_rng(0)
    y = (rng.random(10_000) < 0.02).astype(int)
    proba = y * 0.6 + (1 - y) * 0.2 + rng.random(10_000) * 0.01
    ops = operating_points(y, proba)
    assert set(ops) == {f"alert_budget_{b:.2f}" for b in C.ALERT_BUDGETS}
    for budget, op in ops.items():
        assert op["budget"] == pytest.approx(float(budget.split("_")[-1]))
        assert op["alerts_per_patient_day"] == pytest.approx(
            op["budget"] * 24 * C.TICKS_PER_HOUR, abs=0.05)
        assert 0.0 <= op["precision"] <= 1.0
        assert 0.0 <= op["recall"] <= 1.0
        assert 0.0 < op["risk_threshold"] <= 1.0


def test_operating_points_are_monotone_in_budget():
    """A larger alert budget must never reduce recall."""
    from train_model import operating_points
    rng = np.random.default_rng(1)
    y = (rng.random(20_000) < 0.03).astype(int)
    proba = np.clip(y * 0.5 + (1 - y) * 0.25 + rng.random(20_000) * 0.02, 0, 1)
    recalls = [operating_points(y, proba)[f"alert_budget_{b:.2f}"]["recall"]
               for b in sorted(C.ALERT_BUDGETS)]
    assert all(b >= a for a, b in zip(recalls, recalls[1:])), recalls


# --------------------------------------------------------------------------
# Configuration integrity
# --------------------------------------------------------------------------
def test_config_consistency():
    assert C.LOOKAHEAD_TICKS == C.LOOKAHEAD_HOURS * C.TICKS_PER_HOUR
    assert C.ROLL_WINDOW_TICKS == C.ROLL_WINDOW_HOURS * C.TICKS_PER_HOUR
    assert C.N_TICKS == C.DAYS * 24 * C.TICKS_PER_HOUR
    assert C.MODEL_MAX_ITER and C.MODEL_MAX_DEPTH and C.MODEL_LEARNING_RATE


def test_both_targets_share_hyperparameters():
    """v2 gave the two models different settings, making its runtime note ambiguous."""
    import train_model as tm
    import train_cv_risk_model as tcv
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier

    a = tm.make_pipeline().named_steps["clf"]
    b = tcv.make_pipeline().named_steps["clf"]
    for attr in ("max_iter", "max_depth", "learning_rate", "random_state"):
        assert getattr(a, attr) == getattr(b, attr), (
            f"hyper-parameter '{attr}' differs between the two targets: "
            f"{getattr(a, attr)} vs {getattr(b, attr)}")
        assert getattr(a, attr) == getattr(C, f"MODEL_{attr.upper()}")


# --------------------------------------------------------------------------
# Committed artifacts (skipped until the pipeline has been run)
# --------------------------------------------------------------------------
def _load(name):
    path = f"{C.MODEL_DIR}/{name}"
    if not os.path.exists(path):
        pytest.skip(f"{name} not present; run the pipeline first")
    with open(path) as f:
        return json.load(f)


def test_metrics_are_in_a_plausible_range():
    """RATCHET: enforces the README's design target.

    The v2 failure was a benchmark whose answer was algebraically derivable
    from its own features, which produced AUC 0.953. If a future change closes
    the loop again, the AUC will climb back toward that value and this fails.
    """
    m = _load("metrics.json")
    auc = m["headline"]["roc_auc"]
    assert 0.65 < auc < 0.93, (
        f"headline AUC {auc:.4f} outside the plausible band. Above ~0.93 usually "
        f"means the task has become algebraically closed again -- see README 2.")


def test_top_feature_is_not_the_meal_clock():
    """RATCHET: `hour_of_day` must not be the model's dominant feature.

    v2's giveaway was that its top feature by a wide margin was the clock:
    29.2% importance, because the generator's spikes were a deterministic
    function of time-of-day. The v3 generator jitters meal times, skips meals
    and adds spikes with no observable precursor, so the clock should be a
    genuine but partial prior.

    This is a separate guard from the AUC ratchet on purpose. A model can sit
    at a respectable 0.85 AUC while still being mostly a clock with some
    physiology bolted on, and the AUC band would never notice.
    """
    imp = pd.read_csv(f"{C.MODEL_DIR}/feature_importance.csv")
    top_share = imp["importance"].max()
    hod = imp.loc[imp["feature"] == "hour_of_day", "importance"]
    hod_share = float(hod.iloc[0]) if len(hod) else 0.0

    assert imp.loc[imp["importance"].idxmax(), "feature"] != "hour_of_day", (
        "the meal clock is the top feature again -- the generator's spikes are "
        "probably a deterministic function of time-of-day again")
    assert hod_share < 0.20, (
        f"hour_of_day holds {hod_share:.1%} of importance, over the 20% ceiling; "
        f"top feature is {imp.loc[imp['importance'].idxmax(), 'feature']} at "
        f"{top_share:.1%}")


def test_report_agrees_with_metrics_json():
    """The generated report must not contradict the numbers it was built from.

    v2 quoted 0.953 in one section of its README and 0.955 in another, neither
    matching `metrics.json`. Regenerating the report and diffing it in CI is the
    fix; this asserts the report is currently fresh.
    """
    report_path = f"{C.ROOT}/docs/RESULTS.md"
    if not os.path.exists(report_path):
        pytest.skip("docs/RESULTS.md not present; run src/make_report.py first")
    m = _load("metrics.json")
    text = open(report_path, encoding="utf-8").read()
    for key in ("full_fusion", "shortcut_control", "dynamic_only", "static_only"):
        share = f"{m['ablation'][key]['roc_auc_mean']:.4f}"
        assert share in text, (
            f"docs/RESULTS.md does not contain the {key} AUC {share} from "
            f"metrics.json -- the report is stale, re-run src/make_report.py")


def test_seed_report_headline_is_the_mean_not_one_draw():
    """The headline must be the mean over generator seeds, not a single draw.

    §19 of docs/REVISION_NOTES.md established that the canonical cohort is the
    *maximum* of the five-seed AUC range. v3.1/v3.2 documented that and still
    led with 0.876, so the first number on the README remained a post-hoc
    best-of-five -- which is exactly what the reviewer quoted back.

    This asserts three things, because any one alone is satisfiable by a
    plausible-looking edit:
      1. the mean over `model/seeds/*.json` appears in SEED_ROBUSTNESS.md and
         in the README;
      2. the headline is explicitly labelled as a mean over seeds;
      3. the mean is NOT any individual seed's AUC -- if the sweep ever
         collapses to a single draw, this fails rather than passing silently.
    """
    import glob

    paths = sorted(glob.glob(f"{C.MODEL_DIR}/seeds/metrics_*.json"))
    if not paths:
        pytest.skip("no model/seeds/*.json; run the seed sweep first")

    # Load by full path. `_load()` resolves against C.MODEL_DIR and calls
    # pytest.skip() when the file is absent, so passing it a bare basename here
    # silently skipped this whole test instead of running it -- a green build
    # on a guard that never executed, which is REVISION_NOTES §13.
    aucs = []
    for p in paths:
        assert os.path.exists(p), f"glob returned a nonexistent path: {p}"
        with open(p, encoding="utf-8") as f:
            aucs.append(json.load(f)["ablation"]["full_fusion"]["roc_auc_mean"])

    assert len(aucs) >= 2, (
        "the headline cannot be a mean over seeds if only one snapshot exists")
    mean = sum(aucs) / len(aucs)

    seed_doc = open(f"{C.ROOT}/docs/SEED_ROBUSTNESS.md", encoding="utf-8").read()
    readme = open(f"{C.ROOT}/README.md", encoding="utf-8").read()

    # SEED_ROBUSTNESS.md prints 4 dp; the README rounds to 3 for readability.
    # Both precisions are the generators' own choice -- mirror them rather
    # than inventing a third.
    assert f"{mean:.4f}" in seed_doc, (
        f"SEED_ROBUSTNESS.md does not quote the seed mean {mean:.4f}")
    assert f"{mean:.3f}" in readme, (
        f"README does not quote the seed mean {mean:.3f} as its headline -- if "
        "this regressed to a single draw, the number being quoted is a "
        "selection, not an estimate")
    assert re.search(r"mean\s*[±+]?\s*sd over", seed_doc, re.I), (
        "SEED_ROBUSTNESS.md must label the headline as a mean over seeds")

    for a in aucs:
        assert abs(a - mean) > 1e-9, (
            f"the seed mean {mean:.4f} equals one seed's AUC ({a:.4f}); with "
            "this many draws the headline is a single draw wearing a mean's "
            "label")


def test_strain_label_geometry_has_a_minimum_series_length():
    """The secondary label is impossible on a short series.

    It asks for STRAIN_MIN_CONSECUTIVE_DAYS qualifying days inside a
    STRAIN_LOOKAHEAD_DAYS future window, so a series shorter than
    LOOKAHEAD + MIN_CONSECUTIVE can never contain a positive example. CI used
    to run 10 days, which produced a 0.0 base rate and NaN for every AUC in
    cv_risk_metrics.json -- a green build on a silently meaningless result.
    """
    assert C.STRAIN_MIN_DAYS_REQUIRED == (
        C.STRAIN_LOOKAHEAD_DAYS + C.STRAIN_MIN_CONSECUTIVE_DAYS + 1)
    assert C.DAYS >= C.STRAIN_MIN_DAYS_REQUIRED, (
        f"DAYS={C.DAYS} is below the {C.STRAIN_MIN_DAYS_REQUIRED}-day minimum for "
        f"the strain label, so its base rate will be 0.0 and all AUCs NaN")


def test_too_short_series_is_rejected_loudly():
    """A too-short reduced run must fail with an explanation, not emit NaNs."""
    import subprocess
    env = dict(os.environ)
    env["DIGITAL_TWIN_DAYS"] = str(C.STRAIN_MIN_DAYS_REQUIRED - 1)
    # Put src/ on sys.path inside the child rather than relying on PYTHONPATH,
    # which is not reliably inherited (and is ignored outright under -E / -I).
    proc = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.path.insert(0, {os.path.join(ROOT, 'src')!r}); "
         "import config"],
        env=env, capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode != 0, "a too-short DIGITAL_TWIN_DAYS was accepted"
    assert "too short" in (proc.stdout + proc.stderr), (
        "the failure should name the problem, not just exit non-zero")


def test_committed_cv_risk_metrics_are_not_nan():
    """NaN AUCs in a committed artifact mean the reduced run overwrote it."""
    cv = _load("cv_risk_metrics.json")
    for key, row in cv["ablation"].items():
        for metric in ("roc_auc_mean", "average_precision_mean"):
            value = row.get(metric)
            assert value == value, (  # NaN != NaN
                f"cv_risk_metrics.json: {key}.{metric} is NaN, which means the "
                f"series was too short to produce a single positive label")
    assert cv["positive_rate"] > 0.0, (
        "cv_risk_metrics.json has a 0.0 positive rate -- no usable labels were "
        "generated, so every AUC in it is meaningless")


def test_cv_risk_label_guard_is_loud():
    """A degenerate strain label must fail loudly, never emit NaN AUCs.

    `roc_auc_score` returns NaN (with a warning) when a fold's test partition
    has a single class, so a short reduced series writes a cv_risk_metrics.json
    full of NaN that the pipeline reports as green. Real reduced runs hit both
    shapes below: 40 patients / 10-11 days gave a 0.0 base rate, and 40
    patients / 14-16 days gave single-class folds despite a ~5-6% rate. The
    guard exits with an explanation before any metric is computed.
    """
    import train_cv_risk_model as tcvm

    # Branch 1: identically-zero label (the 40-patient / 11-day case).
    y0 = np.zeros(40)
    g0 = np.array([f"P{i % 10}" for i in range(40)])
    with pytest.raises(SystemExit, match="positive rate"):
        tcvm._assert_label_is_estimable(y0, g0, n_patients=10, n_days=11)

    # Branch 2: positives exist cohort-wide but a fold's test partition lands
    # on no positive patient. A sole positive row guarantees any test partition
    # that excludes its group is single-class; with the fixed BASE_SEED splits
    # this fire of `_fold_splits` trips fold 1 deterministically.
    y1 = np.zeros(80)
    y1[0] = 1  # the only positive, in group 0
    g1 = np.array([f"P{i % 20}" for i in range(80)])
    with pytest.raises(SystemExit, match="single class"):
        tcvm._assert_label_is_estimable(y1, g1, n_patients=20, n_days=12)


def test_permutation_control_is_near_chance():
    """Global label permutation must be ~0.50, else the pipeline leaks."""
    m = _load("metrics.json")
    pc = m["permutation_controls"]
    assert 0.42 < pc["global_label_permutation_auc"] < 0.58, (
        f"global label permutation gave {pc['global_label_permutation_auc']:.4f}; "
        f"anything clearly above 0.5 means the pipeline leaks.")


def test_shortcut_control_is_reported_and_substantially_beaten():
    """The shortcut baseline must exist and the full model must beat it.

    v2 never ran this control, which is how a meal-clock shortcut went
    unexamined.
    """
    m = _load("metrics.json")
    ab = m["ablation"]
    for key in ("shortcut_control", "shortcut_plus_autonomic", "full_fusion"):
        assert key in ab, f"missing required control variant '{key}'"
    assert ab["full_fusion"]["roc_auc_mean"] > ab["shortcut_control"]["roc_auc_mean"], (
        "the full model does not beat the shortcut control")


def test_fusion_claim_is_a_tested_paired_comparison():
    """The headline claim must be a paired test, not a subtraction of means.

    A positive delta on its own is not a result: at 5 folds the fold-to-fold
    spread is wide enough that a real-but-tiny gain and pure noise look
    identical. `metrics.json` has to carry the paired delta AND its bootstrap
    p-value, and the README/report's "significant" wording depends on the
    latter. Without this, a future edit could quietly downgrade the claim to an
    untested difference while the prose still called it supported.
    """
    m = _load("metrics.json")
    dec = m["feature_value_decomposition"]
    claim = dec["full_model_vs_shortcut_control"]
    for field in ("delta_roc_auc", "bootstrap_p"):
        assert field in claim, f"fusion claim is missing '{field}'"
    assert claim["delta_roc_auc"] > 0, "fusion claim is not positive"
    assert claim["bootstrap_p"] < 0.05, (
        f"fusion claim p={claim['bootstrap_p']} is not significant, so the "
        f"report must not describe it as supported")
    assert m["fusion_claim_significant_at_05"] is True

    # Every reported comparison must carry a usable p-value. A p of exactly 0.0
    # is a legitimate bootstrap outcome (all resamples fell on one side) but
    # must never be printed as "p = 0.000", which reads as an impossibility.
    for name, row in dec.items():
        if not isinstance(row, dict) or "delta_roc_auc" not in row:
            continue
        p = row.get("bootstrap_p")
        if p is not None:
            assert 0.0 <= p <= 1.0, f"{name} has an out-of-range p-value {p}"


def test_report_never_prints_p_equal_to_zero():
    """A bootstrap p of 0.000 must be rendered as a bound, not as a number.

    "p = 0.000" is a statement that cannot be true, and printing it invites a
    reader to think the test is infinitely precise. The honest rendering is
    "p < 0.001" -- the resolution implied by 10,000 resamples.
    """
    report = f"{C.ROOT}/docs/RESULTS.md"
    if not os.path.exists(report):
        pytest.skip("docs/RESULTS.md not present; run src/make_report.py first")
    text = open(report, encoding="utf-8").read()
    assert "p = 0.000" not in text, (
        "docs/RESULTS.md prints 'p = 0.000'; render it as a bound instead")
    assert "p = 0.00 " not in text.replace("p = 0.000", "")


def test_calibration_metrics_are_reported():
    """A UI that displays a risk percentage must report calibration."""
    m = _load("metrics.json")
    h = m["headline"]
    for key in ("brier", "ece_10bin", "average_precision"):
        assert key in h and 0.0 <= h[key] <= 1.0


def test_feature_importance_is_not_dominated_by_one_column():
    """v2's top feature was `hour_of_day` at 29%, i.e. it learned the meal clock."""
    path = f"{C.MODEL_DIR}/feature_importance.csv"
    if not os.path.exists(path):
        pytest.skip("feature_importance.csv not present; run the pipeline first")
    imp = pd.read_csv(path)
    total = imp["importance"].sum()
    assert total > 0
    top = imp.sort_values("importance", ascending=False).iloc[0]
    assert top["importance"] < 0.30, (
        f"single feature '{top['feature']}' holds {top['importance']:.1%} of the "
        f"model's importance -- likely a shortcut (v2's `hour_of_day` was 29.2%)")


# --------------------------------------------------------------------------
# Dashboard
# --------------------------------------------------------------------------
def test_dashboard_is_self_contained():
    """v2 claimed self-containment while pulling Chart.js and fonts from CDNs.

    It rendered blank offline, which is exactly when you would want to demo it.
    CI enforces the same check; this makes it catchable locally too.
    """
    path = f"{C.DASHBOARD_DIR}/template.html"
    if not os.path.exists(path):
        pytest.skip("template.html not present")
    with open(path, encoding="utf-8") as f:
        html = f.read()
    remote = re.findall(r'(?:src|href)="https?://[^"]+"', html)
    assert not remote, f"dashboard references remote resources: {remote}"


def test_dashboard_template_has_data_placeholder():
    path = f"{C.DASHBOARD_DIR}/template.html"
    if not os.path.exists(path):
        pytest.skip("template.html not present")
    with open(path, encoding="utf-8") as f:
        assert "__DEMO_DATA__" in f.read()


def test_exported_dashboard_agrees_with_its_metadata():
    """The rendered demo must use the fixed window rule, not a peak search."""
    path = f"{C.DASHBOARD_DIR}/demo_data.json"
    if not os.path.exists(path):
        pytest.skip("demo_data.json not present; run the pipeline first")
    with open(path) as f:
        bundle = json.load(f)
    assert bundle["_meta"]["synthetic_data"] is True
    assert bundle["_meta"]["not_for_clinical_use"] is True
    assert "never selected on model output" in bundle["_meta"]["patient_selection_rule"]


def test_demo_patient_selection_is_deterministic_and_model_free():
    """The roster must be a function of the EHR alone, reproducible exactly.

    v2's central UI sin was choosing which patients to show based on what the
    model predicted. The strongest available guard is to check that the
    selection function takes the EHR and nothing else, and returns the same
    answer every time -- including when the EHR columns are reordered, since a
    rule that secretly depends on row order is still a selection rule, just an
    undeclared one.
    """
    from export_dashboard_data import pick_demo_patients

    ehr = pd.read_csv(f"{C.DATA_DIR}/ehr_patients.csv")
    first = pick_demo_patients(ehr)
    assert first == pick_demo_patients(ehr), "roster is not reproducible"
    assert len(first) == C.N_DEMO_PATIENTS, (
        f"roster has {len(first)} patients, expected {C.N_DEMO_PATIENTS}")

    shuffled = ehr.sample(frac=1.0, random_state=7)
    assert pick_demo_patients(shuffled) == first, (
        "roster depends on the order of rows in ehr_patients.csv")

    assert len(set(first)) == len(first), "roster contains duplicates"

    # The diabetics it picked must be the most severe ones present.
    diabetic = ehr[ehr["type2_diabetes"] == 1].sort_values(
        "hba1c_pct", ascending=False)
    expected = set(diabetic.head(C.N_DEMO_PATIENTS // 2)["patient_id"])
    assert expected.issubset(set(first)), (
        "the highest-HbA1c diabetics are not all in the roster")


def test_exported_dashboard_payload_is_well_formed():
    """Every exported patient must carry the fields the UI dereferences.

    The dashboard reads these unconditionally; a missing key is a blank panel
    in the browser and an unhelpful stack trace at best.
    """
    path = f"{C.DASHBOARD_DIR}/demo_data.json"
    if not os.path.exists(path):
        pytest.skip("demo_data.json not present; run the pipeline first")
    with open(path) as f:
        bundle = json.load(f)
    assert 0.0 <= bundle["cohort"]["alert_rate_all_ticks"] <= 1.0
    assert bundle["cohort"]["alerts_per_patient_day"] > 0.0
    for p in bundle["patients"]:
        assert "contributions" in p and p["contributions"]
        assert "realised_spike_label" in p
        assert p["n_alerts_in_window"] >= len(p["alerts"]), (
            f"{p['patient_id']}: the alert list holds more entries than the "
            f"window total it is a sample of")
        assert p["risk_level"] in ("low", "moderate", "high")


# --------------------------------------------------------------------------
# Alert threshold (v2 hard-coded 0.60, which never fired)
# --------------------------------------------------------------------------
def test_threshold_for_budget_hits_the_declared_budget():
    from export_dashboard_data import threshold_for_budget
    rng = np.random.default_rng(0)
    risk = rng.random(100_000)
    for budget in (0.01, 0.05, 0.20, 0.25):
        cut = threshold_for_budget(risk, budget)
        assert (risk >= cut).mean() == pytest.approx(budget, abs=1e-3)


def test_alert_threshold_is_reachable():
    """v2 hard-coded 0.60 above this model's entire risk range.

    A cut that sits above the 99th percentile of predicted risk can essentially
    never fire, which is a broken alert rule rather than a conservative one.
    This is the guard against regressing to it.
    """
    path = f"{C.DASHBOARD_DIR}/demo_data.json"
    if not os.path.exists(path):
        pytest.skip("demo_data.json not present; run the pipeline first")
    with open(path) as f:
        cohort = json.load(f)["cohort"]
    assert cohort["alert_threshold"] < cohort["p99_risk_all_ticks"] * 1.05, (
        f"alert threshold {cohort['alert_threshold']} is at or above the 99th "
        f"percentile of risk ({cohort['p99_risk_all_ticks']}) -- unreachable")
    assert cohort["alert_rate_all_ticks"] > 0.0, "alert rule never fires at all"


def test_alert_rate_matches_the_declared_budget():
    """The threshold and the achieved rate must be two views of one number."""
    path = f"{C.DASHBOARD_DIR}/demo_data.json"
    if not os.path.exists(path):
        pytest.skip("demo_data.json not present; run the pipeline first")
    with open(path) as f:
        cohort = json.load(f)["cohort"]
    assert cohort["alert_rate_all_ticks"] == pytest.approx(
        cohort["alert_budget_target"], abs=5e-4)
    assert cohort["alerts_per_patient_day"] == pytest.approx(
        cohort["alert_budget_target"] * 24 * C.TICKS_PER_HOUR, abs=0.05)


def test_demo_window_is_not_peak_selected():
    """Every demo window must end at the same index, not at each patient's argmax.

    v2 ended each window at `argmax(risk)`, so all six demo patients were shown
    at their personal maximum. The fixed rule means the last tick is the last
    tick, and the number of alerts in the window is allowed to vary wildly
    (including zero) rather than being equal by construction.
    """
    path = f"{C.DASHBOARD_DIR}/demo_data.json"
    if not os.path.exists(path):
        pytest.skip("demo_data.json not present; run the pipeline first")
    with open(path) as f:
        bundle = json.load(f)
    lengths = {len(p["timeline"]) for p in bundle["patients"]}
    assert lengths == {C.DEMO_WINDOW_HOURS * C.TICKS_PER_HOUR}, lengths
    counts = {p["n_alerts_in_window"] for p in bundle["patients"]}
    assert len(counts) > 1, (
        "every demo patient has an identical alert count, which is what "
        "peak-selection produces")
    assert 0 in counts, (
        "no demo patient has a quiet window -- the demo is still showing only "
        "busy stretches")


# --------------------------------------------------------------------------
# Patient-level cluster bootstrap (the significance engine)
# --------------------------------------------------------------------------
def test_bootstrap_pair_p_floor_and_null():
    """Two-sided cluster-bootstrap p: floors at 1/n, ~1.0 under the null."""
    from train_model import _bootstrap_pair_p

    n = 2000
    a = np.linspace(0.55, 0.95, n)
    b = np.linspace(0.50, 0.70, n)          # a > b on every resample
    assert _bootstrap_pair_p(a, b) == pytest.approx(1.0 / n)
    assert _bootstrap_pair_p(b, a) == pytest.approx(1.0 / n)
    # Identical curves are the null: every delta is exactly 0.
    assert _bootstrap_pair_p(a, a.copy()) == 1.0
    # Empty input must not crash, and must not claim significance.
    assert _bootstrap_pair_p(np.array([]), np.array([])) == 1.0


def test_patient_cluster_bootstrap_resamples_patients_not_rows():
    """The resampling unit must be the patient, not the row.

    If a resample drew ticks instead of patients, multiplying the rows per
    patient would change every resampled AUC. Because whole patients are drawn,
    the AUC of a per-patient-constant label is invariant to how many rows each
    patient has -- this test pins that invariance, which is exactly what the
    p-value's "resampling unit = patient" claim rests on.
    """
    from train_model import _bootstrap_pair_p, _patient_bootstrap_matrix

    n_pat, rows = 30, 6
    codes = np.repeat(np.arange(n_pat), rows)
    y = (np.arange(n_pat) % 2 == 0).repeat(rows).astype(float)
    b6 = _patient_bootstrap_matrix(codes, y, {"a": y.copy(), "b": 1.0 - y},
                                   n_resamples=400, seed=3)

    # Same patients, four times as many rows each.
    rows20 = rows * 4
    codes20 = np.repeat(np.arange(n_pat), rows20)
    y20 = (np.arange(n_pat) % 2 == 0).repeat(rows20).astype(float)
    b20 = _patient_bootstrap_matrix(codes20, y20,
                                    {"a": y20.copy(), "b": 1.0 - y20},
                                    n_resamples=400, seed=3)

    np.testing.assert_array_equal(b6["a"], b20["a"])
    np.testing.assert_array_equal(b6["b"], b20["b"])

    # Perfectly separated variants: two-sided p sits at the floor, never 0.
    p = _bootstrap_pair_p(b6["a"], b6["b"])
    assert 0.0 < p < 0.05


def test_patient_cluster_bootstrap_is_deterministic():
    """Same seed, same data => identical resamples (reproducible p-values)."""
    from train_model import _patient_bootstrap_matrix

    n_pat, rows = 20, 5
    rng = np.random.default_rng(11)
    codes = np.repeat(np.arange(n_pat), rows)
    y = (np.arange(n_pat) % 3 != 0).repeat(rows).astype(float)
    proba = rng.random(n_pat * rows)
    kw = dict(n_resamples=200, seed=5)
    a = _patient_bootstrap_matrix(codes, y, {"v": proba}, **kw)["v"]
    b = _patient_bootstrap_matrix(codes, y, {"v": proba}, **kw)["v"]
    np.testing.assert_array_equal(a, b)


# --------------------------------------------------------------------------
# External-validation harness (real data)
# --------------------------------------------------------------------------
def test_external_validation_rehearsal_schema():
    """The harness runs end-to-end on a small input and emits a sane schema.

    This is a plumbing test, not a metric test: it asserts that
    `validate_real_data.run_validation` survives CV + patient-level bootstrap
    and that no p-value is ever stored as exactly 0.

    Deliberately run WITHOUT a static clinical table. Passing one would enable
    the `full` / `static_only` variants and the three `full_vs_*` deltas, which
    takes this single call from ~29 s to ~104 s: four variants instead of two,
    and four 2000-resample bootstraps instead of one. That branch is covered at
    a fraction of the cost by
    `test_paired_cohort_emits_both_fusion_directions`, which calls `_evaluate`
    -- where the branch actually lives -- on a fixture sized to it.
    """
    from validate_real_data import run_validation

    rng = np.random.default_rng(0)
    frames = []
    for p in range(10):
        n = 150
        g = np.clip(np.cumsum(rng.normal(0, 6, n)) + 120, 60, 320)
        ts = pd.date_range("2021-01-01", periods=n, freq="15min")
        frames.append(pd.DataFrame({
            "patient_id": f"R{p:03d}", "timestamp": ts,
            "glucose_mgdl": g,
            "hrv_rmssd_ms": rng.normal(40, 8, n),
            "heart_rate_bpm": rng.normal(75, 10, n),
            "steps": rng.integers(0, 60, n).astype(float),
            "sleep_stage": rng.integers(0, 4, n),
        }))
    out = run_validation(pd.concat(frames, ignore_index=True), None,
                         "test_rehearsal_schema", "synthetic rehearsal")
    try:
        with open(out) as f:
            ev = json.load(f)
        assert ev["n_rows"] > 0
        assert 0.0 < ev["positive_rate"] < 1.0
        assert "context" in ev["variants"] and "naive" in ev["variants"]
        for s in ev["variants"].values():
            assert np.isfinite(s["roc_auc_mean"])
        d = ev["paired_deltas"]["context_vs_naive"]
        assert d["bootstrap_p"] > 0.0          # floor, never exactly 0
        assert d["n_resamples"] == 2000
        assert ev["tick_minutes"] == 15
        # No static table -> no fusion deltas at all. Emitting
        # `full_vs_static_only` for a cohort with no static arm would be a
        # comparison against a model that was never fitted.
        assert "full_vs_static_only" not in ev["paired_deltas"]
        ops = ev["alert_operating_points"]
        assert ops, "no alert operating points emitted"
        for o in ops.values():
            assert o["observation_patient_days"] > 0
            assert 0.0 <= o["precision"] <= 1.0
    finally:
        os.remove(out)


def test_paired_cohort_emits_both_fusion_directions():
    """A paired cohort must produce BOTH fusion comparisons, in both directions.

    `full - context` and `full - static_only` are the only two numbers a fusion
    claim rests on, and they point in opposite directions. A report emitting just
    one would let a reader infer the wrong sign, so both are emitted and
    asserted exact negatives of each other here.

    This calls `_evaluate` directly rather than `run_validation`, because the
    paired branch is entirely inside `_evaluate`: reaching it through the full
    harness costs ~104 s versus ~3 s here.
    """
    from validate_real_data import _evaluate

    rng = np.random.default_rng(23)
    NP, per = 12, 50
    n = NP * per
    groups = np.repeat(np.arange(NP), per)
    hr = rng.normal(70, 6, n)
    hba1c = np.repeat(5.4 + 0.07 * np.arange(NP), per)
    # Both features must be CENTRED before entering the logit. An uncentred
    # `0.6 * hr` with hr ~ 70 gives a logit of ~42, sigmoid saturates, every row
    # becomes positive, and every CV fold comes back single-class -- which shows
    # up as a NaN delta rather than as an obviously broken fixture.
    sig = 0.12 * (hr - 70) + 1.5 * (hba1c - 5.4) + rng.normal(0, 1, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-sig))).astype(int)
    assert 0.05 < y.mean() < 0.95, "fixture label must not saturate"

    X = pd.DataFrame({"hr_mean_3h": hr, "hba1c_pct": hba1c,
                      "is_male": (groups % 2)})
    variants = {"context": ["hr_mean_3h"], "naive": ["hr_mean_3h"],
                "full": ["hr_mean_3h", "hba1c_pct", "is_male"],
                "static_only": ["hba1c_pct", "is_male"]}

    ev = _evaluate(X, y, groups, variants, ["hba1c_pct", "is_male"])
    for key in ("full_vs_context", "full_vs_static_only", "static_only_vs_full"):
        assert key in ev["paired_deltas"], key
        assert ev["paired_deltas"][key]["bootstrap_p"] > 0.0
    fwd = ev["paired_deltas"]["full_vs_static_only"]["delta_roc_auc"]
    rev = ev["paired_deltas"]["static_only_vs_full"]["delta_roc_auc"]
    assert fwd == pytest.approx(-rev)
    assert set(ev["variants"]) == set(variants)
    for s in ev["variants"].values():
        assert np.isfinite(s["roc_auc_mean"])

    ev2 = _evaluate(X, y, groups, {"context": ["hr_mean_3h"],
                                   "naive": ["hr_mean_3h"]}, None)
    assert "full_vs_static_only" not in ev2["paired_deltas"]
    assert "full_vs_context" not in ev2["paired_deltas"]
    assert "context_vs_naive" in ev2["paired_deltas"]


def test_external_validation_uci_has_no_zero_p():
    """The committed real-data result must keep the p-floor convention too."""
    path = f"{C.MODEL_DIR}/external_validation_uci.json"
    if not os.path.exists(path):
        pytest.skip("run src/convert_uci_diabetes.py + "
                    "src/validate_real_data.py first")
    with open(path) as f:
        ev = json.load(f)
    assert ev["n_patients"] > 0 and ev["n_rows"] > 0
    assert "context" in ev["variants"] and "naive" in ev["variants"]
    for d in ev["paired_deltas"].values():
        assert d["bootstrap_p"] > 0.0
        assert d["p_rendered"].startswith("p <") or d["p_rendered"].startswith("p =")


# --------------------------------------------------------------------------
# Paired real cohort (BIG IDEAs): converter + temporal protocol
# --------------------------------------------------------------------------
def test_big_ideas_sex_is_exact_match_not_substring():
    """Regression: "FEMALE" contains "MALE", so a substring test marks every
    participant male. That silently destroys the sex arm of the EHR comparison
    (and any cohort-level statistic built on it)."""
    import convert_big_ideas as B

    raw = pd.DataFrame({"ID": [1, 2, 3, 4], "Gender": ["FEMALE", "MALE", "Female", "male"],
                        "HbA1c": [5.5, 5.6, 5.9, 6.0]})
    tmp = os.path.join(ROOT, "data", "real", "_tmp_demographics.csv")
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    raw.to_csv(tmp, index=False)
    try:
        out = B.parse_demographics(tmp)
    finally:
        os.remove(tmp)
    # 2 males, 2 females -- the naive .str.contains("MALE") would return 4.
    assert int(out["is_male"].sum()) == 2
    assert out["patient_id"].tolist() == ["001", "002", "003", "004"]


def test_big_ideas_ibi_filter_rejects_implausible_beats():
    """RMSSD is a variance statistic, so a single PPG artefact blows it up.
    Beats outside the physiological window must be dropped, not smoothed."""
    import convert_big_ideas as B

    ts = pd.date_range("2021-01-01", periods=1200, freq="1s")
    ibi = np.full(len(ts), 0.9)                       # 900 ms, plausible
    ibi[600] = 0.02                                    # 20 ms: an artefact
    ibi[601] = 4.0                                     # 4 s: a dropped beat
    df = pd.DataFrame({"datetime": ts, "ibi": ibi})
    tmp = os.path.join(ROOT, "data", "real", "_tmp_ibi.csv")
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    df.to_csv(tmp, index=False)
    try:
        out = B.parse_ibi_hrv(tmp)
    finally:
        os.remove(tmp)
    assert not out.empty
    # A clean 900 ms series has zero beat-to-beat variability. Allowing the
    # 20 ms sample through would give an RMSSD of hundreds of ms.
    assert out["hrv_rmssd_ms"].max() < 1e-6
    assert out["sdnn_ms"].max() < 1e-6


def test_big_ideas_drops_all_nan_dynamic_columns():
    """Regression: Dexcom leaves Rate of Change / Insulin / Carb blank on every
    EGV row. The harness drops any row with a NaN feature, so an all-NaN column
    would delete the entire cohort and surface as a confusing "degenerate
    label" error rather than a source problem."""
    import convert_big_ideas as B

    root = os.path.join(ROOT, "data", "real", "_tmp_bi")
    os.makedirs(root, exist_ok=True)
    # A glucose-only cohort: no HR, no IBI, no food log.
    os.makedirs(os.path.join(root, "001"), exist_ok=True)
    ts = pd.date_range("2021-01-01", periods=200, freq="5min")
    pd.DataFrame({"Index": range(200), "Timestamp (YYYY-MM-DDThh:mm:ss)": ts,
                  "Event Type": "EGV", "Patient Info": "", "Device Info": "",
                  "Source Device ID": "", "Glucose Value (mg/dL)": 100 + np.arange(200) % 7,
                  "Insulin Value (u)": np.nan, "Carb Value (grams)": np.nan,
                  "Duration (hh:mm:ss)": "",
                  "Glucose Rate of Change (mg/dL/min)": np.nan,
                  "Transmitter Time (Long Integer)": 0}).to_csv(
        os.path.join(root, "001", "Dexcom_001.csv"), index=False)
    pd.DataFrame({"ID": [1], "Gender": ["MALE"], "HbA1c": [5.5]}).to_csv(
        os.path.join(root, "Demographics.csv"), index=False)
    try:
        long = os.path.join(root, "out_long.csv")
        s = B.convert(_path(root), _path(long), _path(os.path.join(root, "static.csv")),
                      _path(os.path.join(root, "summary.json")))
        assert s["n_patients"] == 1
        # Every declared stream is absent, so all of them must have been dropped
        # and the glucose column must survive.
        assert s["n_rows"] > 0
        assert "glucose_mgdl" in list(pd.read_csv(long, nrows=1).columns)
        assert set(s["dropped_all_nan_columns"]) == set(B.DYNAMIC_COLS)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _path(p: str) -> "os.PathLike":
    from pathlib import Path
    return Path(p)


def test_harness_rejects_a_degenerate_label():
    """Regression: a positive rate inside (0, 1) is not enough. At 0.01% most
    folds are single-class, roc_auc_score returns NaN, and the bootstrap still
    prints a confident-looking p-value. The harness must refuse instead."""
    from validate_real_data import MIN_CLASS_ROWS, run_validation

    rng = np.random.default_rng(3)
    frames = []
    for p in range(6):
        n = 150
        g = 100 + rng.normal(0, 0.5, n)
        # One short ramp, not an isolated spike. A single +45 sample is
        # annihilated by the 30-min smoothing that the label is defined on, so
        # it produces ZERO positives and trips the rate>0 guard instead of the
        # one under test. A 12-tick ramp yields ~24 positives in 888 rows: past
        # the rate guard, far below the 50-per-class floor.
        g[20:32] += np.linspace(0, 45, 12)
        frames.append(pd.DataFrame({
            "patient_id": f"D{p:03d}",
            "timestamp": pd.date_range("2021-01-01", periods=n, freq="15min"),
            "glucose_mgdl": g}))
    with pytest.raises(SystemExit) as exc:
        run_validation(pd.concat(frames, ignore_index=True), None,
                       "test_degenerate_label", "degenerate fixture")
    msg = str(exc.value)
    assert "degenerate" in msg.lower()
    assert f"{MIN_CLASS_ROWS}" in msg
    # The committed results directory must not gain a file from a failed run.
    assert not os.path.exists(f"{C.MODEL_DIR}/external_validation_test_degenerate_label.json")


def test_harness_rejects_too_few_patients_for_grouped_cv():
    """Regression for the hole MIN_CLASS_ROWS left open.

    That guard checks cohort-level class balance, so a cohort with plenty of
    rows and plenty of positives still slipped through when the *patients* were
    too few: GroupShuffleSplit moves whole patients, so with 3 participants each
    test fold inherits exactly one and comes back single-class. Observed on the
    real BIG IDEas download -- 1,424 rows at a 16% positive rate sailed past the
    50-per-class floor, every AUC came back NaN, and the paired bootstrap still
    printed "p < 0.001" beside "delta +nan".
    """
    from validate_real_data import MIN_FOLD_PATIENTS, _evaluate

    rng = np.random.default_rng(11)
    n_pat, per = 3, 400
    n = n_pat * per
    groups = np.repeat(np.arange(n_pat), per)
    hr = rng.normal(70, 6, n)
    # Both classes present in abundance at the cohort level -- that is the whole
    # point. Only the fold partition is degenerate.
    y = (rng.random(n) < 0.25).astype(int)
    assert min((y == 1).sum(), (y == 0).sum()) > 50, "cohort must be balanced"
    X = pd.DataFrame({"cgm_level": hr})
    variants = {"context": ["cgm_level"], "naive": ["cgm_level"]}

    with pytest.raises(SystemExit) as exc:
        _evaluate(X, y, groups, variants, None)
    msg = str(exc.value)
    assert "single-class" in msg or "degenerate" in msg.lower()
    assert f"{MIN_FOLD_PATIENTS}" in msg
    # It must point at the fix rather than just complain.
    assert "--temporal" in msg


def test_temporal_split_excludes_temporal_leakage_and_says_so():
    """The temporal protocol excludes TEMPORAL leakage, not patient leakage.

    That asymmetry is the whole reason its AUC cannot be compared with the
    cross-patient headline, so it has to be asserted rather than assumed: every
    scored row must postdate every training row for that patient, and each
    patient must in fact appear on BOTH sides (a "no patient leakage" version
    of this protocol would just be GroupShuffleSplit again).
    """
    from validate_real_data import TEMPORAL_CUTS, _evaluate_temporal

    rng = np.random.default_rng(5)
    NP, n = 8, 8 * 96
    groups, stamps, y = [], [], []
    X = {k: np.zeros(NP * n) for k in ("cgm_level", "hr_mean_3h")}
    for p in range(NP):
        t = pd.date_range("2021-01-01", periods=n, freq="15min") + pd.to_timedelta(p * 30, unit="D")
        g = 100 + 40 * np.clip(np.sin(np.arange(n) / 30.0), 0, None) + rng.normal(0, 3, n)
        y.append((g > 120).astype(int))
        groups += [p] * n
        stamps.append(t)
        X["cgm_level"][p * n:(p + 1) * n] = g
        X["hr_mean_3h"][p * n:(p + 1) * n] = rng.normal(60, 5, n)
    Xdf = pd.DataFrame(X)
    y = np.concatenate(y)
    groups = np.concatenate([np.asarray(groups)])
    stamps = pd.Series(pd.DatetimeIndex(np.concatenate([s.to_numpy() for s in stamps])))

    ev = _evaluate_temporal(Xdf, y, groups, stamps, {"a": ["cgm_level"],
                                                     "b": ["cgm_level", "hr_mean_3h"]})
    if ev["status"] != "ok":
        pytest.skip(f"temporal protocol skipped: {ev['reason']}")
    assert ev["n_cuts_scored"] == len(TEMPORAL_CUTS)
    assert ev["n_scored_rows"] > 0
    # The JSON must disclose which leakage it excludes, so a reader cannot
    # mistake this protocol for the cross-patient one.
    assert ev["excludes_temporal_leakage"] is True
    assert ev["excludes_patient_leakage"] is False
    for name, s in ev["variants"].items():
        assert np.isfinite(s["roc_auc_mean"]), name
    for d in ev["paired_deltas"].values():
        assert d["bootstrap_p"] > 0.0

    # Re-derive the split the function used and assert both invariants.
    for cut in TEMPORAL_CUTS:
        tr, te = [], []
        for pid in np.unique(groups):
            idx = np.flatnonzero(groups == pid)
            s = stamps.iloc[idx].sort_values()
            at = s.quantile(cut)
            tr.append(idx[s.to_numpy() <= at])
            te.append(idx[s.to_numpy() > at])
            # Per patient: nothing is predicted from its own future. This must
            # be checked within a patient -- these cohorts are date-shifted, so
            # a global max(training) < min(scoring) comparison across patients
            # with different calendar origins is meaningless.
            assert stamps.iloc[idx][s.to_numpy() <= at].max() \
                < stamps.iloc[idx][s.to_numpy() > at].min(), \
                "a scored row predates that patient's own training rows"
        tr, te = np.concatenate(tr), np.concatenate(te)
        assert not (set(tr) & set(te)), "a row appears on both sides of the split"
        # ... and patients deliberately DO span the split.
        assert set(groups[tr]) & set(groups[te]), \
            "temporal protocol must include each patient on both sides"


def test_operating_points_convert_budget_to_alerts_per_patient_day():
    """A budget quoted as a fraction of rows is not actionable. The reported
    number must be derived from real observation time, so a cohort with gaps
    cannot quietly report an implausible alert rate."""
    from validate_real_data import _operating_points

    stamps = pd.DataFrame({
        "patient_id": ["A"] * 192 + ["B"] * 192,
        # Each patient spans 192 * 15 min = 2 days -> 4 patient-days total.
        "timestamp": list(pd.date_range("2021-01-01", periods=192, freq="15min")) * 2,
    })
    n = len(stamps)
    half = n // 2
    # Labels and scores must agree: the second half is positive AND scores
    # higher, so the top 25% of rows is exactly the top half of the positives.
    y = np.concatenate([np.zeros(half), np.ones(half)]).astype(int)
    pr = np.concatenate([np.linspace(0.2, 0.4, half),
                         np.linspace(0.6, 0.8, half)])
    groups = pd.factorize(stamps["patient_id"])[0]
    ops = _operating_points(y, pr, groups, stamps, budgets=(0.25,))
    op = ops["budget_0.25"]
    # Span is max-minus-min, not count x interval, so 192 quarter-hourly rows
    # is 1.99 patient-days rather than exactly 2.
    assert op["observation_patient_days"] == pytest.approx(3.98, abs=0.05)
    # 25% of 384 rows = 96 alerts over ~3.98 patient-days = ~24/patient/day.
    assert op["alerts_per_patient_day"] == pytest.approx(24.0, abs=0.2)
    assert op["precision"] == pytest.approx(1.0)
    # All 96 alerts are true positives out of 192 positives -> recall 0.5.
    assert op["recall"] == pytest.approx(0.5)


def test_operating_points_overshoot_budget_when_scores_tie():
    """A threshold rule fires on every row at or above the cut, so a block of
    tied scores can raise the realised alert count above the requested budget.
    Reporting only the requested budget would understate the clinician's load,
    so the realised rate is what gets published."""
    from validate_real_data import _operating_points

    stamps = pd.DataFrame({
        "patient_id": ["A"] * 192,
        "timestamp": pd.date_range("2021-01-01", periods=192, freq="15min"),
    })
    y = np.array([0, 1] * 96)
    pr = y.astype(float)                    # every positive ties at 1.0
    groups = pd.factorize(stamps["patient_id"])[0]
    op = _operating_points(y, pr, groups, stamps, budgets=(0.25,))["budget_0.25"]
    assert op["alerts_per_patient_day"] > 24.0
    assert op["precision"] == pytest.approx(1.0)
