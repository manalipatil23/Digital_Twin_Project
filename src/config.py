"""
config.py
---------
Single source of truth for every tunable constant in the project.

Why this exists: v2 of this project had the glucose model and the
cardiovascular model configured independently, and the README quoted
metrics from two different runs (0.953 in one section, 0.955 in another).
Centralising the constants plus auto-generating the results report from
`metrics.json` (see `src/make_report.py`) means documentation can no longer
drift away from what the code actually produced.
"""
from __future__ import annotations

import os

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "model")
DASHBOARD_DIR = os.path.join(ROOT, "dashboard")

# --------------------------------------------------------------------------
# Optional size overrides, used by CI to run the full pipeline in ~2 minutes
# instead of ~10. Set to empty string to use the real defaults.
#
# DAYS has a hard floor, not just a default. The secondary (cardiovascular
# strain) label asks whether a patient has >= STRAIN_MIN_CONSECUTIVE_DAYS bad
# days somewhere inside the next STRAIN_LOOKAHEAD_DAYS days, so a run shorter
# than LOOKAHEAD + MIN_CONSECUTIVE + 1 cannot contain a single positive example.
# Below the floor the label is identically zero and every AUC in
# cv_risk_metrics.json comes out NaN -- exactly what CI did before this floor
# existed. The floor is the *geometric* bound only; it is not guaranteed to be
# sufficient. train_cv_risk_model.py re-checks the generated label at runtime
# and exits loudly if it degenerates (zero positives, or a fold whose test
# partition has a single class), because small reduced cohorts have been
# observed to hit both states above the floor.
# --------------------------------------------------------------------------
N_PATIENTS = int(os.environ.get("DIGITAL_TWIN_N_PATIENTS") or 500)
DAYS = int(os.environ.get("DIGITAL_TWIN_DAYS") or 21)

# Cardiovascular-strain label geometry. Declared here rather than in the
# secondary-target block further down because the reduced-run floor below needs
# it, and forward-referencing a name that is defined 120 lines later turns a
# clear configuration error into a NameError.
STRAIN_LOOKAHEAD_DAYS = 7
STRAIN_MIN_CONSECUTIVE_DAYS = 3
STRAIN_MIN_DAYS_REQUIRED = STRAIN_LOOKAHEAD_DAYS + STRAIN_MIN_CONSECUTIVE_DAYS + 1
if DAYS < STRAIN_MIN_DAYS_REQUIRED:
    raise SystemExit(
        f"config: DIGITAL_TWIN_DAYS={DAYS} is too short.\n"
        f"The cardiovascular-strain label looks for {STRAIN_MIN_CONSECUTIVE_DAYS} "
        f"consecutive qualifying days inside a {STRAIN_LOOKAHEAD_DAYS}-day future "
        f"window, so a series needs at least {STRAIN_MIN_DAYS_REQUIRED} days to "
        f"contain a single positive example. With fewer, the label is identically "
        f"zero and every AUC becomes NaN.\n"
        f"Use DIGITAL_TWIN_DAYS={STRAIN_MIN_DAYS_REQUIRED} or more (the full "
        f"default is 21). Note that this floor is only the geometric bound -- "
        f"train_cv_risk_model.py still exits loudly if the generated label comes "
        f"out degenerate at the chosen reduced setting."
    )

TICKS_PER_HOUR = 4                    # 15-minute resolution (real CGM cadence)
TICKS_PER_DAY = 24 * TICKS_PER_HOUR
N_TICKS = DAYS * TICKS_PER_DAY
SERIES_START = "2026-08-01"

# --------------------------------------------------------------------------
# Clinical thresholds for the glucose-spike label
# --------------------------------------------------------------------------
SPIKE_ABS_THRESHOLD_MGDL = 180.0      # absolute CGM alert threshold
SPIKE_RISE_THRESHOLD_MGDL = 30.0      # rapid-rise alert threshold
SPIKE_RISE_BASELINE_SMOOTH_TICKS = 4  # smooth "current" CGM over 30 min
LOOKAHEAD_HOURS = 2
LOOKAHEAD_TICKS = LOOKAHEAD_HOURS * TICKS_PER_HOUR

# --------------------------------------------------------------------------
# Feature windows (trailing only -- no centred / forward-looking windows)
# --------------------------------------------------------------------------
ROLL_WINDOW_HOURS = 4
ROLL_WINDOW_TICKS = ROLL_WINDOW_HOURS * TICKS_PER_HOUR
SLEEP_LOOKBACK_TICKS = TICKS_PER_DAY

# --------------------------------------------------------------------------
# Generative-process realism knobs (see data/generate_wearable.py)
#
# These control how much of a glucose spike is *knowable in advance* from
# observed wearable signals. v2 of this project had all of these at their
# "everything is predictable" extreme, which made the task algebraically
# closed and the resulting 0.95 AUC meaningless. See README 2.
# --------------------------------------------------------------------------
# AR(1) "metabolic reactivity" -- a HIDDEN process that scales spike magnitude.
# It is never written to any file, so the model can only ever see its average
# effect, not its current value. This is the main irreducible-variance source.
REACTIVITY_AR1_PHI = 0.97
REACTIVITY_AR1_SD = 0.30
REACTIVITY_FLOOR = 0.15

# Multiplicative lognormal noise applied to every spike realisation.
SPIKE_NOISE_LOGSD = 0.80

# Fully exogenous spikes: no observable precursor at all. Models missed
# medication, acute illness, unlogged carbohydrate load, or CGM artefact.
EXOGENOUS_SPIKE_RATE = 0.020
EXOGENOUS_SPIKE_GAMMA_SHAPE = 2.0
EXOGENOUS_SPIKE_GAMMA_SCALE = 9.0

# Meal timing is jittered per patient and meal size varies, so `hour_of_day`
# is a real but imperfect prior rather than a perfect giveaway clock.
MEAL_PHASE_JITTER_H = 1.5
MEAL_SKIP_PROB = 0.10
# Baseline hourly sensor noise (mg/dL).
CGM_NOISE_MGDL = 5.0

# --------------------------------------------------------------------------
# Model / evaluation
# --------------------------------------------------------------------------
N_SPLITS = 5
TEST_SIZE = 0.25
BASE_SEED = 42

# Identical hyper-parameters for BOTH predictive targets. v2 used
# max_iter=200/lr=0.08 for the CV model and 120/0.10 for the glucose model,
# which made the "runtime" note in the README ambiguous.
MODEL_MAX_ITER = 160
MODEL_MAX_DEPTH = 4
MODEL_LEARNING_RATE = 0.10
MODEL_RANDOM_STATE = 42

PERMUTATION_N_REPEATS = 5
PERMUTATION_MAX_ROWS = 20_000

# Operating points reported for the clinician-facing alert, expressed as the
# fraction of all patient-ticks that would raise an alert. A 0.5 threshold is
# meaningless at a 4% base rate, so we report the alert-budget curve instead.
ALERT_BUDGETS = (0.01, 0.05, 0.10, 0.25)

# Number of demo patients and how much history the UI shows.
N_DEMO_PATIENTS = 6
DEMO_WINDOW_HOURS = 48

# --------------------------------------------------------------------------
# Feature column sets
# --------------------------------------------------------------------------
DYNAMIC_FEATURE_COLS = [
    "heart_rate_bpm", "hrv_rmssd_ms", "cgm_mgdl", "steps",
    "heart_rate_bpm_roll_mean", "heart_rate_bpm_roll_std",
    "hrv_rmssd_ms_roll_mean", "hrv_rmssd_ms_roll_std",
    "cgm_mgdl_roll_mean", "cgm_mgdl_roll_std",
    "steps_roll_mean", "steps_roll_std",
    "hrv_trend", "cgm_trend",
    "restorative_sleep_ratio_24h", "low_activity_flag", "hour_of_day",
]

STATIC_FEATURE_COLS = [
    "age", "bmi", "type2_diabetes", "hypertension", "prior_cardiac_event",
    "hba1c_pct", "fasting_glucose_mgdl", "ldl_cholesterol_mgdl", "egfr_ml_min",
    "tcf7l2_risk_allele_count", "apoe4_risk_allele_count",
]

# The "shortcut" control. v2 never ran this, and it is the single control that
# matters most: every one of these columns is available to a clinician with no
# wearable stream and no HRV/sleep/activity reasoning at all. If the full model
# cannot beat this by a real margin, the remaining 25 features are decoration.
SHORTCUT_FEATURE_COLS = [
    "hour_of_day", "cgm_mgdl", "cgm_mgdl_roll_mean", "cgm_trend",
    "fasting_glucose_mgdl", "hba1c_pct",
]

# The autonomic / activity channels -- HRV, sleep stage, heart rate, steps. This
# is the set the README's fusion story is actually about ("can the model read
# metabolic risk off the physiological stream?"), so it deserves its own column.
AUTONOMIC_FEATURE_COLS = [
    "hrv_rmssd_ms", "hrv_rmssd_ms_roll_mean", "hrv_rmssd_ms_roll_std", "hrv_trend",
    "restorative_sleep_ratio_24h",
    "heart_rate_bpm", "heart_rate_bpm_roll_mean", "heart_rate_bpm_roll_std",
    "steps", "steps_roll_mean", "steps_roll_std", "low_activity_flag",
]

# MATCHED control: the shortcut baseline plus exactly the autonomic channels and
# nothing else. The difference between this and SHORTCUT_FEATURE_COLS is a clean,
# isolated estimate of what the wearable physiology channels are worth -- no
# EHR features, no other confounders. This is the number that actually tests the
# fusion claim, and v2 had no equivalent.
SHORTCUT_PLUS_AUTONOMIC_FEATURE_COLS = SHORTCUT_FEATURE_COLS + AUTONOMIC_FEATURE_COLS

# --------------------------------------------------------------------------
# Cardiovascular-strain (secondary) target
# STRAIN_LOOKAHEAD_DAYS and STRAIN_MIN_CONSECUTIVE_DAYS are declared at the top
# of this file, next to the reduced-run floor that depends on them.
# --------------------------------------------------------------------------
HRV_BASELINE_PERCENTILE = 20
HR_ELEVATED_PERCENTILE = 70

CV_STATIC_COLS = [
    "age", "bmi", "hypertension", "prior_cardiac_event", "type2_diabetes",
    "ldl_cholesterol_mgdl", "egfr_ml_min", "apoe4_risk_allele_count",
]
CV_DYNAMIC_COLS = [
    "resting_hr", "hrv_daily_mean", "restorative_sleep_ratio", "steps_daily",
    "hrv_3d_trend", "hr_3d_trend", "hrv_7d_mean", "hr_7d_mean",
]
