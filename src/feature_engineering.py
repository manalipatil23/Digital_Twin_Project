"""
feature_engineering.py
------------------------
Fuses the two data streams into one supervised learning table:

  STATIC   (from EHR, repeated on every row for a patient)
  DYNAMIC  (rolling-window features engineered from the wearable time series)

Label: glucose_spike_2h = 1 if the patient's CGM crosses a clinically
meaningful spike threshold at any point in the next 2 hours, else 0.

  crosses_abs : smoothed CGM exceeds 180 mg/dL  (standard CGM high alert)
  rises_fast  : smoothed CGM rises >= 30 mg/dL above "now"  (rapid-rise alert)

Both are computed on a 30-minute smoothed CGM series rather than raw
15-minute samples. v2 evaluated the rise criterion on a single noisy sample
(`max(future) - cgm[i] >= 30`), which fires on sensor noise as often as on
physiology -- a max over 8 samples each carrying ~5 mg/dL of Gaussian noise
will clear a 30 mg/dL bar often enough to swamp the signal. Smoothing is both
more clinically faithful to how CGM "rapid rise" alerts actually behave and
removes that artefact.

This is the "Digital Twin" fusion step: every training row is one moment in a
patient's life, blending who-they-are (EHR) with what-is-happening-to-them-
right-now (wearable stream), to forecast what happens next.

All feature windows are strictly trailing. There are no centred or
forward-looking windows anywhere in this file, and `tests/test_pipeline.py`
asserts that no feature at time t correlates with the label better than the
label predicts itself under a label-permutation control.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

import config as C


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _trailing_forward_max(values: np.ndarray, horizon: int) -> np.ndarray:
    """For each i, max(values[i+1 : i+1+horizon]). Near the tail this is padded
    and must be discarded -- callers must also apply the `horizon`-row mask."""
    padded = np.concatenate([values, np.full(horizon, values[-1])])
    windows = sliding_window_view(padded, horizon)   # windows[i] = padded[i:i+horizon]
    return windows[1:, -1]                          # result[i] = max(padded[i+1:i+1+horizon])


def _trailing_smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Centred=False rolling mean, min_periods=1, computed without a Python callback."""
    s = pd.Series(values)
    return s.rolling(window, min_periods=1).mean().to_numpy()


# --------------------------------------------------------------------------
# Dynamic features + label
# --------------------------------------------------------------------------
def build_dynamic_features(wear_df: pd.DataFrame) -> pd.DataFrame:
    wear_df = wear_df.sort_values(["patient_id", "timestamp"]).copy()
    wear_df = wear_df.reset_index(drop=True)

    n_rows = len(wear_df)
    horizon = C.LOOKAHEAD_TICKS
    label = np.zeros(n_rows, dtype=np.int8)
    valid = np.zeros(n_rows, dtype=bool)

    pid = wear_df["patient_id"].to_numpy()
    cgm = wear_df["cgm_mgdl"].to_numpy(dtype=float)
    hrv = wear_df["hrv_rmssd_ms"].to_numpy(dtype=float)
    hr = wear_df["heart_rate_bpm"].to_numpy(dtype=float)
    steps = wear_df["steps"].to_numpy(dtype=float)
    sleep_stage = wear_df["sleep_stage"].to_numpy(dtype=int)

    # Index boundaries per patient, so every window below is guaranteed to be
    # intra-patient and never straddles a patient boundary.
    order = np.argsort(pid, kind="stable")
    sorted_pid = pid[order]
    bounds = np.flatnonzero(np.r_[True, sorted_pid[1:] != sorted_pid[:-1], True])
    patient_slices = list(zip(bounds[:-1], bounds[1:]))

    for start, stop in patient_slices:
        idx = order[start:stop]
        n = len(idx)
        if n <= horizon:
            continue

        g_cgm, g_hrv, g_hr = cgm[idx], hrv[idx], hr[idx]
        g_steps, g_sleep = steps[idx], sleep_stage[idx]

        # ---- Trailing rolling statistics = "current physiological state" ----
        for col_name, arr in (("heart_rate_bpm", g_hr), ("hrv_rmssd_ms", g_hrv),
                              ("cgm_mgdl", g_cgm), ("steps", g_steps)):
            s = pd.Series(arr)
            wear_df.loc[idx, f"{col_name}_roll_mean"] = s.rolling(
                C.ROLL_WINDOW_TICKS, min_periods=1).mean().to_numpy()
            wear_df.loc[idx, f"{col_name}_roll_std"] = s.rolling(
                C.ROLL_WINDOW_TICKS, min_periods=2).std().fillna(0).to_numpy()

        shifted_cgm = pd.Series(g_cgm).shift(C.ROLL_WINDOW_TICKS).fillna(0).to_numpy()
        shifted_hrv = pd.Series(g_hrv).shift(C.ROLL_WINDOW_TICKS).fillna(0).to_numpy()
        wear_df.loc[idx, "cgm_trend"] = g_cgm - shifted_cgm
        wear_df.loc[idx, "hrv_trend"] = g_hrv - shifted_hrv

        # Restorative sleep share over the trailing 24h (vectorised -- v2 used a
        # per-row Python callback here, which dominated runtime on 1M rows).
        restorative = np.isin(g_sleep, [2, 3]).astype(float)
        wear_df.loc[idx, "restorative_sleep_ratio_24h"] = pd.Series(restorative).rolling(
            C.SLEEP_LOOKBACK_TICKS, min_periods=1).mean().to_numpy()

        wear_df.loc[idx, "low_activity_flag"] = (
            pd.Series(g_steps).rolling(C.ROLL_WINDOW_TICKS, min_periods=1).mean().to_numpy() < 12
        ).astype(int)
        wear_df.loc[idx, "hour_of_day"] = wear_df.loc[idx, "timestamp"].dt.hour

        # ---- Label ----
        smoothed = _trailing_smooth(g_cgm, C.SPIKE_RISE_BASELINE_SMOOTH_TICKS)
        fwd_max = _trailing_forward_max(smoothed, horizon)
        crosses_abs = fwd_max > C.SPIKE_ABS_THRESHOLD_MGDL
        rises_fast = (fwd_max - smoothed) >= C.SPIKE_RISE_THRESHOLD_MGDL
        patient_label = (crosses_abs | rises_fast).astype(np.int8)

        # A row is only trainable if a FULL `horizon`-tick future window exists.
        # v2 dropped only the final row, so the last 8 ticks of every patient
        # were labelled from a 1-to-7-tick window while being treated as a full
        # 2-hour lookahead. Because the label is a max over future samples, a
        # short window is systematically biased negative -- i.e. silent label
        # noise on 4,000 rows.
        n_valid = n - horizon
        label[idx[:n_valid]] = patient_label[:n_valid]
        valid[idx[:n_valid]] = True

    out = wear_df[valid].copy()
    out["glucose_spike_2h"] = label[valid].astype(int)
    return out


def build_training_table(ehr_df: pd.DataFrame, wear_df: pd.DataFrame) -> pd.DataFrame:
    dyn = build_dynamic_features(wear_df)
    return dyn.merge(ehr_df, on="patient_id", how="left", validate="many_to_one")


if __name__ == "__main__":
    ehr = pd.read_csv(f"{C.DATA_DIR}/ehr_patients.csv")
    wear = pd.read_csv(f"{C.DATA_DIR}/wearable_timeseries.csv", parse_dates=["timestamp"])
    fused = build_training_table(ehr, wear)
    fused.to_parquet(f"{C.DATA_DIR}/fused_training_table.parquet", index=False)

    print(f"Fused table: {fused.shape[0]:,} rows x {fused.shape[1]} cols, "
          f"{fused['patient_id'].nunique()} patients")
    print(f"Positive spike rate: {fused['glucose_spike_2h'].mean():.3f}")
    print(f"Rows dropped per patient (incomplete 2h lookahead): "
          f"{C.N_TICKS - fused.shape[0] // fused['patient_id'].nunique()}")
