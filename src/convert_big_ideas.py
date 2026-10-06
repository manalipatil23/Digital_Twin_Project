"""convert_big_ideas.py
---------------------
Convert the BIG IDEAs Lab Glycemic Variability and Wearable Device Data set
(PhysioNet, Open Access, ODC-By v1.0) into the long-format schema that
``src/validate_real_data.py`` consumes:

    long   patient_id, timestamp, glucose_mgdl, hrv_rmssd_ms, sdnn_ms,
           heart_rate_bpm, mean_ibi_ms, carbs_g, calories, glucose_roc
    static patient_id, hba1c_pct, is_male

Why this cohort matters: it is the only openly-licensed, *paired* real dataset
found that puts continuous CGM, a wrist-worn wearable stream, and a static
clinical record on the same participants. OhioT1DM and MIMIC-IV are
credential-gated and their DUAs forbid redistribution, so they cannot ship.

Source formats, per participant folder ``<id>/``:

  Dexcom_<id>.csv  Dexcom Clarity export. The first rows are metadata
                   (Event Type in {FirstName, LastName, PatientIdentifier});
                   glucose arrives as Event Type == "EGV" with a value, a rate
                   of change, and insulin/carb columns on treatment events.
                   Timestamps: ``YYYY-MM-DDThh:mm:ss``.
  HR_<id>.csv      ``datetime, hr`` -- one or more rows per minute, format
                   ``M/D/YY H:MM`` (no seconds, no zero padding).
  IBI_<id>.csv     ``datetime, ibi`` -- inter-beat interval in SECONDS with
                   microsecond timestamps. This is the HRV source.
  Food_Log_<id>.csv  meal rows with total_carb / sugar / calorie.

All participant timelines are date-shifted, so no two patients share a calendar
period and patient-level separation is exact rather than enforced by a split.

Output is written to ``data/real/`` which is gitignored: no raw real patient
data is ever committed.

Usage:
    python src/convert_big_ideas.py --root data/real/bigideas
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Native cadences. The harness re-grids to 15 min, so emitting the 5-min CGM
# grid is lossless with respect to it and keeps the intermediate small.
# --------------------------------------------------------------------------
BIN = "5min"
CGM_NATIVE_MIN = 5

# E4 beat-detection plausibility window. Outside this the beat is an artefact
# of PPG, not a heartbeat, and letting it into an RMSSD inflates the variance.
IBI_MIN_MS, IBI_MAX_MS = 300.0, 2000.0
MIN_IBI_PER_BIN = 30          # below this, RMSSD is too noisy to report
MIN_IBI_PAIRS = 20            # consecutive pairs needed for RMSSD
MIN_CGM_ROWS = 96             # 8 h of contiguous context

# Optional dynamic columns this converter can emit, in output order. Kept
# explicit so validate_real_data.py and this file cannot disagree.
#
# Note what is NOT here. Dexcom's export leaves "Glucose Rate of Change",
# "Insulin Value" and "Carb Value" blank on every EGV row -- it populates them
# only on discrete Alert/treatment events, and this cohort has no insulin pump
# (participants are prediabetic and diet-managed). Emitting those as columns
# would be an all-NaN column, and the harness drops any row with a NaN feature,
# so it would silently delete the entire cohort. Carbohydrate comes from the
# food log instead, which is populated.
DYNAMIC_COLS = [
    "hrv_rmssd_ms",
    "sdnn_ms",
    "heart_rate_bpm",
    "mean_ibi_ms",
    "carbs_g",
    "calories",
]

# Treatment columns that are EVENT streams, not sampled signals. The food log
# has one row per logged meal; every tick with no logged meal means zero logged
# grams, and must be written as 0.0 rather than NaN. Left as NaN, the harness's
# 3-hour rolling mean (min_periods=3) finds no logged meal in almost any window,
# returns NaN, and the harness's "drop rows with a NaN feature" rule then deletes
# ~90% of the cohort -- a data-source artefact masquerading as a modelling
# result. Filling 0 is the definitional choice, not an imputation: it changes
# the feature's meaning from "carbs eaten" to "carbs LOGGED", so under-logging
# is absorbed into the zero and the feature is treatment adherence, not intake.
EVENT_COLS = ["carbs_g", "calories"]


# --------------------------------------------------------------------------
# Per-file parsers
# --------------------------------------------------------------------------
def _find(root: Path, pid: str, stem: str) -> Path | None:
    hits = sorted((root / pid).glob(f"{stem}_{pid}.csv")) if (root / pid).is_dir() else []
    return hits[0] if hits else None


def parse_dexcom(path: Path) -> pd.DataFrame:
    """EGV rows -> time-indexed glucose, plus insulin/carb treatment events."""
    df = pd.read_csv(path, low_memory=False)
    ev = df["Event Type"].astype("string")
    ts_col = df.columns[1]
    df = df.assign(_ts=pd.to_datetime(df[ts_col], errors="coerce"))

    egv = df[ev.eq("EGV")].copy()
    if egv.empty:
        return pd.DataFrame(columns=["glucose_mgdl", "glucose_roc", "insulin_u", "carbs_g"])
    gcol = [c for c in df.columns if "Glucose Value" in str(c)][0]
    rcol = [c for c in df.columns if "Rate of Change" in str(c)]
    icol = [c for c in df.columns if "Insulin Value" in str(c)][0]
    ccol = [c for c in df.columns if "Carb Value" in str(c)][0]

    out = pd.DataFrame({
        "timestamp": egv["_ts"].to_numpy(),
        "glucose_mgdl": pd.to_numeric(egv[gcol], errors="coerce").to_numpy(),
        "glucose_roc": (pd.to_numeric(egv[rcol[0]], errors="coerce").to_numpy()
                        if rcol else np.nan),
        # Treatment events are carried on their own rows, not on EGV rows.
        "insulin_u": pd.to_numeric(egv[icol], errors="coerce").fillna(0.0).to_numpy(),
        "carbs_g": pd.to_numeric(egv[ccol], errors="coerce").fillna(0.0).to_numpy(),
    })
    return out.dropna(subset=["timestamp", "glucose_mgdl"]).sort_values("timestamp")


def parse_hr(path: Path) -> pd.Series:
    """Heart rate, one value per native E4 sample, averaged into CGM bins."""
    df = pd.read_csv(path)
    ts = pd.to_datetime(df["datetime"], format="mixed", errors="coerce")
    hr = pd.to_numeric(df[df.columns[1]], errors="coerce")
    s = pd.Series(hr.to_numpy(), index=pd.DatetimeIndex(ts)).dropna()
    s = s[~s.index.isna()].sort_index()
    return s.groupby(s.index.floor(BIN)).mean()


def parse_ibi_hrv(path: Path) -> pd.DataFrame:
    """RMSSD / SDNN / mean IBI per CGM bin from the inter-beat interval series.

    RMSSD is the standard short-window HRV statistic and is computed here on
    exactly the same quantity the synthetic generator's ``hrv_rmssd_ms``
    channel represents, so the two pipelines' features mean the same thing.
    """
    df = pd.read_csv(path)
    ts = pd.to_datetime(df["datetime"], format="mixed", errors="coerce")
    ibi_s = pd.to_numeric(df[df.columns[1]], errors="coerce")
    s = pd.Series(ibi_s.to_numpy(), index=pd.DatetimeIndex(ts)).dropna()
    s = s[~s.index.isna()].sort_index()
    s = s[(s * 1000.0).between(IBI_MIN_MS, IBI_MAX_MS)]
    s.index = s.index.floor(BIN)

    rows = []
    for bin_ts, grp in s.groupby(s.index):
        ms = (grp * 1000.0).to_numpy()
        rows.append((bin_ts, float(np.std(ms, ddof=1)) if len(ms) > 1 else np.nan,
                     float(np.mean(ms))))
    out = pd.DataFrame(rows, columns=["timestamp", "sdnn_ms", "mean_ibi_ms"])

    rm = []
    for bin_ts, grp in s.groupby(s.index):
        d = np.diff((grp * 1000.0).to_numpy())
        rm.append((bin_ts, float(np.sqrt(np.mean(d ** 2)))
                   if len(d) >= MIN_IBI_PAIRS else np.nan))
    r = pd.DataFrame(rm, columns=["timestamp", "hrv_rmssd_ms"])

    out = out.merge(r, on="timestamp", how="outer")
    # Bins with too few beats are noise, not physiology.
    counts = s.groupby(s.index).size().rename("n_ibi").reset_index()
    counts.columns = ["timestamp", "n_ibi"]
    out = out.merge(counts, on="timestamp", how="left")
    out.loc[out["n_ibi"] < MIN_IBI_PER_BIN, ["hrv_rmssd_ms", "sdnn_ms"]] = np.nan
    return out


def parse_food_log(path: Path | None) -> pd.DataFrame:
    """Per-meal carbohydrate and energy, expanded onto CGM bins.

    This is a treatment/diet record, not a wearable channel, so it belongs on
    the clinical side of the fusion split rather than the physiological one.
    """
    if path is None:
        return pd.DataFrame(columns=["timestamp", "carbs_g", "calories"])
    df = pd.read_csv(path)
    if "time_begin" not in df.columns:
        return pd.DataFrame(columns=["timestamp", "carbs_g", "calories"])
    ts = pd.to_datetime(df["time_begin"], errors="coerce")
    carbs = pd.to_numeric(df.get("total_carb"), errors="coerce")
    kcal = pd.to_numeric(df.get("calorie"), errors="coerce")
    out = pd.DataFrame({"timestamp": ts, "carbs_g": carbs, "calories": kcal}).dropna(
        subset=["timestamp"])
    out = out[out.index.notna()]
    out["timestamp"] = out["timestamp"].dt.floor(BIN)
    out = out.groupby("timestamp", as_index=False)[["carbs_g", "calories"]].sum()
    return out


def parse_demographics(path: Path) -> pd.DataFrame:
    """The static EHR arm: HbA1c and sex, one row per participant."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    out = pd.DataFrame({
        "patient_id": df["ID"].astype(int).astype(str).str.zfill(3),
        "hba1c_pct": pd.to_numeric(df["HbA1c"], errors="coerce"),
        # Exact match, not substring: "FEMALE" contains "MALE", so
        # .str.contains("MALE") labels all 16 participants male.
        "is_male": df["Gender"].astype(str).str.strip().str.upper().eq("MALE").astype(int),
    })
    return out.dropna(subset=["hba1c_pct"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Per-participant assembly
# --------------------------------------------------------------------------
def build_participant(root: Path, pid: str) -> pd.DataFrame | None:
    """Assemble one participant's 5-min table, or None if unusable."""
    dex = _find(root, pid, "Dexcom")
    hr_p, ibi_p = _find(root, pid, "HR"), _find(root, pid, "IBI")
    if dex is None or not dex.exists():
        return None

    d = parse_dexcom(dex)
    if d.empty:
        return None
    d["timestamp"] = d["timestamp"].dt.floor(BIN)
    g = (d.groupby("timestamp", as_index=False)
         .agg(glucose_mgdl=("glucose_mgdl", "mean"),
              glucose_roc=("glucose_roc", "mean")))

    parts = [g]
    if hr_p and hr_p.exists():
        h = parse_hr(hr_p).rename("heart_rate_bpm").reset_index()
        h.columns = ["timestamp", "heart_rate_bpm"]
        parts.append(h)
    if ibi_p and ibi_p.exists():
        parts.append(parse_ibi_hrv(ibi_p))

    food = parse_food_log(_find(root, pid, "Food_Log"))
    if not food.empty:
        parts.append(food)

    df = parts[0]
    for p in parts[1:]:
        df = df.merge(p, on="timestamp", how="outer")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Restrict to the window the CGM actually covers: HR/IBI start hours before
    # the first glucose reading, and scoring glucose the E4 never covered would
    # hand the model a systematic advantage on rows the patient never had.
    lo, hi = df["glucose_mgdl"].first_valid_index(), df["glucose_mgdl"].last_valid_index()
    if lo is None:
        return None
    df = df.iloc[lo:hi + 1].reset_index(drop=True)

    # Event streams become dense zeros; see EVENT_COLS for why NaN is wrong here.
    for c in EVENT_COLS:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    # Interpolate short wearable dropouts; long gaps stay NaN so the harness
    # drops the row rather than inventing physiology. method="time" needs the
    # timestamps as the index, not a RangeIndex.
    present = [c for c in DYNAMIC_COLS if c in df.columns]
    if present:
        idx = pd.DatetimeIndex(df["timestamp"])
        for c in present:
            df[c] = (pd.Series(df[c].to_numpy(dtype=float), index=idx)
                     .interpolate(method="time", limit=2, limit_area="inside")
                     .to_numpy())
    if len(df.dropna(subset=["glucose_mgdl"])) < MIN_CGM_ROWS:
        return None
    df["patient_id"] = pid
    return df


def convert(root: Path, out_long: Path, out_static: Path, summary_path: Path | None) -> dict:
    demo = parse_demographics(root / "Demographics.csv")
    rows, skipped = [], []
    for pid in demo["patient_id"]:
        try:
            df = build_participant(root, pid)
        except Exception as exc:                      # noqa: BLE001 - report and continue
            skipped.append({"patient_id": pid, "reason": f"{type(exc).__name__}: {exc}"})
            continue
        if df is None:
            skipped.append({"patient_id": pid, "reason": "no usable CGM coverage"})
            continue
        rows.append(df)

    if not rows:
        raise SystemExit("No participant produced a usable table; check the fetch log.")

    long = pd.concat(rows, ignore_index=True)
    for c in DYNAMIC_COLS:
        if c not in long.columns:
            long[c] = np.nan
    long = (long[["patient_id", "timestamp"] + ["glucose_mgdl"] + DYNAMIC_COLS]
            .sort_values(["patient_id", "timestamp"])
            .reset_index(drop=True))

    # The harness drops any row with a NaN in ANY feature. An all-NaN column is
    # therefore not a harmless extra -- it deletes the whole cohort and the run
    # fails with a confusing "degenerate label" error instead of a source
    # problem. Drop such columns loudly, never quietly.
    empty = [c for c in DYNAMIC_COLS if long[c].notna().sum() == 0]
    if empty:
        print(f"WARNING: dropping all-NaN column(s) {empty}; the source files "
              f"do not carry these streams. The harness would otherwise drop "
              f"every row.", flush=True)
        long = long.drop(columns=empty)
    emitted = [c for c in DYNAMIC_COLS if c in long.columns]
    # The harness requires a contiguous, fully-populated glucose column.
    long = long[long["glucose_mgdl"].notna()].reset_index(drop=True)
    long["timestamp"] = long["timestamp"].astype("datetime64[ns]")

    keep = set(long["patient_id"].unique())
    static = demo[demo["patient_id"].isin(keep)].reset_index(drop=True)

    out_long.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(out_long, index=False)
    static.to_csv(out_static, index=False)

    # `hrv_rmssd_ms` is absent whenever the IBI files were unavailable, so the
    # coverage summary must not assume the column it measures still exists.
    agg = {"rows": ("glucose_mgdl", "size"),
           "days": ("timestamp", lambda s: round((s.max() - s.min()).total_seconds() / 86400, 2))}
    if "hrv_rmssd_ms" in long.columns:
        agg["hrv_coverage"] = ("hrv_rmssd_ms", lambda s: round(float(s.notna().mean()), 3))
    per_pt = long.groupby("patient_id").agg(**agg).reset_index()
    summary = {
        "source": "BIG IDEAs Lab Glycemic Variability and Wearable Device Data (PhysioNet, ODC-By v1.0)",
        "n_patients": int(long["patient_id"].nunique()),
        "n_rows": int(len(long)),
        "bin_minutes": CGM_NATIVE_MIN,
        "dynamic_columns": emitted,
        "dropped_all_nan_columns": empty,
        "static_columns": ["hba1c_pct", "is_male"],
        "per_patient": per_pt.to_dict(orient="records"),
        "skipped": skipped,
        "out_long": str(out_long),
        "out_static": str(out_static),
    }
    if summary_path:
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="data/real/bigideas")
    ap.add_argument("--out-long", default="data/real/bigideas_long.csv")
    ap.add_argument("--out-static", default="data/real/bigideas_static.csv")
    ap.add_argument("--summary", default="data/real/bigideas_convert_summary.json")
    args = ap.parse_args()

    s = convert(Path(args.root), Path(args.out_long), Path(args.out_static),
                Path(args.summary))
    print(f"patients {s['n_patients']}  rows {s['n_rows']:,}  "
          f"bin {s['bin_minutes']}min")
    for r in s["per_patient"]:
        cov = f"  hrv_coverage {r['hrv_coverage']}" if "hrv_coverage" in r else "  (no IBI)"
        print(f"  {r['patient_id']}  rows {r['rows']:>5}  days {r['days']:>5}{cov}")
    for r in s["skipped"]:
        print(f"  skipped {r['patient_id']}: {r['reason']}")
    print(f"wrote {args.out_long} and {args.out_static}")


if __name__ == "__main__":
    main()
