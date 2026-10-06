"""
convert_uci_diabetes.py
-----------------------
Converts the UCI "Diabetes" dataset (AIM '94, Michael Kahn; CC BY 4.0,
DOI 10.24432/C5T59G, https://archive.ics.uci.edu/dataset/34/diabetes) into
the long-format schema consumed by `src/validate_real_data.py`:

    patient_id, timestamp, glucose_mgdl

The dataset is 70 patients of tab-separated records `date time code value`.
Only the blood-glucose codes are kept (48, 57-64 per the dataset README);
insulin doses and event codes are dropped. The mirror ships per-patient
files named `data-01` .. `data-70`.

Usage:
    python src/convert_uci_diabetes.py path/to/Diabetes-Data --out data/real/cgm.csv
"""
from __future__ import annotations

import argparse
import glob
import os

import pandas as pd

GLUCOSE_CODES = {48, 57, 58, 59, 60, 61, 62, 63, 64}
SOURCE = ("UCI Diabetes (AIM '94, Kahn; CC BY 4.0) - 70 T1D patients, "
          "1991-1993 capillary glucose records")


def convert(data_dir: str, out_path: str) -> str:
    rows = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "data-*"))):
        pid = os.path.basename(fp).replace("data-", "P")
        with open(fp, encoding="latin-1") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    continue
                date_s, time_s, code = parts[0], parts[1], parts[2]
                try:
                    code_i = int(code)
                    value = int(parts[3].strip())
                except ValueError:
                    continue
                if code_i not in GLUCOSE_CODES:
                    continue
                try:
                    ts = pd.to_datetime(f"{date_s} {time_s}")
                except ValueError:
                    continue
                rows.append((pid, ts, float(value)))
    df = pd.DataFrame(rows, columns=["patient_id", "timestamp",
                                     "glucose_mgdl"])
    # Same-minute duplicate codes (48 + 58 are both glucose) collapse to the
    # mean reading -- the resampler cannot reindex on duplicate labels.
    df = (df.groupby(["patient_id", "timestamp"], as_index=False)
          ["glucose_mgdl"].mean())
    df = df.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}: {len(df):,} glucose readings from "
          f"{df['patient_id'].nunique()} patients ({SOURCE})")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data_dir")
    ap.add_argument("--out", default="data/real/uci_cgm.csv")
    args = ap.parse_args()
    convert(args.data_dir, args.out)