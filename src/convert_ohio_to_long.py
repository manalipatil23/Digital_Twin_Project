"""
convert_ohio_to_long.py
-----------------------
Converts OhioT1DM v1.0 per-day XML files into the long-format schema
consumed by `src/validate_real_data.py`:

    patient_id, timestamp, glucose_mgdl

Requires a lawfully-obtained copy of the dataset (email request + DUA to the
Ohio University custodians, or PhysioNet credentialed access -- see
docs/REAL_DATA.md). No patient data is bundled with this repo, and the
PhysioNet DUA forbids redistributing the raw files.

OhioT1DM v1.0 ships one XML file per patient-day; 5-min CGM samples are
<glucoseLevel ts="..." value="..."/> records among meal/insulin/event tags.

Usage:
    python src/convert_ohio_to_long.py path/to/ohio_xml --out data/real/ohio_cgm.csv
"""
from __future__ import annotations

import argparse
import glob
import os
import xml.etree.ElementTree as ET

import pandas as pd


def _local(tag: str) -> str:
    """Tag name without an XML namespace prefix."""
    return tag.split("}")[-1].lower()


def _child_text(elem, *names: str) -> str | None:
    """First matching direct child's stripped text, or None."""
    for child in elem:
        if _local(child.tag) in names and (child.text or "").strip():
            return child.text.strip()
    return None


def _glucose_records(tree) -> list[tuple[str, str]]:
    """Pull (timestamp, value) pairs out of one OhioT1DM patient-day file.

    Two layouts appear across OhioT1DM releases, and this converter accepts
    both rather than guessing which one the reviewer downloaded:

      * attribute style -- <glucoseLevel ts="..." value="..."/>
      * element style   -- <BGReading><TimeStamp>...</TimeStamp>
                                        <Value>...</Value></BGReading>

    Unknown tags are ignored, so a file that mixes layouts (or carries extra
    sections such as meals, insulin and exercise) still converts.
    """
    records = []
    for elem in tree.iter():
        tag = _local(elem.tag)
        ts = val = None
        if tag in ("glucoselevel", "glucose", "glucosereading", "bgreading"):
            ts = elem.get("ts") or elem.get("timestamp") or elem.get("time")
            val = elem.get("value")
            if ts is None or val is None:
                ts = ts or _child_text(elem, "timestamp", "time", "ts")
                val = val if val is not None else _child_text(elem, "value")
        if ts and val is not None:
            records.append((ts, val))
    return records


def convert(data_dir: str, out_path: str) -> str:
    rows = []
    n_files = 0
    n_skipped = 0
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.xml"))):
        n_files += 1
        # Ohio file names look like "559-ws-training.xml" / "391-ws-testing.xml".
        pid = os.path.basename(fp).split("-")[0]
        try:
            tree = ET.parse(fp)
        except ET.ParseError as exc:
            print(f"  skip {os.path.basename(fp)}: {exc}")
            n_skipped += 1
            continue
        for ts, val in _glucose_records(tree):
            try:
                t = pd.to_datetime(ts)
                v = float(val)
            except (ValueError, TypeError):
                continue
            rows.append((pid, t, v))
    if not rows:
        raise SystemExit(
            f"No glucose readings found under {data_dir} ({n_files} XML "
            "files). Open one file and check its tag layout against "
            "_glucose_records() in this script.")
    df = pd.DataFrame(rows, columns=["patient_id", "timestamp",
                                     "glucose_mgdl"])
    df = (df.groupby(["patient_id", "timestamp"], as_index=False)
          ["glucose_mgdl"].mean())
    df = df.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    df.to_csv(out_path, index=False)
    note = f", {n_skipped} unparseable" if n_skipped else ""
    print(f"Wrote {out_path}: {len(df):,} CGM readings from "
          f"{df['patient_id'].nunique()} patients ({n_files} XML files{note})")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data_dir")
    ap.add_argument("--out", default="data/real/ohio_cgm.csv")
    args = ap.parse_args()
    convert(args.data_dir, args.out)