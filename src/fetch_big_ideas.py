"""Fetch the BIG IDEAs Lab Glycemic Variability and Wearable Device Data set.

PhysioNet, Open Access, Open Data Commons Attribution License v1.0 (ODC-By), so
the data is redistributable -- unlike OhioT1DM / MIMIC-IV, which are
credential-gated and whose DUAs forbid redistribution. Raw files still stay out
of git (``.gitignore`` covers ``data/real/``); only derived metrics are
committed.

This is the cohort that makes a *paired* real-data fusion test possible:

  * ``Dexcom_<id>.csv``   -- G6 interstitial glucose, 5 min, the label source
  * ``HR_<id>.csv``       -- Empatica E4 heart rate, 1 Hz/min, wearable stream
  * ``IBI_<id>.csv``      -- inter-beat intervals, the HRV source
  * ``Food_Log_<id>.csv`` -- meal timing + carbohydrate, a treatment record
  * ``Demographics.csv``  -- gender + HbA1c, the static EHR arm

The raw BVP/ACC/EDA/TEMP streams are deliberately NOT fetched: at ~2.3 GB per
participant for BVP+ACC they are impractical, and the lab's derived HR and IBI
series already carry the autonomic signal the model uses.

Usage:
    python src/fetch_big_ideas.py --out data/real/bigideas
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://physionet.org/files/big-ideas-glycemic-wearable/1.1.3"
ROOT_FILES = ["Demographics.csv", "LICENSE.txt", "SHA256SUMS.txt"]
PER_PATIENTANT = ["Dexcom_{id}.csv", "HR_{id}.csv", "IBI_{id}.csv", "Food_Log_{id}.csv"]
N_PATIENTANTS = 16

# Provenance for the manifest written next to the data. Kept here rather than
# scraped so the committed manifest cannot silently drift from the download.
PROVENANCE = {
    "dataset": "BIG IDEAs Lab Glycemic Variability and Wearable Device Data",
    "physionet_slug": "big-ideas-glycemic-wearable",
    "version": "1.1.3",
    "doi": "10.13026/aw6y-fc44",
    "doi_latest": "10.13026/w591-tp72",
    "access": "Open Access",
    "license": "Open Data Commons Attribution License v1.0",
    "institution": "Duke University (BIG IDEAs Lab, J. P. Dunn)",
    "n_participants": N_PATIENTANTS,
    "wearable": "Empatica E4 (HR, IBI); PPG/EDA/TEMP/ACC present but not fetched",
    "cgm": "Dexcom G6, interstitial glucose, 5 min",
    "static_ehr": "Demographics.csv: gender, HbA1c",
    "fetched_streams": ["Dexcom", "HR", "IBI", "Food_Log", "Demographics"],
    "skipped_streams": ["BVP", "ACC", "EDA", "TEMP"],
    "skip_reason": "~2.3 GB/participant for BVP+ACC; HR+IBI carry the autonomic signal",
}


def _parse_sums(text: str) -> dict[str, str]:
    """Parse a PhysioNet SHA256SUMS.txt into {relative_path: hexdigest}."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        digest, name = parts
        out[name.strip().lstrip("*").replace("\\", "/")] = digest.strip().lower()
    return out


def _download(url: str, dest: Path, expect_len: int | None = None,
              attempts: int = 6) -> int:
    """Stream ``url`` to ``dest``, resuming and retrying.

    PhysioNet streams these files slowly enough that a single 10 MB IBI series
    will intermittently stall mid-transfer and trip a read timeout. Without
    resume, every stall discards the whole file and the run loses a patient's
    HRV series -- which is the channel the model's top feature is built from.
    So: keep partial bytes in a sidecar ``.part`` file, ask for the remainder
    with a Range request, and back off between attempts. Only a file that
    reaches its advertised Content-Length is promoted into place.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_err: Exception | None = None

    for attempt in range(1, attempts + 1):
        have = tmp.stat().st_size if tmp.exists() else 0
        # Already complete, or a server that ignored our Range would append.
        if expect_len is not None and have == expect_len:
            tmp.replace(dest)
            return have
        if expect_len is not None and have > expect_len:
            tmp.unlink(missing_ok=True)
            have = 0

        headers = {"User-Agent": "digital-twin-poc/3.3"}
        if have:
            headers["Range"] = f"bytes={have}-"
        req = urllib.request.Request(url, headers=headers)
        try:
            written = have
            last_log = time.time()
            with urllib.request.urlopen(req, timeout=90) as resp:
                mode = "ab" if (have and resp.status == 206) else "wb"
                if mode == "wb":
                    written = 0
                with tmp.open(mode) as fh:
                    while True:
                        chunk = resp.read(1 << 18)
                        if not chunk:
                            break
                        fh.write(chunk)
                        written += len(chunk)
                        now = time.time()
                        if now - last_log > 10:
                            last_log = now
                            pct = (f"{100 * written / expect_len:5.1f}%"
                                   if expect_len else "")
                            print(f"    {dest.name}: {written / 1e6:8.1f} MB {pct}",
                                  flush=True)
            if expect_len is not None and written < expect_len:
                raise OSError(
                    f"short read: {written} of {expect_len} bytes")
            tmp.replace(dest)
            return written
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last_err = exc
            back = min(2 ** attempt, 30)
            print(f"    retry {attempt}/{attempts} for {dest.name} after "
                  f"{type(exc).__name__}: {exc} (backoff {back}s)", flush=True)
            time.sleep(back)

    raise OSError(f"{url} failed after {attempts} attempts: {last_err}")


def _head_len(url: str) -> int | None:
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "digital-twin-poc/3.3"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            n = resp.headers.get("Content-Length")
            return int(n) if n else None
    except (urllib.error.URLError, ValueError, OSError):
        return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="data/real/bigideas", help="staging directory (gitignored)")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Root files first -- SHA256SUMS.txt is what lets us verify the rest.
    for name in ROOT_FILES:
        dest = out / name
        if dest.exists() and not args.force:
            print(f"  have {name} ({dest.stat().st_size / 1e3:.1f} kB)")
            continue
        print(f"  get  {name}")
        _download(f"{BASE}/{name}", dest)

    sums = _parse_sums((out / "SHA256SUMS.txt").read_text(encoding="utf-8", errors="replace"))
    print(f"  SHA256SUMS.txt lists {len(sums)} files")

    # 2. Per-participant streams. HR/IBI are ~10-13 MB each; Dexcom ~150 kB.
    manifest: list[dict] = []
    failures: list[dict] = []
    verified = mismatched = 0
    for pid in range(1, N_PATIENTANTS + 1):
        s = f"{pid:03d}"
        print(f"participant {s}")
        for pattern in PER_PATIENTANT:
            name = pattern.format(id=s)
            dest = out / s / name
            url = f"{BASE}/{s}/{name}"
            if not dest.exists() or args.force:
                want = _head_len(url)
                try:
                    n = _download(url, dest, want)
                except (urllib.error.URLError, OSError) as exc:
                    print(f"    GAVE UP {name}: {exc}")
                    failures.append({"participant": s, "file": name,
                                     "error": str(exc)[:200]})
                    continue
                print(f"    got {name} ({n / 1e6:.1f} MB)")
            else:
                n = dest.stat().st_size
                print(f"    have {name} ({n / 1e6:.1f} MB)")

            rel = f"{s}/{name}"
            want_sum = sums.get(name) or sums.get(rel) or sums.get(f"./{rel}")
            state = "not-listed"
            if want_sum:
                got = _sha256(dest)
                if got == want_sum:
                    state = "verified"
                    verified += 1
                else:
                    state = "MISMATCH"
                    mismatched += 1
            manifest.append({"participant": s, "file": rel, "bytes": n, "sha256_state": state})

    if mismatched:
        print(f"ERROR: {mismatched} file(s) failed checksum verification", file=sys.stderr)
        return 1

    if failures:
        print(f"\nWARNING: {len(failures)} file(s) could not be fetched after "
              f"retries. Partial bytes are kept alongside as .part; re-run this "
              f"script to resume.", file=sys.stderr)
        for f in failures:
            print(f"  {f['participant']}/{f['file']}: {f['error']}", file=sys.stderr)

    payload = {
        **PROVENANCE,
        "source_url": f"https://physionet.org/content/{PROVENANCE['physionet_slug']}/{PROVENANCE['version']}/",
        "n_files": len(manifest),
        "n_verified": verified,
        "n_unverified": len(manifest) - verified,
        "n_download_failures": len(failures),
        "failures": failures,
        "files": manifest,
    }
    (out / "provenance.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"\n{len(manifest)} files, {verified} checksum-verified, "
        f"{len(failures)} failures -> {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
