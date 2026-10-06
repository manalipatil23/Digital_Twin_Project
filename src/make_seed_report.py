"""
make_seed_report.py
-------------------
Compiles docs/SEED_ROBUSTNESS.md from the per-seed metrics snapshots in
model/seeds/metrics_*.json -- one file per generator master seed from the
seed-robustness sweep (src/scripts in repo root, or read ci.yml for where
the sweep is wired). The model and CV protocols are held fixed across seeds,
so the spread isolates generator randomness.

Generated file -- do not edit by hand; the header is written by this script.
"""
from __future__ import annotations

import glob
import json
import os

import config as C


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _seed_sort_key(path: str) -> tuple[int, int, str]:
    """Canonical run first, then sweep seeds in numeric (not string) order.

    Plain alphabetical sorting would render 123, 21, 7, 84 -- which reads as a
    bug in a document whose entire job is reporting an ordered spread.
    """
    label = os.path.basename(path)[len("metrics_"):-len(".json")]
    if label.startswith("seed"):
        try:
            return (1, int(label[4:]), label)
        except ValueError:
            return (1, 10**6, label)
    return (0, 0, label)


def _fmt_p(p: float | None) -> str:
    """Render a bootstrap p the same way make_report does: a bound, never 0."""
    if p is None:
        return "—"
    if p < 0.001:
        return "< 0.001"
    if p == 0.0005:
        return "0.0005"
    return f"{p:.3f}"


def main() -> None:
    files = sorted(glob.glob(f"{C.MODEL_DIR}/seeds/metrics_*.json"),
                   key=_seed_sort_key)
    if not files:
        raise SystemExit(
            "No model/seeds/metrics_*.json found. Run the seed sweep first:\n"
            "for each seed S: set DIGITAL_TWIN_SEED=S, run the full pipeline,\n"
            "copy model/metrics.json to model/seeds/metrics_<label>.json")

    rows = []
    for fp in files:
        m = _load(fp)
        base = os.path.basename(fp)
        label = base[len("metrics_"):-len(".json")]
        abl = m["ablation"]["full_fusion"]
        perm = m["permutation_controls"]
        dec = m["feature_value_decomposition"]
        rows.append({
            "label": label,
            "auc": abl["roc_auc_mean"],
            "auc_std": abl["roc_auc_std"],
            "pr_auc": abl["average_precision_mean"],
            "base_rate": m["positive_rate"],
            "global_perm": perm["global_label_permutation_auc"],
            "floor": perm["within_patient_permutation_auc"],
            "observed": perm["observed_auc"],
            "d_short": dec["full_model_vs_shortcut_control"]["delta_roc_auc"],
            "p_short": dec["full_model_vs_shortcut_control"].get("bootstrap_p"),
            "d_best": dec.get("full_model_vs_best_single_stream", {}).get("delta_roc_auc"),
            "p_best": dec.get("full_model_vs_best_single_stream", {}).get("bootstrap_p"),
            "runtime": m.get("runtime_seconds"),
        })

    aucs = [r["auc"] for r in rows]
    spread = max(aucs) - min(aucs)
    mean_auc = sum(aucs) / len(aucs)
    sd_auc = (sum((a - mean_auc) ** 2 for a in aucs) / len(aucs)) ** 0.5
    significant = sum(1 for r in rows if (r["p_short"] or 1.0) < 0.05)
    bases = [r["base_rate"] for r in rows]
    best_ns = [r for r in rows if r["p_best"] is not None]
    fusion_pos = sum(1 for r in best_ns if r["d_best"] > 0)
    fusion_sig = sum(1 for r in best_ns if (r["p_best"] or 1.0) < 0.05)
    # Significance must be counted *by direction*. Reporting "1/5 significant"
    # without saying which way it points is how a table ends up reading as
    # support for the very thing it contradicts.
    fusion_sig_neg = sum(1 for r in best_ns
                         if r["d_best"] < 0 and (r["p_best"] or 1.0) < 0.05)
    fusion_sig_pos = fusion_sig - fusion_sig_neg
    n = len(best_ns)

    # Say out loud where the shipped run sits inside its own spread. A reader
    # who has to notice that the canonical seed is the table maximum has been
    # asked to do the author's honesty work for them. Note the headline is NOT
    # that row any more -- see headline_line.
    canon = next((r for r in rows if r["label"] == "canonical"), None)
    if canon is None:
        canonical_note = ""
    elif canon["auc"] >= max(aucs) - 1e-12:
        canonical_note = (
            "**The shipped `.joblib` and dashboard were fitted on the canonical "
            "cohort, whose AUC is the maximum of this range.** It is listed "
            "because it describes the artefact that ships, not because it is "
            "the central estimate. ")
    elif canon["auc"] <= min(aucs) + 1e-12:
        canonical_note = (
            "**The shipped `.joblib` and dashboard were fitted on the "
            "canonical cohort, whose AUC is the minimum of this range.** ")
    else:
        canonical_note = (
            f"The shipped artefact's cohort sits inside this range, "
            f"{canon['auc'] - mean_auc:+.4f} from the mean. ")

    # The headline is the mean over draws, not one draw. Naming any single seed
    # as the headline is a selection decision, and the canonical seed turns out
    # to be the maximum of the range -- so the only selection-free summary of
    # generator randomness is the mean across all of them.
    headline_line = (
        f"**Headline: {mean_auc:.4f} ± {sd_auc:.4f} ROC-AUC, mean ± sd over "
        f"{len(rows)} generator seeds.** This is the number quoted in the "
        "README. It is deliberately not a single seed: the sweep exists "
        "precisely because one draw is not a central estimate, and picking the "
        "seed that reads best would reintroduce the selection the sweep was "
        "run to remove. The shipped model artefact was fitted on the canonical "
        "cohort, so its own score is a per-draw number and is reported as such "
        f"({canon['auc']:.4f}) wherever an artefact-level figure is needed."
        if canon is not None else
        "run to remove. No canonical snapshot is present, so there is no "
        "artefact-level figure to report.")
    if fusion_pos <= n / 2:
        fusion_verdict = (
            f"fusion is **below** the best non-fusion subset in "
            f"{n - fusion_pos}/{n} cohorts, and significantly below in "
            f"{fusion_sig_neg}/{n}. The shipped canonical run is in the "
            f"minority where fusion is ahead, so its positive delta is a "
            f"property of that cohort draw, not of the method.")
    else:
        fusion_verdict = (
            f"fusion is above the best non-fusion subset in {fusion_pos}/{n} "
            f"cohorts (significantly so in {fusion_sig_pos}/{n}); the delta "
            f"is small relative to its own spread across draws.")

    L = [
        "# Seed robustness",
        "",
        "Generated from `model/seeds/` by `src/make_seed_report.py` — do not edit.",
        "",
        "Protocol: the full 500-patient / 21-day pipeline run under each "
        "generator master seed in `DIGITAL_TWIN_SEED` (both generator stages "
        "draw from it; per-patient streams are still hashed from the "
        "identifier). The model, the 5-fold grouped-CV protocol and the "
        "estimator seeds are held fixed, so the spread below isolates "
        "generator randomness — this is the answer to “did one lucky data "
        "draw carry the headline?”.",
        "",
        "| seed | headline AUC | PR-AUC | base rate | global perm | "
        "within-patient floor | observed (same split) | +Δ vs shortcut (p) | "
        "+Δ vs best non-fusion (p) | train (s) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        L.append(
            f"| {r['label']} | {r['auc']:.4f} ± {r['auc_std']:.4f} "
            f"| {r['pr_auc']:.3f} | {r['base_rate']:.2%} "
            f"| {r['global_perm']:.3f} | {r['floor']:.3f} "
            f"| {r['observed']:.3f} | {r['d_short']:+.4f} ({_fmt_p(r['p_short'])}) "
            f"| {r['d_best']:+.4f} ({_fmt_p(r['p_best'])}) "
            f"| {r['runtime']:.0f} |")
    L += [
        "",
        headline_line,
        "",
        f"**Spread across seeds.** Headline AUC ranges {min(aucs):.4f}–{max(aucs):.4f} "
        f"(Δ {spread:.4f}); mean ± sd = {mean_auc:.4f} ± {sd_auc:.4f}. "
        f"{canonical_note}{'The headline is stable across cohort redraws (Δ < 0.02).' if spread < 0.02 else 'Treat any single-seed AUC as one draw, not the law.'}",
        "",
        f"**Significance stability.** Fusion vs shortcut p < 0.05 in "
        f"{significant}/{len(rows)} seeds. "
        f"{'The paired gain is a load-bearing, reproducible signal.' if significant == len(rows) else 'The paired gain does not reproduce at p < 0.05 in every seed — significance is seed-dependent and should be re-tested on real data.'}",
        "",
        f"**Fusion over the best non-fusion subset is not supported.** The "
        f"delta ranges {min(r['d_best'] for r in best_ns):+.4f} to "
        f"{max(r['d_best'] for r in best_ns):+.4f}; {fusion_verdict}",
        "",
        f"**Base rate** ranges {min(bases):.2%}–{max(bases):.2%}; the label and "
        "its controls stay in their designed operating regime across seeds.",
        "",
        "The `metrics_*.json` snapshots are committed alongside this report, so "
        "every number above is auditable from `model/seeds/`.",
        "",
    ]

    out = f"{C.ROOT}/docs/SEED_ROBUSTNESS.md"
    os.makedirs(f"{C.ROOT}/docs", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"Wrote {out} ({len(L)} lines, {len(rows)} seeds)")


if __name__ == "__main__":
    main()