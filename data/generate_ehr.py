"""
generate_ehr.py
----------------
Generates a synthetic Electronic Health Record (EHR) population.

This mimics the *kind* of fields a real tool like Synthea or an anonymized
extract from MIMIC-IV would provide, WITHOUT using any real patient data
(per DPDP Act / HIPAA constraints of the challenge sandbox).

Fields:
  Demographics       : age, sex, bmi
  Past diagnoses      : type2_diabetes, hypertension, prior_cardiac_event
  Lab results         : hba1c, fasting_glucose, ldl_cholesterol, egfr
  Genetic markers     : tcf7l2_risk_allele (T2D risk), apoe4_risk_allele (CV risk)

Output: data/ehr_patients.csv  (one row per patient)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

# Allow `python data/generate_ehr.py` from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import config as C  # noqa: E402

RNG_SEED = 42


def generate_ehr(n_patients: int = C.N_PATIENTS, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    patient_id = [f"P{100000 + i}" for i in range(n_patients)]
    age = rng.integers(22, 85, size=n_patients)
    sex = rng.choice(["F", "M"], size=n_patients)

    # BMI correlated loosely with age
    bmi = rng.normal(26 + (age - 50) * 0.03, 4.5, size=n_patients).clip(16, 48)

    # Genetic markers (minor allele carriage, population-plausible frequencies)
    tcf7l2_risk_allele = rng.choice([0, 1, 2], size=n_patients, p=[0.49, 0.42, 0.09])
    apoe4_risk_allele = rng.choice([0, 1, 2], size=n_patients, p=[0.60, 0.32, 0.08])

    # Diagnosis probability is a function of age, bmi, and genetic risk.
    t2d_logit = -6.5 + 0.045 * age + 0.11 * (bmi - 25) + 0.9 * tcf7l2_risk_allele
    type2_diabetes = rng.binomial(1, 1 / (1 + np.exp(-t2d_logit)))

    htn_logit = -4.2 + 0.05 * age + 0.05 * (bmi - 25) + 0.4 * type2_diabetes
    hypertension = rng.binomial(1, 1 / (1 + np.exp(-htn_logit)))

    cardiac_logit = (-5.5 + 0.05 * age + 0.7 * apoe4_risk_allele
                     + 0.6 * hypertension + 0.4 * type2_diabetes)
    prior_cardiac_event = rng.binomial(1, 1 / (1 + np.exp(-cardiac_logit)))

    # Labs, shifted upward for diabetic / at-risk patients
    hba1c = rng.normal(5.2 + 1.8 * type2_diabetes + 0.01 * (age - 40), 0.4,
                       size=n_patients).clip(4.5, 12.5)
    fasting_glucose = rng.normal(90 + 45 * type2_diabetes + 0.3 * (bmi - 25), 10,
                                 size=n_patients).clip(70, 260)
    ldl_cholesterol = rng.normal(100 + 20 * hypertension + 0.4 * (age - 40), 20,
                                 size=n_patients).clip(50, 220)
    egfr = rng.normal(95 - 0.4 * age - 5 * hypertension, 12,
                      size=n_patients).clip(15, 130)

    return pd.DataFrame({
        "patient_id": patient_id,
        "age": age,
        "sex": sex,
        "bmi": bmi.round(1),
        "type2_diabetes": type2_diabetes,
        "hypertension": hypertension,
        "prior_cardiac_event": prior_cardiac_event,
        "hba1c_pct": hba1c.round(2),
        "fasting_glucose_mgdl": fasting_glucose.round(1),
        "ldl_cholesterol_mgdl": ldl_cholesterol.round(1),
        "egfr_ml_min": egfr.round(1),
        "tcf7l2_risk_allele_count": tcf7l2_risk_allele,
        "apoe4_risk_allele_count": apoe4_risk_allele,
    })


if __name__ == "__main__":
    # DIGITAL_TWIN_SEED drives both generator CLIs together: it lets the same
    # cohort be re-drawn under a different master seed for seed-robustness runs
    # (see src/make_seed_report.py). Unset preserves the canonical default.
    seed = int(os.environ.get("DIGITAL_TWIN_SEED") or RNG_SEED)
    df = generate_ehr(seed=seed)
    df.to_csv(f"{C.DATA_DIR}/ehr_patients.csv", index=False)
    print(f"Wrote {len(df)} synthetic patients -> data/ehr_patients.csv")
    print(f"  type2_diabetes prevalence: {df['type2_diabetes'].mean():.1%}")
    print(f"  hypertension prevalence  : {df['hypertension'].mean():.1%}")
    print(f"  prior_cardiac_event      : {df['prior_cardiac_event'].mean():.1%}")
