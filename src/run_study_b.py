"""Study B: does a DP-protected sex label preserve the true subgroup AUC
gap, or wash it out? See CLAUDE.md, Study B — Core Contribution.

Reads exactly two files from Study A's frozen output —
results/study_a/predictions_90_10.csv and results/study_a/patient_split.csv
(confirmation only, to check we're scoring the same test set — never to
re-split or resample) — both hardcoded below, never globbed (see
CLAUDE.md, "What results/study_a/ actually contains after the merge": 26
other CSVs live alongside these two and must not be read).

Does not import src/train.py, src/data_loading.py, or src/metrics.py —
those are Study A's. The patient-level aggregation rule they encode (label
by max, score by mean per patient — CLAUDE.md's Multi-image aggregation
decision) is reimplemented directly below instead of reused as a
dependency, so Study B's only import from Study A's code is none at all;
its only shared import is src/dp_mechanisms.py.

Subgroup assignment mechanism (see CLAUDE.md, Subgroup Assignment
Mechanism — reworked 2026-08-19 after a technical review found the
original aggregate-count-reassignment design didn't actually produce
privacy-protected data): every test patient's true sex label is passed
independently through dp_mechanisms.privatize_categorical_label
(randomized response). The resulting per-patient labels are genuine DP
output, not a reconstruction from an aggregate release, so subgroup AUC
computed on them directly tests what the study asks.

Run with: python src/run_study_b.py
"""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import dp_mechanisms as dp

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDICTIONS_PATH = os.path.join(REPO_ROOT, "results", "study_a", "predictions_90_10.csv")
SPLIT_PATH = os.path.join(REPO_ROOT, "results", "study_a", "patient_split.csv")
OUTPUT_PATH = os.path.join(REPO_ROOT, "results", "study_b", "epsilon_sweep_results.csv")

MAJORITY_SEX = "M"
MINORITY_SEX = "F"

SURVIVAL_PCT_THRESHOLD = 0.15
# Seeds every per-epsilon label-privatization draw, deterministically, in
# EPSILON_SWEEP order — see run_epsilon_sweep.
BASE_SEED = 42


def _patient_table(df):
    """One row per patient: label by max, score by mean, sex constant per
    patient — CLAUDE.md's Multi-image aggregation rule, reimplemented here
    rather than imported (see module docstring)."""
    return df.groupby("patient_id").agg(
        label=("true_label", "max"),
        score=("predicted_score", "mean"),
        sex=("true_sex", "first"),
    )


def _subgroup_auc_gap(patient_table, sex_series):
    """Majority-minority patient-level AUC gap for the given per-patient
    sex assignment — `sex_series` may be the true sex or a DP-privatized
    one; `patient_table`'s label/score columns are always the frozen,
    unmodified ones.

    At low enough epsilon, randomized response can (rarely) flip an entire
    subgroup to size 0 (AUC undefined) — this is itself a meaningful
    outcome (total subgroup erasure, not a subtler washing-out), not an
    error case to hide, so it returns None rather than letting
    roc_auc_score raise.
    """
    majority_mask = sex_series == MAJORITY_SEX
    minority_mask = sex_series == MINORITY_SEX
    if majority_mask.sum() == 0 or minority_mask.sum() == 0:
        return None
    majority_auc = roc_auc_score(patient_table.loc[majority_mask, "label"], patient_table.loc[majority_mask, "score"])
    minority_auc = roc_auc_score(patient_table.loc[minority_mask, "label"], patient_table.loc[minority_mask, "score"])
    return majority_auc - minority_auc


def load_frozen_test_set():
    """Loads the two allowed files and returns the patient-level test-set
    table, filtered to patient_split.csv's test patients (confirmation
    only — this is Study A's already-frozen split, not re-derived here)."""
    predictions = pd.read_csv(PREDICTIONS_PATH)
    split = pd.read_csv(SPLIT_PATH)

    test_patients = set(split.loc[split["split"] == "test", "patient_id"])
    predictions_test = predictions[predictions["patient_id"].isin(test_patients)]
    assert not predictions_test.empty, "no rows matched the test split — check patient_split.csv join"

    patients = _patient_table(predictions_test)
    assert set(patients["sex"].unique()) <= {MAJORITY_SEX, MINORITY_SEX}, patients["sex"].unique()
    return patients


def privatize_subgroups(patients, epsilon, seed):
    """DP-protected per-patient subgroup assignment: every patient's true
    sex label is passed independently through
    dp_mechanisms.privatize_categorical_label (randomized response) — a
    genuine per-record DP draw for every patient, not a reconstruction
    from an aggregate release. See module docstring and CLAUDE.md's
    Subgroup Assignment Mechanism for why this replaced the original
    aggregate-count-based design.
    """
    noisy = dp.privatize_categorical_label(
        patients["sex"].tolist(), epsilon, value0=MAJORITY_SEX, value1=MINORITY_SEX, random_state=seed
    )
    return pd.Series(noisy, index=patients.index)


def run_epsilon_sweep():
    patients = load_frozen_test_set()
    true_gap = _subgroup_auc_gap(patients, patients["sex"])

    master_rng = np.random.default_rng(BASE_SEED)
    rows = []
    for epsilon in dp.EPSILON_SWEEP:
        seed = int(master_rng.integers(0, 2**31 - 1))

        dp_sex = privatize_subgroups(patients, epsilon, seed)
        dp_gap = _subgroup_auc_gap(patients, dp_sex)

        if dp_gap is None:
            # A whole subgroup was privatized to size 0 — total erasure,
            # not a subtler washing-out. Recorded as NaN/False rather than
            # skipped, so it's visible in the CSV, not silently dropped.
            direction_match, pct_diff, survived = False, float("nan"), False
        else:
            direction_match = (dp_gap > 0) == (true_gap > 0)
            pct_diff = abs(dp_gap - true_gap) / abs(true_gap)
            survived = bool(direction_match and pct_diff <= SURVIVAL_PCT_THRESHOLD)

        rows.append({
            "epsilon": epsilon,
            "true_gap": true_gap,
            "dp_gap": dp_gap if dp_gap is not None else float("nan"),
            "direction_match": bool(direction_match),
            "pct_diff": pct_diff,
            "survived": survived,
        })

    return pd.DataFrame(rows)


def crossover_epsilon(results):
    """Smallest epsilon at/above which every larger-or-equal epsilon in the
    (ascending) sweep also survives — more noise (lower epsilon) is
    expected to wash the gap out more, not less, so this is the point
    where "survived" becomes reliably True from there up. Returns None if
    no epsilon in the sweep survives; if the pattern isn't actually
    monotonic, this returns the last point from which it's True all the
    way up, and the caller should look at the full table, not just this
    number."""
    ordered = results.sort_values("epsilon")
    survived = ordered["survived"].to_numpy()
    epsilons = ordered["epsilon"].to_numpy()
    for i in range(len(survived)):
        if survived[i:].all():
            return epsilons[i]
    return None


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    results = run_epsilon_sweep()
    results.to_csv(OUTPUT_PATH, index=False)

    crossover = crossover_epsilon(results)
    print(results.to_string(index=False))
    print(f"\nCrossover epsilon (smallest epsilon at/above which all larger epsilons also survive): {crossover}")


if __name__ == "__main__":
    main()
