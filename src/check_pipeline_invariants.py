"""Study A pre-flight sanity checks: verifies data_loading.py's and
metrics.py's core invariants against the real metadata/split before any
GPU-hours are spent on the actual training sweep. Not part of CLAUDE.md's
documented Study A file layout — added as a reviewer-requested pre-flight
gate; see CHANGELOG.md for the entry recording this addition.

Each check is independent (a failure in one doesn't stop the others from
running) so a single run reports the full picture. Exits 1 if any check
fails.

Run with: python src/check_pipeline_invariants.py
"""

import os
import sys
import traceback

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import data_loading as dl
import metrics as m
import train as tr

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


@check
def image_files_exist_on_disk():
    """Every image referenced in the metadata CSV has a corresponding file
    in DEFAULT_IMAGE_DIR — a missing file would otherwise surface as a
    crash hours into an unattended training run.
    """
    metadata = dl.load_metadata()
    csv_images = set(metadata["image_id"])
    dir_images = {f for f in os.listdir(dl.DEFAULT_IMAGE_DIR) if f.endswith(".png")}
    missing = csv_images - dir_images
    assert not missing, f"{len(missing)} images referenced in metadata are missing on disk, e.g. {list(missing)[:5]}"


@check
def patient_sex_is_consistent():
    """Every patient_id maps to exactly one true_sex value — a patient
    split by sex (majority/minority pools) depends on this.
    """
    metadata = dl.load_metadata()
    n_values_per_patient = metadata.groupby("patient_id")["true_sex"].nunique()
    inconsistent = n_values_per_patient[n_values_per_patient > 1]
    assert len(inconsistent) == 0, f"{len(inconsistent)} patients have >1 distinct sex value"


@check
def canonical_split_has_no_leakage():
    """No patient appears in more than one of train/val/test, and every
    patient in the metadata is assigned to exactly one split.
    """
    metadata = dl.load_metadata()
    split_df = dl.get_patient_split(metadata)

    dupes = split_df["patient_id"][split_df["patient_id"].duplicated()]
    assert len(dupes) == 0, f"{len(dupes)} patient_ids appear more than once in patient_split.csv"

    all_patients = set(metadata["patient_id"].unique())
    split_patients = set(split_df["patient_id"])
    assert all_patients == split_patients, (
        f"split covers {len(split_patients)} patients but metadata has {len(all_patients)} "
        f"— {len(all_patients - split_patients)} missing, {len(split_patients - all_patients)} extra"
    )


@check
def undersampled_training_sets_stay_within_train_split():
    """build_training_set's chosen patients, for every arm, are a subset of
    the canonical split's train patients — undersampling must never pull
    from val/test.
    """
    metadata = dl.load_metadata()
    split_df = dl.get_patient_split(metadata)
    train_patients = set(split_df.loc[split_df["split"] == "train", "patient_id"])

    for arm in dl.IMBALANCE_RATIOS:
        train_df = dl.build_training_set(metadata, split_df, arm)
        chosen = set(train_df["patient_id"])
        assert chosen <= train_patients, f"arm {arm}: {len(chosen - train_patients)} chosen patients fall outside the train split"


@check
def training_budget_is_fixed_across_arms():
    """Total training-patient count (N_total) is identical across all three
    imbalance arms, per CLAUDE.md's Design Decisions (undersampling caps
    N_total the same way for every arm — only composition varies).
    """
    metadata = dl.load_metadata()
    split_df = dl.get_patient_split(metadata)

    patient_counts = {}
    for arm in dl.IMBALANCE_RATIOS:
        train_df = dl.build_training_set(metadata, split_df, arm)
        patient_counts[arm] = train_df["patient_id"].nunique()

    assert len(set(patient_counts.values())) == 1, f"training patient counts differ across arms: {patient_counts}"


@check
def achieved_sex_ratio_matches_requested_ratio():
    """Each arm's undersampled training set actually hits its target
    male/female ratio (within a small rounding tolerance).
    """
    metadata = dl.load_metadata()
    split_df = dl.get_patient_split(metadata)

    for arm, ratio in dl.IMBALANCE_RATIOS.items():
        train_df = dl.build_training_set(metadata, split_df, arm)
        sex_counts = train_df.drop_duplicates("patient_id")["true_sex"].value_counts()
        n_total = sex_counts.sum()
        achieved_minority_frac = sex_counts.get(dl.MINORITY_SEX, 0) / n_total
        target_minority_frac = ratio[dl.MINORITY_SEX]
        assert abs(achieved_minority_frac - target_minority_frac) < 0.01, (
            f"arm {arm}: achieved minority fraction {achieved_minority_frac:.4f} vs. "
            f"target {target_minority_frac:.4f}"
        )


@check
def eval_sets_are_fixed_and_representative_across_arms():
    """val/test sets don't depend on which arm is training (they're built
    from split_df alone, not from any arm-specific undersampling), and
    their sex composition matches the overall split's — i.e. they're not
    accidentally rebalanced like the training set is.
    """
    metadata = dl.load_metadata()
    split_df = dl.get_patient_split(metadata)

    for split_name in ("val", "test"):
        eval_df = dl.get_fixed_eval_set(metadata, split_df, split_name)
        eval_sex_counts = eval_df.drop_duplicates("patient_id")["true_sex"].value_counts(normalize=True)

        split_patients = split_df.loc[split_df["split"] == split_name, "patient_id"]
        full_sex = metadata.drop_duplicates("patient_id").set_index("patient_id")["true_sex"]
        overall_sex_counts = full_sex.reindex(split_patients).value_counts(normalize=True)

        for sex in (dl.MAJORITY_SEX, dl.MINORITY_SEX):
            assert abs(eval_sex_counts.get(sex, 0) - overall_sex_counts.get(sex, 0)) < 0.01, (
                f"{split_name} set's {sex} fraction ({eval_sex_counts.get(sex, 0):.4f}) doesn't match "
                f"the split's overall {sex} fraction ({overall_sex_counts.get(sex, 0):.4f})"
            )


@check
def split_generation_is_deterministic_per_seed():
    """Same seed -> identical split (idempotent, since patient_split.csv is
    only ever written once and loaded thereafter). Different seed ->
    different assignment (confirms the seed actually has an effect, not a
    no-op).
    """
    patient_ids = np.array([f"p{i}" for i in range(2000)])

    split_a = dl._generate_split_df(patient_ids, seed=999)
    split_b = dl._generate_split_df(patient_ids, seed=999)
    pd.testing.assert_frame_equal(split_a, split_b)

    split_c = dl._generate_split_df(patient_ids, seed=998)
    assert not split_a["patient_id"].equals(split_c["patient_id"]), "different seeds produced an identical patient order"


@check
def patient_level_auc_matches_hand_computed_value():
    """Sanity check patient_level_auc/subgroup_auc_gap's aggregation (max
    label, mean score) against a small hand-constructed example with a
    known-by-hand AUC, so a future change to the aggregation logic can't
    silently break without a test noticing.
    """
    df = pd.DataFrame(
        [
            # Patient p1: two images, true_label max=1, score mean=0.9 -> clearly positive-scored
            {"patient_id": "p1", "image_id": "p1_0", "true_label": 0, "predicted_score": 0.8, "true_sex": "M"},
            {"patient_id": "p1", "image_id": "p1_1", "true_label": 1, "predicted_score": 1.0, "true_sex": "M"},
            # Patient p2: negative, low score
            {"patient_id": "p2", "image_id": "p2_0", "true_label": 0, "predicted_score": 0.1, "true_sex": "M"},
            # Patient p3: positive, low score (the "wrong" one, to make AUC non-trivial)
            {"patient_id": "p3", "image_id": "p3_0", "true_label": 1, "predicted_score": 0.3, "true_sex": "M"},
        ]
    )
    # Aggregated: p1 -> (label=1, score=0.9), p2 -> (label=0, score=0.1), p3 -> (label=1, score=0.3)
    # Positives {p1: 0.9, p3: 0.3}, negative {p2: 0.1}. Both positives rank above the negative -> AUC = 1.0
    expected_auc = roc_auc_score([1, 0, 1], [0.9, 0.1, 0.3])
    got_auc = m.patient_level_auc(df)
    assert abs(got_auc - expected_auc) < 1e-9, f"expected {expected_auc}, got {got_auc}"
    assert abs(got_auc - 1.0) < 1e-9, f"hand-computed expectation itself should be 1.0, got {expected_auc}"


@check
def bootstrap_ci_is_internally_coherent():
    """The bootstrap CI's point estimate falls within its own reported
    interval, and it returns the requested number of resamples — a basic
    coherence check on bootstrap_gap_ci, not a statistical claim.
    """
    rng = np.random.RandomState(0)
    n_maj, n_min = 200, 80
    rows = []
    for i in range(n_maj):
        label = int(rng.rand() < 0.1)
        score = rng.rand() * 0.5 + (0.3 if label else 0.0)
        rows.append({"patient_id": f"M{i}", "image_id": f"M{i}_0", "true_label": label, "predicted_score": score, "true_sex": "M"})
    for i in range(n_min):
        label = int(rng.rand() < 0.1)
        score = rng.rand() * 0.9
        rows.append({"patient_id": f"F{i}", "image_id": f"F{i}_0", "true_label": label, "predicted_score": score, "true_sex": "F"})
    df = pd.DataFrame(rows)

    result = m.bootstrap_gap_ci(df, n_resamples=200)
    assert result["n_resamples"] == 200
    assert result["ci_lower"] <= result["point_estimate"] <= result["ci_upper"], (
        f"point estimate {result['point_estimate']} outside its own CI "
        f"[{result['ci_lower']}, {result['ci_upper']}]"
    )


def main():
    results = []
    for fn in CHECKS:
        try:
            fn()
            results.append((fn.__name__, True, None))
        except Exception as exc:  # noqa: BLE001 — collecting all failures, not just the first
            results.append((fn.__name__, False, f"{type(exc).__name__}: {exc}"))

    print(f"{'CHECK':<50} {'RESULT'}")
    print("-" * 65)
    for name, passed, error in results:
        print(f"{name:<50} {'PASS' if passed else 'FAIL'}")
        if error:
            print(f"    {error}")

    n_failed = sum(1 for _, passed, _ in results if not passed)
    print("-" * 65)
    print(f"{len(results) - n_failed}/{len(results)} checks passed")

    if n_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
