"""Study A robustness-note computations: the cross-training-seed gap spread
(Seed Replication) and the cross-split gap spread (Split Sensitivity). Both
are non-gating notes reported in CHANGELOG.md — the pass/fail test oracle
stays the single canonical seed=42, canonical-split 90/10 run; this module
never decides pass/fail, only summarizes robustness across the extra runs.

See CLAUDE.md, "Study A — Seed Replication" and "Study A — Split
Sensitivity" for the rules this module implements.
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import data_loading as dl
import train as tr


def patient_level_auc(df, score_col="predicted_score", label_col="true_label"):
    """Patient-level AUC from a predictions dataframe: ground truth
    aggregated by max, score aggregated by mean, across a patient's images
    (see CLAUDE.md, Multi-image aggregation).
    """
    agg = df.groupby("patient_id").agg(
        label=(label_col, "max"), score=(score_col, "mean")
    )
    return roc_auc_score(agg["label"], agg["score"])


def subgroup_auc_gap(df, sex_col="true_sex"):
    """Majority-minority patient-level AUC gap (majority - minority) — the
    quantity the Larrazabal-gap test oracle and both robustness notes check
    the direction/magnitude of.
    """
    majority_auc = patient_level_auc(df[df[sex_col] == dl.MAJORITY_SEX])
    minority_auc = patient_level_auc(df[df[sex_col] == dl.MINORITY_SEX])
    return majority_auc - minority_auc


def _gap_spread(canonical_gap, other_gaps):
    all_gaps = [canonical_gap] + list(other_gaps)
    return {
        "canonical_gap": canonical_gap,
        "other_gaps": list(other_gaps),
        "mean_gap": float(np.mean(all_gaps)),
        "min_gap": float(np.min(all_gaps)),
        "max_gap": float(np.max(all_gaps)),
        "direction_agreement": bool(
            all(np.sign(g) == np.sign(canonical_gap) for g in all_gaps)
        ),
    }


def cross_seed_gap_spread(canonical_gap, replicate_gaps):
    """Cross-training-seed robustness note (CLAUDE.md, Seed Replication):
    mean, range, and direction agreement of the 90/10 gap across the 5
    canonical-split seeds (42 canonical + 43-46 replicate), vs. the
    canonical seed=42 gap the oracle actually gates on. Non-gating.
    """
    return _gap_spread(canonical_gap, replicate_gaps)


def cross_split_gap_spread(canonical_gap, alternate_split_gaps):
    """Cross-split robustness note (CLAUDE.md, Split Sensitivity): mean,
    range, and direction agreement of the 90/10 gap across the 3 alternate
    70/15/15 splits (seeds 101-103, canonical training seed=42), vs. the
    canonical split's gap. Checks that the gap isn't an artifact of which
    patients landed in test, as distinct from cross_seed_gap_spread, which
    checks training stochasticity on the same (canonical) split. Non-gating.
    """
    return _gap_spread(canonical_gap, alternate_split_gaps)


def _load_predictions(path):
    return pd.read_csv(path, dtype={"patient_id": str, "image_id": str})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--note", choices=["seed", "split", "both"], default="both",
        help="which robustness note to compute",
    )
    args = parser.parse_args()

    canonical_path = tr._output_path(tr.SPLIT_SENSITIVITY_ARM, dl.SEED)
    canonical_gap = subgroup_auc_gap(_load_predictions(canonical_path))
    print(f"canonical (seed=42, canonical split) gap: {canonical_gap:.4f}")

    if args.note in ("seed", "both"):
        replicate_seeds = [s for s in tr.REPLICATION_SEEDS["90_10"] if s != dl.SEED]
        replicate_gaps = [
            subgroup_auc_gap(_load_predictions(tr._output_path("90_10", seed)))
            for seed in replicate_seeds
        ]
        result = cross_seed_gap_spread(canonical_gap, replicate_gaps)
        print("cross-seed gap spread:", result)

    if args.note in ("split", "both"):
        alternate_gaps = [
            subgroup_auc_gap(
                _load_predictions(
                    os.path.join(
                        tr.SPLIT_SENSITIVITY_DIR,
                        f"predictions_{tr.SPLIT_SENSITIVITY_ARM}_split{seed}.csv",
                    )
                )
            )
            for seed in dl.SPLIT_SENSITIVITY_SEEDS
        ]
        result = cross_split_gap_spread(canonical_gap, alternate_gaps)
        print("cross-split gap spread:", result)


if __name__ == "__main__":
    main()
