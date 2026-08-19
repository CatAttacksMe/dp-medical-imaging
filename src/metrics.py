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
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import data_loading as dl
import train as tr

# Dedicated bootstrap seed — distinct from the 42-46 training-seed pool and
# the 101-103 split-seed pool, so none of the three kinds of seed in this
# study are ever confused. See CLAUDE.md, Test oracle scope.
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 1042


def patient_level_auc(df, score_col="predicted_score", label_col="true_label"):
    """Patient-level AUC from a predictions dataframe: ground truth
    aggregated by max, score aggregated by mean, across a patient's images
    (see CLAUDE.md, Multi-image aggregation). Delegates to train.py's
    _patient_level_auc rather than reimplementing the aggregation, so the
    rule that early stopping optimizes against (train.py) and the rule the
    reported oracle/robustness gaps are computed with (here) can't
    silently diverge.
    """
    return tr._patient_level_auc(df["patient_id"], df[label_col], df[score_col])


def subgroup_auc_gap(df, sex_col="true_sex"):
    """Majority-minority patient-level AUC gap (majority - minority) — the
    quantity the Larrazabal-gap test oracle and both robustness notes check
    the direction/magnitude of.
    """
    majority_auc = patient_level_auc(df[df[sex_col] == dl.MAJORITY_SEX])
    minority_auc = patient_level_auc(df[df[sex_col] == dl.MINORITY_SEX])
    return majority_auc - minority_auc


def _patient_table(df, sex_col="true_sex"):
    """One row per patient: label aggregated by max, score by mean — the same
    rule as train.py's _patient_level_auc (CLAUDE.md, Multi-image
    aggregation) — plus sex (constant per patient). Computed once so
    bootstrap_gap_ci's ~1,000 resamples draw from patient-level rows
    directly instead of re-aggregating from images every iteration.
    """
    return df.groupby("patient_id").agg(
        label=("true_label", "max"),
        score=("predicted_score", "mean"),
        sex=(sex_col, "first"),
    )


def bootstrap_gap_ci(
    df, n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED, ci_level=0.95, sex_col="true_sex"
):
    """Patient-level bootstrap CI on the majority-minority AUC gap (CLAUDE.md,
    Test oracle scope): resamples patients with replacement, independently
    within each sex group (stratified, so each resample keeps both groups'
    sample sizes fixed), recomputes the gap each time, and reports a
    percentile CI. Non-gating — a robustness note alongside
    cross_seed_gap_spread/cross_split_gap_spread, not part of the pass/fail
    oracle.
    """
    patients = _patient_table(df, sex_col=sex_col)
    majority = patients[patients["sex"] == dl.MAJORITY_SEX]
    minority = patients[patients["sex"] == dl.MINORITY_SEX]

    rng = np.random.RandomState(seed)
    gaps = np.empty(n_resamples)
    for i in range(n_resamples):
        maj_sample = majority.sample(n=len(majority), replace=True, random_state=rng)
        min_sample = minority.sample(n=len(minority), replace=True, random_state=rng)
        gaps[i] = roc_auc_score(maj_sample["label"], maj_sample["score"]) - roc_auc_score(
            min_sample["label"], min_sample["score"]
        )

    alpha = (1 - ci_level) / 2
    ci_lower, ci_upper = float(np.quantile(gaps, alpha)), float(np.quantile(gaps, 1 - alpha))
    return {
        "point_estimate": subgroup_auc_gap(df, sex_col=sex_col),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_level": ci_level,
        "n_resamples": n_resamples,
        "excludes_zero": bool(ci_lower > 0 or ci_upper < 0),
    }


def check_oracle_direction(gap):
    """Automated half of CLAUDE.md's test oracle: same-direction check only
    (majority AUC > minority AUC, i.e. gap > 0 under the majority-minus-
    minority convention subgroup_auc_gap uses).

    The magnitude half of the oracle ("within 2 AUC points of Larrazabal et
    al.'s reported gap") is deliberately NOT automated. Larrazabal et al.
    (2020) report Pneumothorax results only as box plots (Fig. 1, panels
    B-2/C-2) across female-training ratios 0/25/50/75/100% — no 90/10
    point, and no numeric table in the text or SI Appendix. Hardcoding a
    number read off that figure would be false precision from a source that
    doesn't support it. The magnitude comparison stays a human judgment
    call against Figure 1 directly.
    """
    return gap > 0


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


def _load_predictions(path, run_hint):
    """Loads a predictions CSV, or raises with a pointer to the training
    command that produces it — a bare FileNotFoundError here wouldn't say
    which of train.py's several output paths is missing or why.
    """
    if not os.path.exists(path):
        raise RuntimeError(f"{path} not found — run `{run_hint}` first")
    return pd.read_csv(path, dtype={"patient_id": str, "image_id": str})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--note", choices=["seed", "split", "both"], default="both",
        help="which robustness note to compute",
    )
    args = parser.parse_args()

    canonical_path = tr._output_path(tr.SPLIT_SENSITIVITY_ARM, dl.SEED)
    canonical_df = _load_predictions(canonical_path, "python train.py --arm 90_10")
    canonical_gap = subgroup_auc_gap(canonical_df)
    print(f"canonical (seed=42, canonical split) gap: {canonical_gap:.4f}")

    direction_ok = check_oracle_direction(canonical_gap)
    print(f"oracle direction check (majority AUC > minority AUC): {'PASS' if direction_ok else 'FAIL'}")
    print(
        "oracle magnitude check (within 2 AUC points of Larrazabal et al.'s reported "
        "Pneumothorax gap) is NOT automated — compare canonical_gap above by eye against "
        "Larrazabal et al. 2020, Figure 1 panels B-2/C-2; the paper reports no numeric "
        "table and no 90/10 condition, so no reliable number exists to hardcode here."
    )

    bootstrap_result = bootstrap_gap_ci(canonical_df)
    print(
        f"bootstrap {bootstrap_result['ci_level']:.0%} CI ({bootstrap_result['n_resamples']} "
        f"resamples): [{bootstrap_result['ci_lower']:.4f}, {bootstrap_result['ci_upper']:.4f}], "
        f"excludes_zero={bootstrap_result['excludes_zero']}"
    )

    if args.note in ("seed", "both"):
        # 90/10 is the oracle-gated arm (full 5-seed spread); 70/30 and
        # 50/50 get a lighter 3-seed spread (canonical + 43-44) added
        # 2026-08-19 to check the cross-arm gap trend isn't a single-run
        # artifact — see CLAUDE.md, Study A Seed Replication. All of these
        # stay non-gating notes regardless of arm.
        for arm in tr.ARMS:
            seeds = tr.REPLICATION_SEEDS.get(arm)
            if not seeds:
                continue
            arm_canonical_gap = subgroup_auc_gap(
                _load_predictions(tr._output_path(arm, dl.SEED), f"python train.py --arm {arm}")
            )
            replicate_gaps = [
                subgroup_auc_gap(
                    _load_predictions(tr._output_path(arm, seed), f"python train.py --arm {arm}")
                )
                for seed in seeds if seed != dl.SEED
            ]
            result = cross_seed_gap_spread(arm_canonical_gap, replicate_gaps)
            print(f"cross-seed gap spread [{arm}]:", result)

    if args.note in ("split", "both"):
        alternate_gaps = [
            subgroup_auc_gap(
                _load_predictions(
                    os.path.join(
                        tr.SPLIT_SENSITIVITY_DIR,
                        f"predictions_{tr.SPLIT_SENSITIVITY_ARM}_split{seed}.csv",
                    ),
                    "python train.py --split-sensitivity",
                )
            )
            for seed in dl.SPLIT_SENSITIVITY_SEEDS
        ]
        result = cross_split_gap_spread(canonical_gap, alternate_gaps)
        print("cross-split gap spread:", result)

    if not direction_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
