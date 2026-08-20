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
from scipy.stats import norm
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
# EPSILON_SWEEP order — see run_epsilon_sweep. The canonical single-draw
# sweep and the seed-replication sweep use separate seed pools (BASE_SEED
# vs. REPLICATION_BASE_SEED) so neither can accidentally reproduce the
# other's draws.
BASE_SEED = 42
REPLICATION_BASE_SEED = 2000
N_REPLICATION_SEEDS = 30


def _patient_table(df):
    """One row per patient: label by max, score by mean, sex constant per
    patient — CLAUDE.md's Multi-image aggregation rule, reimplemented here
    rather than imported (see module docstring)."""
    return df.groupby("patient_id").agg(
        label=("true_label", "max"),
        score=("predicted_score", "mean"),
        sex=("true_sex", "first"),
    )


def _keep_probability(epsilon):
    """P(randomized response keeps the true value) for privatize_categorical_label's
    Binary mechanism — same formula as src/dp_mechanisms.py's docstring, not
    imported from there since it's not exposed as its own function."""
    return float(np.exp(epsilon) / (1 + np.exp(epsilon)))


def _correct_population_rate(observed_rate, keep_prob):
    """Inverts the randomized-response flip rate to estimate the true
    population majority-share from the *already-released* per-record
    labels' aggregate rate. This is pure post-processing on DP output (Dwork
    & Roth's post-processing immunity) — the correction is a deterministic
    function of only the already-released privatized labels, never of
    true_sex, so it costs no additional privacy budget and needs no
    separate epsilon.

    Classic Warner-style inversion: observed_rate = keep_prob * true_rate +
    (1 - keep_prob) * (1 - true_rate), solved for true_rate. Unstable as
    keep_prob -> 0.5 (epsilon -> 0) since it divides by (2*keep_prob - 1);
    clipped to [0, 1] rather than further guarded, and the resulting
    high-variance correction at low epsilon is expected, not a bug — see
    CLAUDE.md / the paper draft's discussion of the debiased estimator.
    """
    denom = 2 * keep_prob - 1
    if abs(denom) < 1e-9:
        return 0.5
    return float(np.clip((observed_rate - (1 - keep_prob)) / denom, 0.0, 1.0))


def _posterior_majority_weight(observed_is_majority, keep_prob, prior_majority):
    """Per-record Bayesian posterior P(true=majority | observed label,
    mechanism), given the population prior above. `observed_is_majority` is
    a boolean array over the *privatized* labels — the posterior is a
    deterministic function of that already-released output and the
    mechanism's known (public) keep_prob, so, like the prior above, this is
    post-processing and adds no privacy cost."""
    p_obs_given_majority = np.where(observed_is_majority, keep_prob, 1 - keep_prob)
    p_obs_given_minority = np.where(observed_is_majority, 1 - keep_prob, keep_prob)
    numerator = p_obs_given_majority * prior_majority
    denominator = numerator + p_obs_given_minority * (1 - prior_majority)
    denominator = np.where(denominator == 0, 1.0, denominator)  # guards prior_majority in {0, 1}
    return numerator / denominator


def _debiased_subgroup_auc_gap(patient_table, dp_sex_series, epsilon):
    """Soft-membership-weighted subgroup AUC gap: rather than hard-assigning
    each patient to majority/minority by their privatized label, weight
    every patient's contribution to each subgroup's AUC by their Bayesian
    posterior probability of true membership (see
    _posterior_majority_weight), using sklearn's roc_auc_score(sample_weight=...)
    to compute the weighted AUC per subgroup.

    Built entirely from the already-released privatized labels
    (dp_sex_series) and the mechanism's known keep probability — never
    reads true_sex — so this is a free post-processing improvement on the
    same epsilon-DP release, not a second, separately-budgeted query. See
    CLAUDE.md, Study B, for why this doesn't reintroduce the discarded
    aggregate-count mechanism's privacy-accounting flaw: unlike that
    mechanism, no *fresh* aggregate query is made against true_sex here —
    the only aggregate involved (the observed rate used for
    _correct_population_rate) is itself computed from data already
    released under the record's own epsilon.

    Returns None if either weighted AUC is undefined (e.g. one true_label
    class carries ~all the posterior weight) — same convention as
    _subgroup_auc_gap.
    """
    keep_prob = _keep_probability(epsilon)
    observed_is_majority = (dp_sex_series == MAJORITY_SEX).to_numpy()
    prior_majority = _correct_population_rate(observed_is_majority.mean(), keep_prob)

    weight_majority = _posterior_majority_weight(observed_is_majority, keep_prob, prior_majority)
    weight_minority = 1.0 - weight_majority

    labels = patient_table["label"].to_numpy()
    scores = patient_table["score"].to_numpy()
    if weight_majority.sum() <= 0 or weight_minority.sum() <= 0:
        return None
    try:
        majority_auc = roc_auc_score(labels, scores, sample_weight=weight_majority)
        minority_auc = roc_auc_score(labels, scores, sample_weight=weight_minority)
    except ValueError:
        # weighted analogue of the one-class-present degeneracy
        # _subgroup_auc_gap guards for directly
        return None
    return majority_auc - minority_auc


def _subgroup_auc_gap(patient_table, sex_series):
    """Majority-minority patient-level AUC gap for the given per-patient
    sex assignment — `sex_series` may be the true sex or a DP-privatized
    one; `patient_table`'s label/score columns are always the frozen,
    unmodified ones.

    At low enough epsilon, randomized response can (rarely) flip an entire
    subgroup to size 0, or leave a nonempty subgroup with only one
    true_label value present (both make AUC undefined — roc_auc_score
    raises "Only one class present in y_true" for the latter, confirmed
    directly, not assumed). Both are meaningful outcomes (total subgroup
    erasure or a degenerate one-class subgroup, not a subtler
    washing-out), not error cases to hide, so this returns None rather
    than letting roc_auc_score raise.
    """
    majority_mask = sex_series == MAJORITY_SEX
    minority_mask = sex_series == MINORITY_SEX
    if majority_mask.sum() == 0 or minority_mask.sum() == 0:
        return None
    if patient_table.loc[majority_mask, "label"].nunique() < 2:
        return None
    if patient_table.loc[minority_mask, "label"].nunique() < 2:
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


def _score_gap(gap, true_gap):
    """Direction/pct_diff/survived scoring shared by the naive and
    debiased estimators, so the two can't silently diverge in how
    "survived" is computed. `gap=None` (a whole subgroup privatized to
    size 0, or left with only one true_label value present — total
    erasure/degeneracy, not a subtler washing-out) is recorded as
    NaN/False rather than skipped, so it's visible in the output, not
    silently dropped."""
    if gap is None:
        return {"dp_gap": float("nan"), "direction_match": False, "pct_diff": float("nan"), "survived": False}
    direction_match = (gap > 0) == (true_gap > 0)
    pct_diff = abs(gap - true_gap) / abs(true_gap)
    survived = bool(direction_match and pct_diff <= SURVIVAL_PCT_THRESHOLD)
    return {"dp_gap": gap, "direction_match": bool(direction_match), "pct_diff": pct_diff, "survived": survived}


def _evaluate_draw(patients, true_gap, epsilon, seed):
    """One epsilon's worth of the sweep: privatize once, then score both
    the naive hard-assignment gap and the debiased soft-weighted gap
    (Section "Debiased estimator" in CLAUDE.md / the paper draft) against
    the *same* privatized draw — a paired comparison, not two independent
    draws, so any difference between them reflects the estimator, not
    draw-to-draw noise. Shared by the canonical single-draw sweep and the
    seed-replication sweep so the two can't silently diverge in how
    "survived" is computed. Debiased columns are prefixed `debiased_`.
    """
    dp_sex = privatize_subgroups(patients, epsilon, seed)

    naive_gap = _subgroup_auc_gap(patients, dp_sex)
    debiased_gap = _debiased_subgroup_auc_gap(patients, dp_sex, epsilon)

    row = _score_gap(naive_gap, true_gap)
    row.update({f"debiased_{k}": v for k, v in _score_gap(debiased_gap, true_gap).items()})
    return row


def run_epsilon_sweep():
    patients = load_frozen_test_set()
    true_gap = _subgroup_auc_gap(patients, patients["sex"])

    master_rng = np.random.default_rng(BASE_SEED)
    rows = []
    for epsilon in dp.EPSILON_SWEEP:
        seed = int(master_rng.integers(0, 2**31 - 1))
        row = {"epsilon": epsilon, "true_gap": true_gap}
        row.update(_evaluate_draw(patients, true_gap, epsilon, seed))
        rows.append(row)

    return pd.DataFrame(rows)


def run_seed_replication():
    """N_REPLICATION_SEEDS independent draws per epsilon (separate seed
    pool from the canonical sweep — see REPLICATION_BASE_SEED), so the
    canonical single-draw crossover can be checked against a distribution
    instead of trusted as a point estimate. See CLAUDE.md, Study B — Seed
    Replication for why this was added: a development-time diagnostic
    found real, large per-draw variance near the transition (pct_diff
    ranging <1% to 150%+ at the same epsilon), which a single canonical
    draw can't distinguish from a stable result.
    """
    patients = load_frozen_test_set()
    true_gap = _subgroup_auc_gap(patients, patients["sex"])

    master_rng = np.random.default_rng(REPLICATION_BASE_SEED)
    rows = []
    for epsilon in dp.EPSILON_SWEEP:
        for replicate in range(N_REPLICATION_SEEDS):
            seed = int(master_rng.integers(0, 2**31 - 1))
            row = {"epsilon": epsilon, "seed": seed, "true_gap": true_gap}
            row.update(_evaluate_draw(patients, true_gap, epsilon, seed))
            rows.append(row)

    return pd.DataFrame(rows)


def _wilson_ci(successes, n, confidence=0.95):
    """Wilson score interval for a binomial proportion (successes/n) — used
    for the per-epsilon survival-rate CIs below. Added after the draft
    write-up noted the borderline-epsilon survival rates (e.g. 90.0% at
    epsilon=4/5, n=30) carry real sampling uncertainty — binomial SE
    ~5.5% at p=0.9, n=30 — that a bare point estimate hides; see CLAUDE.md
    / the paper draft's Discussion."""
    if n == 0:
        return float("nan"), float("nan")
    z = norm.ppf(1 - (1 - confidence) / 2)
    phat = successes / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = (z * np.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def seed_replication_summary(replication_df):
    """Per-epsilon spread across the replication draws: mean/std dp_gap,
    mean/std pct_diff, direction-agreement rate, and survival rate
    (fraction of replicate draws that independently satisfied "survived")
    — for both the naive and debiased estimators (paired on the same
    draws, see _evaluate_draw), plus a Wilson 95% CI on each survival
    rate. Non-gating — mirrors Study A's cross-seed gap spread pattern
    (see CLAUDE.md, Study A Seed Replication), reported in CHANGELOG.md
    alongside the canonical sweep, not replacing it.
    """
    summary = replication_df.groupby("epsilon").agg(
        dp_gap_mean=("dp_gap", "mean"),
        dp_gap_std=("dp_gap", "std"),
        pct_diff_mean=("pct_diff", "mean"),
        pct_diff_std=("pct_diff", "std"),
        direction_agreement_rate=("direction_match", "mean"),
        survival_rate=("survived", "mean"),
        debiased_dp_gap_mean=("debiased_dp_gap", "mean"),
        debiased_dp_gap_std=("debiased_dp_gap", "std"),
        debiased_pct_diff_mean=("debiased_pct_diff", "mean"),
        debiased_pct_diff_std=("debiased_pct_diff", "std"),
        debiased_direction_agreement_rate=("debiased_direction_match", "mean"),
        debiased_survival_rate=("debiased_survived", "mean"),
        n_replicates=("seed", "count"),
    ).reset_index()

    for prefix in ("", "debiased_"):
        ci = summary.apply(
            lambda r, prefix=prefix: _wilson_ci(
                round(r[f"{prefix}survival_rate"] * r["n_replicates"]), int(r["n_replicates"])
            ),
            axis=1,
        )
        summary[f"{prefix}survival_rate_ci_lower"] = [c[0] for c in ci]
        summary[f"{prefix}survival_rate_ci_upper"] = [c[1] for c in ci]

    return summary


def crossover_epsilon(survived_by_epsilon):
    """Smallest epsilon at/above which every larger-or-equal epsilon also
    "survives" — more noise (lower epsilon) is expected to wash the gap
    out more, not less, so this is the point where survival becomes
    reliable from there up. `survived_by_epsilon` is a DataFrame with
    "epsilon" and a boolean-like "survived" column — the caller decides
    what "survived" means (a single draw's boolean, or
    survival_rate >= some threshold for the replication-based version).
    Returns None if no epsilon survives; if the pattern isn't actually
    monotonic, this returns the last point from which it's True all the
    way up, and the caller should look at the full table, not just this
    number."""
    ordered = survived_by_epsilon.sort_values("epsilon")
    survived = ordered["survived"].to_numpy().astype(bool)
    epsilons = ordered["epsilon"].to_numpy()
    for i in range(len(survived)):
        if survived[i:].all():
            return epsilons[i]
    return None


def _majority_crossover(summary_df, survival_col):
    survived = summary_df[["epsilon", survival_col]].copy()
    survived["survived"] = survived[survival_col] >= 0.5
    return crossover_epsilon(survived[["epsilon", "survived"]])


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    results = run_epsilon_sweep()
    results.to_csv(OUTPUT_PATH, index=False)

    crossover = crossover_epsilon(results[["epsilon", "survived"]])
    debiased_crossover = crossover_epsilon(
        results[["epsilon", "debiased_survived"]].rename(columns={"debiased_survived": "survived"})
    )
    print(results.to_string(index=False))
    print(f"\nCanonical-draw crossover epsilon (naive): {crossover}")
    print(f"Canonical-draw crossover epsilon (debiased): {debiased_crossover}")

    replication_dir = os.path.join(REPO_ROOT, "results", "study_b", "seed_replication")
    os.makedirs(replication_dir, exist_ok=True)
    replication = run_seed_replication()
    replication.to_csv(os.path.join(replication_dir, "dp_gap_replication.csv"), index=False)

    summary = seed_replication_summary(replication)
    replication_crossover = _majority_crossover(summary, "survival_rate")
    debiased_replication_crossover = _majority_crossover(summary, "debiased_survival_rate")

    print(f"\n{N_REPLICATION_SEEDS}-seed replication summary:")
    print(summary.to_string(index=False))
    print(f"\nReplication-based crossover epsilon (naive, majority of {N_REPLICATION_SEEDS} draws survive): {replication_crossover}")
    print(f"Replication-based crossover epsilon (debiased, majority of {N_REPLICATION_SEEDS} draws survive): {debiased_replication_crossover}")


if __name__ == "__main__":
    main()
