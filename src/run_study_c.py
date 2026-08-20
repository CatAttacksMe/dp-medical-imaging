"""Study C: how small can a subgroup be before DP noise makes its
underrepresentation undetectable? See CLAUDE.md, Study C — Objection
Response.

Fully synthetic — no dependency on real ChestX-ray14 data, Study A, or
Study B's results/checkpoints. Imports only src/dp_mechanisms.py.

Reference-group design (CLAUDE.md, Study C — Method, revised 2026-08-20):
the "reference group" is a fixed, public, non-privatized reference
proportion (REFERENCE_PROPORTION below), representing an external
expected/unbiased share — not a second synthetic group drawn from the
same audited cohort. Only the audited cohort's own subgroup count is
sensitive and gets privatized; the reference figure needs no protection
and no privacy budget. An earlier "self-complement" reading (subgroup vs.
everyone else in one cohort, e.g. 2% vs. 98%) was rejected before
implementation: at that separation a difference test would report
"detected" at almost any epsilon, so it wouldn't actually test
detectability of underrepresentation — just confirm the subgroup is a
minority by construction.

Detection criterion: a one-sample test. privatize_categorical_proportions
is called with a single category ({"subgroup": true_count}) and an
explicit total (the audited cohort's fixed, public size) — same "cohort
size is public, only the statistic is privatized" convention as
privatize_age_mean. "Detected" = the subgroup's own closed-form CI
excludes the known constant REFERENCE_PROPORTION.

Run with: python src/run_study_c.py
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import norm

import dp_mechanisms as dp

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(REPO_ROOT, "results", "study_c", "detection_floor.csv")

# Fixed cohort size, tied to a realistic audit scale rather than an
# arbitrary synthetic N — Study B's frozen test cohort size (4,620
# patients, confirmed in CHANGELOG.md). The mechanism's CI half-width
# scales with 1/total, so this choice is what makes the "detection floor"
# meaningful for the paper's actual claims. Study C is fully synthetic and
# does not read any Study A/B file to get this number — it's hardcoded
# here as a scale decision, not a data dependency.
REFERENCE_TOTAL = 4620

# The swept true prevalences of the audited cohort's subgroup, and the
# fixed external reference proportion they're tested against. The top of
# the sweep doubles as the false-positive control cell (true_prevalence ==
# REFERENCE_PROPORTION, i.e. no real gap) — see CLAUDE.md, False-positive
# control — rather than needing a separately-run condition.
PREVALENCES = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
REFERENCE_PROPORTION = max(PREVALENCES)

CONFIDENCE = 0.95

# Reuses dp_mechanisms.EPSILON_SWEEP for consistency with the other
# studies' epsilon axis in the final paper, even though that sweep was
# re-derived specifically for privatize_categorical_label's
# randomized-response noise profile (CLAUDE.md, Study B — Subgroup
# Assignment Mechanism), not for privatize_categorical_proportions's
# Laplace-count mechanism used here. If the results below turn out
# saturated at both ends of this sweep (no resolution near a transition),
# that's a signal to run a Study-B-style diagnostic pass and add
# Study-C-specific points, not to silently reuse this list forever.
EPSILON_SWEEP = dp.EPSILON_SWEEP

# Own seed pool (distinct range from Study A's 42-46/101-103 and Study
# B's 42/2000) since Study C is independent and shares no reproducibility
# anchor with either.
N_REPLICATION_SEEDS = 30
BASE_SEED = 5000


def _evaluate_draw(true_prevalence, epsilon, seed):
    """One replicate draw: privatize the subgroup's own count and test
    whether its CI excludes the known reference proportion."""
    true_count = round(true_prevalence * REFERENCE_TOTAL)
    result = dp.privatize_categorical_proportions(
        {"subgroup": true_count},
        epsilon=epsilon,
        total=REFERENCE_TOTAL,
        confidence=CONFIDENCE,
        random_state=seed,
    )["subgroup"]
    detected = not (result["ci_lower"] <= REFERENCE_PROPORTION <= result["ci_upper"])
    return {
        "noisy_count": result["noisy_count"],
        "noisy_proportion": result["proportion"],
        "ci_lower": result["ci_lower"],
        "ci_upper": result["ci_upper"],
        "detected": bool(detected),
    }


def run_replication():
    """N_REPLICATION_SEEDS independent draws per (epsilon, true_prevalence)
    cell — not a single draw, see CLAUDE.md, Study C — Replication: a
    boundary-region boolean outcome from one stochastic draw can't
    distinguish "this epsilon/prevalence reliably detects" from "this
    particular draw happened to."
    """
    master_rng = np.random.default_rng(BASE_SEED)
    rows = []
    for true_prevalence in PREVALENCES:
        is_null = bool(np.isclose(true_prevalence, REFERENCE_PROPORTION))
        for epsilon in EPSILON_SWEEP:
            for _ in range(N_REPLICATION_SEEDS):
                seed = int(master_rng.integers(0, 2**31 - 1))
                row = {
                    "true_prevalence": true_prevalence,
                    "epsilon": epsilon,
                    "seed": seed,
                    "is_null_condition": is_null,
                }
                row.update(_evaluate_draw(true_prevalence, epsilon, seed))
                rows.append(row)
    return pd.DataFrame(rows)


def _wilson_ci(successes, n, confidence=0.95):
    """Wilson score interval for a binomial proportion — same
    construction as src/run_study_b.py's _wilson_ci, reimplemented here
    since Study C's allowed imports are src/dp_mechanisms.py only (not
    another study's script)."""
    if n == 0:
        return float("nan"), float("nan")
    z = norm.ppf(1 - (1 - confidence) / 2)
    phat = successes / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = (z * np.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def detection_summary(replication_df):
    """Per-(epsilon, true_prevalence) detection rate across the replicate
    draws, with a Wilson 95% CI. For the true_prevalence ==
    REFERENCE_PROPORTION cell this rate is the false-positive rate (see
    CLAUDE.md, False-positive control); for every other cell it's the
    power to detect a real gap of that size at that epsilon.
    """
    summary = (
        replication_df.groupby(["true_prevalence", "epsilon", "is_null_condition"])
        .agg(detection_rate=("detected", "mean"), n_replicates=("seed", "count"))
        .reset_index()
    )

    ci = summary.apply(
        lambda r: _wilson_ci(round(r["detection_rate"] * r["n_replicates"]), int(r["n_replicates"])),
        axis=1,
    )
    summary["detection_rate_ci_lower"] = [c[0] for c in ci]
    summary["detection_rate_ci_upper"] = [c[1] for c in ci]
    return summary


def detection_floor(summary_df, rate_threshold=0.5):
    """Smallest true_prevalence, per epsilon, whose detection_rate is >=
    rate_threshold — the study's headline number. Excludes the null
    condition row (its "detection rate" is a false-positive rate, not a
    power estimate, so it doesn't belong in a floor computed over real
    gaps). Returns {epsilon: smallest_detected_prevalence_or_None}.
    """
    real_gap = summary_df[~summary_df["is_null_condition"]]
    floors = {}
    for epsilon, group in real_gap.groupby("epsilon"):
        detected = group[group["detection_rate"] >= rate_threshold]
        floors[epsilon] = detected["true_prevalence"].min() if not detected.empty else None
    return floors



# --- Fine-grained floor localization (exploratory) -------------------------
#
# The primary sweep above (PREVALENCES x dp.EPSILON_SWEEP) came back
# almost entirely saturated at 100% detection (see CHANGELOG.md) -- the
# widest tested gap-to-epsilon combination in this design overwhelms
# EPSILON_SWEEP's noise scale almost everywhere in its 0.1-10 range, the
# same way Study B's *original* epsilon sweep undershot its mechanism's
# real transition before being re-derived (CLAUDE.md, Study B -- Subgroup
# Assignment Mechanism). A diagnostic pass (60 trials/cell) confirmed the
# real transition for this test sits at epsilon <~1 and gaps under ~1
# percentage point from REFERENCE_PROPORTION -- see CLAUDE.md, Study C --
# Fine-Grained Floor Localization. This is an additive exploratory
# analysis, not a replacement for the primary deliverable: it keeps
# EPSILON_SWEEP fixed for the primary sweep (cross-study epsilon-axis
# comparability in the final paper) and adds a separate, finer grid here
# instead of quietly redefining the shared constant's meaning for Study C
# alone. Writes to its own file, not detection_floor.csv.
FLOOR_EPSILONS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
FLOOR_PREVALENCES = [0.005, 0.01, 0.02, 0.03, 0.035, 0.04, 0.042, 0.045, 0.048, 0.049]
FLOOR_OUTPUT_PATH = os.path.join(REPO_ROOT, "results", "study_c", "floor_localization.csv")
FLOOR_BASE_SEED = 6000


def run_floor_localization():
    """Same replicate-draw design as run_replication, on the finer grid
    above instead of the primary (PREVALENCES, EPSILON_SWEEP) grid. Own
    seed pool (FLOOR_BASE_SEED) so it can't reproduce the primary sweep's
    draws."""
    master_rng = np.random.default_rng(FLOOR_BASE_SEED)
    rows = []
    for true_prevalence in FLOOR_PREVALENCES:
        is_null = bool(np.isclose(true_prevalence, REFERENCE_PROPORTION))
        for epsilon in FLOOR_EPSILONS:
            for _ in range(N_REPLICATION_SEEDS):
                seed = int(master_rng.integers(0, 2**31 - 1))
                row = {
                    "true_prevalence": true_prevalence,
                    "epsilon": epsilon,
                    "seed": seed,
                    "is_null_condition": is_null,
                }
                row.update(_evaluate_draw(true_prevalence, epsilon, seed))
                rows.append(row)
    return pd.DataFrame(rows)


def floor_prevalence_by_epsilon(summary_df, rate_threshold=0.5):
    """For each epsilon, the largest true_prevalence (smallest gap from
    REFERENCE_PROPORTION) at/below which detection_rate stays >=
    rate_threshold for every smaller prevalence too. Mirrors
    src/run_study_b.py's crossover_epsilon, applied to the prevalence axis
    instead of the epsilon axis: a bigger gap from the reference should
    only get easier to detect, not harder, so this is the boundary from
    which detection holds reliably all the way down to the smallest
    tested prevalence. Returns {epsilon: boundary_prevalence_or_None}.
    """
    floors = {}
    for epsilon, group in summary_df.groupby("epsilon"):
        ordered = group.sort_values("true_prevalence", ascending=False)
        rates = ordered["detection_rate"].to_numpy()
        prevalences = ordered["true_prevalence"].to_numpy()
        floors[epsilon] = None
        for i in range(len(rates)):
            if (rates[i:] >= rate_threshold).all():
                floors[epsilon] = prevalences[i]
                break
    return floors


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    replication = run_replication()
    replication.to_csv(OUTPUT_PATH, index=False)

    summary = detection_summary(replication)
    print(summary.to_string(index=False))

    null_summary = summary[summary["is_null_condition"]]
    print(f"\nFalse-positive rate at true_prevalence={REFERENCE_PROPORTION} (null condition), by epsilon:")
    print(
        null_summary[
            ["epsilon", "detection_rate", "detection_rate_ci_lower", "detection_rate_ci_upper"]
        ].to_string(index=False)
    )

    floors = detection_floor(summary)
    print("\nDetection floor (smallest true_prevalence with detection_rate >= 50%), by epsilon:")
    for epsilon in EPSILON_SWEEP:
        print(f"  epsilon={epsilon}: {floors.get(epsilon)}")

    floor_localization = run_floor_localization()
    floor_localization.to_csv(FLOOR_OUTPUT_PATH, index=False)
    floor_summary = detection_summary(floor_localization)
    print("\nFine-grid floor localization summary:")
    print(floor_summary.to_string(index=False))

    boundary = floor_prevalence_by_epsilon(floor_summary)
    print("\nFloor localization: boundary true_prevalence (smallest gap from "
          f"{REFERENCE_PROPORTION} still reliably detected), by epsilon:")
    for epsilon in FLOOR_EPSILONS:
        print(f"  epsilon={epsilon}: {boundary.get(epsilon)}")


if __name__ == "__main__":
    main()
