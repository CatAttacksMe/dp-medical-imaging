"""Shared differential-privacy mechanisms for Study B and Study C.

Reviewed infrastructure (see CLAUDE.md, Shared Infrastructure) — changes
here must still pass src/check_dp_mechanisms.py before either study branch
pulls in a new version. All mechanisms use diffprivlib, pure epsilon-DP (no
delta).

Two different neighboring-database (adjacency) models are in play here, and
the functions below split cleanly along that line — mixing them up (using
the wrong function for the wrong question) is the kind of mistake this note
exists to prevent:

- **Add/remove adjacency** — protects whether an individual is *in the
  dataset at all*. `privatize_categorical_counts`, `privatize_age_mean`,
  `privatize_age_histogram`, and `privatize_categorical_proportions` all
  assume this model: they release *aggregate* statistics (a count, a mean,
  a histogram) via the Laplace mechanism, and lean on parallel composition
  across disjoint category bins — valid under add/remove adjacency because
  removing one record changes exactly one bin's count. This is the right
  tool for "how many patients of each kind exist" questions.
- **Substitute-one (attribute) adjacency** — protects what a *known*
  individual's attribute value is, while their presence in the dataset is
  already public. This is the actual threat model for Study B's use case:
  `results/study_a/patient_split.csv` is committed to git, so the test
  cohort's membership is already fully public — what's sensitive is each
  patient's sex, not whether they're in the study. Under this model, a
  single attribute substitution changes *two* aggregate bin counts at
  once (one down, one up), which breaks parallel composition's
  precondition that a record affects at most one bin — so the aggregate
  functions above are the wrong tool here; releasing both bin counts
  independently at "full epsilon each" would actually cost ~2x epsilon
  under this model, not epsilon. `privatize_categorical_label` (below)
  sidesteps the question entirely by protecting one record's attribute
  directly, with a single well-defined per-record epsilon-DP guarantee
  that doesn't depend on cohort size or which adjacency model applies to
  the rest of the dataset.
"""

import numpy as np

# diffprivlib==0.6.6's top-level __init__ unconditionally imports
# diffprivlib.models, which imports diffprivlib.models.forest, which imports
# private (non-public) sklearn.tree._tree Cython symbols that newer
# scikit-learn releases removed. requirements.txt pins scikit-learn==1.4.2
# specifically so `import diffprivlib` succeeds with no workaround needed
# here — see CHANGELOG.md, [Shared] diffprivlib/sklearn fix. If that pin is
# ever revisited, re-verify `import diffprivlib` still works before
# assuming this module still does.
from diffprivlib.mechanisms import Binary, Laplace

EPSILON_SWEEP = [0.1, 0.5, 1, 2, 3, 4, 5, 6, 8, 10]


def _split_seeds(random_state, n):
    """n independent sub-seeds from one reproducibility seed, or n Nones."""
    if random_state is None:
        return [None] * n
    rng = np.random.default_rng(random_state)
    return rng.integers(0, 2**31 - 1, size=n).tolist()


def _laplace_count_noise(true_count, epsilon, random_state=None):
    """Laplace-mechanism noise for a single count query, sensitivity=1.

    Sensitivity 1 assumes an add/remove-one neighboring relation on a query
    that counts individuals in one category — removing or adding a single
    patient changes this specific count by at most 1. Returns
    (noisy_value, scale), where scale = sensitivity / epsilon is the
    Laplace distribution's scale parameter b.
    """
    mech = Laplace(epsilon=epsilon, sensitivity=1.0, random_state=random_state)
    return mech.randomise(float(true_count)), 1.0 / epsilon


def privatize_categorical_counts(counts, epsilon, random_state=None):
    """DP-protected per-category counts (e.g. sex counts for Study B).

    `counts` must be counts over categories that partition the population
    (each individual contributes to exactly one category) — e.g. a sex
    breakdown of one fixed cohort. Under that condition, the categories'
    noise draws are independent by *parallel* composition: each category's
    Laplace mechanism can use the *full* `epsilon`, not `epsilon /
    len(counts)`, because no single individual's record influences more
    than one category's count. Do not reuse this function for overlapping
    or non-exhaustive category sets without revisiting this assumption.

    Returns {category: noisy_count}, rounded to the nearest non-negative
    integer — rounding/clipping is post-processing on an already-released
    value and does not consume additional privacy budget.
    """
    seeds = _split_seeds(random_state, len(counts))
    noisy_counts = {}
    for (category, true_count), seed in zip(counts.items(), seeds):
        noisy, _ = _laplace_count_noise(true_count, epsilon, random_state=seed)
        noisy_counts[category] = max(0, round(noisy))
    return noisy_counts


def privatize_age_mean(ages, epsilon, bounds=(0, 100), random_state=None):
    """DP-protected mean age over a fixed-size cohort.

    `ages` are clipped to `bounds` before computing the true mean —
    clipping is a fixed, public transformation (not data-dependent), so it
    doesn't consume privacy budget, and it's what bounds the mechanism's
    sensitivity: for a mean of n clipped values, sensitivity = (hi - lo) /
    n under an add/remove-one neighboring relation. Cohort size `len(ages)`
    is treated as public (e.g. the frozen test-set patient count already
    published via patient_split.csv) — only the mean itself is privatized.
    """
    ages = np.asarray(ages, dtype=float)
    lo, hi = bounds
    clipped = np.clip(ages, lo, hi)
    true_mean = float(clipped.mean())
    sensitivity = (hi - lo) / len(clipped)
    mech = Laplace(epsilon=epsilon, sensitivity=sensitivity, random_state=random_state)
    return mech.randomise(true_mean)


def privatize_age_histogram(ages, epsilon, bins=10, bounds=(0, 100), random_state=None):
    """DP-protected age histogram.

    Ages are clipped to `bounds` before binning so every individual falls
    into exactly one bin (the bins partition the clipped range) — same
    parallel-composition argument as privatize_categorical_counts applies:
    each bin's Laplace mechanism uses the full `epsilon`.

    Returns (noisy_counts, bin_edges); noisy_counts is rounded to the
    nearest non-negative integer per bin.
    """
    ages = np.asarray(ages, dtype=float)
    lo, hi = bounds
    clipped = np.clip(ages, lo, hi)
    true_counts, bin_edges = np.histogram(clipped, bins=bins, range=bounds)

    seeds = _split_seeds(random_state, len(true_counts))
    noisy_counts = np.array([
        max(0, round(_laplace_count_noise(c, epsilon, random_state=s)[0]))
        for c, s in zip(true_counts, seeds)
    ])
    return noisy_counts, bin_edges


def privatize_categorical_proportions(counts, epsilon, total=None, confidence=0.95, random_state=None):
    """DP-protected subgroup proportions, with a Laplace-derived CI (Study C).

    Only the per-category *counts* are privatized (same parallel-composition
    argument as privatize_categorical_counts). `total` — the denominator
    used to convert counts to proportions — defaults to sum(counts.values())
    and is treated as public, not privatized itself; pass an explicit
    `total` if the reference population size should come from elsewhere.
    This matches how Study C frames the question (a subgroup's share of an
    already-known population size), not a scenario where the total itself
    is sensitive.

    The confidence interval is closed-form from the Laplace mechanism's
    known noise scale, not a resampling estimate: for scale b, P(|noise| >
    t) = exp(-t/b), so a (1 - alpha) two-sided interval has half-width t =
    b * ln(1/alpha), evaluated on the proportion scale (b / total).

    Returns {category: {"true_count", "noisy_count", "proportion",
    "ci_lower", "ci_upper"}}.
    """
    if total is None:
        total = sum(counts.values())
    if total <= 0:
        raise ValueError("total must be positive")

    alpha = 1 - confidence
    seeds = _split_seeds(random_state, len(counts))

    results = {}
    for (category, true_count), seed in zip(counts.items(), seeds):
        noisy, scale = _laplace_count_noise(true_count, epsilon, random_state=seed)
        noisy_count = max(0, round(noisy))
        # proportion/CI use the unrounded noisy value, not noisy_count — the
        # CI's half-width is derived for the continuous Laplace variable, so
        # rounding the count first (then building the CI around the rounded
        # value) desyncs the interval from the coverage it's supposed to
        # guarantee. Confirmed via check_dp_mechanisms.py's coverage check:
        # rounding-first measured ~91-92% empirical coverage against a
        # nominal 95%, unrounded measured ~95% as expected.
        proportion = max(0.0, noisy) / total
        half_width = (scale / total) * np.log(1 / alpha)
        results[category] = {
            "true_count": true_count,
            "noisy_count": noisy_count,
            "proportion": float(np.clip(proportion, 0.0, 1.0)),
            "ci_lower": float(np.clip(proportion - half_width, 0.0, 1.0)),
            "ci_upper": float(np.clip(proportion + half_width, 0.0, 1.0)),
        }
    return results


def privatize_categorical_label(values, epsilon, value0, value1, random_state=None):
    """Per-record randomized response for a binary attribute (e.g. sex) —
    substitute-one (attribute) adjacency, see module docstring.

    Each record's true value is independently passed through
    diffprivlib's classic Binary mechanism: kept as-is with probability
    e^epsilon / (1 + e^epsilon), flipped to the other value otherwise.
    This is a single well-defined epsilon-DP query per record — there is
    no aggregate count, no cross-bin composition question, and no
    dependence on whether the rest of the cohort's membership is public.

    Every record in the output is a genuine randomized-response draw (not
    "true unless selected for adjustment") — this is the primitive to use
    when downstream analysis needs a full per-record labeling (e.g.
    recomputing a subgroup AUC), where an aggregate count alone can't
    supply group membership for any specific record.

    `values` — any sequence of `value0`/`value1` labels. Returns a list of
    the same length, each entry independently randomized.
    """
    seeds = _split_seeds(random_state, len(values))
    result = []
    for value, seed in zip(values, seeds):
        mech = Binary(epsilon=epsilon, value0=value0, value1=value1, random_state=seed)
        result.append(mech.randomise(value))
    return result
