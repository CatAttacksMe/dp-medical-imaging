"""Unit/invariant checks for src/dp_mechanisms.py — the shared, reviewed DP
infrastructure Study B and Study C both import. CLAUDE.md requires this
file's checks pass before either study branch pulls in a new version of
dp_mechanisms.py. Not itself part of either study's file layout.

Each check is independent (a failure in one doesn't stop the others from
running) so a single run reports the full picture. Exits 1 if any check
fails.

Run with: python src/check_dp_mechanisms.py
"""

import sys

import numpy as np

import dp_mechanisms as dp

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


@check
def epsilon_sweep_matches_spec():
    """CLAUDE.md fixes this exact sweep; Study B must reference it, not a
    hardcoded copy — so a silent edit here would silently change Study B."""
    assert dp.EPSILON_SWEEP == [0.1, 0.5, 1, 2, 3, 4, 5, 6, 8, 10], dp.EPSILON_SWEEP


@check
def counts_add_real_noise():
    """At low epsilon, repeated draws from the same true count should not
    all collapse to the same (or the true) value — a passthrough bug would
    otherwise go unnoticed."""
    counts = {"M": 1000, "F": 200}
    draws = [dp.privatize_categorical_counts(counts, epsilon=0.1, random_state=s)["M"] for s in range(20)]
    assert len(set(draws)) > 1, "no variation across seeds — noise is not being applied"
    assert not all(d == counts["M"] for d in draws), "every draw exactly matches the true count"


@check
def counts_are_nonnegative_integers():
    counts = {"M": 5, "F": 1}
    for epsilon in dp.EPSILON_SWEEP:
        noisy = dp.privatize_categorical_counts(counts, epsilon=epsilon, random_state=42)
        for category, value in noisy.items():
            assert isinstance(value, int), f"{category}: {value!r} is not an int"
            assert value >= 0, f"{category}: {value} is negative"


@check
def counts_reproducible_with_same_seed():
    counts = {"M": 1000, "F": 200}
    a = dp.privatize_categorical_counts(counts, epsilon=1.0, random_state=123)
    b = dp.privatize_categorical_counts(counts, epsilon=1.0, random_state=123)
    assert a == b, f"same random_state gave different results: {a} vs {b}"


@check
def counts_use_parallel_composition_not_split_epsilon():
    """Each category should get the full epsilon (parallel composition over
    a partition), not epsilon / n_categories. Verify empirically: a
    single-category noisy count's variance at epsilon `e` should match a
    2-category dict's per-category variance at the same `e`, not the
    variance a naively-split epsilon (e/2) would produce."""
    true_count, epsilon, n_trials = 1000, 1.0, 4000
    rng = np.random.default_rng(0)

    one_cat_draws = [
        dp.privatize_categorical_counts({"X": true_count}, epsilon=epsilon, random_state=int(s))["X"]
        for s in rng.integers(0, 2**31 - 1, size=n_trials)
    ]
    two_cat_draws = [
        dp.privatize_categorical_counts({"X": true_count, "Y": 50}, epsilon=epsilon, random_state=int(s))["X"]
        for s in rng.integers(0, 2**31 - 1, size=n_trials)
    ]

    var_one, var_two = np.var(one_cat_draws), np.var(two_cat_draws)
    # Laplace(scale=1/epsilon) has variance 2/epsilon^2 = 2.0 here; allow
    # generous Monte Carlo slack rather than pin an exact value.
    expected = 2.0 / epsilon**2
    for name, var in [("one-category", var_one), ("two-category", var_two)]:
        assert 0.5 * expected <= var <= 1.5 * expected, (
            f"{name} variance {var:.3f} far from expected {expected:.3f} — "
            f"looks like epsilon is being split across categories"
        )


@check
def age_mean_sensitivity_scales_with_cohort_size():
    """Noise magnitude should shrink as the cohort grows (sensitivity =
    (hi-lo)/n) — a fixed-sensitivity bug wouldn't show this pattern."""
    rng = np.random.default_rng(1)
    small_cohort = rng.uniform(20, 80, size=20)
    large_cohort = rng.uniform(20, 80, size=2000)

    def spread(ages, n_trials=300):
        draws = [dp.privatize_age_mean(ages, epsilon=0.5, bounds=(0, 100), random_state=s) for s in range(n_trials)]
        return np.std(draws)

    assert spread(small_cohort) > spread(large_cohort), "small-cohort noise should be larger than large-cohort noise"


@check
def age_mean_clips_out_of_bounds_values():
    ages = [-50, 500, 40, 60]
    mean = dp.privatize_age_mean(ages, epsilon=1000, bounds=(0, 100), random_state=42)
    # epsilon=1000 -> negligible noise; clipped true mean is (0+100+40+60)/4 = 50
    assert abs(mean - 50) < 1, f"expected ~50 (clipped mean) at high epsilon, got {mean}"


@check
def age_histogram_shape_and_edges():
    rng = np.random.default_rng(2)
    ages = rng.uniform(0, 100, size=500)
    noisy_counts, bin_edges = dp.privatize_age_histogram(ages, epsilon=1.0, bins=10, bounds=(0, 100), random_state=7)
    assert len(noisy_counts) == 10
    assert len(bin_edges) == 11
    assert bin_edges[0] == 0 and bin_edges[-1] == 100
    assert all(c >= 0 for c in noisy_counts)


@check
def proportions_default_total_is_sum_of_counts():
    counts = {"rare": 30, "common": 970}
    result = dp.privatize_categorical_proportions(counts, epsilon=5.0, random_state=42)
    # At epsilon=5 noise is small; proportions should be close to true (30/1000, 970/1000)
    assert abs(result["rare"]["proportion"] - 0.03) < 0.02
    assert abs(result["common"]["proportion"] - 0.97) < 0.02


@check
def proportions_bounds_stay_in_unit_interval():
    counts = {"tiny": 1, "rest": 9}
    for epsilon in dp.EPSILON_SWEEP:
        result = dp.privatize_categorical_proportions(counts, epsilon=epsilon, random_state=1)
        for category, stats in result.items():
            for key in ("proportion", "ci_lower", "ci_upper"):
                v = stats[key]
                assert 0.0 <= v <= 1.0, f"{category}.{key} = {v} outside [0, 1] at epsilon={epsilon}"
            assert stats["ci_lower"] <= stats["ci_upper"]


@check
def proportions_ci_achieves_nominal_coverage():
    """The core correctness check for Study C's whole 'detected' methodology:
    a 95% CI should contain the true proportion ~95% of the time. Wrong
    sign, wrong log, or a missing /total scaling would show up as coverage
    far from 0.95, not as a crash — this is the check that would catch it."""
    true_count, total, epsilon, confidence, n_trials = 50, 1000, 1.0, 0.95, 3000
    true_proportion = true_count / total

    hits = 0
    for seed in range(n_trials):
        result = dp.privatize_categorical_proportions(
            {"subgroup": true_count}, epsilon=epsilon, total=total, confidence=confidence, random_state=seed
        )["subgroup"]
        if result["ci_lower"] <= true_proportion <= result["ci_upper"]:
            hits += 1

    coverage = hits / n_trials
    # Binomial SE at p=0.95, n=3000 is ~0.4%; allow +/-2% (5 SE) slack.
    assert abs(coverage - confidence) < 0.02, f"empirical coverage {coverage:.3f}, expected ~{confidence}"


@check
def proportions_ci_achieves_nominal_coverage_at_low_counts():
    """Study C's low-prevalence sweep (0.5% of a ~4,620-patient cohort is
    ~23 records) implies much smaller counts than
    proportions_ci_achieves_nominal_coverage above validates (true_count=50
    of total=1000). Clipping to [0, 1] engages more often at small
    counts/low epsilon, which can bias coverage away from nominal in a way
    the larger-count check wouldn't reveal — this re-runs the same
    coverage test at Study C's actual low end rather than assuming the
    check above generalizes down. See CLAUDE.md, Study C — CI coverage
    check."""
    total, confidence, n_trials = 4620, 0.95, 3000
    true_count = round(0.005 * total)  # ~23 — Study C's lowest swept prevalence
    true_proportion = true_count / total

    for epsilon in (0.5, 1.0, 2.0):
        hits = 0
        for seed in range(n_trials):
            result = dp.privatize_categorical_proportions(
                {"subgroup": true_count}, epsilon=epsilon, total=total, confidence=confidence, random_state=seed
            )["subgroup"]
            if result["ci_lower"] <= true_proportion <= result["ci_upper"]:
                hits += 1
        coverage = hits / n_trials
        # Binomial SE at p=0.95, n=3000 is ~0.4%; allow +/-3% (looser than
        # the large-count check's +/-2%, since this checks 3 epsilons
        # against one true_count rather than one epsilon).
        assert abs(coverage - confidence) < 0.03, (
            f"epsilon={epsilon}: empirical coverage {coverage:.3f} at true_count={true_count} "
            f"(Study C's low end), expected ~{confidence}"
        )


@check
def label_flip_probability_matches_theory():
    """The core correctness check for privatize_categorical_label: kept
    probability should match e^epsilon / (1 + e^epsilon), the standard
    randomized-response calibration — a wrong flip probability would still
    produce plausible-looking M/F output with no crash, exactly the kind
    of bug that needs an empirical check to catch."""
    n_trials = 4000
    for epsilon in [0.01, 0.1, 1, 2, 5]:
        draws = dp.privatize_categorical_label(["M"] * n_trials, epsilon, "M", "F", random_state=1)
        empirical_kept = sum(d == "M" for d in draws) / n_trials
        theoretical_kept = np.exp(epsilon) / (1 + np.exp(epsilon))
        # Binomial SE at n=4000 is at most ~0.8%; allow +/-3% slack.
        assert abs(empirical_kept - theoretical_kept) < 0.03, (
            f"epsilon={epsilon}: empirical kept-rate {empirical_kept:.3f}, theoretical {theoretical_kept:.3f}"
        )


@check
def label_output_domain_is_value0_or_value1():
    draws = dp.privatize_categorical_label(["M", "F", "M", "M", "F"] * 20, epsilon=0.5, value0="M", value1="F", random_state=7)
    assert set(draws) <= {"M", "F"}, set(draws)
    assert len(draws) == 100


@check
def label_reproducible_with_same_seed():
    values = ["M", "F", "M"] * 10
    a = dp.privatize_categorical_label(values, epsilon=1.0, value0="M", value1="F", random_state=123)
    b = dp.privatize_categorical_label(values, epsilon=1.0, value0="M", value1="F", random_state=123)
    assert a == b, "same random_state gave different results"


@check
def label_adds_real_noise_at_low_epsilon():
    """At epsilon near 0, output should look close to a coin flip, not a
    passthrough of the true value."""
    n_trials = 4000
    draws = dp.privatize_categorical_label(["M"] * n_trials, epsilon=0.001, value0="M", value1="F", random_state=2)
    kept_rate = sum(d == "M" for d in draws) / n_trials
    assert abs(kept_rate - 0.5) < 0.03, f"expected near-coin-flip at epsilon=0.001, got kept-rate={kept_rate:.3f}"


@check
def nonpositive_epsilon_raises():
    for fn, args in [
        (dp.privatize_categorical_counts, ({"A": 1}, 0)),
        (dp.privatize_age_mean, ([1, 2, 3], -1)),
        (dp.privatize_categorical_label, (["M", "F"], 0, "M", "F")),
    ]:
        try:
            fn(*args)
        except Exception:
            continue
        raise AssertionError(f"{fn.__name__} did not raise for a non-positive epsilon")


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
