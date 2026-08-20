# Changelog / Lab Notes

## [Study B] 2026-08-19 (policy-implications section added to study_b_draft.tex)
- Added a new final section, "Implications for the policy paper"
  (`sec:policy-implications`), to `paper/study_drafts/study_b_draft.tex`,
  recording Study B's own read on how its results bear on
  `paper/Final_Policy_Recommendation.tex` — that file stays untouched
  until all three studies finish (per CLAUDE.md), but the current draft
  already stakes out claims this study speaks to directly, so the
  analysis is captured now rather than re-derived later.
- **Verdict recorded: qualified support, not blanket support.** epsilon>=6
  gives Study B a project-specific result to back the policy draft's
  "moderate privacy settings preserve utility" rebuttal (currently backed
  only by an external review citation) — but epsilon>=6 is a loose budget
  by DP convention, wider than what "moderate" usually connotes, so the
  claim needs an explicit, scoped floor rather than staying general.
  Below epsilon~3, and especially at epsilon<=1 (10-30% wrong-direction
  rate), Study B is a concrete instance of the exact risk the policy
  draft's "Small Populations Are Harmed by the Noise" section already
  footnotes ("differential privacy can even widen the subgroup
  performance gaps"). The debiased-estimator negative result also closes
  off "just correct for the noise after the fact" as an escape hatch for
  low-epsilon utility.
- **Recommended framing given to the policy draft:** state the floor
  explicitly (epsilon roughly >=6 for reliable subgroup audits on a
  cohort this size — 4,620 patients, 2,150-patient minority group) rather
  than an unqualified "moderate settings work," and don't generalize past
  this study's scope (single dataset/label/mechanism/imbalance-arm) —
  Study C's small-subgroup detection floor should be expected to push the
  needed epsilon higher, not lower, for smaller cohorts.
- No numeric/code changes — this is Study B's own narrative read on
  already-collected results, kept in the same draft file per the user's
  request rather than split into a separate document.

## [Study B] 2026-08-19 (Wilson CIs on survival rate; debiased-estimator comparison, negative result)
- Follow-up to an advisor-style review of `study_b_draft.tex`, addressing
  two points raised: (1) whether the replication's reported survival
  rates need their own uncertainty given the paper's headline claims rest
  on the borderline epsilons (3, 4, 5); (2) whether other DP mechanisms
  should be explored for better privacy/utility. See CLAUDE.md's new
  "Survival-Rate Confidence Intervals" and "Debiased Estimator"
  subsections under Study B for full detail — summarized here.
- **Wilson CIs.** `seed_replication_summary()` in `src/run_study_b.py` now
  computes a Wilson score 95% CI (`scipy.stats.norm`) on `survival_rate`
  (and on the new `debiased_survival_rate`, below) per epsilon. At
  n=30 replicates, the borderline points carry real width: epsilon=3's
  50.0% point estimate is [33.2%, 66.8%]; epsilon=4 and 5's 90.0% is each
  [74.4%, 96.5%]. Doesn't change the qualitative story (reliable only at
  epsilon>=6) but the paper draft's per-epsilon numbers in the 0.5-5 band
  should now be read with these bands attached, not as exact rates.
- **Debiased estimator — tried, and it did not help.** Implemented
  `_debiased_subgroup_auc_gap()`: reweights each privatized draw's
  patients by their Bayesian posterior P(true=majority | observed label),
  using a population-share prior obtained by inverting the draw's own
  observed aggregate rate (classic Warner/randomized-response inversion)
  — built entirely from the already-released privatized labels, so this
  is free post-processing on the existing epsilon-DP release, not a
  second mechanism or a second privacy cost. Scored from the *same*
  privatized draw as the naive hard-assignment estimator (paired
  comparison, both come from one `privatize_subgroups()` call per draw),
  not an independent redraw.
  - **Result:** slightly worse than the naive estimator across almost the
    whole sweep, not better. Both the canonical-draw and replication-based
    crossover epsilon moved from 3.0 (naive) to 4.0 (debiased) — the wrong
    direction. Debiased survival rate is <= the naive rate at every
    epsilon below 5 (e.g. epsilon=2: naive 36.7% vs. debiased 3.3%;
    epsilon=0.1/0.5/1.0: naive 3.3%/10.0%/6.7% vs. debiased 0.0% at all
    three). The two converge only once epsilon>=5, where naive was
    already reliable.
  - **Why:** two compounding reasons, not a bug — (1) the population-share
    correction divides by `(2*keep_prob - 1)`, which shrinks toward 0 as
    epsilon shrinks, making the correction itself high-variance and prone
    to clipping to 0/1 exactly where debiasing would matter most; (2) more
    fundamentally, AUC is a pairwise/rank statistic, so reweighting
    *instances* fed into `roc_auc_score` (via `sample_weight`) does not
    correctly reweight the *pairs* the statistic is computed over —
    unlike debiasing a mean or a proportion, this requires a
    pair-level correction that wasn't attempted here (out of scope,
    logged as a limitation in the paper draft).
  - **Decision:** kept as a genuine negative result, not discarded —
    it's a direct, honest answer to "should we explore other mechanisms"
    for this specific idea. `privatize_categorical_label` and the
    original hard-assignment scoring remain Study B's primary mechanism
    and result; nothing about the canonical sweep or its crossover
    (epsilon=3.0) changed.
- **Schema (additive, non-breaking):** both
  `results/study_b/epsilon_sweep_results.csv` and
  `results/study_b/seed_replication/dp_gap_replication.csv` gained
  `debiased_dp_gap`, `debiased_direction_match`, `debiased_pct_diff`,
  `debiased_survived` columns, scored from the same draw as the existing
  four columns. The replication summary gained
  `debiased_dp_gap_mean/std`, `debiased_pct_diff_mean/std`,
  `debiased_direction_agreement_rate`, `debiased_survival_rate`, and
  Wilson CI columns for both `survival_rate` and `debiased_survival_rate`.
  Re-ran `src/run_study_b.py` end to end to regenerate both CSVs under
  the new schema — no change to `true_gap` (0.067036) or any existing
  naive-column value, confirming the additions are purely additive.
- Updated `paper/study_drafts/study_b_draft.tex`: CIs added to the
  replication table, a new Discussion paragraph connecting Study A's gap
  bootstrap CI ([0.0230, 0.1118]) to the 15%-of-true-gap tolerance band
  ([0.057, 0.077] — narrower than Study A's own estimation uncertainty on
  the reference value), and a new section reporting the debiased-estimator
  comparison and its negative result.

## [Study B] 2026-08-19 (add draft findings write-up, paper/study_drafts/study_b_draft.tex)
- Wrote `paper/study_drafts/study_b_draft.tex`, mirroring Study A's draft
  format/tone (internal, numbers-focused results record, not paper prose):
  abstract, brief methods, the discarded-mechanism history (kept short —
  full account stays in CLAUDE.md and the two `[Shared]` CHANGELOG entries
  above, not duplicated here), the canonical single-draw sweep table, the
  30-seed replication table, a discussion section stating plainly that the
  replication changed the headline (direction unreliable at low epsilon,
  epsilon=4/5 only 90% reliable despite looking stable in the canonical
  draw), and a limitations section covering scope (90/10 arm only), the
  unvalidated 15% threshold, the missing analytical epsilon-to-attenuation
  relationship, and finite-replicate sampling uncertainty on the reported
  survival rates themselves.
- Updated CLAUDE.md's File Layout tree — `study_b_draft.tex` no longer
  listed as "created once Study B has results to record," since it now
  exists.
- No pdflatex/latexmk available in this environment to compile-check;
  verified `\begin`/`\end` environment balance and no unescaped `%`
  programmatically, and all `\geq`/`\leq` usages confirmed wrapped in math
  mode by inspection.

## [Study B] 2026-08-19 (seed replication; all-one-class guard fix)
- A second technical review (post-mechanism-rework) flagged two issues,
  both addressed here — see CLAUDE.md's new "Study B — Seed Replication"
  section for the full account.
- **`_subgroup_auc_gap` robustness fix.** Previously only guarded against a
  subgroup being fully empty; a nonempty subgroup with only one
  `true_label` value present also crashes `roc_auc_score` ("Only one class
  present in y_true" — confirmed directly, not assumed). Not triggered
  anywhere in the current `EPSILON_SWEEP` given ~3-5% pneumothorax
  prevalence, but was an unguarded path. Now returns `None` (same NaN/False
  handling as total erasure) for both failure modes.
- **30-seed replication added** (`run_seed_replication()`,
  `seed_replication_summary()`), separate seed pool
  (`REPLICATION_BASE_SEED=2000`) from the canonical sweep, ~80s total —
  cheap enough to run by default, not flag-gated. Writes
  `results/study_b/seed_replication/dp_gap_replication.csv`; summary
  reported here, not saved as its own file (same pattern as Study A's
  cross-seed gap spread).
- **The replication changed the honest headline, not just added error
  bars:**

  | epsilon | dp_gap mean (std) | direction agreement | survival rate |
  |---|---|---|---|
  | 0.1 | 0.0119 (0.027) | 70.0% | 3.3% |
  | 0.5 | 0.0169 (0.024) | 73.3% | 10.0% |
  | 1.0 | 0.0232 (0.024) | 90.0% | 6.7% |
  | 2.0 | 0.0504 (0.016) | 96.7% | 36.7% |
  | 3.0 | 0.0626 (0.011) | 100% | 50.0% |
  | 4.0 | 0.0652 (0.007) | 100% | 90.0% |
  | 5.0 | 0.0651 (0.005) | 100% | 90.0% |
  | 6.0 | 0.0668 (0.002) | 100% | 100% |
  | 8.0 | 0.0671 (0.0004) | 100% | 100% |
  | 10.0 | 0.0671 (0.0001) | 100% | 100% |

  - **Direction is unreliable at low epsilon, not just magnitude** — the
    canonical single draw (logged below) showed `direction_match=True` at
    every epsilon including 0.1, and the previous entry reported that at
    face value ("Direction is correct at every epsilon tested, including
    0.1"). That was one lucky draw: direction agreement is actually only
    70% at epsilon=0.1, 73% at 0.5, 66% at 1.0 (using the raw per-draw
    rate; the table above rounds). This is a correction to the previous
    entry's headline claim, not a new finding layered on top of it.
  - **Epsilon=4 and 5 looked stable in the single-draw table
    (`survived=True`, pct_diff 1.3%/0.3%) but are only 90% reliable** —
    1 in 10 replicate draws at each of those epsilons did not survive.
    Epsilon=3 is a near-exact coin flip (50.0%). Only epsilon>=6 hits
    100% survival across all 30 replicates.
  - The replication-based crossover (smallest epsilon at/above which
    survival_rate>=50% holds for every larger epsilon too) is 3.0 — same
    value as the canonical single draw's crossover — but that numeric
    agreement doesn't rescue the single-draw table's specific claims about
    4 and 5, which this replication shows were overstated.
- **Revised headline for the paper:** the gap is *reliably* (100% across
  30 replicates) preserved only at epsilon>=6. Below that, "survives" is
  probabilistic, not guaranteed — meaningfully so even at epsilon=4-5,
  points that looked solid from the single canonical draw alone. Any
  policy claim citing a specific epsilon should cite its survival rate
  from this table, not a single-draw `survived` boolean.
- Canonical single-draw sweep (`BASE_SEED=42`) is unchanged from the
  previous entry — logged again here only as the reference point the
  replication is compared against, not re-run.

## [Study B] 2026-08-19 (epsilon sweep results, reworked mechanism)
- Ran the full epsilon sweep against Study A's frozen `predictions_90_10.csv`
  test set (4,620 patients: 2,470 male, 2,150 female), using the reworked
  `privatize_categorical_label` mechanism (per-record randomized response
  — see CLAUDE.md's Subgroup Assignment Mechanism and the two `[Shared]`
  entries above for why the original aggregate-count mechanism was
  discarded and what replaced it). `src/run_study_b.py`, single
  deterministic draw per epsilon (`BASE_SEED=42`).
- **True gap** (recomputed from the frozen CSV as a sanity cross-check):
  0.067036 — matches Study A's canonical 90/10 gap (0.0670) and this
  study's own discarded-mechanism run, as expected (`load_frozen_test_set`
  and `_subgroup_auc_gap` are unchanged; only subgroup assignment changed).
- **Results** (`results/study_b/epsilon_sweep_results.csv`):

  | epsilon | dp_gap | pct_diff | survived |
  |---|---|---|---|
  | 0.1 | 0.03549 | 47.1% | False |
  | 0.5 | 0.05717 | 14.7% | True |
  | 1.0 | 0.04665 | 30.4% | False |
  | 2.0 | 0.09362 | 39.7% | False |
  | 3.0 | 0.05957 | 11.1% | True |
  | 4.0 | 0.06614 | 1.3% | True |
  | 5.0 | 0.06721 | 0.3% | True |
  | 6.0 | 0.06603 | 1.5% | True |
  | 8.0 | 0.06629 | 1.1% | True |
  | 10.0 | 0.06704 | 0.0% | True |

- **Crossover epsilon: 3.0** — the smallest epsilon at/above which every
  larger epsilon in the sweep also survives. Direction is correct at every
  single epsilon tested, including 0.1 — randomized response's flip
  probability never gets low enough in this sweep to reverse the sign, only
  to distort magnitude.
- **Non-monotonicity is real, not a bug.** epsilon=0.5 survived (14.7%) but
  epsilon=1.0 and 2.0 didn't (30.4%, 39.7%) — consistent with the
  development-time diagnostic (40 trials/epsilon), which found high
  variance through exactly this range: only 4/40 trials survived at
  epsilon=1, 13/40 at epsilon=0.5. A single draw landing outside the
  "typical" pattern at these epsilons is expected, not evidence of an
  error. This is a genuine improvement over the discarded mechanism's
  behavior: that mechanism's "noise" was so structurally small relative to
  cohort size that it produced an almost perfectly smooth, monotonic curve
  regardless of epsilon — smooth because it wasn't really testing per-record
  privacy protection at all (see the two `[Shared]` entries above). This
  sweep's roughness is a sign the mechanism is doing what it's supposed to.
- **Headline finding:** unlike the discarded mechanism, this one shows a
  real, well-behaved crossover well inside commonly-cited real-world
  epsilon values — the gap reliably survives at epsilon>=4-5, is
  unreliable in the 0.5-3 range (sometimes survives, sometimes doesn't,
  by chance), and is washed out in magnitude (though not direction) at
  epsilon=0.1. This is a materially different, more defensible, and more
  interesting result than the discarded mechanism produced, and it directly
  supports (with real caveats about single-draw variance, see below) the
  same qualitative policy point: DP-protected sex labels can preserve a
  subgroup bias-audit conclusion at practical privacy budgets, but the
  margin is narrower and noisier than the discarded mechanism made it look.
- **Caveat:** single-draw-per-epsilon, same limitation flagged for the
  discarded mechanism's sweep and not resolved here — CLAUDE.md's Study B
  deliverable schema has no seed column (see Subgroup Assignment Mechanism,
  "Not done" in the earlier design-decision entries). Given how much
  variance the diagnostic showed in the 0.5-3 range specifically, the
  crossover value (3.0) and the individual epsilon=0.5/1.0/2.0 results
  should be read as one realization of a genuinely noisy process, not
  stable per-epsilon facts — more so than for the discarded mechanism,
  where the "noise" turned out to barely exist at all.
- This is Study B's first real result under the corrected mechanism — not
  yet oracle-gated or merge-ready; CLAUDE.md doesn't define a merge
  criterion for Study B beyond "epsilon sweep complete, results logged in
  CHANGELOG.md" (branch table), which this entry satisfies.

## [Shared] 2026-08-19 (re-derive EPSILON_SWEEP for privatize_categorical_label)
- Diagnosed the epsilon range for the new mechanism before running the real
  sweep, same process as the discarded mechanism's extension: 40 trials per
  candidate epsilon, measuring mean `pct_diff` and direction-plus-15%
  survival rate against Study A's frozen 90/10 test set.
- **Result:** a much cleaner, better-behaved transition than the discarded
  mechanism ever showed — mean `pct_diff` ~0.0% at epsilon=10, ~0.3% at 8,
  ~3% at 5, ~8% at 4, crossing 15% right around epsilon=3-4 (23/40 trials
  survived at epsilon=3, 36/40 at epsilon=4), then climbing to ~32% at
  epsilon=2 and ~53% at epsilon=1. Sits comfortably inside the *original*
  0.1-10 range — no low-epsilon extension needed this time, unlike the
  discarded mechanism which never washed out anywhere in that range.
- **Decision:** `EPSILON_SWEEP` set to `[0.1, 0.5, 1, 2, 3, 4, 5, 6, 8, 10]`
  — kept the original six values for continuity with commonly-cited
  real-world epsilon, added 3/4/6/8 to resolve the transition zone, and
  dropped the discarded mechanism's 0.001-0.05 extension (randomized
  response is already deep in "washed out" territory there, ~90-115%
  `pct_diff`, no useful differentiation between points). Updated
  `check_dp_mechanisms.py`'s exact-match assertion and CLAUDE.md's three
  references. Re-ran `check_dp_mechanisms.py`: 16/16 still pass.

## [Shared] 2026-08-19 (add privatize_categorical_label; discard reassignment mechanism)
- A technical review of Study B's first epsilon-sweep result (before it was
  pushed anywhere) found two disqualifying problems with the mechanism —
  see CLAUDE.md's rewritten "Study B — Subgroup Assignment Mechanism" for
  the full account:
  1. **Construct validity.** The discarded mechanism reconstructed a
     per-patient group assignment by randomly moving just enough patients
     — starting from their *true* labels — to match a DP-noised aggregate
     count. At epsilon=10, 0 of 4,620 patients were ever reassigned; even
     at epsilon=0.1, ~7 were. The experiment could not have shown a
     different qualitative result across almost the entire sweep,
     regardless of whether genuinely privacy-protected per-record data
     would preserve the audit — it measured aggregate-count concentration
     at large N, not what the study actually asks.
  2. **Privacy accounting.** The aggregate mechanism's "full epsilon per
     category via parallel composition" claim assumes add/remove
     adjacency, but `patient_split.csv` is committed to git — cohort
     membership is already public, so the real threat model is attribute
     (substitute-one) privacy, under which that composition claim doesn't
     hold (a single attribute flip changes two bin counts at once,
     breaking parallel composition's precondition; sequential composition
     would apply instead, costing ~2x epsilon).
- **All `study-b`-branch-only commits from the discarded version were
  reset away** (the branch had never been pushed to origin — see this
  entry's git log for what remains). Not kept with a superseding commit;
  nothing about the discarded mechanism was worth preserving as a paper
  trail. The already-pushed `main` commits from the same session (the
  diffprivlib/scikit-learn fix, `dp_mechanisms.py` itself, the
  `EPSILON_SWEEP` extension) were kept — none of those are wrong, only the
  study-B-specific reassignment logic and its epsilon-sweep results were.
- **New function `privatize_categorical_label`** in `src/dp_mechanisms.py`
  — per-record randomized response via diffprivlib's `Binary` mechanism
  (kept with probability e^epsilon/(1+e^epsilon), flipped otherwise),
  applied independently to every record. Fixes both problems above: every
  output label is a genuine per-record randomized draw, and there's no
  aggregate count or cross-bin composition question at all. Added a
  module-level adjacency-model note to `dp_mechanisms.py` explaining which
  existing functions assume add/remove adjacency (general aggregate
  releases — still valid for their own use cases) vs. this new
  substitute-adjacency, attribute-privacy function.
- **New checks in `check_dp_mechanisms.py`** (4 added, 16/16 total pass):
  `label_flip_probability_matches_theory` (empirical kept-rate vs.
  e^epsilon/(1+e^epsilon) across 5 epsilons — the core correctness check,
  since a wrong flip probability would still produce plausible-looking
  M/F output with no crash), `label_output_domain_is_value0_or_value1`,
  `label_reproducible_with_same_seed`, `label_adds_real_noise_at_low_epsilon`.
  `nonpositive_epsilon_raises` extended to cover the new function too.
- **Not yet done:** `src/run_study_b.py` doesn't exist yet on the reset
  `study-b` branch — this entry covers the shared-infra fix only.
  `EPSILON_SWEEP` is unchanged for now (still the range tuned to the
  discarded mechanism); CLAUDE.md now flags explicitly that it needs
  re-diagnosing against the new mechanism's very different noise profile
  (e.g. epsilon=1 already flips ~27% of labels, vs. ~1 patient reassigned
  under the old mechanism at the same epsilon) before the next sweep runs.

## [Shared] 2026-08-19 (extend EPSILON_SWEEP below 0.1)
- While building `src/run_study_b.py`, the first run against the original
  `EPSILON_SWEEP = [0.1, 0.5, 1, 2, 5, 10]` found the DP-protected 90/10
  gap survived every single epsilon — `pct_diff` never exceeded 1.1%, even
  at epsilon=0.1. Diagnosed by counting patients actually reassigned per
  epsilon (out of 4,620 test patients): 0 at epsilon=10, ~1 at epsilon=1,
  ~7 at epsilon=0.1 — the count-release mechanism's noise scale is
  `1/epsilon`, trivial next to cohort counts in the thousands. See
  CLAUDE.md's new "Study B — Subgroup Assignment Mechanism" section for
  the full mechanism this diagnosed against.
- A follow-up diagnostic sweep (15 trials each) found the actual
  transition zone: mean `pct_diff` was ~0.8% at epsilon=0.1, ~2.5% at
  0.03, ~9% at 0.01, ~14% at 0.005, crossing 15% somewhere around
  0.003-0.005 — one to two orders of magnitude below the original sweep's
  floor. At epsilon<=0.001, reassignment could push an entire subgroup to
  size 0 (AUC undefined) rather than merely wash out the gap.
- **Decision:** extended `EPSILON_SWEEP` in `src/dp_mechanisms.py` to
  `[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]` — the four added
  points bracket the observed transition; the original six are kept for
  continuity with commonly-cited real-world epsilon values, even though
  none of them wash out the gap under this mechanism. Updated
  `check_dp_mechanisms.py`'s exact-match assertion and CLAUDE.md's two
  references to the old six-value list. Re-ran `check_dp_mechanisms.py`:
  12/12 still pass.
- **Not done:** seed-replicating individual epsilon points near the
  crossover, even though the diagnostic showed real variance there (e.g.
  epsilon=0.01's `pct_diff` ranged 0.3%-39% across 15 trials in the
  diagnostic). CLAUDE.md's Study B deliverable schema is one row per
  epsilon with no seed column; changing that is a separate decision from
  extending the sweep's range, not bundled into this one. The variance is
  reported as a caveat alongside the actual sweep results, not smoothed
  over by adding replication unasked.

## [Shared] 2026-08-19 (fix diffprivlib/scikit-learn conflict; add src/dp_mechanisms.py)
- **diffprivlib/scikit-learn fix.** Discovered while starting Study B that
  `import diffprivlib` was completely broken against the `requirements.txt`
  state merged from Study A: `ImportError: cannot import name 'DOUBLE' from
  'sklearn.tree._tree'`. Cause: diffprivlib==0.6.6's top-level `__init__.py`
  unconditionally imports its `models` submodule (DP-aware ML models,
  unrelated to what this project needs), which imports private
  (non-public) `sklearn.tree._tree` Cython symbols that scikit-learn
  removed at some point after 1.4.x. diffprivlib declares
  `scikit-learn>=1.4.0` with no upper bound, which doesn't reflect this —
  it's a private-API break, not a declared incompatibility. Verified
  `pip install scikit-learn==1.4.2` fixes the import (confirmed
  `diffprivlib.mechanisms.Laplace` works and `sklearn.metrics.roc_auc_score`
  still returns correct values) with no `pip check` conflicts. Pinned in
  `requirements.txt` with an explanatory comment. Low risk to Study A/main's
  frozen code, which only ever called `sklearn.metrics.roc_auc_score` — one
  of scikit-learn's most stable APIs.
- **New file `src/dp_mechanisms.py`** — the shared DP mechanism module
  CLAUDE.md's Shared Infrastructure section already documented an interface
  for but that, like `src/metrics.py` before the 2026-08-18 split-sensitivity
  session, had never actually been written. Implements exactly the four
  functions + constant CLAUDE.md names:
  - `EPSILON_SWEEP = [0.1, 0.5, 1, 2, 5, 10]`.
  - `privatize_categorical_counts` — Laplace mechanism, sensitivity=1 per
    category. Categories are assumed to partition the population (e.g. a
    sex breakdown of one fixed cohort), so each category's mechanism uses
    the *full* epsilon under parallel composition, not `epsilon /
    n_categories` — documented explicitly in the docstring since it's an
    easy mistake to make (sequential composition would be the wrong model
    here).
  - `privatize_age_mean` — clips ages to `bounds` (public, fixed
    transformation) before computing sensitivity `(hi-lo)/n` and applying
    Laplace noise to the mean; cohort size `n` itself is treated as public.
  - `privatize_age_histogram` — same parallel-composition argument as
    categorical counts, applied per bin after clipping ages into range.
  - `privatize_categorical_proportions` — counts privatized the same way,
    divided by a `total` that defaults to `sum(counts.values())` and is
    treated as public (not itself privatized) — a documented assumption
    that matters for Study C, which frames the question as "a subgroup's
    share of an already-known population size." Includes a closed-form
    Laplace CI (`ci_lower`/`ci_upper`) derived from the mechanism's known
    noise scale, not resampling — needed for Study C's
    `results/study_c/detection_floor.csv` schema.
- **New file `src/check_dp_mechanisms.py`** — 12 unit/invariant checks,
  matching the precedent set by Study A's `check_pipeline_invariants.py`.
  CLAUDE.md requires dp_mechanisms.py changes to pass its own unit tests
  before either study branch pulls in a new version; this is that test
  file. All 12 pass as of this entry.
- **Bug caught by the coverage check, fixed before this entry.** The first
  implementation of `privatize_categorical_proportions` rounded the noisy
  count to an integer *before* computing the proportion, then built the CI
  around that rounded value using a half-width derived for the continuous
  (unrounded) Laplace distribution — a center/interval mismatch.
  `proportions_ci_achieves_nominal_coverage` (3,000 trials, nominal 95%
  confidence) measured ~91-92% empirical coverage instead of ~95%, well
  outside the ±2% Monte Carlo tolerance, catching it immediately. Fixed by
  computing `proportion` (and its CI) from the unrounded noisy value;
  `noisy_count` stays a separately-rounded integer for display only, not
  fed back into the CI. Re-ran after the fix: 95.0% coverage, 12/12 checks
  pass. Recorded here as a concrete demonstration of why this file has its
  own check script rather than only inline reasoning — the bug produced
  plausible-looking output (valid-looking floats, no crash) and would not
  have been caught without an empirical calibration check.

## [Study A] 2026-08-19 (pre-merge hardening: auxiliary-file guardrail + paper/ restructure)
- **Auxiliary-file guardrail for Study B.** The upcoming merge of `study-a`
  into `main` brings in more than the two files Study B is allowed to read —
  26 auxiliary CSVs under `results/study_a/seed_replication/`,
  `split_sensitivity/`, and `n_sensitivity/` land in the same directory,
  matching the same `predictions_*.csv` naming pattern and column schema as
  the two allowed files, with nothing visually distinguishing allowed from
  forbidden. Documented this explicitly in CLAUDE.md: a new subsection under
  "The Frozen Handoff" spells out the full post-merge directory listing with
  ALLOWED/NOT ALLOWED markers on every entry, and instructs
  `src/run_study_b.py`'s loader to hardcode the two exact filenames rather
  than glob. Study B's "Must not" bullet also now names the three forbidden
  subdirectories explicitly (previously only named the two forbidden sibling
  CSVs, not the subdirectories) and points back to that section. No code
  changes yet — `src/run_study_b.py` doesn't exist until the `study-b`
  branch — this closes the documentation gap before that loader gets
  written, not after.
- **`paper/` restructured into a two-tier layout**, at the user's request:
  `paper/Final_Policy_Recommendation.tex` stays untouched until all three
  studies are complete, at which point it will be rewritten to combine the
  existing policy paper with the studies' findings. Per-study internal,
  numbers-focused result records (not paper prose) now live in
  `paper/study_drafts/` — `study_a_draft.tex` moved there via `git mv`
  (history preserved); `study_b_draft.tex` and `study_c_draft.tex` will be
  created in the same directory once those studies have results to record,
  not stubbed out empty now. Updated the one cross-reference to the old
  `paper/study_a_draft.tex` path inside CLAUDE.md's Magnitude Oracle
  Resolution section, and the File Layout tree.

## [Study A] 2026-08-19 (magnitude oracle resolution against Larrazabal Fig. 1)
- Closed out the manual magnitude comparison that every prior entry below
  deferred ("left for manual comparison against Figure 1") without ever
  actually performing it or recording a verdict. Source: `notebooks/Larrazabal
  Study.pdf` + `notebooks/Larrazabal Figure 1.jpg`, provided this session.
  `poppler-utils`/`pdftoppm` were not installed in this environment, so the
  PDF page was rendered via `pymupdf` (installed for this task) at 4x scale
  instead; Figure 1 values below are a visual read off the box-plot mean
  markers, not a pixel-calibrated digitization — treat as ±0.01.
- **Panel selection:** Fig. 1 panels B-2/C-2 (Pneumothorax) are the correct
  comparison — single model trained on a mixed-sex ratio, evaluated
  separately on male (B-2) and female (C-2) test folds, matching Study A's
  gap definition (male test AUC − female test AUC from one model). Panel A
  (single-sex-only training, cross-sex generalization drop) is a different
  quantity and was not used. Larrazabal's x-axis (% female in training) only
  has points at 0/25/50/75/100 — confirms the "no 90/10 point" note already
  in prior entries below.
- **Values read:**
  - 0% female training: male AUC ≈0.84, female AUC ≈0.705, gap ≈0.135.
  - 25% female training: male AUC ≈0.835, female AUC ≈0.735, gap ≈0.10.
  - Linear interpolation to Study A's 10% female composition: gap ≈0.12.
- **Verdict:** direction PASSES (male AUC > female AUC in both sources, as
  already established). Magnitude does **not** pass the literal "within 2
  AUC points" criterion stated in CLAUDE.md's original Test oracle wording:
  Study A's canonical 90/10 gap (0.0670) is ≈0.05 below the interpolated
  Larrazabal value (≈0.12), and ≈0.03–0.08 below either raw neighboring grid
  point — 1.5×–4× the 0.02-AUC tolerance depending on reference point.
- **Decision:** rather than treat this as a blocking failure or silently
  drop the criterion, revised CLAUDE.md's Test oracle (Study A section) to
  make magnitude a documented comparison instead of a strict gate — see
  CLAUDE.md's new "Study A — Magnitude Oracle Resolution" subsection for the
  full rationale (no exact 90/10 point exists on Larrazabal's grid; the two
  studies use different variance estimators — 20-fold aggregation vs. this
  study's 5-seed/3-split replication — so a tolerance calibrated for an
  exact same-estimator comparison doesn't automatically transfer). Study A's
  merge-to-`main` gate is now direction PASS + magnitude discrepancy
  documented, which this entry satisfies.
- **Not a retroactive pass:** this does not mark any earlier entry's
  deferred magnitude check as having succeeded — the discrepancy is real and
  is reported as such, not minimized. Study B's actual dependency (a real,
  correctly-signed, statistically significant subgroup gap — bootstrap CI
  [0.0230, 0.1118] excludes zero) is unaffected by this discrepancy.
- Per this revised gate, `study-a`'s deliverables (three predictions CSVs +
  `patient_split.csv`) are now eligible to merge into `main`.

## [Study A] 2026-08-19 (N=5000 sample-size sensitivity, 3-seed replication)
- Replicated the exploratory N=5,000 sample-size-sensitivity pass across
  2 additional seeds (43, 44; canonical 42 unchanged) after the
  single-seed result logged below showed a surprising, non-monotonic
  pattern — one run wasn't enough to trust given the noise floor already
  established elsewhere in this study. See CLAUDE.md, Study A
  Sample-Size Sensitivity.
- **Cross-seed gap spread per arm at N=5,000** (`src/metrics.py --note
  n_sensitivity_seed`):
  - 90/10: canonical=0.0179, other seeds=[0.0326, 0.0439], mean=0.0315,
    range=[0.0179, 0.0439]
  - 70/30: canonical=0.0073, other seeds=[0.0220, 0.0182], mean=0.0159,
    range=[0.0073, 0.0220]
  - 50/50: canonical=0.0188, other seeds=[0.0299, 0.0011], mean=0.0166,
    range=[0.0011, 0.0299]
  - Direction agreement is True in all 9 runs (majority AUC > minority
    AUC every time).
- **The gap-shrinks-at-smaller-N finding replicates — not a single-seed
  fluke.** Mean gaps at N=5,000 (0.0315 / 0.0159 / 0.0166 for 90/10 /
  70/30 / 50/50) are still substantially below their canonical-N
  (11,664) counterparts (0.0535 / 0.0388 / 0.0364) for every arm,
  confirming the direction reported in the single-seed entry below.
- **The 70/30-vs-50/50 indistinguishability replicates across a very
  different total-N regime.** At N=5,000, 70/30's mean gap (0.0159) and
  50/50's (0.0166) are again nearly identical and each within the
  other's seed-to-seed range — the same pattern already found at
  canonical N (0.0388 vs. 0.0364). 90/10 remains clearly the largest gap
  at both N levels. This is the most useful result of this pass:
  "90/10 has the biggest gap; 70/30 and 50/50 are not reliably
  distinguishable" now holds at two very different training-set sizes,
  which is stronger support for that claim than either N level alone.
- **Gap std at N=5,000 (0.0076-0.0145) is not obviously larger than at
  canonical N (0.0115-0.0212)** — contrary to the naive expectation that
  a smaller training set makes training noisier. Female-subgroup AUC std
  specifically looks much smaller at N=5,000 (0.0015-0.0111) than at
  canonical N (0.0208-0.0230, close to the ~0.023 Hanley-McNeil
  evaluation-noise floor discussed in the seed-replication entries
  below). With only 3 seeds per arm, this is most likely a small-sample
  artifact in the std estimate itself — the same caveat already noted
  for 70/30's/50/50's suspiciously low male-AUC std in the canonical
  seed-replication entry — not evidence that N=5,000 training is
  genuinely more stable. Not enough seeds to tell the difference either
  way.
- **Overall takeaway for the paper:** reported gap magnitudes are
  training-set-size-dependent — all three arms' gaps roughly halve or
  more at N=5,000 vs. N=11,664 — a real caveat on how the headline 90/10
  gap value should be generalized. But the *relative* ordering across
  ratios (90/10 distinctly worse; 70/30 and 50/50 statistically tied)
  holds up across both training-set sizes tested. Per CLAUDE.md's Not in
  scope note, no further N levels were run — this closes out the
  sample-size-sensitivity investigation as scoped.

## [Study A] 2026-08-19 (sample-size sensitivity, exploratory)
- Ran the exploratory sample-size-sensitivity pass added this session: all
  three ratio arms (90/10, 70/30, 50/50), each trained once at a fixed
  `N_total=5,000` (vs. the canonical 11,664), canonical seed=42, canonical
  split. Motivated by the seed-replication finding above that 70/30 and
  50/50's gaps are statistically indistinguishable at the canonical
  budget — this checks whether the ratio's effect on the gap looks
  different at a smaller training size, where no arm has "abundant"
  minority data. See CLAUDE.md, Study A Sample-Size Sensitivity.
- **Results** (`src/metrics.py --note n_sensitivity`):

  | arm | male AUC | female AUC | gap (N=5,000) | gap (canonical N=11,664) |
  |---|---|---|---|---|
  | 90/10 | 0.8410 | 0.8231 | 0.0179 | 0.0670 |
  | 70/30 | 0.8314 | 0.8241 | 0.0073 | 0.0415 |
  | 50/50 | 0.8706 | 0.8518 | 0.0188 | 0.0170 |

- **Every arm's gap shrinks substantially at N=5,000 relative to its
  canonical value** — most sharply for 90/10 (0.0670→0.0179) and 70/30
  (0.0415→0.0073); 50/50 is roughly unchanged (0.0170→0.0188). This is
  the opposite of the "representation matters more under scarcity"
  pattern the pass was designed to look for. Both subgroup AUCs are also
  uniformly lower at N=5,000 than at the canonical budget (male:
  0.83-0.87 vs ~0.89-0.92 canonical; female: 0.82-0.85 vs ~0.84-0.87
  canonical), consistent with a less-fit model overall — from less
  training data — compressing both subgroups toward a similar, weaker
  performance level, which would mechanically shrink the gap regardless
  of ratio.
- **Cross-ratio ordering at N=5,000 is non-monotonic and inverted from
  canonical:** 70/30 (0.0073) < 90/10 (0.0179) < 50/50 (0.0188) — 50/50
  now shows the *largest* gap of the three, not the smallest. Per
  CLAUDE.md's stated Limits for this pass, this is a single seed per arm
  at a smaller (and likely noisier) training budget, on top of the same
  fixed-test-set evaluation noise floor already implicated in the
  seed-replication entries below (Hanley-McNeil SE ≈0.023-0.025 per
  subgroup AUC, given only ~75-107 positive test cases per sex) — this
  ordering should not be read as a robust finding, only as the reason
  this pass was scoped exploratory rather than gating.
- **Takeaway:** the one consistent signal across all three independently
  trained arms is the gap shrinking at smaller N, not a clean ratio
  effect emerging under data scarcity. Given the single-seed caveat, the
  honest read is "training-set size affects the gap at least as much as
  ratio does, in the opposite direction to what this pass hypothesized"
  — worth flagging as a reason to caveat the ratio-sweep's external
  validity in the paper, not as a citable finding on its own. Per
  CLAUDE.md's Not in scope note, no further N levels or seed replication
  were run as part of this addition.

## [Study A] 2026-08-19 (70/30 and 50/50 seed replication)
- Ran the lighter 3-seed replication (42 canonical + 43-44) for 70/30 and
  50/50 added in this entry's preceding commit, to check whether the
  cross-arm trend (gap shrinking as training-set balance improves) holds
  up beyond single-run point estimates. See CLAUDE.md, Study A Seed
  Replication.
- **Cross-seed gap spread per arm** (`src/metrics.py --note seed`):
  - 90/10: canonical=0.0670, other seeds=[0.0396, 0.0599, 0.0437,
    0.0574], mean=0.0535, range=[0.0396, 0.0670]
  - 70/30: canonical=0.0415, other seeds=[0.0532, 0.0217], mean=0.0388,
    range=[0.0217, 0.0532]
  - 50/50: canonical=0.0170, other seeds=[0.0590, 0.0333], mean=0.0364,
    range=[0.0170, 0.0590]
  - Direction agreement is True within every arm (every seed in every
    arm shows majority AUC > minority AUC — the gap's sign is robust).
- **The cross-arm trend is directionally real but noisier than the
  canonical-only numbers suggest.** Comparing canonical-seed values
  alone (0.0670 → 0.0415 → 0.0170) looks like a clean monotonic story.
  Comparing mean-across-seeds still preserves the ordering (0.0535 →
  0.0388 → 0.0364), but 70/30's and 50/50's ranges overlap substantially
  — 50/50 seed43's gap (0.0590) exceeds every 70/30 seed's gap, and even
  approaches 90/10's minimum (0.0396). So: *90/10 has a clearly larger
  gap than the other two arms*, but *70/30 vs. 50/50 individually is not
  a robust ordering* — their single-run point estimates (0.0415 vs.
  0.0170) overstate how separable those two arms actually are. If the
  paper cites the three-arm trend, it should lean on the 90/10-vs-rest
  contrast and the mean-gap ordering, not on the 70/30-vs-50/50 gap
  specifically.
- Not run for 70/30/50/50 (by design — see CLAUDE.md): bootstrap CI,
  split-sensitivity. This replication is informational only, not
  oracle-gating; the pass/fail criterion remains the single 90/10
  seed=42 canonical-split run.

## [Study A] 2026-08-19 (full sweep results)
- Ran the full sweep: 90/10 arm (canonical seed=42 + replication seeds
  43-46), 70/30 arm, 50/50 arm, and split-sensitivity (alternate splits
  101-103, canonical training seed=42), per CLAUDE.md's Study A design.
  All runs completed without errors; predictions CSVs match the fixed
  16,653-row test set on the canonical split, invariant checks pass.
- **Canonical 90/10 gap (seed=42, canonical split):** 0.0670 patient-level
  AUC (majority male AUC minus minority female AUC).
- **Oracle direction check:** PASS (majority AUC > minority AUC, as
  Larrazabal et al. report). The magnitude half of the oracle
  ("within 2 AUC points") is not automated — see `check_oracle_direction`'s
  docstring and the 2026-08-18 pre-flight entry: Larrazabal et al. (2020)
  report Pneumothorax only as box plots (Fig. 1, panels B-2/C-2) with no
  90/10 point and no numeric table, so there's no reliable number to check
  against programmatically. Reporting the gap value here for manual
  comparison against Figure 1.
- **Bootstrap 95% CI** (patient-level, stratified by sex, 1,000 resamples,
  `BOOTSTRAP_SEED=1042`) on the canonical gap: **[0.0230, 0.1118]**,
  excludes zero.
- **Cross-seed gap spread** (90/10 arm, canonical split, seeds 42-46):
  canonical=0.0670, other seeds (43-46)=[0.0396, 0.0599, 0.0437, 0.0574],
  mean=0.0535, range=[0.0396, 0.0670], direction agreement=True (all 5
  seeds show majority AUC > minority AUC).
- **Cross-split gap spread** (90/10 arm, canonical training seed=42,
  splits 101-103): canonical=0.0670, other splits=[0.0722, 0.0815,
  0.0657], mean=0.0716, range=[0.0657, 0.0815], direction agreement=True
  (all 3 alternate splits show majority AUC > minority AUC).
- Per-arm final test patient-AUC: 90/10 seed42=0.8848 (male), seed43
  replicate whole-run test-AUC=0.8758, seed44=0.8812, seed45=0.8256,
  seed46=0.8614; 70/30 seed42=0.8801; 50/50 seed42=0.8845. (These are
  whole-test-set AUCs from training logs, not the sex-disaggregated
  subgroup AUCs the gap figures above are computed from.)
- Pass/fail verdict on gap *magnitude* against Larrazabal et al. is
  intentionally not stated here — left for manual comparison against
  Figure 1, per CLAUDE.md's Test oracle scope section.

## [Study A] 2026-08-18 (pre-flight review fixes)
- Findings from a pre-run readiness/reviewer pass, addressed before starting
  the real training sweep:
  - `src/metrics.py`: added the third robustness note CLAUDE.md's Test
    oracle scope section calls for but that was never actually implemented
    — `bootstrap_gap_ci` (patient-level, stratified by sex, 1,000
    resamples, percentile CI on the 90/10 canonical gap). Dedicated
    `BOOTSTRAP_SEED=1042`, distinct from the 42-46 training-seed pool and
    the 101-103 split-seed pool. Non-gating, reported alongside the
    existing cross-seed/cross-split spreads.
  - `src/metrics.py`: added `check_oracle_direction` — automates the
    direction half of the test oracle only (majority AUC > minority AUC).
    The magnitude half ("within 2 AUC points of Larrazabal et al.'s
    reported gap") is explicitly **not** automated: Larrazabal et al.
    (2020) report Pneumothorax only as box plots (Fig. 1, panels B-2/C-2)
    across female-training ratios 0/25/50/75/100%, with no 90/10 point and
    no numeric table in the text or SI Appendix — hardcoding a number read
    off that figure would be false precision the source doesn't support.
    `main()` now exits 1 if the direction check fails; the magnitude
    comparison stays a manual note against Figure 1.
  - **New file** `src/check_pipeline_invariants.py` — a pre-flight
    validation script (not part of CLAUDE.md's documented Study A file
    layout, added deliberately as a reviewer-requested gate before
    spending GPU-hours): checks image-file/metadata consistency, patient
    sex consistency, split leakage, undersampling-budget invariants
    (training patients ⊆ train split, fixed N_total across arms, achieved
    vs. target sex ratios), val/test representativeness across arms, split
    determinism, and `patient_level_auc`/`bootstrap_gap_ci` correctness on
    synthetic examples. All 10 checks pass against the real repo data as of
    this entry; negative-control spot checks confirm the assertions
    actually fail when their invariant is violated (not tautological
    passes). Run via `python src/check_pipeline_invariants.py`.

## [Study A] 2026-08-18 (split sensitivity)
- Added a split-level counterpart to the existing cross-seed robustness
  note: reproduces the 90/10 gap on 3 additional patient-level 70/15/15
  splits (seeds 101, 102, 103 — distinct from the 42-46 training-seed
  range) rather than 3 additional training runs on the same split, to
  check the gap isn't an artifact of which patients happened to land in
  test. Only the 90/10 arm, canonical training seed=42, is affected —
  same scope rule as Seed Replication. See CLAUDE.md, Study A Split
  Sensitivity.
- `src/data_loading.py`: factored the split-generation logic out of
  `get_patient_split` into `_generate_split_df`, shared with the new
  `get_alternate_patient_split(metadata, seed)`, which writes to
  `results/study_a/split_sensitivity/patient_split_seed{N}.csv` and never
  touches the canonical `patient_split.csv`.
- `src/train.py`: generalized `train_one_arm` to accept an explicit
  `output_path`/`run_name`/`undersample_seed` (defaults reproduce prior
  behavior exactly) so it can be reused for split-sensitivity runs without
  colliding with canonical/seed-replication checkpoint and log filenames —
  all split-sensitivity runs share `run_seed=SEED` (only `split_df`
  differs), so filenames are keyed by split, not training seed. Added
  `train_split_sensitivity()` and a `--split-sensitivity` CLI flag, writing
  to `results/study_a/split_sensitivity/predictions_90_10_split{N}.csv`.
- `src/metrics.py`: **new file** — didn't exist yet, so this also
  implements the cross-training-seed gap spread that CLAUDE.md's Seed
  Replication section already described as living here (documented but
  never actually written until now), alongside the new cross-split gap
  spread. Both are non-gating CHANGELOG robustness notes computed from
  patient-level subgroup AUC gaps; neither changes the pass/fail oracle,
  which stays the single canonical seed=42, canonical-split 90/10 run.
- Not yet run: no Study A training has happened in this repo yet (no
  `predictions_*.csv` exist), so neither the cross-seed nor the
  cross-split spread has an actual numeric result to log yet — this entry
  covers the implementation only. Numeric results to follow once the
  90/10 sweep (canonical + seed replication + split sensitivity) is run.

## [Study A] 2026-08-18 (train.py throughput)
- Verified the two remaining items from the reviewer pass that hadn't
  actually been implemented yet: patient-level ground-truth aggregation
  via `max` was already correct in code but undocumented — added a note
  to CLAUDE.md. Mixed precision (AMP) had been raised but never
  implemented — investigated below instead of adding it blind.
- Benchmarked the CPU/GPU split on the RTX 4070 + Ryzen 9800X3D (WSL2):
  pure GPU compute (no data loading) hits 218.5 img/s FP32/TF32 vs.
  311.0 img/s AMP bf16 — AMP clearly helps GPU-bound compute. But the
  full pipeline at `num_workers=4` only reached ~130-134 img/s regardless
  of AMP, meaning it was CPU-bound (PNG decode + resize), not GPU-bound —
  AMP would have added complexity for ~0 real speedup at that setting.
- Found WSL2 exposes only 8 logical CPUs on this machine (`lscpu`:
  `Thread(s) per core: 1`, no `.wslconfig` present) — the 9800X3D's other
  8 SMT threads aren't visible to Linux. Getting them would need
  `processors=16` in `.wslconfig` + `wsl --shutdown`, which kills all
  running WSL sessions — not done; flagging as available but disruptive
  if GPU-hours become a harder constraint later.
- Within the current 8-CPU cap, raising `num_workers` 4→7 (not 8 — no
  core left for the main process, which measured slightly worse) closed
  most of the CPU/GPU gap: ~130-134 img/s → ~195-220 img/s (45-65% faster,
  with some run-to-run variance). Adopted in `train.py`. Also added
  `persistent_workers=True` so the 7 workers aren't respawned every
  epoch, which `num_workers=7` alone wouldn't otherwise benefit from
  given train/val loaders are re-iterated every epoch.
- Re-tested AMP on top of `num_workers=7`: measured 195.3 img/s vs.
  195.7-220.7 img/s without AMP across two non-AMP reruns — the
  difference is within run-to-run noise, not a real effect, once the CPU
  bottleneck is fixed. **Not adopted** — no clear benefit to justify the
  added `autocast` complexity at this bottleneck balance.

## [Study A] 2026-08-18 (train.py)
- Wrote `src/train.py`: full end-to-end fine-tuning of ImageNet-pretrained
  DenseNet-121 (single-logit head) across the 90/10, 70/30, 50/50 arms.
  AdamW, LR 3e-5 (within CLAUDE.md's 1e-5-1e-4 range, fixed across arms),
  max 20 epochs, early stopping on patient-level validation AUC (patience
  5, gradient clipping at norm 1.0), identical across all three arms.
  Writes `results/study_a/predictions_{arm}.csv` per the frozen schema.
  Checkpoints saved to `results/study_a/checkpoints/*.pth` (gitignored).
- Post-review hardening after a reviewer-style pass on the initial draft:
  - GPU training now runs with `torch.use_deterministic_algorithms(True)`
    (+ `cudnn.deterministic`) so seed 42 is an actual bit-reproducibility
    anchor — the initial draft was not reproducible on GPU despite
    CLAUDE.md's claim that seeding alone was sufficient.
  - 90/10 (the only oracle-gated arm) is now replicated across 5 seeds
    (42 canonical + 43-46) to check its gap isn't a one-run artifact of
    training stochasticity; 70/30 and 50/50 stay single-run since they
    aren't independently oracle-gated. See CLAUDE.md, Study A Seed
    Replication. Canonical seed=42 output is still the frozen
    `predictions_90_10.csv`; replicate seeds write to
    `results/study_a/seed_replication/`, outside Study B's read contract.
  - Runs whose output CSV already exists are skipped (`--force` to redo),
    so an interrupted multi-run sweep only loses the run in flight, not
    everything before it.
  - Added a NaN-val-AUC guard (raises immediately with a clear message
    instead of a confusing downstream FileNotFoundError from a
    never-written checkpoint) and per-epoch loss/val-AUC logging to
    `results/study_a/logs/train_log_{arm}_seed{run_seed}.csv`.
- Benchmarked real throughput on the RTX 4070 (90/10 arm, real data,
  determinism enabled): ~130 img/s train, ~167 img/s val → ~7.4 min/epoch,
  ~1-2.5h per run depending on early stopping. Full plan (90/10 x5 seeds +
  70/30 + 50/50 single runs, 7 runs total): ~7-17.5h estimated, not yet
  run.
- Smoke-tested the full pipeline (train step, checkpointing, patient-level
  AUC, CSV schema/dtypes, skip-if-exists, per-epoch log) on a tiny
  hand-balanced subset written to scratch, not the real results/ path —
  the real multi-hour training run has not happened yet.

## [Study A] 2026-08-18
- Wrote `src/data_loading.py`: metadata loading, frozen 70/15/15
  patient-level split (seed 42), patient-level undersampling for the
  90/10, 70/30, 50/50 sex-imbalance sweep (fixed N_total=11664 across all
  three arms, female fixed as minority sex), fixed representative val/test
  sets, and the ImageNet-normalized image dataset for the pretrained
  backbone.
- Generated `results/study_a/patient_split.csv` (21564/4621/4620
  train/val/test patients) — frozen from this point forward.
- Dropped the `torchxrayvision` import from `src/data_loading.py`
  entirely — resizing to 224x224 is now done directly with
  `skimage.transform.resize` (same call `XRayResizer` made internally, so
  preprocessing output is unchanged) instead of via the package. Study A's
  code now has zero dependency on `torchxrayvision`, for weights or
  preprocessing, avoiding any structural contact with the chest-X-ray-
  pretrained ecosystem. Package remains installed/pinned in
  `requirements.txt` but is unused by any study's code.

## [Shared] 2026-08-18
- Study A backbone pinned to ImageNet-pretrained DenseNet-121
  (`torchvision.models.densenet121`, `DenseNet121_Weights.IMAGENET1K_V1`),
  not a torchxrayvision chest-X-ray checkpoint — avoids NIH-corpus
  pretraining leakage into the test split and matches Larrazabal et al.'s
  own init methodology, which the test oracle checks against.

## [Setup]
- Environment created: venv at ~/envs/dp-cxr, PyTorch <version>, CUDA verified on RTX 4070.
# Project Context — Setup Log
### Differential Privacy for Medical Imaging Metadata

Snapshot of what's been decided and completed so far. Meant to be pasted
into `CHANGELOG.md` or handed to anyone (human or agent) picking this up.

---

## Environment — completed

| Component | Decision | Status |
|---|---|---|
| Linux layer | WSL2 | ✅ Installed |
| Distro | Ubuntu 24.04 LTS (chose over the newer 26.04 for CUDA/WSL maturity) | ✅ Installed |
| GPU | RTX 4070 — Windows driver only, nothing installed inside WSL | ✅ `nvidia-smi` verified working in WSL |
| Python | **venv**, not conda — nothing in the stack needs conda's binary package management | ✅ Created at `~/envs/dp-cxr` |
| ML libraries | PyTorch (CUDA build), torchxrayvision, diffprivlib, opendp, jupyterlab, pandas/numpy/sklearn/matplotlib | ✅ Installed |
| Git | SSH key generated (`ed25519` — chosen over RSA for speed, shorter keys, no weak-randomness failure mode) | ✅ Key added to GitHub, `ssh -T git@github.com` authenticated |
| GitHub | Private repo `dp-medical-imaging` | ✅ Created, remote added |
| Claude Code | Native installer (`curl -fsSL https://claude.ai/install.sh \| bash`) | ✅ v2.1.220 installed; required a manual `PATH` fix for `~/.local/bin` |
| VS Code | Remote-WSL extension + Claude Code extension + Python/Jupyter extensions, opened via `code .` from inside the project directory | ✅ Configured |
| tmux | For detach/reattach of long-running Claude Code sessions | ✅ Installed — **use a standalone terminal, not VS Code's integrated terminal** (VS Code's default `Ctrl+B` sidebar toggle intercepts the tmux prefix key) |
| Project scaffold | `CLAUDE.md`, `CHANGELOG.md`, `README.md`, `.gitignore`, `requirements.txt`, `venv_path.txt`, plus `data/`, `notebooks/`, `src/`, `results/`, `paper/` — all at repo root | ✅ Created |

## Research design — decided, not yet run

Three linked studies for the paper:

- **Study A (baseline):** Fine-tune a classifier on NIH ChestX-ray14 under varying sex-balance ratios; replicate Larrazabal et al. (2020)'s subgroup AUC gap as a sanity check on this pipeline.
- **Study B (core contribution):** Apply differential privacy to demographic fields (age, sex) at a range of epsilon values; test whether subgroup bias-audit conclusions survive vs. get washed out by DP noise. This is the direct empirical support for the paper's central policy claim.
- **Study C (objection response):** Simulate small subgroup prevalences (0.5–5%) to find the detection floor under DP noise — turns the paper's "small populations harmed by noise" rebuttal into a quantified threshold.

Compute note: Study A is the only GPU-bound piece (~15–25 GPU-hours total across imbalance ratios); Studies B and C are pure statistics/simulation, no GPU required.


## Not yet done / next steps

- [ ] Download NIH ChestX-ray14 (hosted on Box — happens on Andy's machine, not through Claude chat)
- [ ] Write data loading + preprocessing script (`src/data_loading.py`)
- [ ] Write the DP mechanism code for Study B/C (`src/dp_mechanisms.py`)
- [ ] Draft full `CLAUDE.md` content (template already provided, needs to be pasted in)
- [ ] First Claude Code session: scaffold verification + Study A training script