# Project: Differential Privacy for Medical Imaging Metadata

## Goal
Empirically support a policy paper arguing that differential privacy applied
to demographic/tabular metadata (age, sex) — not image pixels — preserves
enough utility for (a) representativeness checks and (b) subgroup bias
audits, using NIH ChestX-ray14.

## Environment
Always run this before anything else in a new session:
```bash
source ~/envs/dp-cxr/bin/activate
```
(`venv_path.txt` at the repo root points here — `cd` into the repo should
auto-activate if the shell hook from Setup Documentation is in place; if not,
activate manually.) Verify GPU visibility before Study A work:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Repository & Branch Structure

One repo, one paper. Each study gets its own branch so work on one can never
silently leak into another. `main` holds shared infrastructure and each
study's *frozen, completed* deliverables — never a study's in-progress state.

| Branch | Purpose | Merges into `main` when |
|---|---|---|
| `main` | CLAUDE.md, README, requirements.txt, `paper/`, shared `src/dp_mechanisms.py`, and each study's frozen deliverables | trunk — not merged anywhere |
| `study-a` | Baseline training, sex-balance sweep | Passes its test oracle (direction reproduced; magnitude compared and documented — see Magnitude Oracle Resolution) |
| `study-b` | DP vs. true demographics on frozen Study A output | Epsilon sweep complete, results logged in CHANGELOG.md |
| `study-c` | Synthetic small-subgroup detection floor | Complete, results logged in CHANGELOG.md |

**Rules:**
- Before touching any file, confirm the checked-out branch matches the study
  you were asked to work on. If it doesn't, stop and ask — don't switch
  branches or guess which one was meant.
- Never merge a study branch into `main` until that study's own test oracle
  has passed and the result is logged in CHANGELOG.md.
- `study-b` and `study-c` branch off `main`, never off `study-a` directly —
  so they only ever inherit Study A's frozen, merged output, never its
  uncommitted or in-progress state.
- `study-c` has no dependency on Study A or B and may be branched and worked
  at any time.
- Don't edit another study's owned files "in passing" to fix something you
  noticed — note it in CHANGELOG.md under that study's tag and leave it for
  that study's own session.

---

## Shared Infrastructure (lives on `main`)

**`src/dp_mechanisms.py`** — the only place DP noise logic is implemented.
Study B and Study C both import from it; neither reimplements or modifies
noise logic locally. Treat this file as reviewed infrastructure, not
experimental code — changes must still pass its own unit tests before either
study branch pulls in a new version.

- `privatize_categorical_label(...)` — per-record randomized response on a
  binary attribute (Study B's actual subgroup-AUC mechanism, see Subgroup
  Assignment Mechanism below). Substitute-one/attribute adjacency.
- `privatize_categorical_counts(...)` — aggregate category counts, general
  purpose (e.g. representativeness checks) — no longer Study B's
  subgroup-AUC mechanism as of the 2026-08-19 rework. Add/remove
  adjacency — see the adjacency-model note at the top of
  `src/dp_mechanisms.py` before using this for anything attribute-privacy
  shaped.
- `privatize_age_mean(...)` / `privatize_age_histogram(...)` — age, general
  purpose. Add/remove adjacency, same caveat as above.
- `privatize_categorical_proportions(...)` — subgroup prevalence (Study C).
  Add/remove adjacency, same caveat as above.
- `EPSILON_SWEEP = [0.1, 0.5, 1, 2, 3, 4, 5, 6, 8, 10]` — the fixed sweep.
  Re-derived on 2026-08-19 after `privatize_categorical_label` replaced
  the discarded aggregate-count mechanism (see Subgroup Assignment
  Mechanism) — randomized response's noise profile is entirely different,
  and the real transition (diagnosed empirically, 40 trials/epsilon) sits
  at epsilon 3-4, comfortably inside the original 0.1-10 range this time;
  points 3/4/6/8 were added around that transition for resolution, and the
  low-epsilon extension from the discarded mechanism (0.001-0.05) was
  dropped — at that noise level randomized response is already deep in
  "washed out" territory (~90-115% pct_diff) with no useful
  differentiation. If this changes, update it here only; Study B
  references this constant rather than hardcoding its own list.

---

## The Frozen Handoff: Study A → Study B

Study B never touches Study A's model checkpoint, training code, or raw
data. It reads exactly **one** artifact:

**`results/study_a/predictions_90_10.csv`** — output of the 90/10
sex-imbalance run, chosen because it has the largest, most clearly
replicated subgroup gap. Study B does not sweep across imbalance ratios —
that's Study A's question, not Study B's.

Frozen schema (do not alter without updating this file *and* Study B's
loader):

| column | meaning |
|---|---|
| `patient_id` | patient identifier |
| `image_id` | image identifier (multiple images per patient) |
| `true_label` | ground-truth pneumothorax label (0/1) |
| `predicted_score` | Study A model's output probability |
| `true_sex` | ground-truth sex |
| `true_age` | ground-truth age (years) |

Also frozen: **`results/study_a/patient_split.csv`** (`patient_id`, `split`)
— the exact patient-level 70/15/15 split, seed 42. Study B reads this only
to confirm it's scoring the same test set; it never re-splits or resamples.

Study A also produces `predictions_70_30.csv` and `predictions_50_50.csv`
for its own internal ratio comparison. **These are Study A's alone — Study B
must not read them.**

Model checkpoints are never committed to git on any branch (see Conventions)
— the CSV above is the *only* channel between Study A and Study B, by
design. This makes it structurally impossible for Study B to retrain,
re-infer, or accidentally use a different model than the one that passed the
test oracle.

**What `results/study_a/` actually contains after the merge — read this
before writing Study B's loader.** The merge brings in Study A's full
history, not just the two permitted files. After merging, `results/study_a/`
on `main` contains:

```
results/study_a/
├── patient_split.csv                  # ALLOWED (confirmation only)
├── predictions_90_10.csv              # ALLOWED — the only real input
├── predictions_70_30.csv              # NOT ALLOWED — Study A's own ratio comparison
├── predictions_50_50.csv              # NOT ALLOWED — Study A's own ratio comparison
├── seed_replication/                  # NOT ALLOWED — training-seed robustness runs (see Seed Replication)
│   └── predictions_{arm}_seed{N}.csv
├── split_sensitivity/                 # NOT ALLOWED — alternate-split robustness runs (see Split Sensitivity)
│   ├── patient_split_seed{N}.csv
│   └── predictions_90_10_split{N}.csv
└── n_sensitivity/                     # NOT ALLOWED — N=5,000 sample-size sensitivity pass
    └── predictions_{arm}_N5000[_seed{N}].csv
```

26 auxiliary CSVs sit alongside the 2 allowed files, all matching the same
`predictions_*.csv` naming pattern and column schema — nothing about the
directory listing or file contents visually distinguishes an allowed file
from a forbidden one. **`src/run_study_b.py`'s loader must hardcode the two
exact filenames above (`predictions_90_10.csv`, `patient_split.csv`) — never
glob `results/study_a/*.csv`, never glob `predictions_*.csv`, and never walk
into `seed_replication/`, `split_sensitivity/`, or `n_sensitivity/`.** If
Study B's own robustness checks later need multiple predictions files (e.g.
to sanity-check epsilon-sweep stability), generate them within
`results/study_b/`, from the one allowed input — do not reach back into
Study A's auxiliary directories for a shortcut.

---

## Study A — Baseline

**Branch:** `study-a`
**Owns:** `src/data_loading.py`, `src/train.py`, `src/metrics.py`,
`notebooks/study_a/`, `results/study_a/`
**May read:** `data/` (gitignored raw/processed NIH ChestX-ray14)
**Must not:** import anything from Study B/C, or reference epsilon/DP
mechanisms at all — Study A is DP-free by design; it's the pre-DP ground
truth everything else is measured against.

- **Label:** Pneumothorax only — the label Larrazabal et al. report their
  headline sex-gap result on. Report macro-averaged AUC per sex group, not
  averaged across all 14 pathology labels.
- **Sweep:** sex-balance ratios 90/10, 70/30, 50/50.
- **Split:** fixed 70/15/15, at the *patient* level (not image-level —
  ChestX-ray14 has multiple images per patient; leakage across splits
  invalidates the gap). Seed 42. Write once to
  `results/study_a/patient_split.csv`; never regenerate it.
- **Test oracle:** reproduce Larrazabal et al. (2020)'s subgroup AUC gap,
  same direction (automated, required to pass) and magnitude within 2 AUC
  points (documented comparison against their Figure 1, not a strict gate
  — see Magnitude Oracle Resolution below for why, and what was actually
  found). Until direction passes and the magnitude comparison is
  documented, do not merge into `main`, and no Study B/C result should be
  treated as meaningful.
- **Deliverable to merge:** the three predictions CSVs + `patient_split.csv`,
  per the schema above. Checkpoints are never committed.
  ### Study A — Design Decisions (finalized)

- **Test/val composition:** fixed and sex-representative (unmanipulated) across
  all three ratio arms, drawn once from the frozen 70/15/15 patient split.
  Only the training set's sex composition varies per arm.
- **Imbalancing method:** undersampling only (no duplication, no ad hoc
  discarding). Total training N is fixed across all three arms, capped by
  the 50/50 arm's demand: `N_total = min(available_majority, 2 ×
  available_minority)`. Majority is undersampled within that fixed budget
  to hit each ratio. Minority sex identity is fixed and stated explicitly,
  not relabeled per arm. **Minority sex = female** — the 90/10, 70/30, and
  50/50 arms all skew male-majority, female-minority (matches Larrazabal et
  al.'s most-cited scenario, needed for the test oracle's direction check).
  Undersampling operates at the *patient* level (whole patients dropped,
  not individual images), consistent with the patient-level split.
- **Fine-tuning:** full end-to-end fine-tuning of the backbone (not
  frozen) — see Backbone Initialization below for which backbone. LR
  1e-5–1e-4, early stopping on validation AUC, shared max-epoch cap across
  all three arms. Hyperparameters and epoch budget identical across arms —
  only training composition changes.
- **Multi-image aggregation:** patient-level AUC computed by averaging
  `predicted_score` across a patient's images before ranking. Patient-level
  ground truth is aggregated by `max` (a patient counts as positive if any
  of their images does) — standard, low-risk, noted here since it wasn't
  previously written down explicitly.
- **Preprocessing:** resize to 224x224 (the resolution convention
  torchxrayvision's models use, though the package itself is not a
  dependency — see Backbone Initialization below), replicate grayscale to
  3 channels, normalize with ImageNet mean/std. Not torchxrayvision's own
  single-channel normalization range — that original note assumed an
  xrv-pretrained backbone; superseded by the Backbone Initialization
  decision below, which requires ImageNet-style input statistics for the
  pretrained conv1 weights to be valid.
- **Seeding:** seed 42 covers the patient split AND weight init AND
  data-loader shuffling for the training run — the only reproducibility
  anchor, since checkpoints are never committed. GPU training explicitly
  enables `torch.use_deterministic_algorithms(True)` +
  `cudnn.deterministic = True` (`cudnn.benchmark = False`), so this is an
  actual bit-reproducibility guarantee, not just a nominal one — plain
  seeding alone is not sufficient on GPU (cuDNN's default conv algorithms
  are non-deterministic regardless of seed).
- **Class weighting:** pneumothorax prevalence handling decided once
  (inverse-frequency `pos_weight` in `BCEWithLogitsLoss`), applied
  identically across all three arms.
- **Test oracle scope:** only the 90/10 arm is checked against Larrazabal's
  reported gap (same direction, magnitude documented — see Magnitude
  Oracle Resolution below), using its canonical seed=42 run. 70/30 and
  50/50 are internal comparison points, not independently oracle-gated. A
  patient-level bootstrap CI (~1,000 resamples) on the 90/10 gap, the
  cross-seed gap spread (see Seed Replication below), and the cross-split
  gap spread (see Split Sensitivity below), are logged in CHANGELOG.md as
  robustness notes — all non-gating; the pass/fail criterion stays the
  single seed=42 run's direction check.
- **`true_age` column:** carried for completeness/future reference; not a
  manipulated variable or part of Study A's oracle.

### Study A — Backbone Initialization (finalized)

- **Backbone init:** ImageNet-pretrained DenseNet-121 via
  `torchvision.models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)`.
  NOT a torchxrayvision chest-X-ray checkpoint — `-nih`/`-all` were
  pretrained on the same NIH corpus this study splits into train/test
  (leakage risk), and even non-NIH checkpoints (`-chex`, `-mimic_ch`, etc.)
  deviate from Larrazabal et al.'s own ImageNet-init methodology, which is
  what the test oracle checks against. The `torchxrayvision` package is
  **not a dependency of Study A at all** — not for weights, and not for
  data loading/preprocessing either, to keep the pipeline structurally
  free of any contact with the chest-X-ray-pretrained ecosystem.
  Resizing to 224x224 is done directly with `skimage.transform.resize`;
  `src/data_loading.py` does not import `torchxrayvision`.
- **Weights enum:** pin `DenseNet121_Weights.IMAGENET1K_V1` explicitly in
  `src/train.py` — do not use `weights="DEFAULT"` or `weights=True`, since
  torchvision's default IMAGENET1K weights enum has changed across
  versions and an unpinned default is not reproducible.

### Study A — Seed Replication (finalized)

- **Why:** the test oracle checks a single stochastic training run's
  subgroup AUC gap against a published number. A single run can't
  distinguish "the imbalance causes this gap" from "this particular
  weight-init/batch-order draw happened to produce this gap" — only the
  arm actually checked against an external claim needs this defended.
- **Scope:** the 90/10 arm is replicated across 5 seeds (42, 43, 44, 45,
  46) — it's the only oracle-gated arm (see Test oracle scope above), so
  it gets full robustness treatment (this section, plus bootstrap CI and
  Split Sensitivity). 70/30 and 50/50 additionally get a **lighter**
  3-seed replication (42, 43, 44) as of 2026-08-19 — not because they're
  oracle-gated, but because the cross-arm trend (gap shrinking as
  training-set balance improves, an internal Study A finding) is itself a
  claim worth defending against seed noise: the 90/10 5-seed spread
  showed a real range (0.0396-0.0670), so a single point estimate per arm
  risked overstating how clean the trend is. This addition is
  informational only — no bootstrap CI and no split-sensitivity run for
  70/30 or 50/50, matching the effort to what the claim needs rather than
  giving them full parity with 90/10.
- **What varies vs. what stays fixed across replicate seeds:** only
  weight init (the new classifier head) and data-loader shuffling order
  vary by seed. The patient split and the 90/10 arm's undersampled
  training patients are always drawn with the canonical seed 42,
  identical across all 5 runs — otherwise composition noise and training
  noise would be conflated, defeating the point of the replication.
- **Output contract:** the canonical seed=42 run for each arm still
  writes the frozen `results/study_a/predictions_{arm}.csv` — Study B's
  input contract (90/10 only) is unchanged. Replicate seeds write to
  `results/study_a/seed_replication/predictions_{arm}_seed{N}.csv`
  (43-46 for 90/10; 43-44 for 70/30 and 50/50), which is explicitly
  **not** part of the Study A → Study B handoff (see Frozen Handoff
  above) — Study B reads only `predictions_90_10.csv`.
- **Reporting:** the cross-seed gap spread (mean, range, direction
  agreement) is a non-gating robustness note in CHANGELOG.md for each
  replicated arm — 90/10 from all 5 runs, 70/30 and 50/50 from their 3
  runs each — the pass/fail oracle criterion itself stays the single
  90/10 seed=42 run, to keep it simple and avoid re-litigating what
  "passing" means. `src/metrics.py` computes this, not `src/train.py`,
  which only produces predictions CSVs.
- **Auxiliary outputs** (`results/study_a/checkpoints/`,
  `results/study_a/logs/`, `results/study_a/seed_replication/`) are not
  frozen deliverables — only the three `predictions_*.csv` and
  `patient_split.csv` are (see Deliverable to merge above).

### Study A — Split Sensitivity (finalized)

- **Why:** the test oracle checks the reproduced gap on a single
  patient-level 70/15/15 split (seed 42). A single split can't distinguish
  "the imbalance causes this gap" from "this particular set of patients
  happened to land in test" — this is the split-level counterpart to Seed
  Replication above, which instead checks training stochasticity on that
  same fixed split.
- **Scope:** only the 90/10 arm, canonical training seed=42, is affected.
  70/30 and 50/50 are out of scope, for the same reason they're out of
  scope for Seed Replication (see Test oracle scope above).
- **Splits used:** 3 additional patient-level 70/15/15 splits generated
  with seeds 101, 102, 103 — distinct from the 42-46 range already used
  for weight-init/data-loader-shuffle seeding, so the two kinds of seed are
  never confused. These are **new splits** (different patients land in
  train/val/test each time), not repeated training runs on the same
  split — this checks split sensitivity, not training stochasticity.
- **Procedure:** for each of the 3 alternate splits, rebuild the
  undersampled 90/10 training set from that split (same undersampling
  procedure as the canonical split), train once with the canonical
  training seed=42, and recompute the subgroup AUC gap (same
  direction/magnitude check as the main oracle).
- **Output contract:** the canonical split's canonical seed=42 run still
  writes the frozen `results/study_a/predictions_90_10.csv` and
  `results/study_a/patient_split.csv` — Study B's input contract is
  unchanged. The 3 alternate splits write to
  `results/study_a/split_sensitivity/patient_split_seed{N}.csv` and
  `results/study_a/split_sensitivity/predictions_90_10_split{N}.csv`,
  which are explicitly **not** part of the Study A → Study B handoff (see
  Frozen Handoff above) — Study B reads only `predictions_90_10.csv` and
  `patient_split.csv`.
- **Reporting:** the cross-split gap spread (mean, range, direction
  agreement across the 3 alternate splits vs. the canonical split) is a
  non-gating robustness note in CHANGELOG.md, alongside the existing
  cross-training-seed spread — the pass/fail oracle criterion itself stays
  the single canonical seed=42, canonical-split run. `src/metrics.py`
  computes this, following the same pattern as the cross-seed spread
  computation.
- **Auxiliary outputs** (`results/study_a/split_sensitivity/`) are not a
  frozen deliverable — only the three `predictions_*.csv` and
  `patient_split.csv` from the canonical split are (see Deliverable to
  merge above).

### Study A — Sample-Size Sensitivity (exploratory, 2026-08-19)

- **Why:** the canonical sweep fixes total training N at
  `min(available_majority, 2 × available_minority)` = 11,664 across all
  three ratio arms (see Design Decisions above), so moving toward balance
  simultaneously grows the minority training set *and* shrinks the
  majority training set — the two effects are coupled, not isolated.
  Comparing female-subgroup AUC across arms showed no gain from the
  extra minority data added between 70/30 (3,499 female patients, mean
  AUC 0.8650 across 3 seeds) and 50/50 (5,832 female patients, mean AUC
  0.8560) — consistent with either measurement noise (the Hanley-McNeil
  standard error for an AUC estimate on ~107 female test positives is
  ~0.023, close to the observed cross-seed spread) or a saturating
  minority-data learning curve, since the female-AUC jump *was* large
  going 90/10→70/30 (1,166→3,499 patients, mean AUC 0.8369→0.8650) but
  flat going 70/30→50/50. This exploratory pass checks whether the
  ratio's effect on the subgroup gap looks different at a smaller, fixed
  training-set size, where no arm has "abundant" minority data — a
  data-scarcity regime where representation might matter more.
- **Scope:** all three ratio arms (90/10, 70/30, 50/50), trained at a
  fixed `N_total=5,000` (`dl.N_SENSITIVITY_TOTAL`) — well within both
  the available majority pool (11,664) and minority pool (9,900) at
  every ratio; 90/10 is the tightest case, needing 4,500 majority / 500
  minority patients. Canonical undersampling seed=42, canonical split
  throughout. Initially single-seed only; extended same-day to a
  **lighter 3-seed replication** (canonical 42 + 43-44,
  `tr.N_SENSITIVITY_REPLICATION_SEEDS`) — same seed range and rationale
  as 70/30's/50/50's Seed Replication above — after the single-seed
  result showed a non-monotonic, opposite-direction pattern from the
  canonical-N sweep, too surprising to report on a single run per
  CLAUDE.md's own noise-floor findings elsewhere in this file. Only
  weight init/data-loader order vary across the 3 seeds; the
  undersampling draw stays canonical (seed=42) throughout, same rule as
  the main Seed Replication section.
- **Not in scope:** additional N_total levels (e.g. 1,000, 3,000,
  10,000) — considered and deferred as disproportionate to what Study A
  needs. A full N×ratio grid at reliable seed counts would cost several
  times the GPU-hours already spent on the canonical sweep, for a
  question (does the ratio effect interact with data scale) that is
  secondary to Study A's actual job of clearing the test oracle for
  Study B. N=1,000 specifically is also likely infeasible to interpret:
  at 90/10 it implies ~100 female training patients, ~5 expected
  positives at the dataset's ~5% pneumothorax prevalence — probably too
  few to fine-tune on at all, not just noisier.
- **Output contract:** canonical seed=42 writes
  `results/study_a/n_sensitivity/predictions_{arm}_N5000.csv`; replicate
  seeds 43-44 write `predictions_{arm}_N5000_seed{N}.csv`, same
  directory (`dl.N_SENSITIVITY_DIR`, via `python train.py
  --n-sensitivity` / `--n-sensitivity-replicate`) — explicitly **not**
  part of the Study A → Study B handoff (see Frozen Handoff above);
  Study B reads only `predictions_90_10.csv` and `patient_split.csv`.
- **Reporting:** male/female subgroup AUC and gap per arm at N=5,000
  (single-seed point estimate), compared against each arm's canonical
  (N=11,664) gap — `n_sensitivity_report()`, `--note n_sensitivity` —
  plus the cross-seed gap spread across the 3-seed replication —
  `n_sensitivity_seed_spread()`, `--note n_sensitivity_seed` — both
  logged as non-gating, exploratory notes in CHANGELOG.md. No CI, no
  direction check, no pass/fail criterion.
- **Limits:** even with the 3-seed replication, this remains a lighter
  check than the canonical sweep gets (no bootstrap CI, no split
  sensitivity, matching the same reduced-scope precedent already set for
  70/30/50/50 vs. 90/10) — a scoping look at whether ratio and total-N
  interact, not a robustness-checked finding on its own.

### Study A — Magnitude Oracle Resolution (finalized, 2026-08-19)

- **Why:** the Test oracle above requires "same direction, within 2 AUC
  points" against Larrazabal et al. (2020). Direction was automated and
  passed robustly (18+ independent runs across canonical, seed, split, and
  N-sensitivity checks, zero reversals). Magnitude had been repeatedly
  deferred in CHANGELOG.md as "manual comparison against Figure 1" without
  that comparison ever actually being performed or a verdict recorded —
  this entry closes that out.
- **Comparison performed:** Larrazabal et al.'s Fig. 1 panels B-2/C-2
  (Pneumothorax; a single model trained on a mixed-sex ratio, evaluated
  separately on male [B-2] and female [C-2] test folds) are the panels
  methodologically comparable to Study A's design — one model, mixed
  training ratio, gap = male test AUC − female test AUC. Panel A is a
  different quantity (single-sex-only training, cross-sex generalization
  drop) and was not used. Their x-axis (% female in training) only has
  points at 0/25/50/75/100 — no 90/10 point exists, confirming what prior
  CHANGELOG entries already noted.
- **Values read** (visual read off the box-plot mean markers in Fig. 1, PDF
  page rendered via PyMuPDF since `poppler-utils` was unavailable in this
  environment; treat as ±0.01, not pixel-calibrated):
  - 0% female training: male AUC ≈0.84, female AUC ≈0.705, gap ≈0.135.
  - 25% female training: male AUC ≈0.835, female AUC ≈0.735, gap ≈0.10.
  - Linearly interpolating to Study A's 10% female composition gives a
    Larrazabal-equivalent gap of ≈0.12.
- **Verdict:** direction PASSES (male AUC > female AUC in both sources).
  Magnitude does **not** pass the literal 2-AUC-point tolerance: Study A's
  canonical 90/10 gap (0.0670) is ≈0.05 below the interpolated Larrazabal
  value (≈0.12), and ≈0.03–0.08 below either raw neighboring point (0.135
  at 0%, 0.10 at 25%) — 1.5×–4× the 0.02 tolerance, whichever reference
  point is used.
- **Why this is not treated as a blocking failure** (two structural
  reasons, not an attempt to explain the discrepancy away):
  1. No exact 90/10 point exists on Larrazabal's grid (0/25/50/75/100
     only) — any comparison already requires interpolation or a
     neighboring-point proxy, which the original "within 2 AUC points"
     wording did not anticipate.
  2. The two studies estimate the gap with different estimators of
     related but not identical quantities: Larrazabal's box plots
     aggregate 20 folds; Study A's spread comes from 5 training seeds + 3
     alternate splits on one fixed architecture/hyperparameter
     configuration. A 2-AUC-point tolerance calibrated for an exact,
     same-estimator comparison is not automatically the right tolerance
     across two different variance-estimation protocols.
- **Decision:** the magnitude criterion in Test oracle above is revised
  from a strict pass/fail gate to a documented comparison. Study A's
  merge-to-`main` gate is now: direction PASS (automated,
  `check_oracle_direction`) + the magnitude discrepancy explicitly
  recorded here and in CHANGELOG.md, rather than requiring the
  discrepancy to fall under 2 AUC points. This is a one-time, logged
  revision to the gating criterion itself — it does not retroactively
  mark any prior CHANGELOG entry as having "passed" magnitude; those
  entries correctly reported the check as not yet done.
- **What this does and doesn't mean for Study B:** Study A's own gap is
  still correctly signed and statistically significant (bootstrap 95% CI
  [0.0230, 0.1118] excludes zero — see Canonical oracle result in
  `paper/study_drafts/study_a_draft.tex`). That is what Study B actually needs — a
  real, non-trivial, correctly-signed subgroup gap to test whether
  DP-protected demographics preserve or wash it out. It is not a claim
  that this pipeline exactly reproduces Larrazabal et al.'s reported
  magnitude, and the paper should not imply otherwise.

---

## Study B — Core Contribution: DP vs. True Demographics

**Branch:** `study-b` (branch from `main` only after Study A's deliverables
are merged)
**Owns:** `notebooks/study_b/`, `results/study_b/`, and the script that
applies `src/dp_mechanisms.py` to the frozen predictions file (e.g.
`src/run_study_b.py`)
**May read:** `results/study_a/predictions_90_10.csv`,
`results/study_a/patient_split.csv` (confirmation only), `src/dp_mechanisms.py`
**Must not:** call `src/train.py` or `src/data_loading.py`; touch `data/`;
retrain, re-infer, or regenerate any Study A prediction; read
`predictions_70_30.csv` / `predictions_50_50.csv`, or anything under
`seed_replication/`, `split_sensitivity/`, or `n_sensitivity/`; vary the
imbalance ratio (fixed to the single input file above). **Before writing
`src/run_study_b.py`'s loader, read "What `results/study_a/` actually
contains after the merge" under The Frozen Handoff above** — the directory
holds 26 auxiliary CSVs beyond the 2 allowed files, all matching the same
naming pattern; the loader must hardcode the two exact filenames, never
glob.

- **Question:** does a DP-protected demographic label preserve the true
  subgroup AUC gap, or wash it out?
- **Epsilon sweep:** fixed to `EPSILON_SWEEP` in `src/dp_mechanisms.py` —
  `{0.1, 0.5, 1, 2, 3, 4, 5, 6, 8, 10}`, re-derived on 2026-08-19 for
  `privatize_categorical_label`'s noise profile (see Subgroup Assignment
  Mechanism below). No ad hoc epsilon values without updating that
  constant.
- **Mechanism:** `privatize_categorical_label` (per-record randomized
  response, diffprivlib's `Binary` mechanism) applied independently to
  every test patient's sex label — not an aggregate Laplace release. See
  Subgroup Assignment Mechanism below for why this replaced the original
  "Laplace mechanism, applied once per release" design.
- **Test oracle:** at each epsilon, recompute the subgroup AUC gap using
  the per-record DP-protected sex labels directly (`true_label` /
  `predicted_score` are unchanged from the frozen CSV — only the sex label
  used to assign subgroups is noised, and it's noised for every patient,
  not reconstructed from an aggregate). "Survives" = same direction +
  magnitude within 15% of the true-demographic gap. "Washed out" = wrong
  direction or >15% off. Record the crossover epsilon.
- **Deliverable:** `results/study_b/epsilon_sweep_results.csv` (`epsilon`,
  `true_gap`, `dp_gap`, `direction_match`, `pct_diff`, `survived`) + the
  crossover epsilon noted in CHANGELOG.md. A single canonical draw isn't
  enough to trust that crossover on its own — see Seed Replication below,
  and cite the replication's survival rate, not the canonical draw's
  `survived` column, for any per-epsilon claim. As of 2026-08-19 the file
  additionally carries `debiased_dp_gap`, `debiased_direction_match`,
  `debiased_pct_diff`, `debiased_survived` — an additive, non-breaking
  schema extension for the debiased-estimator comparison, see Debiased
  Estimator below. These columns are scored from the *same* per-record
  privatized draw as the original four (paired comparison), not a second
  independent draw.

### Study B — Subgroup Assignment Mechanism (finalized, 2026-08-19, reworked same day)

- **First version (discarded) and why:** the original design released
  DP-noised aggregate M/F counts once per epsilon
  (`privatize_categorical_counts`), then reconstructed a per-patient
  group assignment by randomly moving just enough patients between groups
  — starting from the *true* assignment — to match the DP-implied target
  size. A technical review before pushing found two disqualifying
  problems, not minor caveats:
  1. **Construct validity.** The artifact the AUC gap was computed on
     wasn't privacy-protected data — it was the true per-patient labels
     with minimal random perturbation. Diagnostic: at epsilon=10, 0 of
     4,620 test patients were ever reassigned; even at epsilon=0.1, ~7
     were. The experiment couldn't have shown a different qualitative
     result at any epsilon where noise is small relative to cohort size
     (i.e. almost the whole sweep) regardless of whether genuinely
     privacy-protected per-record data would preserve the audit — it was
     measuring "DP aggregate counts concentrate at large N," not "DP
     labels preserve bias-audit conclusions."
  2. **Privacy accounting.** `privatize_categorical_counts`'s "each
     category gets the full epsilon via parallel composition" claim holds
     under add/remove adjacency, but the actual threat model here is
     attribute privacy on an already-public cohort (`patient_split.csv`
     is committed to git — membership isn't secret, sex is) — substitute-
     one adjacency, under which one patient's sex flip changes two bin
     counts at once, breaking parallel composition's precondition.
     Releasing both counts independently at "full epsilon each" under
     that model actually costs ~2x epsilon via sequential composition.
     See the adjacency-model note at the top of `src/dp_mechanisms.py`.
  All of that version's commits were discarded (reset on the `study-b`
  branch, which had never been pushed) rather than kept with a
  superseding commit — nothing about the discarded mechanism is worth
  preserving as a paper trail; this entry is the record of what happened
  and why.
- **Corrected mechanism:** `privatize_categorical_label` (new in
  `src/dp_mechanisms.py`) applies diffprivlib's `Binary` mechanism
  (classic randomized response) independently to *every* test patient's
  true sex label — kept with probability e^epsilon / (1 + e^epsilon),
  flipped otherwise. This fixes both problems at once: every patient's
  released label is a genuine randomized draw (not "true unless swept up
  in a rare reassignment"), and because it's a single per-record query
  with no aggregate count involved, there's no cross-bin composition
  question and no dependence on which adjacency model applies to the rest
  of the cohort.
- **Epsilon sweep re-derived, not reused:** randomized response's noise
  profile is entirely different from the discarded aggregate-count
  mechanism's — at epsilon=1, ~27% of labels flip (vs. ~1 patient out of
  4,620 reassigned under the old mechanism at the same epsilon). A
  diagnostic (40 trials/epsilon) found a clean transition: mean `pct_diff`
  is ~0.3% at epsilon=8, ~3% at 5, ~8% at 4, crossing 15% around
  epsilon=3-4 (23/40 trials survived at exactly 3), then climbing past 30%
  by epsilon=2 and 50%+ by epsilon=1 — comfortably inside the *original*
  0.1-10 range this time, unlike the discarded mechanism. `EPSILON_SWEEP`
  updated to `{0.1, 0.5, 1, 2, 3, 4, 5, 6, 8, 10}` — kept the original six
  for continuity, added 3/4/6/8 to resolve the transition, dropped the
  discarded mechanism's low-epsilon extension (0.001-0.05) since
  randomized response is already deep in "washed out" territory there
  with no useful differentiation between points.

### Study B — Seed Replication (finalized, 2026-08-19)

- **Why:** a technical review of the single-canonical-draw sweep flagged
  that, unlike the discarded mechanism, `privatize_categorical_label`'s
  variance is real, not an artifact — AUC is rank-sensitive to *which*
  specific patients get relabeled, not just how many, so the
  concentration you'd expect from 4,620 independent coin flips doesn't
  straightforwardly stabilize the reported gap. A single draw per epsilon
  can't distinguish "this epsilon reliably preserves the gap" from "this
  epsilon happened to preserve it this time" — the same reasoning as
  Study A's Seed Replication, applied to Study B's own stochastic
  mechanism instead of training stochasticity.
- **Scope:** all ten `EPSILON_SWEEP` points, 30 independent replicate
  draws each (`N_REPLICATION_SEEDS`), using a separate seed pool
  (`REPLICATION_BASE_SEED=2000`) from the canonical single-draw sweep
  (`BASE_SEED=42`) so neither can reproduce the other's draws. Cheap
  enough (no GPU, pure per-record simulation, ~80s total) to just run by
  default in `main()` rather than gating behind a flag the way Study A's
  GPU-hour-costly replication had to.
- **What the replication found that the single canonical draw hid:**
  running it changed the honest headline, not just added error bars.
  - **Direction is unreliable at low epsilon too, not just magnitude** —
    the canonical draw showed `direction_match=True` at every epsilon
    including 0.1, which the original write-up reported at face value.
    Across 30 replicates, direction agreement is only 70% at epsilon=0.1,
    73% at 0.5, 66% at 1.0 (rising to 100% by epsilon=3). The single draw
    happening to get direction right at 0.1 was luck, not a property of
    that epsilon.
  - **The apparent stability at epsilon=4 and 5 in the single-draw table
    was also luck.** Survival *rate* across 30 replicates is 90% at both
    — not the 100%-looking "True, pct_diff=1.3%/0.3%" the canonical draw
    reported. Epsilon=3 is a near-exact coin flip (50.0% survival rate).
    Only epsilon>=6 hits 100% survival across all 30 replicates.
  - The canonical-draw crossover (3.0) and the replication-based crossover
    (smallest epsilon at/above which survival_rate>=50% holds for every
    larger epsilon too) landed on the same value this time — but that
    agreement doesn't rescue the single-draw table's per-epsilon claims at
    4 and 5, which the replication shows were overstated.
- **Output contract:** `run_seed_replication()` writes
  `results/study_b/seed_replication/dp_gap_replication.csv` (`epsilon`,
  `seed`, `true_gap`, `dp_gap`, `direction_match`, `pct_diff`, `survived`,
  plus the `debiased_*` columns described under Deliverable above — one
  row per epsilon×replicate). Not part of the frozen deliverable — the
  canonical `epsilon_sweep_results.csv` schema (plus its additive
  `debiased_*` columns) is unchanged by this file's existence. The
  summary (mean/std `dp_gap` and `pct_diff`, direction-agreement rate,
  survival rate per epsilon, for both estimators) is computed by
  `seed_replication_summary()` and reported in CHANGELOG.md, not saved as
  its own file — same pattern as Study A's cross-seed gap spread.
- **Reporting:** if the paper cites a crossover epsilon or a "the gap
  survives at epsilon=X" claim, it should cite the replication's survival
  rate at that epsilon, not the canonical single draw's `survived`
  column — the canonical sweep stays useful as a reproducible point
  reference, not as the epsilon-by-epsilon narrative.

### Study B — Survival-Rate Confidence Intervals (finalized, 2026-08-19)

- **Why:** a review of the draft write-up (`paper/study_drafts/study_b_draft.tex`)
  noted that the replication's per-epsilon survival *rates* — the numbers
  actually driving the paper's headline claims at the borderline epsilons
  (3, 4, 5) — were reported as bare point estimates with no uncertainty of
  their own, even though `N_REPLICATION_SEEDS=30` gives a binomial SE
  large enough to matter (e.g. ~5.5% at p=0.9) at exactly the epsilons the
  discussion leans on most.
- **What was added:** `seed_replication_summary()` now computes a Wilson
  score 95% CI on `survival_rate` and on `debiased_survival_rate` (see
  Debiased Estimator below) per epsilon, using `scipy.stats.norm` for the
  z-value — `_wilson_ci()` in `src/run_study_b.py`. Not saved as its own
  file; reported alongside the summary in CHANGELOG.md and cited directly
  in the paper draft's replication table, same non-gating status as the
  rest of the replication summary.
- **What it shows:** the CIs are wide enough at the borderline epsilons to
  matter for how confidently the paper can state a specific rate — e.g.
  epsilon=3's 50.0% point estimate carries a 95% CI of roughly
  [33%, 67%], and epsilon=4/5's 90.0% carries roughly [74%, 97%]. This
  doesn't change the qualitative story (reliable only at epsilon>=6) but
  means per-epsilon numbers in that borderline band should be read as
  "consistent with," not "equal to," the point estimate.

### Study B — Debiased Estimator (exploratory, 2026-08-19)

- **Why:** an advisor-style review of the draft asked whether other DP
  mechanisms might give better privacy/utility than plain per-record
  randomized response. Rather than reopening the mechanism search (the
  discarded aggregate-count design already showed the risk of that path —
  see Subgroup Assignment Mechanism above), this explores a
  *post-processing* improvement on the existing mechanism's own output:
  since `privatize_categorical_label`'s flip probability is exactly known
  (public), the released per-record labels can in principle be
  reweighted to reduce the AUC gap's attenuation — at zero extra privacy
  cost, since post-processing on already-released DP output is free under
  DP's post-processing immunity (never touches `true_sex`).
- **Method tried:** for each privatized draw, invert the classic
  Warner/randomized-response formula on the draw's own *aggregate*
  observed rate to estimate the true population majority-share (a
  deterministic function of the already-released labels, not of
  `true_sex` — still free post-processing), then compute each patient's
  Bayesian posterior P(true=majority | observed label, known flip
  probability, that population estimate), and recompute the subgroup AUC
  gap by calling `sklearn.roc_auc_score(..., sample_weight=posterior_weight)`
  instead of hard-assigning subgroups from the raw privatized label.
  Implemented as `_debiased_subgroup_auc_gap()` in `src/run_study_b.py`,
  scored from the *same* privatized draw as the naive estimator (paired
  comparison — both `_evaluate_draw()` outputs come from one
  `privatize_subgroups()` call), so any difference reflects the
  estimator, not draw-to-draw noise.
- **Result: this did not help — it was slightly worse across almost the
  entire sweep, not better.** The debiased canonical-draw and
  replication-based crossover epsilon both moved from 3.0 to 4.0 (worse,
  not better), and at every epsilon below 5 the debiased survival rate
  was equal to or lower than the naive one (e.g. epsilon=2: naive 36.7%
  vs. debiased 3.3%; epsilon=0.1-1: naive 3.3-10.0% vs. debiased 0.0% at
  all three). The two estimators only converge once epsilon>=5, where the
  naive estimator was already reliable anyway.
- **Why it didn't help (two compounding reasons, not a bug):**
  1. The population-share correction inverts observed_rate by dividing by
     `(2*keep_prob - 1)`, which shrinks toward 0 as epsilon shrinks toward
     0 (keep_prob toward 0.5) — exactly where debiasing would matter most,
     the correction itself becomes a high-variance, easily-clipped
     estimate (confirmed: it collapses to 0 or 1 often enough at low
     epsilon to make the posterior weight nearly degenerate).
  2. More fundamentally, AUC is a pairwise/rank statistic, not a linear
     functional of per-record labels — unlike a mean or a proportion,
     reweighting the *instances* fed into `roc_auc_score` does not
     correctly reweight the *pairs* the statistic is actually computed
     over. Because the posterior weight here takes only two values across
     the whole cohort (a function solely of each patient's own observed
     label, not of anything else about them), the weighted statistic
     decomposes into a mixture of the observed-majority-only AUC, the
     observed-minority-only AUC, and cross-group pair terms — not a clean
     debiased estimate of the true-majority-only AUC. A correct
     pairwise-level correction would need to reweight comparisons, not
     instances; that was judged out of scope here (see Limitations in the
     paper draft) rather than attempted.
- **Decision:** kept in the codebase and reported as a genuine negative
  result — it directly answers the "should we try other mechanisms"
  question with a concrete, honest no for this variant, and the paired
  same-draw design means the comparison itself is trustworthy even though
  the estimator isn't an improvement. `EPSILON_SWEEP` and the primary
  mechanism (`privatize_categorical_label`, hard-assignment scoring) are
  unchanged; this is purely an additional analysis on already-collected
  draws, not a replacement.

---

## Study C — Objection Response: Small-Subgroup Detection Floor

**Branch:** `study-c` (independent — may branch off `main` at any time)
**Owns:** `notebooks/study_c/`, `results/study_c/`, simulation script (e.g.
`src/run_study_c.py`)
**May read:** `src/dp_mechanisms.py` only
**Must not:** touch `data/`, `results/study_a/`, `results/study_b/`,
checkpoints, `src/train.py`, `src/data_loading.py`. Study C is fully
synthetic — zero dependency on real ChestX-ray14 data or any trained model.

- **Question:** how small can a subgroup be before DP noise makes its
  underrepresentation undetectable?
- **Threat model:** add/remove adjacency, matching
  `privatize_categorical_proportions`'s own assumption (see the
  adjacency-model note at the top of `src/dp_mechanisms.py`) — Study C asks
  "how many members of this subgroup exist in the cohort," a membership
  question, not "what is a known cohort member's sensitive attribute" (that
  second question is Study B's, and needed a different
  mechanism/adjacency model entirely — see Study B's Subgroup Assignment
  Mechanism above for what went wrong there). Stated explicitly here so
  Study C doesn't inherit `privatize_categorical_proportions` by default
  without checking it's the right tool for its own question, the way
  Study B initially didn't.
- **Method:** simulate subgroup prevalences 0.5%–5% against a larger
  synthetic reference group, using `privatize_categorical_proportions`,
  with the reference cohort's total size fixed to a realistic audit scale
  (e.g. Study B's ~4,620-patient test cohort) rather than an arbitrary
  synthetic N — the mechanism's CI half-width scales with `1/total`, so
  the detection floor's meaning depends on this choice being tied to
  something the paper can defend.
- **Detection criterion (revised):** a single closed-form CI on the
  *difference* `proportion_subgroup − proportion_reference`, combining
  both categories' known Laplace noise scales (both counts come from the
  same `privatize_categorical_proportions` call, so both scales are
  known). "Detected" = that difference-CI excludes zero. This replaces an
  earlier draft criterion ("CI excludes zero" / "CI excludes the reference
  group's proportion," used interchangeably) — those are two different
  tests with opposite biases (excludes-zero is nearly always true and
  makes DP look better at detection than it is; excludes-reference was
  ambiguous about whether it meant the reference's point estimate or its
  own CI). Separately, checking whether two independent per-group CIs
  merely *overlap* each other — a tempting substitute — is a known-invalid
  proxy for a proper difference test (roughly equivalent to testing at
  alpha≈0.006 instead of the nominal 0.05), which would have overstated
  how much separation DP noise requires to "hide" a subgroup.
- **Replication:** multiple independent draws per `(epsilon,
  true_prevalence)` cell, not a single draw — same rationale as Study B's
  Seed Replication above: a boundary-region boolean outcome from one
  stochastic draw can't distinguish "this epsilon/prevalence reliably
  detects the gap" from "this particular draw happened to." Report a
  detection *rate* per cell with a Wilson 95% CI (same construction as
  `_wilson_ci()` in `src/run_study_b.py`), not a single boolean.
- **False-positive control:** also run cells where `true_prevalence`
  equals the reference proportion (no real gap), and report the rate at
  which the difference-CI test still (incorrectly) excludes zero. Without
  this, a detection rate reported at a real gap has no Type-I-error
  baseline to be interpreted against.
- **CI coverage check:** before trusting results, verify
  `privatize_categorical_proportions`'s CI coverage specifically at the
  low counts implied by this study's low-prevalence end (extend
  `check_dp_mechanisms.py`'s existing coverage check into that regime) —
  the existing check validated coverage at counts relevant to Study B's
  use case, not necessarily the single-digit counts a 0.5% prevalence can
  imply here.
- **Deliverable:** `results/study_c/detection_floor.csv`
  (`true_prevalence`, `epsilon`, `seed`, `is_null_condition`, `true_diff`,
  `noisy_diff`, `ci_lower`, `ci_upper`, `detected`) — one row per replicate
  draw, covering both the real-gap cells and the null-condition cells.
  The per-cell detection rate (with Wilson CI) and, for the
  null-condition rows, the false-positive rate, are computed as a summary
  and logged in CHANGELOG.md rather than saved as a separate file — same
  canonical-file-plus-summary pattern as Study B's replication (see Study
  B — Seed Replication above).

---

## Conventions

- Commit after every meaningful unit of work, on the correct branch.
- Log every run in `CHANGELOG.md`, prefixed with a study tag: `[Study A]`,
  `[Study B]`, `[Study C]`, or `[Shared]` for infra changes on `main`.
- Never commit files under `data/` or model checkpoint files, on any branch.
- If it's unclear which study an instruction belongs to, or which branch
  should be active, stop and ask rather than guessing.
- Never add Co-Authored-By or Generated-with trailers to commits

---

## File Layout
```
├── CLAUDE.md
├── CHANGELOG.md
├── README.md
├── requirements.txt
├── .gitignore
├── venv_path.txt          # gitignored
│
├── data/                  # gitignored — Study A only
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── study_a/
│   ├── study_b/
│   └── study_c/
│
├── src/
│   ├── data_loading.py    # Study A only
│   ├── train.py           # Study A only
│   ├── metrics.py         # Study A only
│   ├── dp_mechanisms.py   # shared — Study B and C, main branch
│   ├── run_study_b.py     # Study B only
│   └── run_study_c.py     # Study C only
│
├── results/
│   ├── study_a/           # predictions_90_10/70_30/50_50.csv, patient_split.csv — frozen once merged
│   │   ├── seed_replication/     # NOT part of the Study A → B handoff — see Frozen Handoff
│   │   ├── split_sensitivity/    # NOT part of the Study A → B handoff — see Frozen Handoff
│   │   └── n_sensitivity/        # NOT part of the Study A → B handoff — see Frozen Handoff
│   ├── study_b/           # epsilon_sweep_results.csv
│   └── study_c/           # detection_floor.csv
│
└── paper/
    ├── Final_Policy_Recommendation.tex   # untouched until all 3 studies are done — then rewritten to
    │                                     # combine the original policy paper with the studies' findings
    └── study_drafts/                     # internal, numbers-focused per-study records (not paper prose)
        ├── study_a_draft.tex
        ├── study_b_draft.tex
        └── study_c_draft.tex             # created once Study C has results to record
```