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
| `study-a` | Baseline training, sex-balance sweep | Passes its test oracle (reproduces Larrazabal gap within 2 AUC points) |
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

- `privatize_categorical_counts(...)` — sex counts (Study B)
- `privatize_age_mean(...)` / `privatize_age_histogram(...)` — age (Study B)
- `privatize_categorical_proportions(...)` — subgroup prevalence (Study C)
- `EPSILON_SWEEP = [0.1, 0.5, 1, 2, 5, 10]` — the fixed sweep. If this
  changes, update it here only; Study B references this constant rather
  than hardcoding its own list.

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
  same direction, within 2 AUC points. Until this passes, do not merge into
  `main`, and no Study B/C result should be treated as meaningful.
- **Deliverable to merge:** the three predictions CSVs + `patient_split.csv`,
  per the schema above. Checkpoints are never committed.

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
`predictions_70_30.csv` / `predictions_50_50.csv`; vary the imbalance ratio
(fixed to the single input file above).

- **Question:** does a DP-protected demographic label preserve the true
  subgroup AUC gap, or wash it out?
- **Epsilon sweep:** fixed to `EPSILON_SWEEP` in `src/dp_mechanisms.py` —
  `{0.1, 0.5, 1, 2, 5, 10}`. No ad hoc epsilon values without updating that
  constant.
- **Mechanism:** Laplace mechanism (diffprivlib), applied once per release.
  No repeated querying of the same privatized release — a future second
  query against the same data requires explicit composition accounting, not
  an ad hoc second call.
- **Test oracle:** at each epsilon, recompute the subgroup AUC gap using
  DP-protected sex labels to assign subgroups (`true_label` /
  `predicted_score` are unchanged from the frozen CSV — only the subgroup
  assignment is noised). "Survives" = same direction + magnitude within 15%
  of the true-demographic gap. "Washed out" = wrong direction or >15% off.
  Record the crossover epsilon.
- **Deliverable:** `results/study_b/epsilon_sweep_results.csv` (`epsilon`,
  `true_gap`, `dp_gap`, `direction_match`, `pct_diff`, `survived`) + the
  crossover epsilon noted in CHANGELOG.md.

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
- **Method:** simulate subgroup prevalences 0.5%–5% against a larger
  synthetic reference group, using `privatize_categorical_proportions`.
  "Detected" = the noisy proportion's confidence interval no longer overlaps
  zero / the reference group's proportion.
- **Deliverable:** `results/study_c/detection_floor.csv`
  (`true_prevalence`, `epsilon`, `detected`, `ci_lower`, `ci_upper`).

---

## Conventions

- Commit after every meaningful unit of work, on the correct branch.
- Log every run in `CHANGELOG.md`, prefixed with a study tag: `[Study A]`,
  `[Study B]`, `[Study C]`, or `[Shared]` for infra changes on `main`.
- Never commit files under `data/` or model checkpoint files, on any branch.
- If it's unclear which study an instruction belongs to, or which branch
  should be active, stop and ask rather than guessing.

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
│   ├── study_a/           # predictions_*.csv, patient_split.csv — frozen once merged
│   ├── study_b/           # epsilon_sweep_results.csv
│   └── study_c/           # detection_floor.csv
│
└── paper/
    └── Final_Policy_Recommendation.tex
```