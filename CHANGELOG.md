# Changelog / Lab Notes

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