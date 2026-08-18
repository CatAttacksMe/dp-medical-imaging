# Changelog / Lab Notes

## [Study A] 2026-08-18
- Wrote `src/data_loading.py`: metadata loading, frozen 70/15/15
  patient-level split (seed 42), patient-level undersampling for the
  90/10, 70/30, 50/50 sex-imbalance sweep (fixed N_total=11664 across all
  three arms, female fixed as minority sex), fixed representative val/test
  sets, and the ImageNet-normalized image dataset for the pretrained
  backbone.
- Generated `results/study_a/patient_split.csv` (21564/4621/4620
  train/val/test patients) — frozen from this point forward.

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