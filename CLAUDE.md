# Project: Differential Privacy for Medical Imaging Metadata

## Goal
Empirically support a policy paper arguing that differential privacy applied
to demographic/tabular metadata (age, sex) — not image pixels — preserves
enough utility for (a) representativeness checks and (b) subgroup bias
audits, using NIH ChestX-ray14.

## Three studies (see /paper for the full writeup)
1. **Study A (baseline):** Fine-tune a classifier under varying sex-balance
   ratios (90/10, 70/30, 50/50); measure resulting subgroup AUC gap.
   Target: reproduce a gap in the same direction/order of magnitude as
   Larrazabal et al. 2020.
2. **Study B (core):** Apply a DP mechanism to demographic fields at a range
   of epsilon values; recompute subgroup AUC and representativeness stats
   using DP-protected vs. true demographics. Find the epsilon range where
   conclusions survive vs. get washed out.
3. **Study C (objection-response):** Simulate small subgroup prevalences
   (0.5%–5%); find the detection floor under DP noise.

## Success criteria / test oracle
Before trusting any new Study B/C result, reproduce Larrazabal et al.'s
published subgroup AUC gap on our own held-out split within ~2 AUC points.
If that check fails, stop and debug before proceeding.

## Conventions
- Commit after every meaningful unit of work.
- Log every run (config, result, dead ends) in CHANGELOG.md before moving on.
- Never commit files under data/ or checkpoint files.

## File layout
- data/        raw and processed data (gitignored)
- notebooks/   exploratory analysis
- src/         reusable code (data loading, DP mechanisms, training loop)
- results/     saved metrics, tables, figures
- paper/       LaTeX source