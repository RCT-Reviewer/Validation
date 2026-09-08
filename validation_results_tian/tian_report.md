# Tier E — RCT-Reviewer vs human RoB reference standard (Tian 2024)

Generated 2026-09-04T03:21:23.

Reference: Tian et al., Res Synth Methods 2024;15(6):1111-1119 — 1,955 RCTs with human-consensus RoB + original RobotReviewer labels (OSF: https://osf.io/k6w9q/).

- Trials analysed with RCT-Reviewer: **313** (of 1,955; limited by open-access PDF availability)
- Domain-level comparisons: **1252** (4 domains per trial)

**Why the evaluated subset is smaller than Tian's 1,955.** Tian et al. assessed publisher PDFs obtained through their own review pipeline. This harness can only legitimately retrieve **open-access** full texts: of 1955 trials, 1820 resolved to PubMed records, 427 have a PMC record, and 340 yielded an open-access PDF (PMC + Unpaywall). The remaining trials are paywalled (JAMA, NEJM, Ann Oncol, etc.) and are excluded rather than scraped. Bounded human-referenced evaluations are an established design: Hirt 2021 used n=190, Armijo-Olivo 2020 used n=393. Note: some trials visible in PMC could not be downloaded because they are author manuscripts whose publisher licence (Lancet, NEJM, JAMA, etc.) forbids programmatic PDF delivery — Europe PMC returns HTTP 500 for them; this is a licensing restriction, not a tool failure.
Per-domain agreement of RCT-Reviewer with the human consensus, against the same statistics for the original RobotReviewer on the identical subset (computed from the deposited data) and Tian's published full-sample values:

| Domain | n | RCT-Rev vs human: concordance (95% CI) | κ (95% CI) | PPA | NPA | Original RR vs human (same subset): concordance | κ | Tian published κ (full n=1955) |
|---|---|---|---|---|---|---|---|---|
| Random sequence generation | 313 | 62.6 (57.1–67.8) | 0.26 (0.15–0.36) | 0.68 | 0.58 | 70.0 | 0.40 | 0.46 |
| Allocation concealment | 313 | 60.4 (54.9–65.6) | 0.20 (0.10–0.30) | 0.58 | 0.66 | 61.7 | 0.25 | 0.25 |
| Blinding of participants and personnel | 313 | 76.0 (71.0–80.4) | 0.48 (0.38–0.57) | 0.72 | 0.90 | 83.4 | 0.58 | 0.59 |
| Blinding of outcome assessment | 313 | 63.3 (57.8–68.4) | 0.12 (0.01–0.23) | 0.48 | 0.67 | 73.5 | 0.31 | 0.27 |

External fidelity check: on the identical subset, RCT-Reviewer's judgements agree with the *original RobotReviewer's deposited labels* (generated from publisher PDFs) in 78.9% of domain comparisons on average across the four domains. Because Tier C proved the two implementations' SVM RoB pipelines judgement-identical on identical inputs (the original runs through the shim in the same SVM-only configuration as RCT-Reviewer), this residual reflects PDF-source differences (open-access PMC versions + PyMuPDF extraction vs publisher PDFs + the original stack), not an implementation difference.


Figure: figure_tier_e_human_concordance.(png|svg|pdf)

## Control — original implementation on the same PMC text

The original 2017 BiasRobot (via the compatibility shim, run in the same SVM-only configuration as RCT-Reviewer) was re-judged on the identical open-access PDF text that RCT-Reviewer judged. This is a like-for-like comparison: identical inputs, identical SVM pipeline, so agreement between the two should be near-100% and the original should show the SAME reduced human agreement as the refactored tool — isolating PDF source as the only difference vs Tian's published run. The original's TensorFlow 1.x CNN is bypassed by the shim and never executes here (its ensemble-level contribution is quantified separately in Tier B from stored outputs).

| Domain | n | original(shim, PMC text) vs RCT-Reviewer agreement | original(shim, PMC text) vs human κ | RCT-Reviewer vs human κ (from Tier E) |
|---|---|---|---|---|
| Random sequence generation | 313 | 100.0% | 0.26 | 0.26 |
| Allocation concealment | 313 | 100.0% | 0.21 | 0.21 |
| Blinding of participants and personnel | 313 | 100.0% | 0.48 | 0.48 |
| Blinding of outcome assessment | 313 | 100.0% | 0.12 | 0.12 |
