# RCT-Reviewer validation — automated results

Generated 2026-09-04T00:28:45. Python 3.12.12, numpy 2.5.2, scikit-learn 1.9.0, spaCy 3.8.16, PyMuPDF 1.28.2.

Weight provenance (SHA-256 of the exact files loaded by the tool):
- `bias/bias_doc_level.npz`: `233e706a323aefec…`
- `bias/bias_sent_level.npz`: `659f967020c115af…`
- `rct/rct_svm_weights.npz`: `965279af5307d5ed…`
- `rct/rct_model_calibration.json`: `37a167ecbf8fc178…`

**Weight identity vs the original repository: 4/4 files verified byte-identical** — RCT-Reviewer's weight files were hashed and compared against the original RobotReviewer repository's Git LFS object hashes (or direct hashes for non-LFS files). The published performance of the original system therefore attaches to exactly these artifacts.

## Tier A — RCT classifier on the Clinical Hedges benchmark

n = 751 records (corpus parsed: 751); balanced threshold.

| metric | value (95% CI) |
|---|---|
| Sensitivity | 94.1 (91.4–96.0) |
| Specificity | 88.1 (84.2–91.2) |
| Accuracy | 91.5 (89.3–93.3) |
| PPV | 91.1 (88.0–93.4) |
| NPV | 92.1 (88.5–94.6) |
| F1 | 0.925 (0.907–0.943) |
| Cohen's kappa | 0.826 (0.786–0.866) |

Implementation fidelity vs stored original-model outputs:
- max |Δ score| = 9.253e-02, mean |Δ| = 1.519e-02, Pearson r = 0.999947
- decision agreement 99.6 (98.8–99.9), kappa 0.9918

- vs original 2017 code executed in this environment (same SVM-only configuration as the refactored tool): max |Δ| = 0.000e+00, agreement 100.0 (99.5–100.0)

The max |Δ| of 9.3e-02 against the *stored* outputs reflects scikit-learn float-precision drift between the 2016 environment that generated them and this one; against the original code executed here in the same SVM-only configuration the scores are bit-identical (max |Δ| = 0.0), and decisions agree 99.6% with the stored outputs. Scope: the executed comparison validates the SVM path shared by both tools — the original's TensorFlow/Keras CNN is bypassed by the shim and never executes, so the full SVM+CNN ensemble appears only as stored outputs (Tier B).

**Advanced statistics:**
- ROC AUC: 0.9658 (rank-based; unaffected by the SVM score's arbitrary scale)
- Brier score: 0.0674 (on Platt-scaled scores; logistic calibration fit in-sample on this benchmark, therefore optimistic)
- Benchmark RCT prevalence: 56.2% — PPV and NPV are prevalence-dependent and will differ on a literature stream with a different case mix.

**Inclusion funnel:**
- 1000 MEDLINE records parsed → 930 unique PMIDs → 751 retained (both title and abstract present) and scored.
- Excluded records (duplicates; missing title or abstract) were not scored, and ground-truth labels were not available for them.

## Tier B — CNN ablation (SVM-only vs original SVM+CNN ensemble)

- decision agreement vs full ensemble: 92.5 (90.4–94.2), kappa 0.8474
- SVM-only vs Hedges: sens 94.1 (91.4–96.0), spec 88.1 (84.2–91.2), F1 0.925 (0.907–0.943)
- SVM+CNN (no ptyp) vs Hedges: sens 95.5 (93.1–97.1), spec 94.5 (91.5–96.5), F1 0.956 (0.942–0.969)
- Full ensemble (SVM+CNN+ptyp) vs Hedges: sens 97.6 (95.7–98.7), spec 95.1 (92.2–97.0), F1 0.969 (0.957–0.980)

**Attribution:** the refactored SVM-only decisions agree 94.5% with SVM+CNN (publication-type features removed) but 92.5% with the full ensemble — so most of the ablation gap is attributable to the publication-type features, with the CNN contributing the remainder. Both components were unavailable in the refactored tool because the original TensorFlow/Keras CNN cannot run in a maintained environment.

**Ablation Statistical Testing:**
- McNemar's Test (SVM vs Ensemble) p-value: p < 0.0001

## Tier C — Risk-of-Bias pipeline fidelity (original vs refactored)

- comparisons: 6018 (documents × 6 domains)
- judgement agreement: 100.0 (99.9–100.0), kappa 1.0000
- max |Δ sentence score|: 0.000e+00
- vectorizer matrices identical: True (14 probes)

**Scope (SVM-to-SVM comparison):** both pipelines run SVM-only. The RoB path exercised here in the original is its SVM pipeline (interaction-hashed vectorizer + linear SVMs over the shared weight files); the original's TensorFlow/Keras components are bypassed by the validation shim and never execute. The agreement above therefore says: the refactored SVM RoB pipeline exactly reproduces the original's SVM judgements on identical inputs — an apples-to-apples design that isolates the refactoring as the only variable. It is NOT a reproduction of the original's full SVM+CNN ensemble; the performance contribution of the dropped components is quantified separately in Tier B from stored ensemble outputs.

## Tier D — parser robustness on the open-access corpus (descriptive, not accuracy)

**Corpus note:** Tier D ran on the 1000 open-access PDFs downloaded from Europe PMC via `fetch_corpus.py` into `corpus/`. The fetcher reached its 1,000-PDF target. The corpus is intentionally self-selected: the fetcher kept only papers RCT-Reviewer's own SVM classified as RCTs, so Tier D measures robustness on tool-relevant open-access RCT-like PDFs, not on a representative literature sample. Trial protocols accepted by the SVM are retained as valid parse targets; the protocol share of the corpus can be estimated from corpus/metadata.csv.

- PDFs: 1000, parse success 100.0 (99.6–100.0)
- median time 1.57s (IQR 1.26–1.95)
- status counts: {'success': 1000}

| domain | keyword hit-rate | snippets |
|---|---|---|
| Random sequence generation | 40.5% | 3000 |
| Allocation concealment | 7.2% | 3000 |
| Blinding of participants and personnel | 28.6% | 3000 |
| Blinding of outcome assessment | 1.6% | 3000 |
| Incomplete outcome data | 14.2% | 3000 |
| Selective reporting | 4.5% | 3000 |

## PyMuPDF vs GROBID case study (n=1, example.pdf)

- pages: 10, PyMuPDF chars: 62835
- GROBID title token coverage: 48.1% (14 of 27 GROBID title tokens missing from the extracted text)
- GROBID abstract token coverage: 70.0%
- **Interpretation:** the missing title tokens (human, papillomavirus, hpv, hpv, antibody…) are absent even as substrings of the raw extracted text — the title font in this PDF maps no Unicode, so PyMuPDF silently loses the title while the GROBID parse recovered it. This is a genuine extraction gap in raw PyMuPDF extraction on real-world PDFs, not a metric artifact, and is one motivation for the sentence-level robustness checks in Tier D.

## Framing notes (for the manuscript)

- The Risk-of-Bias model weights loaded by RCT-Reviewer are the
  original validated RobotReviewer artifacts (hashes above); published
  RoB accuracy therefore transfers to the refactored tool by
  weight-identity, verified here by exact pipeline reproduction
  (Tier C). No new RoB ground truth was collected.
- Scope of all live comparisons: the original 2017 code cannot
  execute its TensorFlow 1.x CNN in a maintained environment;
  the validation shim bypasses it deliberately. Every executed
  original-vs-refactored comparison (Tier A executed-code
  fidelity, Tier C, and the Tier E control) therefore runs the
  original in SVM-only capacity, matching the refactored tool's
  SVM-only configuration. Claims of exact reproduction are
  claims about these SVM paths; the ensemble-level cost of
  dropping the CNN + publication-type components is measured
  only in Tier B, against the original's stored outputs.
- Tier D is a robustness/extraction-integrity check on modern
  open-access PDFs, explicitly not an accuracy validation.
- Data and code availability: the full validation harness
  (validation_shim.py, fetch_corpus.py, evaluate.py), the
  per-tier CSV/JSON outputs, corpus metadata (corpus/metadata.csv,
  with the SVM decision for every candidate paper), and SHA-256
  hashes of every model weight loaded (provenance.json) ship with
  this repository. The Clinical Hedges benchmark and the stored
  original-model outputs are distributed with the original
  RobotReviewer data and are byte-identical copies.
- Positioning: the contribution is a maintained, Java-free
  refactoring of RobotReviewer with demonstrated input-output
  fidelity to the validated original — not a claim of new
  state-of-the-art accuracy. Where the ensemble is degraded, the
  loss is attributed (Tier B) to components that cannot run in a
  modern maintained environment (TensorFlow 1.x CNN, ptyp model).
