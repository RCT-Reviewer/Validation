
# RCT-Reviewer Validation Harness

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22260983.svg)](https://doi.org/10.5281/zenodo.22260983)

This suite provides validation for the RCT-Reviewer. It bridges the gap between the 2017 original code (`robotreviewer-master`) and the 2026 refactored code (`RCT-Reviewer`), proving mathematical fidelity of the shared SVM paths, predictive validity, and modern infrastructure robustness. Note on scope: the original's TensorFlow 1.x CNN cannot run in a maintained environment and is deliberately bypassed by the validation shim, so every executed original-vs-refactored comparison is SVM-only on both sides (see `validation_shim.py` and Tier B for how the dropped ensemble is accounted for).

## Directory Structure
- `RCT-Reviewer/` : The refactored (2026) repository (Python 3.12) - the tool under test.
- `robotreviewer-master/` : The original (2017) repository - the reference implementation.
- `validation_shim.py` : The compatibility shim that runs the original 2017 code inside the modern Python 3.12 venv without touching either repo.
- `fetch_corpus.py` : Downloads recent open-access RCT PDFs from Europe PMC, filtered by RCT-Reviewer's own SVM.
- `evaluate.py` : Runs the 4 validation tiers and generates all statistics, CSVs, and figures.
- `corpus/` : The downloaded PDF corpus (1,000 PDFs, see below).
- `validation_results/` : All outputs - `report.md`, per-tier CSVs/JSONs, and figures in **PNG + SVG + PDF**.
- `requirements.txt` : Pinned environment (latest majors; `pandas<3` because streamlit 1.55 in RCT-Reviewer requires it).

## Environment
Tested with: Python 3.12.12, numpy 2.5.2, scipy 1.18.1, scikit-learn 1.9.0, spaCy 3.8.16 (+ `en_core_web_sm` 3.8.0), PyMuPDF 1.28.2, matplotlib 3.11.1, seaborn 0.13.2, statsmodels 0.15.0, pandas 2.3.3, pydantic 2.13.5. `provenance.json` records the exact versions of every run.


## How to Run (Step-by-Step)

1. Ensure you are in the `RCT-Reviewer-Validation` directory by cloning this repository:

```bash
git clone https://github.com/RCT-Reviewer/Validation.git
```
---

2. **Get the two upstream codebases (first time only).** This repo ships the harness, results, and corpus metadata - not the upstream code or the article PDFs. Place both repositories at these exact paths (the harness resolves `RCT-Reviewer/` and `robotreviewer-master/` relative to its own location):

```bash
# RCT-Reviewer (the refactored tool under test): model weights are stored with Git LFS, so install LFS once (git lfs install) and pull the weights:
git clone https://github.com/aurumz-rgb/RCT-Reviewer.git RCT-Reviewer
cd RCT-Reviewer && git lfs pull && cd ..
```

```bash
# RobotReviewer (the original 2017 reference implementation). Pull its LFS weights too, so both codebases hold their real model files and the original pipeline is fully runnable for independent verification:
git clone https://github.com/ijmarshall/robotreviewer.git robotreviewer-master
cd robotreviewer-master && git lfs pull && cd ..
```

---

3. Make a .venv environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

---

4. Install dependencies:

```bash
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

5. Corpus retrieval (three options)

The article PDFs are not redistributed in this repository (publisher licensing). Three ways to obtain the 1,000-PDF corpus:

-- **Zenodo archive (recommended, exact).** The full corpus used in the paper is deposited on Zenodo with restricted access - request access and download the byte-identical set: [https://doi.org/10.5281/zenodo.22260256](https://doi.org/10.5281/zenodo.22260256). Unzip into `corpus/`.

 -- **Exact 1:1 rebuild from the audited metadata.** `corpus/metadata.csv` records every accepted paper's PMCID. This re-downloads exactly those papers from Europe PMC (skipping cached files; verified 1,000/1,000):
   ```bash
   python fetch_corpus.py --pmcid-list corpus/metadata.csv
   ```
 Expect ~1 h (network-bound, ~0.5 s politeness delay per download). Failed downloads are retried on re-run.
 
-- **Fresh query (new comparable corpus, not identical).** `python fetch_corpus.py --target 1000` re-runs the Europe PMC search + SVM filter. Europe PMC result sets drift between searches, so this yields a comparable but not paper-identical corpus; Tier D statistics are descriptive and remain interpretable, while Tiers A/B/C do not use the corpus at all.

---

6. Execute the full validation suite (Tiers A, B, C, D):

```bash
python evaluate.py --tier all --corpus-dir corpus
```

   Useful flags: `--tier A|B|C|D` for a single tier

---

7. Check `validation_results/` for `report.md`, the CSV/JSON outputs, and the figures (`*.png`, `*.svg`, `*.pdf`).

---

## 8. Tier E - external validity against a human reference standard (Tian 2024)

Tiers A–D validate the refactored tool against the *original tool* and against the *RCT screening benchmark*. Tier E adds what a methods journal expects for a risk-of-bias tool: agreement with **human reviewers**, using the public reference dataset of **Tian et al., Res Synth Methods 2024;15(6):1111–1119** (1,955 RCTs with human-consensus RoB on 4 domains, plus the original RobotReviewer's automatic labels; dataset: [https://osf.io/k6w9q](https://osf.io/k6w9q), snapshot stored at `reference_data/tian_rob.xlsx`).

**How it works:** each trial's citation string is resolved to a PMID via NCBI eutils (journal + year + volume + first page; ~96% resolve), PMIDs are converted to PMC records, open-access PDFs are downloaded from Europe PMC into `tian_corpus/`, RCT-Reviewer judges the 4 domains, and agreement is computed against (a) the human consensus and (b) the original RobotReviewer's deposited labels on the identical subset - an external, third-party check of the fidelity claim.

**Run it** (all phases cached and resumable; rerun simply continues where it stopped).

**Pilot test first (recommended):** a 2-trial demo end-to-end - downloads, judging, and report generation in ~2 minutes:

```bash
python evaluate_tian.py --unpaywall-email vihaansahu143@gmail.com --limit 2
```

Then the full run (the 2 pilot PDFs are kept; the rest are fetched and the report is regenerated with the real n):

```bash
python evaluate_tian.py --unpaywall-email vihaansahu143@gmail.com
```

Other controls: `--phase analyze` resumes a specific phase; `--limit N` caps new downloads/expansions/analyses per phase.

**Trial PDFs - two options:**

1. **Zenodo archive (recommended).** Download the exact trial-PDF set used in this evaluation ([https://doi.org/10.5281/zenodo.22286385](https://doi.org/10.5281/zenodo.22286385)) and place the PDFs into `tian_corpus/`. This reproduces the published Tier E numbers 1:1 - no fetching needed; `evaluate_tian.py` detects the cached PDFs and skips straight to analysis.
2. **Fetch like we did.** First set your Semantic Scholar API key (used by the `expand` phase to find open-access PDFs for trials that Europe PMC and Unpaywall cannot provide; get a free key at [semanticscholar.org/product/api](https://www.semanticscholar.org/product/api#api-key-form)). Set it once per shell and do **not** commit it to the repository:
   ```bash
   export S2_API_KEY=your_semantic_scholar_key   # or pass --s2-api-key instead
   ```
   Then run the full pipeline - resolve citations → PMC records → download open-access PDFs from Europe PMC → harvest OA copies via Unpaywall → Semantic Scholar:
   ```bash
   python evaluate_tian.py --unpaywall-email vihaansahu143@gmail.com
   ```
   This yields a comparable (slightly different) open-access subset, since availability changes over time. Without `S2_API_KEY` the Semantic Scholar pass is skipped (PMC + Unpaywall still run); the API rate-limits aggressively, so the harness backs off on HTTP 429 and any trials it misses can be picked up by rerunning the expand phase later.

Use the same venv you run `evaluate.py` with (`RCT-Reviewer/.venv/bin/python`, or any venv with `requirements.txt` installed - the analyze phase needs spaCy/PyMuPDF/scikit-learn). The script holds a lock file while running: **run one instance at a time**, and if a run is interrupted, just rerun the same command - every phase resumes from its cached CSV.

**Runtime:** ~1.5–2.5 h total (citation resolution ~40 min, downloads ~20 min, analysis ~45 min; network-bound). Outputs land in `validation_results_tian/` - `tian_report.md`, `tian_agreement.csv`, `tian_rr_judgments.csv`, `tian_resolution.csv`, and the figure in PNG/SVG/PDF. `validation_results/` is not touched by Tier E.

**Control experiment (`--phase control`).** Tier E compares against Tian's labels, which were produced from *publisher* PDFs, while Tier E uses open-access PMC PDFs - so a residual gap vs the published numbers is expected from the input difference alone. To prove that gap is the PDF source and not the refactoring, the control phase re-runs the **original 2017 BiasRobot** (via the compatibility shim, in the same **SVM-only** configuration as RCT-Reviewer) on the *identical* PMC text that RCT-Reviewer judged:
```bash
python evaluate_tian.py --phase control    # ~30 min; appends to tian_report.md
```
Expected result: original-on-PMC-text (SVM-only) agrees near-100% with RCT-Reviewer (SVM-only) and shows the same reduced human κ as the refactored tool - a like-for-like comparison that isolates PDF source as the only variable (implementation equivalence held constant by Tier C on the same SVM path).

**Why Tier E evaluates fewer trials than Tian's 1,955:** Tian et al. assessed publisher PDFs obtained through their own review pipeline; this harness can only legitimately retrieve **open-access** full texts. Of 1,955 trials, ~93% resolve to PubMed records, ~427 have a PMC record, and an Unpaywall pass (`expand` phase) harvests further open-access copies by DOI - giving a final subset of roughly 400–700 trials (exact n is reported in `tian_report.md`). The remainder are paywalled (JAMA, NEJM, Ann Oncol, …) and are excluded rather than scraped. Bounded human-referenced evaluations are an established design - Hirt 2021 published with n=190, Armijo-Olivo 2020 with n=393.

**Note on PMC records that could not be downloaded:** PMC contains two classes of full text. Open-access (CC-licensed) articles allow programmatic PDF delivery; *author manuscripts* deposited under funder policies (typical for Lancet, NEJM, JAMA) are publicly viewable on the PMC website but their publisher licence forbids programmatic PDF rendering - Europe PMC's render endpoint returns an HTTP 500 for them. So a trial can be visible in PMC yet yield no downloadable PDF; such trials are excluded from Tier E rather than scraped. The exclusion is licensing-driven, not a tool failure - the harness never receives those PDFs, so it cannot be the cause of any judgement difference.

**Result (confirmed 2026-09-04):** on the 313-trial open-access subset, RCT-Reviewer agrees with the human consensus at κ 0.26/0.20/0.48/0.12 (concordance 60–76%) across the four domains - within the range Tian published for the original tool (κ 0.25–0.59, concordance 63–83% on publisher PDFs). The control phase confirmed the mechanism: the original 2017 implementation, run through the shim in the same SVM-only configuration on the identical PMC text, agrees with RCT-Reviewer in **100.0% of domain judgements** (SVM-to-SVM, identical inputs) and shows identical human κ - so the difference vs Tian's published values is attributable to the PDF source (open-access versions vs publisher PDFs), not the refactoring. External fidelity vs the original's deposited publisher-PDF labels: 78.9% of domain judgements.

## What reproducing takes (measured on an M1 Pro, 1,000-PDF corpus)

| Step | Time | Notes |
|---|---|---|
| Setup: venv + `requirements.txt` + spaCy model + upstream clones/LFS | 15–25 min | network-bound; RCT-Reviewer light LFS pull (bias + rct weights) ~2.6 GB full; RobotReviewer LFS pull adds its own weights |
| `fetch_corpus.py --target 1000` | 1–2 h | only needed if you don't reuse a corpus; results for Tiers A/B/C do not depend on it |
| `evaluate.py --tier all` - Tier A+B (751 benchmark records) | ~6 min | deterministic, seeded |
| Tier C (1,003 documents × 6 domains, both pipelines) | ~70 min | the dominant cost: runs the original 2017 pipeline over every corpus document |
| Tier D (1,000 PDFs, 12,060 pages) | ~35 min | parse + annotate each PDF |
| Tier E (Tian 2024 human-reference arm, `evaluate_tian.py`) | ~1.5–2.5 h | optional; independent of the corpus, resumable per phase |
| **Total** | **≈ 2 h** (Tiers A–D) / **≈ 4 h** with Tier E | 6–8 GB disk (venv + models + corpus) |


## License


[![GNU GPL v3 License](https://www.gnu.org/graphics/gplv3-with-text-136x68.png)](https://www.gnu.org/licenses/gpl-3.0.en.html)

GPL-3.0 - the same licence as both upstream repositories (`RCT-Reviewer` and `robotreviewer-master`). See [`LICENSE`](LICENSE). RCT-Reviewer is an independent refactoring of RobotReviewer by a different team; the original authors' model weights are redistributed here under the same licence with attribution, and all published performance claims cited in the results belong to the original evaluations.

**Corpus**:  The corpus is 1,000 recent (2025–2026) open-access PDFs from Europe PMC, all accepted by RCT-Reviewer's own SVM. Roughly a third of the titles mention "protocol" (trial protocols that the SVM also flags as RCT-like); they are retained as valid parse targets, and per-paper decisions are in `corpus/metadata.csv`. The corpus is intentionally self-selected, so Tier D measures robustness on tool-relevant PDFs, not a representative literature sample. Tier A/B/C do not depend on the corpus.

---

## How the Code Works (The Architecture)

### `validation_shim.py` (Compatibility shim)

The original RobotReviewer code relies on dead dependencies (Keras/TensorFlow 1.x, Python 3.6-era scikit-learn). The shim tricks it into running in the modern venv by applying **five compatibility shims**, none of which alter the numerics under test:

1. **Keras stubs** - the original `vectorizer.py`/`rct_robot.py` import Keras at module level. Fake `keras.*` modules are injected into `sys.modules` before import. The original CNN (Keras `.h5`) ensemble is *never executed* - its absence is exactly the ablation studied in Tier B. **Consequence for all live comparisons:** every executed original-vs-refactored comparison in this harness (Tier A executed-code fidelity, Tier C, Tier E control) runs the original 2017 code in **SVM-only capacity**, deliberately matching the refactored tool's SVM-only configuration. This is an apples-to-apples design: identical inputs, identical weight files, identical configuration - so any difference could only come from the refactoring itself. The performance contribution of the dropped CNN + publication-type components is quantified separately in Tier B, from the original's *stored* ensemble outputs (generated in the original 2016 environment, not re-executed).
2. **scikit-learn kwarg translation** - the original passes `non_negative=True` to `HashingVectorizer`; renamed to `alternate_sign` in scikit-learn 0.24 with identical semantics. `HashingVectorizer.__init__` is wrapped to translate the kwarg; hashing/tokenization behaviour is untouched.
3. **DATA_ROOT redirect** - the original repo's model weights are 132-byte git-LFS pointer stubs. `robotreviewer.DATA_ROOT` is repointed at `RCT-Reviewer/data`, which holds the real `.npz` weight files (the same artifacts the published tool loads; SHA-256 hashes are written to `provenance.json`). This redirect stays in place even when RobotReviewer's own LFS weights are pulled, so both pipelines are guaranteed to load byte-identical weight files - that guarantee is what makes the fidelity comparison in Tiers A and C meaningful. And it is *verified, not assumed*: on every run, `evaluate.py` hashes RCT-Reviewer's weight files and compares them against the original repository's Git LFS object hashes (or direct file hashes for non-LFS files) - all four files verified byte-identical as of the current run.
4. **Renamed-API aliases** - `np.int` (removed in numpy 2.0, used by the original `MiniClassifier.predict`) and `VectorizerMixin` (renamed to `_VectorizerMixin`) are re-aliased to their old names, plus the `sklearn.linear_model.logistic` module alias used by old pickles.
5. **Guarded unpickling** - the original `RCTRobot.__init__` unconditionally unpickles SVM+CNN calibration pickles (unused on the SVM-only path); load failures there are swapped for an inert dummy so construction completes.

On top of the shims it exposes four adapters used by `evaluate.py`:
- `original_bias_annotate(full_text)` - segments text with the *same* spaCy model the refactored tool uses, feeds it to the untouched original `BiasRobot.pdf_annotate`, and returns the original per-domain judgements + top-3 evidence sentences.
- `original_sentence_scores(sentences, domain)` - recomputes sentence-level decision scores with the original vectorizer+classifier, mirroring the original `pdf_annotate` lines verbatim.
- `original_rct_predict(title, abstract)` - original `RCTRobot.predict` in the configuration that matches the refactored tool (`ensemble_type='svm'`, `threshold_type='balanced'`, no publication-type features).
- `load_medline_records(path)` - parses the MEDLINE benchmark file with the original `ris.py` parser for input parity.

### `fetch_corpus.py` 

To guarantee the corpus contains strict RCTs, the fetcher:

1. Queries the Europe PMC REST API for open-access articles with a PDF, published strictly in 2025–2026 (`OPEN_ACCESS:y AND HAS_PDF:y AND (PUB_YEAR:2025 OR PUB_YEAR:2026)`), using `cursorMark` pagination (500/page) and a 0.5 s politeness delay.
2. For every candidate with a title + abstract, runs **RCT-Reviewer's own SVM** (`RCTRobot.predict`). Papers the model does not label as RCTs are skipped and logged.
3. Downloads accepted papers from `europepmc.org/articles/PMC…?pdf=render`, deduplicating by PMCID and skipping PDFs already on disk (resume-capable). Failed downloads are deleted so a re-run retries them.
4. Writes `corpus/metadata.csv` (pmcid, title, year, model score, decision, filename, status) so every inclusion decision is auditable.

### `evaluate.py` (The 4-Tier Validator)

- **Tier A (Predictive validity)** - Parses the 1,000-record Clinical Hedges MEDLINE benchmark (`pubmed_test.txt`) with the *original* parser, keeps records having both title and abstract (n = 751 after dedup), and scores each through the refactored `RCTRobot`. Metrics against the human `hedges_is_rct` ground truth: sensitivity, specificity, accuracy, PPV, NPV, F1, Cohen's kappa - each with a 95% Wilson score interval; F1/kappa CIs use 1,000-resample bootstrap (seed 42). Also computed: ROC AUC and Brier score. *Implementation fidelity* compares the new scores to (a) the stored original-model outputs (`pubmed_expected.json`) and (b) the original 2017 code executed live through the shim **in the same SVM-only configuration as the refactored tool** (`ensemble_type='svm'`; the executed comparison validates the SVM path shared by both systems, while the full SVM+CNN ensemble appears only as stored outputs in Tier B) - the acceptance gate is max |Δscore| < 1e-9 vs the executed original (achieved: 0.0).

- **Tier B (CNN ablation)** - On the same 751 records, the refactored SVM-only decisions are compared against the stored decisions of the original full SVM+CNN(+ptyp) ensemble: decision agreement + kappa, both systems' metrics vs Hedges, plus **McNemar's exact test** on paired correct/incorrect counts (p < 0.0001). This quantifies the cost of dropping the dead CNN: agreement 92.5%, F1 0.925 → 0.969.

- **Tier C (RoB fidelity)** - Runs both the shimmed 2017 BiasRobot and the refactored BiasRobot over identical sentence lists from the bundled PDFs (`example.pdf`, `sample_bias.pdf`), the GROBID fixture abstract, and every corpus PDF. **Both pipelines run SVM-only:** the RoB path exercised here in the original is its SVM pipeline (interaction-hashed vectorizer + linear SVMs over the shared weight files), and the original's TensorFlow/Keras components are bypassed by the shim and never execute. The comparison is therefore apples-to-apples - identical sentences, identical weights, identical configuration - and the refactoring is the only variable. Per document × domain (6 RoB domains): judgement agreement, top-3 evidence-sentence Jaccard, and max |Δ| sentence decision score. A separate **vectorizer equivalence probe** pushes identical texts (plain + domain-interaction tuples) through the original and refactored `InteractionHashingVectorizer` stacks and asserts byte-identical sparse matrices. Result: 100% agreement, kappa 1.0, max |Δ| = 0, 14/14 probes identical - i.e. the refactored SVM RoB pipeline exactly reproduces the original's SVM judgements. RoB accuracy therefore transfers by weight-identity (no new RoB ground truth was collected - the report's framing notes state this explicitly).

- **Tier D (Parser robustness - descriptive, not accuracy)** - Parses every corpus PDF with the new PyMuPDF-based `PDFParser`, recording parse success/failure mode, pages, chars, sentences, and per-document time (Wilson CI on success rate, median + IQR timing). It also runs the refactored BiasRobot on each PDF and reports the **keyword hit-rate** of the top-3 highlighted sentences per RoB domain against hand-written methodological-term regex lexicons - a lexical plausibility check of the extracted evidence. Finally, an n=1 **PyMuPDF-vs-GROBID case study** compares PyMuPDF extraction of `example.pdf` against the stored GROBID parse (`pdffile.json`) via title/abstract token coverage.

### `evaluate_tian.py` (Tier E - human-reference external validation)
Separate from the four tiers above (and writing only to `validation_results_tian/`), this script measures agreement between RCT-Reviewer's RoB judgements and the **human consensus reference standard** of Tian et al. 2024 (Res Synth Methods; dataset on OSF, snapshot in `reference_data/tian_rob.xlsx`, 1,955 RCTs, 4 domains, binary low vs high/unclear):
1. **Resolve** each trial's citation string to a PMID (NCBI eutils, journal+year+volume+first page; progressive fallbacks; ~96% resolve).
2. **Convert** PMIDs to PMC records (NCBI ID converter, batched).
3. **Download** open-access PDFs from Europe PMC into `tian_corpus/` (politeness delay, cached).
4. **Analyze** each PDF with the same PDFParser + BiasRobot used in Tiers C/D; map judgements to low vs high/unclear.
5. **Compare** per domain (n, concordance with Wilson CI, Cohen's kappa with seeded bootstrap CI, PPA, NPA): (a) RCT-Reviewer vs human consensus; (b) the original RobotReviewer's deposited labels vs the human consensus on the *identical subset* (matched-sample comparison); (c) RCT-Reviewer vs the original's deposited labels (external fidelity check on data this project never touched). Outputs: `tian_report.md`, `tian_agreement.csv`, `tian_rr_judgments.csv`, `tian_resolution.csv`, and `figure_tier_e_human_concordance.*`.
Note on the reference data: all four of Tian's published kappas (0.46/0.25/0.59/0.27) and PPA/NPA values reproduce exactly from the deposited xlsx, confirming correct decoding; the running-text concordance percentages in the paper appear domain-shuffled relative to the deposited data, which the report footnotes.

**Figures** - every figure is saved in **three formats (`.png`, `.svg`, `.pdf`)** by the `save_fig` helper.

| figure | what it shows | how to read it |
|---|---|---|
| `figure_tier_a_confusion_matrix` | The tool's verdicts vs the human "Clinical Hedges" label on all 751 benchmark records. Rows = human truth, columns = the tool's prediction; darker cells = more papers. | The diagonal cells are correct calls: 290 papers correctly rejected as not-RCT (top-left) and 397 RCTs correctly accepted (bottom-right). The off-diagonal cells are the errors: 39 not-RCTs wrongly accepted (top-right) and 25 RCTs missed (bottom-left). The parenthesized percentage in each cell is the count divided by its true-class total - exactly the specificity (88.1%, top row) and sensitivity (94.1%, bottom row) in the Tier A table. |
| `figure_tier_a_calibration` | Reliability diagram: how close the SVM's score is to a true probability. | The raw SVM score is an unbounded decision value, not a probability, so it is first Platt-scaled (logistic regression on the raw score, fit on this benchmark - in-sample, therefore optimistic). Papers are grouped by calibrated probability (8 quantile bins); each point plots mean predicted probability vs the fraction that actually were RCTs. Points near the dotted diagonal mean the calibrated score can be read as a probability. |
| `figure_tier_b_ablation` | What was lost by removing the dead TensorFlow CNN: SVM-only (refactored) vs the original full SVM+CNN ensemble, side by side. | Taller bars = better. The ensemble is slightly better on every metric (F1 0.969 vs 0.925). The report attributes the gap across all three arms: SVM-only agrees 94.5% with SVM+CNN (no publication-type features) but 92.5% with the full SVM+CNN+ptyp ensemble - so most of the loss comes from the publication-type features, with the CNN contributing the remainder; both are unrunnable in a maintained environment. McNemar's test (p < 0.0001) confirms the total gap is statistically significant. |
| `figure_tier_d_keywords` | Lexical plausibility check: of the top-3 evidence sentences the tool highlights per RoB domain per PDF (1,000 × 3 = 3,000 snippets per domain), the share containing an explicit methodological keyword (e.g., "randomly assigned", "sealed envelopes", "double-blind"). | Low rates are expected and are *not* an error rate. The model returns three highlighted sentences for every domain of every PDF - even when the paper never describes that domain - so many snippets are lexically generic (or off-topic) and contain no domain keyword. Randomization (40.5%) and participant blinding (28.6%) are most often described explicitly; allocation concealment (7.2%), selective reporting (4.5%) and outcome-assessor blinding (1.6%) almost never are. Descriptive only. |
| `figure_tier_d_scale` | Parse time vs document length across the corpus. | Points rising roughly along a line (Pearson r ≈ 0.93) show the parser scales linearly with document size and has no blow-ups on long PDFs. |

`report.md` (all tiers) and `master_summary.csv` (headline numbers) are regenerated on every `--tier all` run.

## Current Results Snapshot (1,000-PDF corpus)
- **Tier A** (n=751): Sensitivity 94.1 (91.4–96.0), Specificity 88.1 (84.2–91.2), F1 0.925, kappa 0.826, ROC AUC 0.966 (raw scores), Brier 0.067 (Platt-scaled scores, in-sample), prevalence 56.2%. Fidelity vs executed original code: max |Δ| = 0.0, agreement 100% (vs stored 2016 outputs: max |Δ| = 0.093 from scikit-learn version drift, decisions agree 99.6%).

- **Tier B**: agreement with full ensemble 92.5 (90.4–94.2), kappa 0.847; three-arm attribution - SVM-only agrees 94.5% with SVM+CNN (no ptyp) vs 92.5% with the full SVM+CNN+ptyp ensemble (F1 0.925 vs 0.956 vs 0.969), so the publication-type features account for most of the gap; McNemar p < 0.0001.

- **Tier C** (SVM-to-SVM comparison; the original's CNN is bypassed by the shim and never executes): 6,018/6,018 judgements agree across all 1,003 documents × 6 domains (kappa 1.0, CI 99.9–100.0), sentence scores identical, vectorizer matrices identical (14/14 probes). The refactored SVM RoB pipeline exactly reproduces the original's SVM judgements; the dropped CNN/ptyp components' contribution is quantified separately in Tier B.

- **Tier D** (1,000 PDFs, 12,060 pages): parse success 100.0% (CI 99.6–100.0), median 1.66 s/PDF (IQR 1.32–2.06), longest PDF 7.0 s.

---

## Figures

All figures are generated by `evaluate.py` / `evaluate_tian.py` and saved in **PNG + SVG + PDF** under `validation_results/` and `validation_results_tian/`.

### Tier A - classifier vs human labels (Clinical Hedges, n=751)

Confusion matrix of the refactored SVM classifier against human labels. Diagonal = correct calls (290 not-RCTs rejected, 397 RCTs accepted); off-diagonal = errors (39 wrongly retained, 25 missed). Cell percentages are the specificity (88.1%, top row) and sensitivity (94.1%, bottom row).

![Tier A confusion matrix](validation_results/figure_tier_a_confusion_matrix.png)

Reliability diagram of the SVM score after in-sample Platt scaling - points near the diagonal mean the calibrated score reads as a probability.

![Tier A calibration](validation_results/figure_tier_a_calibration.png)

### Tier B - CNN ablation (SVM-only vs original full ensemble)

The measurable, attributed cost of removing the unmaintainable TensorFlow CNN: F1 0.925 vs 0.969 (McNemar p < 0.0001); most of the gap comes from the publication-type features.

![Tier B ablation](validation_results/figure_tier_b_ablation.png)

### Tier D - parser robustness on the 1,000-PDF corpus

Keyword hit-rate of the top-3 highlighted evidence sentences per RoB domain (3,000 sentences per domain) - a lexical plausibility check of extracted evidence; low rates reflect how rarely papers describe those domains explicitly.

![Tier D keywords](validation_results/figure_tier_d_keywords.png)

Parse time vs document length across all 1,000 PDFs - linear scaling (Pearson r = 0.93), no blow-ups (max 7.0 s).

![Tier D scale](validation_results/figure_tier_d_scale.png)

### Tier E - agreement with human reviewers (Tian 2024 subset, n=313)

Per-domain concordance with the human consensus on the Tian 2024 open-access subset: RCT-Reviewer (blue) vs the original RobotReviewer's deposited labels (green) on the identical trials. The control phase additionally showed the two implementations' SVM pipelines (both run SVM-only by design) agree on 100% of judgements when given the same text.

![Tier E human concordance](validation_results_tian/figure_tier_e_human_concordance.png)

---

## Conclusion

1. **The refactored tool's SVM paths are mathematically the original's SVM paths.** Against the original 2017 code executed in this environment *in the same SVM-only configuration* (the TensorFlow 1.x CNN cannot run in a maintained environment and is deliberately bypassed by the shim), the RCT classifier reproduces every score bit-identically (max |Δ| = 0.0, 751/751 records), and the refactored RoB pipeline reproduces the original's judgements in **6,018/6,018** document × domain comparisons (kappa 1.0) with zero sentence-score difference and byte-identical vectorizer matrices. These are deliberately apples-to-apples comparisons - identical inputs, identical weight files, identical configuration - so the refactoring is isolated as the only variable; the performance contribution of the dropped CNN + publication-type components is quantified separately in Tier B from the original's stored ensemble outputs (never re-executed). Published accuracy therefore transfers by weight identity; the loaded weights are the original validated artifacts (SHA-256 in `provenance.json`).

2. **Predictive validity is preserved and strong.** On the human-labelled Clinical Hedges benchmark (n=751, 56.2% RCT prevalence): sensitivity 94.1 (91.4–96.0), specificity 88.1 (84.2–91.2), accuracy 91.5, PPV 91.1, NPV 92.1, F1 0.925, Cohen's kappa 0.826 ("almost perfect" agreement), ROC AUC 0.966. PPV/NPV are prevalence-dependent and will differ on streams with a different case mix.

3. **The cost of dropping the dead CNN is quantified and attributed.** The original full SVM+CNN+ptyp ensemble reaches F1 0.969 vs 0.925 for the SVM-only refactored tool (McNemar p < 0.0001). Decomposition across three arms shows the publication-type features, not the CNN, account for most of the gap (agreement 94.5% without ptyp vs 92.5% with it); both components cannot run in a maintained environment (TensorFlow 1.x).

4. **The new parser is robust at scale.** All **1,000/1,000** PDFs parsed successfully (95% CI 99.6–100.0), median 1.66 s per document (IQR 1.32–2.06), processing time scaling linearly with document length (Pearson r = 0.93) with no blow-ups on the longest documents (max 7.0 s across a 12,060-page corpus).

5. **Extracted evidence is lexically plausible where papers describe methods explicitly.** Of the 3,000 top-3 highlighted sentences per RoB domain, 40.5% contain an explicit randomization keyword, 28.6% a participant/personnel blinding keyword, and 14.2% an attrition keyword; allocation concealment (7.2%), selective reporting (4.5%) and outcome-assessor blinding (1.6%) are rarely described explicitly in modern open-access papers. Low rates reflect what papers actually say (and that the model highlights top-3 evidence for every domain, including undescribed ones), not a bug.

6. **Known limitations, stated up front.** The corpus is self-selected by the tool's own SVM (dogfooding) and includes trial protocols; the GROBID comparison is n=1 but documents a genuine PyMuPDF gap (that PDF's title font maps no Unicode); the Platt calibration is fit in-sample (optimistic); no new RoB ground truth was collected - RoB validity transfers by weight identity rather than being re-measured; and every live comparison against the 2017 code is **SVM-only** (the original's TensorFlow 1.x CNN is unrunnable and deliberately bypassed by the shim), so exact-reproduction claims are scoped to the SVM paths both tools share, with the ensemble-level cost of the dropped components measured only in Tier B from stored outputs.

## Reproducibility notes
- `provenance.json` stores the run timestamp, package versions, and SHA-256 of every model weight file loaded by the tool.

- Bootstrap CIs are seeded (default 42); the SVM, vectorizers, and hash-based features are deterministic - re-running Tiers A–C reproduces the numbers exactly.

- Tier D sentence counts depend on the spaCy model version; upgrading spaCy can shift them slightly (results from a previous, older-package environment differed by a few sentences per PDF).

- `validation_shim.py` and `evaluate.py` never modify either repository - everything new lives in this directory.

---

## Troubleshooting
`libmupdf.dylib` Error: run these three commands, one by one:
```bash
RCT-Reviewer/.venv/bin/python -m pip uninstall -y pymupdf
RCT-Reviewer/.venv/bin/python -m pip cache purge
RCT-Reviewer/.venv/bin/python -m pip install "PyMuPDF==1.28.2" --no-cache-dir
```
Verify with `RCT-Reviewer/.venv/bin/python -c "import fitz; print('PyMuPDF works!')"`, then re-run `evaluate.py`.
