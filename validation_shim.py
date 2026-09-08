"""
validation_shim.py — run the ORIGINAL RobotReviewer (2022 master) model code
inside RCT-Reviewer's Python 3.12 venv, without modifying either repository.

Three documented compatibility shims are applied:

1. Keras stubs. robotreviewer/ml/vectorizer.py imports keras at module level
   and robots/rct_robot.py imports keras in __init__. The Risk-of-Bias and
   SVM-only code paths never execute keras operations, so stub modules are
   injected into sys.modules before importing the package. The original CNN
   (Keras .h5) ensemble is NOT executed — that is the ablation under study.

2. scikit-learn kwarg translation. The original vectorizer passes
   non_negative=True to HashingVectorizer. The parameter was renamed to
   alternate_sign in scikit-learn 0.24 with identical semantics (hash signs
   clamped to non-negative). HashingVectorizer.__init__ is wrapped to
   translate the old kwarg; no hashing/tokenization behaviour is altered.

3. DATA_ROOT redirect. The robotreviewer-master checkout's model weight files
   are git-LFS pointer stubs (132 bytes). robotreviewer.DATA_ROOT is pointed
   at RCT-Reviewer/data, which holds the real weight files — the same files
   the published RCT-Reviewer tool loads.

Consequence for every comparison built on this shim: the original 2017 code
runs in SVM-ONLY capacity. All original-vs-refactored comparisons (Tier A
executed-code fidelity, Tier C, and the Tier E control in evaluate_tian.py)
are therefore apples-to-apples SVM-to-SVM comparisons — identical inputs,
identical weight files (DATA_ROOT redirect), identical configuration — so the
refactoring is the only variable. Exact-reproduction claims are claims about
these shared SVM paths; the original's full SVM+CNN(+ptyp) ensemble is never
re-executed anywhere, and its performance contribution is quantified
separately in Tier B from the stored outputs generated in the original
2016 environment.

Weight provenance: the .npz MiniClassifier weights (bias_doc_level,
bias_sent_level, rct_svm_weights) were copied unconverted into RCT-Reviewer;
they are the original RobotReviewer artifacts. Their SHA-256 hashes are
recorded in validation_results/provenance.json by evaluate.py.
"""

import logging
import os
import pickle as _pickle
import sys
import types
from pathlib import Path

log = logging.getLogger("validation_shim")

BASE = Path(__file__).resolve().parent
RR_REPO = BASE / "robotreviewer-master"
RCT_REPO = BASE / "RCT-Reviewer"
RCT_DATA = RCT_REPO / "data"

if not RR_REPO.exists():
    raise RuntimeError(f"Original repo not found: {RR_REPO}")
if not RCT_DATA.exists():
    raise RuntimeError(f"RCT-Reviewer data dir not found: {RCT_DATA}")

_state = {"installed": False, "robotreviewer": None, "bias_robot": None,
          "rct_robot": None, "nlp": None}


class _StubModule(types.ModuleType):
    """Module whose missing attributes resolve to inert placeholder classes.
    The original code imports many keras names at module/init level; none are
    executed on the code paths under test (SVM-only, bias)."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        attr = type(name, (), {"__call__": lambda self, *a, **k: None})
        setattr(self, name, attr)
        return attr


def _install_keras_stubs():
    if "keras" in sys.modules:
        return
    names = ["keras", "keras.preprocessing", "keras.preprocessing.sequence",
             "keras.preprocessing.text", "keras.models", "keras.layers",
             "keras.regularizers", "keras.backend"]
    for name in names:
        sys.modules[name] = _StubModule(name)
    # Original RCTRobot.__init__ loads the 10 CNN .h5 files via load_model.
    # The stub returns an inert object; the CNN ensemble is never invoked on
    # the code paths under test (predict with ensemble_type='svm' skips all
    # CNN branches). This is exactly the ablation under study.
    sys.modules["keras.models"].load_model = lambda f: object()
    sys.modules["keras.preprocessing.sequence"].pad_sequences = \
        lambda seqs, maxlen=None: seqs
    log.info("Keras stub modules installed (CNN ensemble never invoked)")


def _patch_hashing_vectorizer():
    from sklearn.feature_extraction.text import HashingVectorizer
    if getattr(HashingVectorizer, "_nn_translated", False):
        return
    _orig_init = HashingVectorizer.__init__

    def _patched_init(self, *args, **kwargs):
        if "non_negative" in kwargs:
            nn = kwargs.pop("non_negative")
            kwargs.setdefault("alternate_sign", bool(nn))
        _orig_init(self, *args, **kwargs)

    HashingVectorizer.__init__ = _patched_init
    HashingVectorizer._nn_translated = True
    log.info("HashingVectorizer.__init__ patched: non_negative -> alternate_sign")


def install():
    """Import the original package with shims applied. Idempotent."""
    if _state["installed"]:
        return _state["robotreviewer"]

    _install_keras_stubs()
    _patch_hashing_vectorizer()

    # Same alias the refactored tool uses for old pickles (rct_reviewer/ml/rct_robot.py)
    import sklearn.linear_model
    if "sklearn.linear_model.logistic" not in sys.modules:
        sys.modules["sklearn.linear_model.logistic"] = sklearn.linear_model

    # numpy removed the deprecated Python-builtin aliases in 2.0; the original
    # MiniClassifier.predict uses np.int. Aliasing to the builtin restores the
    # exact original behaviour (astype(int)).
    import numpy as np
    if not hasattr(np, "int"):
        np.int = int

    # VectorizerMixin was renamed _VectorizerMixin in recent scikit-learn; the
    # original rct_robot.py imports it under the old public name.
    import sklearn.feature_extraction.text as _sk_text
    if not hasattr(_sk_text, "VectorizerMixin"):
        if hasattr(_sk_text, "_VectorizerMixin"):
            _sk_text.VectorizerMixin = _sk_text._VectorizerMixin
        else:
            raise RuntimeError("cannot expose VectorizerMixin for original code")

    sys.path.insert(0, str(RR_REPO))
    import robotreviewer
    robotreviewer.DATA_ROOT = str(RCT_DATA)
    # robotreviewer.get_data() joins the module-global DATA_ROOT at call time,
    # so the redirect above redirects every weight load.

    _state["installed"] = True
    _state["robotreviewer"] = robotreviewer
    log.info("robotreviewer imported; DATA_ROOT -> %s", RCT_DATA)
    return robotreviewer


def _get_nlp():
    """spaCy model shared with the refactored pipeline (same model, same
    segmentation), so both pipelines see identical sentences."""
    if _state["nlp"] is None:
        sys.path.insert(0, str(RCT_REPO))
        import spacy
        _state["nlp"] = spacy.load("en_core_web_sm")
    return _state["nlp"]


def get_original_bias_robot():
    """Instantiate the ORIGINAL BiasRobot (weights load from RCT-Reviewer data
    via the DATA_ROOT redirect)."""
    if _state["bias_robot"] is None:
        rr = install()
        from robotreviewer.robots.bias_robot import BiasRobot
        _state["bias_robot"] = BiasRobot()
    return _state["bias_robot"]


def original_bias_annotate(full_text: str, top_k: int = 3):
    """Run the original BiasRobot.pdf_annotate on raw text.

    The text is segmented with the same spaCy model the refactored pipeline
    uses, and passed to the untouched original method via its MultiDict
    input. Returns the original structured_data:
      [{'domain', 'judgement', 'annotations': [{'content', 'position', ...}]}]
    """
    robot = get_original_bias_robot()
    from robotreviewer.data_structures import MultiDict
    nlp = _get_nlp()
    spacy_doc = nlp(full_text)
    md = MultiDict()
    md.data["gold"]["text"] = full_text
    md.data["gold"]["parsed_text"] = spacy_doc
    robot.pdf_annotate(md)
    return md.data["ml"]["bias"]


def original_sentence_scores(sent_texts, domain, top_k=3):
    """Sentence-level decision scores from the ORIGINAL model components,
    mirroring bias_robot.pdf_annotate lines 80-97 verbatim."""
    robot = get_original_bias_robot()
    doc_domains = [domain] * len(sent_texts)
    robot.vec.builder_clear()
    robot.vec.builder_add_docs(sent_texts)
    robot.vec.builder_add_docs(list(zip(sent_texts, doc_domains)))
    X = robot.vec.builder_transform()
    return robot.sent_clf.decision_function(X)


def get_original_rct_robot():
    """Instantiate the ORIGINAL RCTRobot.

    __init__ unconditionally unpickles the SVM+CNN calibration LogisticRegression
    models (unused on the SVM-only path). Those pickles were written for
    scikit-learn >=0.20/Python 3.6 and may fail to load under scikit-learn 1.9;
    during __init__ only, pickle.load failures are substituted with a dummy so
    construction completes. The SVM path itself (MiniClassifier +
    HashingVectorizer + calibration JSON) is untouched original code.
    """
    if _state["rct_robot"] is not None:
        return _state["rct_robot"]
    rr = install()
    from robotreviewer.robots.rct_robot import RCTRobot

    class _DummyCalibration:
        def predict_proba(self, X):
            raise RuntimeError("calibration unused on SVM-only path")

    real_load = _pickle.load

    def guarded_load(*args, **kwargs):
        try:
            return real_load(*args, **kwargs)
        except Exception as e:  # old sklearn estimator pickles
            log.warning("pickle.load failed during original RCTRobot init (%s); "
                        "substituting dummy (unused on SVM-only path)", e)
            return _DummyCalibration()

    _pickle.load = guarded_load
    try:
        _state["rct_robot"] = RCTRobot()
    finally:
        _pickle.load = real_load
    return _state["rct_robot"]


def original_rct_predict(title: str, abstract: str):
    """Original RCTRobot SVM-only prediction, the configuration that matches
    the refactored tool (ensemble_type='svm', no ptyp, 'balanced' threshold)."""
    robot = get_original_rct_robot()
    row = robot.predict({"title": title, "abstract": abstract},
                        ensemble_type="svm", threshold_type="balanced",
                        auto_use_ptyp=False)[0]
    return {"score": float(row["score"]), "is_rct": bool(row["is_rct"]),
            "model": row["model"], "threshold_value": float(row["threshold_value"])}


def load_medline_records(path):
    """Parse a MEDLINE-format file with the ORIGINAL ris parser."""
    rr = install()
    from robotreviewer.parsers import ris
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    records = ris.loads(text)
    return records
