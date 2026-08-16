"""Timeline, metrics and method configuration -- everything that does not need a GPU.

Kept free of torch so it can be imported and tested on any machine; train.py
holds the parts that need a model.
"""

import csv
import os

import numpy as np

# The notebooks encode labels this way in every file. It is the reason the
# published "Macro-F1" is really the binary F1 of the FIXED class: sklearn's
# f1_score defaults to average='binary', pos_label=1.
LABEL_MAP = {"VULNERABLE": 0, "FIXED": 1}
VULNERABLE, FIXED = 0, 1

BACKWARD_LAGS = (1, 3, 5, 6)

# Written per evaluation. Unlike the original logs this records both classes and
# the true macro average, so no downstream step has to guess which F1 it holds.
METRIC_FIELDS = [
    "method", "model", "seed", "granularity", "train_window", "eval_window", "direction",
    "accuracy",
    "macro_f1", "f1_vulnerable", "f1_fixed",
    "precision_vulnerable", "recall_vulnerable",
    "precision_fixed", "recall_fixed",
    "f1_binary_pos1_LEGACY",
    "n_vulnerable", "n_fixed", "n_train", "n_replay",
]


def make_windows(months, start_year=2018, end_year=2024):
    """Window tags over [start_year, end_year], `months` wide.

    The bi-monthly case reproduces the notebooks' tags exactly:
        2018_M01-02, 2018_M03-04, ... 2024_M11-12
    """
    if 12 % months:
        raise ValueError(f"granularity must divide 12, got {months}")
    tags = []
    for year in range(start_year, end_year + 1):
        for m in range(1, 13, months):
            tags.append(f"{year}_M{m:02d}" if months == 1
                        else f"{year}_M{m:02d}-{m + months - 1:02d}")
    return tags


def compute_metrics(y_true, y_pred):
    """Per-class and macro metrics, plus the legacy number for comparability.

    `f1_binary_pos1_LEGACY` is what the notebooks logged as "f1" and the paper
    printed as "Macro-F1". It is kept so old and new runs can be lined up, and
    named so nobody mistakes it for a macro average again.
    """
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    kw = dict(labels=[VULNERABLE, FIXED], zero_division=0)
    p = precision_score(y_true, y_pred, average=None, **kw)
    r = recall_score(y_true, y_pred, average=None, **kw)
    f = f1_score(y_true, y_pred, average=None, **kw)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", **kw),
        "f1_vulnerable": f[VULNERABLE], "f1_fixed": f[FIXED],
        "precision_vulnerable": p[VULNERABLE], "recall_vulnerable": r[VULNERABLE],
        "precision_fixed": p[FIXED], "recall_fixed": r[FIXED],
        "f1_binary_pos1_LEGACY": f1_score(y_true, y_pred, zero_division=0),
        "n_vulnerable": int((y_true == VULNERABLE).sum()),
        "n_fixed": int((y_true == FIXED).sum()),
    }


class MetricsWriter:
    """Append-only CSV writer. Flushes every row so a crashed run keeps its results."""

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        new = not os.path.exists(path) or os.path.getsize(path) == 0
        self.fh = open(path, "a", newline="", encoding="utf-8")
        self.w = csv.DictWriter(self.fh, fieldnames=METRIC_FIELDS, extrasaction="ignore")
        if new:
            self.w.writeheader()
            self.fh.flush()

    def write(self, **row):
        self.w.writerow(row)
        self.fh.flush()

    def close(self):
        self.fh.close()


# --- method registry ---------------------------------------------------------
#
# `adapter` is the single most consequential axis and the one the conference
# code is inconsistent about:
#
#   "inherit" -- load the previous window's LoRA adapter and keep training it.
#                This is what makes a method continual.
#   "fresh"   -- build a new adapter every window, discarding everything learned.
#
# As committed, replay-3p and olora use "fresh". They therefore carry no
# knowledge across windows at all, which is a plain explanation for their poor
# scores that has nothing to do with buffer size or orthogonality. Both are
# reproduced here as-published; --adapter overrides the default so the corrected
# variant can be run and the two separated.
#
# `replay` describes what past data is mixed into the training set:
#   none | full:<k windows> | uncertain:<k samples>  (see build_train_set)
#
# `paper_mismatch` records where the conference text describes something the
# code does not do.
METHODS = {
    "window-only": dict(
        adapter="inherit", replay="none",
        desc="Fine-tune on the current window only."),
    "cumulative": dict(
        adapter="inherit", replay="cumulative",
        desc="Train on all windows up to t. Expensive upper baseline."),
    "replay-1p": dict(
        adapter="inherit", replay="full:1",
        desc="Current window plus the whole previous window."),
    "replay-3p": dict(
        adapter="fresh", replay="full:2",
        desc="Current window plus the previous two, whole.",
        paper_mismatch="Published as a replay-buffer variant, but the committed "
                       "code rebuilds the adapter each window, so it is not "
                       "continual. Its deficit is not evidence that larger "
                       "buffers hurt."),
    "casr": dict(
        adapter="inherit", replay="uncertain:100",
        desc="Replay the highest-entropy samples from the previous window."),
    "hybrid-casr": dict(
        adapter="inherit", replay="uncertain-balanced:100+25",
        desc="Highest-entropy samples, equal counts per class, from t-1 and t-2.",
        paper_mismatch="The paper describes a 70/30 split between uncertain and "
                       "uniformly-drawn samples. The code takes k//2 by entropy "
                       "from each class and only draws at random to fill a class "
                       "that is short."),
    "lbcl": dict(
        adapter="inherit", replay="none", orthogonal_init=True,
        desc="Orthogonally-initialised LoRA (QR), no replay.",
        paper_mismatch="Described as class-weighted cross-entropy. The committed "
                       "code contains no class weighting whatsoever -- it is an "
                       "orthogonal-initialisation method."),
    "olora": dict(
        adapter="fresh", replay="none", orthogonalise_against_history=True,
        desc="Fresh adapter each window, Gram-Schmidt orthogonalised against all "
             "previous LoRA A matrices.",
        paper_mismatch="The paper describes a soft regulariser beta*L_orth with "
                       "beta=0.1. The code performs hard Gram-Schmidt projection "
                       "at initialisation and starts from a fresh adapter, so the "
                       "method never accumulates knowledge."),
    "zero-shot": dict(
        adapter="none", replay="none",
        desc="Evaluate the base model untrained."),
}


def resolve_replay(spec, index, windows):
    """Which past windows feed the training set, and how they are sampled.

    Returns (window_tags, mode, budget). `mode` is one of
    none | full | cumulative | uncertain | uncertain-balanced.
    """
    if spec == "none":
        return [], "none", 0
    if spec == "cumulative":
        return windows[:index], "cumulative", 0
    kind, _, arg = spec.partition(":")
    if kind == "full":
        k = int(arg)
        return [windows[index - j] for j in range(1, k + 1) if index - j >= 0], "full", 0
    if kind == "uncertain":
        return ([windows[index - 1]] if index >= 1 else []), "uncertain", int(arg)
    if kind == "uncertain-balanced":
        prev, two = (int(x) for x in arg.split("+"))
        tags, budgets = [], []
        if index >= 1:
            tags.append(windows[index - 1]); budgets.append(prev)
        if index >= 2:
            tags.append(windows[index - 2]); budgets.append(two)
        return tags, "uncertain-balanced", budgets
    raise ValueError(f"unknown replay spec: {spec}")
