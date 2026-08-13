"""Tests for the GPU-free half of the pipeline: timeline, metrics, replay resolution.

These cover the parts that decide what a run *means* -- which windows exist,
what feeds each training set, and which F1 gets reported -- so a mistake there
is caught here rather than after hours on a GPU.

    python experiments/test_common.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from common import (METHODS, MetricsWriter, compute_metrics,  # noqa: E402
                    make_windows, resolve_replay)

FAILURES = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILURES.append(msg)


def test_windows():
    print("[windows]")
    w = make_windows(2)
    check(len(w) == 42, f"bi-monthly 2018-2024 gives 42 windows (got {len(w)})")
    check(w[0] == "2018_M01-02", f"first tag matches the notebooks (got {w[0]})")
    check(w[-1] == "2024_M11-12", f"last tag matches the notebooks (got {w[-1]})")
    # Table 3 of the paper: N windows -> N-1 forward evaluations.
    for months, expected in [(1, 84), (2, 42), (3, 28), (6, 14), (12, 7)]:
        got = len(make_windows(months))
        check(got == expected, f"{months}-month granularity yields {expected} windows (got {got})")
    check(make_windows(1)[0] == "2018_M01", "monthly tags have no range suffix")
    try:
        make_windows(5)
        check(False, "granularity not dividing 12 is rejected")
    except ValueError:
        check(True, "granularity not dividing 12 is rejected")


def test_metrics():
    print("[metrics]")
    # 0 = VULNERABLE, 1 = FIXED, as the notebooks encode it.
    y_true = [0, 0, 0, 0, 1, 1, 1, 1]
    y_pred = [0, 0, 1, 1, 1, 1, 1, 0]
    m = compute_metrics(y_true, y_pred)
    # class 1: TP=3 FP=2 FN=1 -> P=0.6 R=0.75 F1=2/3
    # class 0: TP=2 FP=1 FN=2 -> P=2/3 R=0.5 F1=4/7
    check(abs(m["f1_fixed"] - 2 / 3) < 1e-9, "f1_fixed computed for label 1")
    check(abs(m["f1_vulnerable"] - 4 / 7) < 1e-9, "f1_vulnerable computed for label 0")
    check(abs(m["macro_f1"] - (2 / 3 + 4 / 7) / 2) < 1e-9, "macro_f1 averages both classes")
    check(abs(m["f1_binary_pos1_LEGACY"] - m["f1_fixed"]) < 1e-12,
          "legacy field reproduces the notebooks' number (binary F1 of FIXED)")
    check(m["macro_f1"] != m["f1_binary_pos1_LEGACY"],
          "macro and legacy differ -- the whole point of the correction")
    check(m["n_vulnerable"] == 4 and m["n_fixed"] == 4, "support counted per class")

    # A degenerate window: predicting one class must not crash or silently
    # produce NaN, since real windows do collapse this way.
    m2 = compute_metrics([0, 0, 1, 1], [1, 1, 1, 1])
    check(m2["f1_vulnerable"] == 0.0, "absent predicted class scores 0, not NaN")
    check(0 < m2["macro_f1"] < 1, "macro_f1 stays finite when a class is never predicted")

    m3 = compute_metrics([0, 1, 0, 1], [0, 1, 0, 1])
    check(m3["macro_f1"] == 1.0 and m3["accuracy"] == 1.0, "perfect prediction scores 1.0")


def test_replay():
    print("[replay resolution]")
    w = make_windows(2)
    check(resolve_replay("none", 5, w) == ([], "none", 0), "no replay yields nothing")

    tags, mode, _ = resolve_replay("full:1", 5, w)
    check(tags == [w[4]] and mode == "full", "replay-1p pulls the immediately previous window")

    tags, _, _ = resolve_replay("full:2", 5, w)
    check(tags == [w[4], w[3]], "replay-3p pulls the previous two windows")

    tags, _, _ = resolve_replay("full:2", 0, w)
    check(tags == [], "no history at index 0")
    tags, _, _ = resolve_replay("full:2", 1, w)
    check(tags == [w[0]], "only one window of history available at index 1")

    tags, mode, budget = resolve_replay("uncertain:100", 5, w)
    check(tags == [w[4]] and budget == 100, "casr budget parsed")

    tags, mode, budget = resolve_replay("uncertain-balanced:100+25", 5, w)
    check(tags == [w[4], w[3]] and budget == [100, 25],
          "hybrid-casr takes 100 from t-1 and 25 from t-2")
    tags, _, budget = resolve_replay("uncertain-balanced:100+25", 1, w)
    check(tags == [w[0]] and budget == [100], "hybrid-casr degrades to one window at index 1")

    tags, mode, _ = resolve_replay("cumulative", 5, w)
    check(tags == w[:5] and mode == "cumulative", "cumulative pulls all prior windows")


def test_method_registry():
    print("[method registry]")
    check(len(METHODS) == 9, f"nine strategies registered (got {len(METHODS)})")
    # The audit finding, pinned so it cannot regress unnoticed.
    fresh = {k for k, v in METHODS.items() if v["adapter"] == "fresh"}
    check(fresh == {"replay-3p", "olora"},
          f"replay-3p and olora are the non-continual pair (got {fresh})")
    check(METHODS["zero-shot"]["adapter"] == "none", "zero-shot trains nothing")
    for name, spec in METHODS.items():
        check(bool(spec.get("desc")), f"{name} documents itself")
        try:
            resolve_replay(spec["replay"], 5, make_windows(2))
        except Exception as e:
            check(False, f"{name} replay spec parses ({e})")
    documented = {k for k, v in METHODS.items() if v.get("paper_mismatch")}
    check(documented == {"replay-3p", "olora", "lbcl", "hybrid-casr"},
          f"code/paper mismatches recorded (got {documented})")


def test_writer():
    print("[metrics writer]")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "nested", "metrics.csv")
        w = MetricsWriter(path)
        w.write(method="m", seed=1, granularity=2, train_window="a", eval_window="b",
                direction="forward", **compute_metrics([0, 1], [0, 1]))
        w.close()
        MetricsWriter(path).close()  # reopening must not duplicate the header
        lines = open(path).read().strip().split("\n")
        check(len(lines) == 2, f"header written once across reopens (got {len(lines)} lines)")
        check(lines[0].split(",") == common.METRIC_FIELDS, "header matches METRIC_FIELDS")
        check("macro_f1" in lines[0] and "f1_binary_pos1_LEGACY" in lines[0],
              "both the corrected and legacy metrics are logged")


def main():
    for t in (test_windows, test_metrics, test_replay, test_method_registry, test_writer):
        t()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
