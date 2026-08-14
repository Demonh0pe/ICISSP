"""Turn downloaded commit patches into temporal window JSONL files.

Two modes, because the published pipeline and the published *description* of
that pipeline are not the same thing.

--mode faithful
    Reproduces vuln_LLM(phi2)_2morth.py exactly: every '-' line in the patch is
    concatenated into one VULNERABLE sample, every '+' line into one FIXED
    sample, across all files and all hunks of the commit, with context lines
    discarded. No deduplication, no language filter. Use this to rebuild the
    dataset behind the conference numbers.

--mode hunk  (default)
    One sample per hunk per file, reconstructing the hunk's before-image
    (context + removed lines) and after-image (context + added lines). This is
    still not whole functions -- a patch does not contain them -- but it is
    contiguous code with its surrounding context rather than a bag of changed
    lines, so the label cannot be read off "was this line added or removed".

Why the distinction matters: under `faithful`, a VULNERABLE sample is by
construction the set of deleted lines and a FIXED sample the set of added lines
from the same commit. Those two populations differ in surface form regardless
of whether anything is a vulnerability, so a classifier can score well on a cue
that has nothing to do with security.

Deduplication and language filtering default to on in hunk mode and off in
faithful mode, matching what each pipeline actually did. The paper describes
both as applied; neither is in the committed code.

Usage:
    python data/build_from_patches.py --patches data/patches \
        --nvd-dir data/nvd/cleaned --out data/splits_hunk

    # rebuild the conference dataset
    python data/build_from_patches.py --patches data/patches \
        --nvd-dir data/nvd/cleaned --out data/splits_faithful --mode faithful
"""

import argparse
import collections
import glob
import hashlib
import json
import os
import re
import sys

PROMPT_PREFIX = "Please classify the following function as VULNERABLE or FIXED:\n"
CVE_RE = re.compile(r"(CVE-\d{4}-\d+)")
DIFF_HEADER_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)$")
HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@")
_WS = re.compile(r"\s+")

# Extension -> language, for the C/C++ filter the paper describes.
LANG_BY_EXT = {
    ".c": "c", ".h": "c", ".cc": "c++", ".cpp": "c++", ".cxx": "c++", ".hpp": "c++",
    ".hh": "c++", ".hxx": "c++", ".java": "java", ".py": "python", ".js": "javascript",
    ".ts": "typescript", ".go": "go", ".rb": "ruby", ".php": "php", ".rs": "rust",
    ".cs": "c#", ".swift": "swift", ".kt": "kotlin", ".scala": "scala", ".m": "objective-c",
}


def language_of(path):
    return LANG_BY_EXT.get(os.path.splitext(path)[1].lower(), "unknown")


def load_dates(nvd_dir):
    """cve_id -> YYYY-MM-DD disclosure date, from NVD 1.1 feeds.

    Read from each record's publishedDate. The feeds are named by CVE ID year,
    which is not the disclosure year -- nvdcve-1.1-2008.json holds entries
    disclosed as late as 2023 -- so the filename is never used as the date.
    """
    dates = {}
    files = sorted(glob.glob(os.path.join(nvd_dir, "**", "*.json"), recursive=True))
    print(f"parsing {len(files)} NVD feed file(s) -- about a minute for the full set")
    for n, path in enumerate(files, 1):
        print(f"  [{n}/{len(files)}] {os.path.basename(path)}", flush=True)
        try:
            blob = json.load(open(path, encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        for item in blob.get("CVE_Items", []):
            try:
                cid = item["cve"]["CVE_data_meta"]["ID"]
                pub = item.get("publishedDate", "").split("T")[0]
            except (KeyError, TypeError):
                continue
            if pub and (cid not in dates or pub < dates[cid]):
                dates[cid] = pub
    print(f"{len(dates)} disclosure dates from {len(files)} feed file(s)")
    return dates


def parse_faithful(lines):
    """The original extract_functions_from_patch, unchanged in behaviour."""
    old, new = [], []
    for line in lines:
        if line.startswith("---") or line.startswith("+++") or line.startswith("@@"):
            continue
        if line.startswith("-") and not line.startswith("--"):
            old.append(line[1:].rstrip())
        elif line.startswith("+") and not line.startswith("++"):
            new.append(line[1:].rstrip())
    return [("\n".join(old), "\n".join(new), None)] if (old or new) else []


def parse_hunks(lines):
    """Per-file, per-hunk (before_image, after_image, path).

    Context lines appear in both images, so the two differ only where the commit
    changed something -- which is the point.
    """
    out = []
    path = None
    before, after, changed = [], [], False

    def flush():
        if changed and (before or after):
            out.append(("\n".join(before), "\n".join(after), path))
        before.clear(); after.clear()

    for raw in lines:
        line = raw.rstrip("\n")
        header = DIFF_HEADER_RE.match(line)
        if header:
            flush(); changed = False
            path = header.group(2)
            continue
        if HUNK_RE.match(line):
            flush(); changed = False
            continue
        if line.startswith("--- ") or line.startswith("+++ ") or line.startswith("index "):
            continue
        if not line:
            continue
        tag, body = line[0], line[1:]
        if tag == "-":
            before.append(body); changed = True
        elif tag == "+":
            after.append(body); changed = True
        elif tag == " ":
            before.append(body); after.append(body)
        # Anything else (\ No newline at end of file, commit message prose
        # before the first diff header) is not part of a hunk.
    flush()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patches", required=True)
    ap.add_argument("--nvd-dir", required=True, help="directory of NVD feeds (cleaned or raw)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["hunk", "faithful"], default="hunk")
    ap.add_argument("--granularity", type=int, default=2)
    ap.add_argument("--start-year", type=int, default=2015)
    ap.add_argument("--end-year", type=int, default=2024)
    ap.add_argument("--languages", default="",
                    help="comma-separated (e.g. 'c,c++'); empty means keep all. "
                         "Only meaningful in hunk mode, where file paths are known.")
    ap.add_argument("--dedup", dest="dedup", action="store_true", default=None)
    ap.add_argument("--no-dedup", dest="dedup", action="store_false")
    ap.add_argument("--min-code-chars", type=int, default=0,
                    help="drop samples whose code (excluding the prompt prefix) is "
                         "shorter than this. The trainer's len(prompt)>10 filter is a "
                         "no-op because the prefix alone is 63 characters.")
    args = ap.parse_args()

    if 12 % args.granularity:
        sys.exit(f"--granularity must divide 12, got {args.granularity}")
    if args.dedup is None:
        args.dedup = (args.mode == "hunk")
    want_langs = {s.strip().lower() for s in args.languages.split(",") if s.strip()}
    if want_langs and args.mode == "faithful":
        print("note: --languages is ignored in faithful mode (the original pipeline "
              "merged all files of a commit into one sample, losing the paths)")
        want_langs = set()

    dates = load_dates(args.nvd_dir)
    files = sorted(glob.glob(os.path.join(args.patches, "*.patch")))
    print(f"{len(files)} patch files, mode={args.mode}, dedup={args.dedup}")
    if not files:
        sys.exit(f"no .patch files in {args.patches} -- run data/fetch_patches.py first")

    stats = collections.Counter()
    langs = collections.Counter()
    candidates = []

    for path in files:
        stats["patches"] += 1
        m = CVE_RE.search(os.path.basename(path))
        if not m:
            stats["dropped_no_cve_in_filename"] += 1
            continue
        cve_id = m.group(1)
        pub = dates.get(cve_id)
        if not pub:
            stats["dropped_no_date"] += 1
            continue
        with open(path, encoding="utf-8", errors="ignore") as fh:
            lines = fh.readlines()

        pieces = parse_faithful(lines) if args.mode == "faithful" else parse_hunks(lines)
        if not pieces:
            stats["dropped_empty_patch"] += 1
            continue

        for before, after, fpath in pieces:
            lang = language_of(fpath) if fpath else "unknown"
            if fpath:
                langs[lang] += 1
            if want_langs and lang not in want_langs:
                stats["dropped_language"] += 1
                continue
            for code, label in ((before, "VULNERABLE"), (after, "FIXED")):
                if not code.strip() or len(code.strip()) < args.min_code_chars:
                    stats["dropped_empty_side"] += 1
                    continue
                candidates.append((pub, code, label, cve_id))

    # Earliest first, so dedup keeps the first appearance along the timeline and
    # a later copy cannot pull a sample forward into a future window.
    candidates.sort(key=lambda t: t[0])
    kept = []
    if args.dedup:
        seen = {}
        for pub, code, label, cve in candidates:
            key = hashlib.sha1(_WS.sub(" ", code).strip().encode("utf-8", "replace")).hexdigest()
            if key in seen:
                if seen[key][0] != label:
                    seen[key] = (seen[key][0], True)
                    stats["dropped_contradictory"] += 1
                else:
                    stats["dropped_duplicate"] += 1
                continue
            seen[key] = (label, False)
            kept.append((pub, code, label, cve, key))
        bad = {k for k, v in seen.items() if v[1]}
        before_n = len(kept)
        kept = [t for t in kept if t[4] not in bad]
        stats["dropped_contradictory"] += before_n - len(kept)
    else:
        kept = [(p, c, l, cid, None) for p, c, l, cid in candidates]

    buckets = collections.defaultdict(list)
    for pub, code, label, cve, _ in kept:
        try:
            year, month = int(pub[:4]), int(pub[5:7])
        except (ValueError, IndexError):
            stats["dropped_bad_date"] += 1
            continue
        if not (args.start_year <= year <= args.end_year):
            stats["dropped_out_of_range"] += 1
            continue
        start = ((month - 1) // args.granularity) * args.granularity + 1
        tag = (f"{year}_M{start:02d}" if args.granularity == 1
               else f"{year}_M{start:02d}-{start + args.granularity - 1:02d}")
        buckets[tag].append({"prompt": PROMPT_PREFIX + code, "response": label,
                             "cve_id": cve, "published": pub})

    os.makedirs(args.out, exist_ok=True)
    counts = {}
    for tag, recs in sorted(buckets.items()):
        with open(os.path.join(args.out, f"{tag}.jsonl"), "w", encoding="utf-8") as fh:
            for rec in recs:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        counts[tag] = {"vulnerable": sum(r["response"] == "VULNERABLE" for r in recs),
                       "fixed": sum(r["response"] == "FIXED" for r in recs)}

    summary = {"mode": args.mode, "dedup": args.dedup, "languages": sorted(want_langs) or "all",
               "granularity_months": args.granularity,
               "dropped": {k: v for k, v in sorted(stats.items()) if k.startswith("dropped")},
               "patches_read": stats["patches"],
               "kept": sum(c["vulnerable"] + c["fixed"] for c in counts.values()),
               "languages_seen": dict(langs.most_common()), "windows": counts}
    json.dump(summary, open(os.path.join(args.out, "build_stats.json"), "w"),
              indent=1, ensure_ascii=False)

    print(f"\nkept {summary['kept']} samples from {stats['patches']} patches")
    for k, v in summary["dropped"].items():
        print(f"  {k:28s} {v}")
    print(f"\n{len(counts)} windows -> {args.out}")
    degenerate = [t for t, c in counts.items() if c["vulnerable"] == 0 or c["fixed"] == 0]
    if degenerate:
        print(f"WARNING: {len(degenerate)} single-class window(s): {', '.join(degenerate[:8])}")
    print("\nNext: python data/verify_dataset.py --data-dir " + args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
