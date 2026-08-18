#!/usr/bin/env bash
# Does the replay buffer's budget explain why Hybrid-CASR does not lead on Qwen?
#
# The published buffer (100 from t-1, 25 from t-2) was fixed on phi-2 and never
# re-tuned. Every other hyperparameter had to stay pinned so that backbone
# identity was the only variable -- which also means the second backbone ran at
# the first one's budget. This script is how that gets tested.
#
# Only the buffer changes, which is what makes this cheap: window-only has no
# buffer, so the existing three-seed baseline in runs/swap/metrics.csv stays
# valid and does not need re-running. One run here buys one comparison.
#
#   bash run_tune.sh                  # the ladder at seed 42
#   SEEDS="43 44" bash run_tune.sh b2x   # add seeds to one variant
#
# Resumable at run granularity: a run whose full forward sweep is already in
# runs/tune/metrics.csv is skipped. A run that died part-way restarts from its
# first window -- train.py keeps no mid-run checkpoint, and a partial sweep is
# not a result, so finishing it means starting it again.
#
# Both backbones are cached after the first run, but transformers still calls
# huggingface.co to revalidate on every launch, which hangs on a slow link.
# Export HF_HUB_OFFLINE=1 to skip that call once the weights are local.
#
# Read the probe before spending more. Three outcomes, three decisions:
#   delta goes positive  -> add seeds 43/44, it becomes a result
#   delta goes to ~0     -> add seeds 43/44, "indistinguishable" gets stronger
#   delta goes more      -> stop. That is a clean answer too: the deficit is
#   negative                not the buffer being too small, and one sentence in
#                           the threats section says so with evidence.

set -uo pipefail

DATA="${DATA:-data/splits_faithful}"
OUT="${OUT:-runs/tune}"
QWEN="${QWEN:-Qwen/Qwen2.5-Coder-1.5B}"
SEEDS="${SEEDS:-42}"
ONLY="${1:-}"

# Same settings as run_swap.sh. They have to match or the new numbers cannot be
# put next to the old ones.
COMMON=(--data-dir "$DATA" --out "$OUT" --group-by-length --dtype fp32 --quiet)

# buffer spec | tag | what it asks
RUNS=(
  "uncertain-balanced:200+50|b2x|double the budget: is 125 samples simply too few here"
  "uncertain-balanced:400+100|b4x|quadruple it: does the trend continue or turn over"
)

mkdir -p "$OUT"
METRICS="$OUT/metrics.csv"

# Skip only a run that FINISHED. Keying on "has any row" would strand a run
# that died in window 3: it would be skipped on every relaunch and never
# complete. The bar is the full forward sweep -- one evaluation per window
# transition, i.e. one fewer than the number of window files.
done_already() {
  [ -f "$METRICS" ] || return 1
  python - "$METRICS" "$1" "$2" "$3" "$DATA" <<'EOF'
import csv, glob, os, sys
path, method, model, seed, data_dir = sys.argv[1:6]
want = len(glob.glob(os.path.join(data_dir, "*.jsonl"))) - 1
if want < 1:
    print(f"  cannot count windows in {data_dir}; not skipping")
    sys.exit(1)
have = 0
with open(path, newline="") as fh:
    seen = set()
    for row in csv.DictReader(fh):
        if (row.get("method") == method and row.get("model") == model
                and row.get("seed") == seed and row.get("direction") == "forward"):
            seen.add(row.get("eval_window"))
    have = len(seen)
if have >= want:
    sys.exit(0)
if have:
    print(f"  {have}/{want} windows present -- restarting this run from the top")
sys.exit(1)
EOF
}

model_tag="${QWEN##*/}"
ran=0

for entry in "${RUNS[@]}"; do
  IFS='|' read -r replay tag why <<< "$entry"
  [ -n "$ONLY" ] && [ "$ONLY" != "$tag" ] && continue

  for seed in $SEEDS; do
    echo "=============================================================="
    echo "hybrid-casr+$tag  seed $seed  replay=$replay"
    echo "  $why"
    echo "=============================================================="

    if done_already "hybrid-casr+$tag" "$model_tag" "$seed"; then
      echo "  already in $METRICS -- skipping"
      continue
    fi

    start=$(date +%s)
    python -u experiments/train.py --method hybrid-casr --model "$QWEN" \
        --replay "$replay" --tag "$tag" --seed "$seed" "${COMMON[@]}"
    status=$?
    mins=$(( ($(date +%s) - start) / 60 ))
    ran=$((ran + 1))

    if [ $status -ne 0 ]; then
      echo "  FAILED after ${mins}m (exit $status) -- continuing"
      echo "hybrid-casr+$tag seed=$seed exit=$status" >> "$OUT/failures.txt"
    else
      echo "  done in ${mins}m"
    fi
  done
done

echo
echo "=============================================================="
if [ "$ran" -eq 0 ]; then
  echo "Nothing to run -- every requested run is already in $METRICS."
else
  echo "Runs finished. Compare against the untuned runs:"
fi
echo
echo "  python analysis/aggregate_runs.py \\"
echo "      --metrics runs/swap/metrics.csv $METRICS \\"
echo "      --out figures/tune --metric f1_binary_pos1_LEGACY"
echo
echo "The tuned variants appear as hybrid-casr+b2x / +b4x, each compared"
echo "against window-only on the same backbone. What matters is whether their"
echo "delta is less negative than the untuned -0.0056, and whether it clears"
echo "the seed spread (0.0066) rather than just changing sign."
[ -f "$OUT/failures.txt" ] && echo && echo "Some runs failed:" && cat "$OUT/failures.txt"
echo "=============================================================="
