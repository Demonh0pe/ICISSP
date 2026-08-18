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
# Resumable: anything already in runs/tune/metrics.csv is skipped, so
# re-launching after a disconnect picks up where it stopped.
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

done_already() {
  [ -f "$METRICS" ] || return 1
  python - "$METRICS" "$1" "$2" "$3" <<'EOF'
import csv, sys
path, method, model, seed = sys.argv[1:5]
with open(path, newline="") as fh:
    for row in csv.DictReader(fh):
        if (row.get("method") == method and row.get("model") == model
                and row.get("seed") == seed):
            sys.exit(0)
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
