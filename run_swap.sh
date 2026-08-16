#!/usr/bin/env bash
# Cross-architecture run set: does the continual-learning ranking hold on a
# different backbone?
#
# Ordered by what each run buys, so stopping early still leaves something
# publishable. Runs are sequential -- two of these will not fit on one card.
#
#   bash run_swap.sh                 # everything, in priority order
#   bash run_swap.sh 3               # only the first 3 runs
#
# Resumable: a run whose metrics are already in runs/swap/metrics.csv is
# skipped, so re-launching after a disconnect picks up where it stopped.
#
# Settings are fixed here rather than passed in, because the comparison is only
# valid if every run shares them:
#   fp32              bf16 is ~1.5x faster but moved probe windows by up to 0.08,
#                     which would confound precision with the architecture change
#   group-by-length   ~1.95x faster, moved the same windows by 0.005 -- noise
#   batch 32, 3 epochs, lr 2e-4, LoRA r=16 alpha=32  as published

set -uo pipefail

DATA="${DATA:-data/splits_faithful}"
OUT="${OUT:-runs/swap}"
PHI="${PHI:-microsoft/phi-2}"
QWEN="${QWEN:-Qwen/Qwen2.5-Coder-1.5B}"
LIMIT="${1:-99}"

COMMON=(--data-dir "$DATA" --out "$OUT" --group-by-length --dtype fp32 --quiet)

# model | method | why it earns its slot
RUNS=(
  "$PHI|window-only|anchor: does the rebuilt data reproduce the published baseline"
  "$QWEN|window-only|the same baseline on the new backbone -- the comparison itself"
  "$QWEN|hybrid-casr|does the proposed method still lead on a different backbone"
  "$QWEN|olora|worst performer as published; is that architecture-independent"
  "$QWEN|replay-3p|second worst; same question"
  "$QWEN|casr|separates the class-balancing half of hybrid-casr"
  "$QWEN|replay-1p|best backward retention as published"
  "$QWEN|lbcl|completes the table"
)

mkdir -p "$OUT"
METRICS="$OUT/metrics.csv"

done_already() {
  # Keyed on method AND model: the same method runs under both backbones, and
  # train.py records the model tag in its own column for exactly this reason.
  [ -f "$METRICS" ] || return 1
  local tag="${1##*/}"
  python - "$METRICS" "$2" "$tag" <<'EOF'
import csv, sys
path, method, model = sys.argv[1:4]
with open(path, newline="") as fh:
    for row in csv.DictReader(fh):
        if row.get("method") == method and row.get("model") == model:
            sys.exit(0)
sys.exit(1)
EOF
}

i=0
for entry in "${RUNS[@]}"; do
  i=$((i + 1))
  [ "$i" -gt "$LIMIT" ] && break
  IFS='|' read -r model method why <<< "$entry"

  echo "=============================================================="
  echo "[$i/${#RUNS[@]}] $method on $model"
  echo "  $why"
  echo "=============================================================="

  if done_already "$model" "$method"; then
    echo "  already in $METRICS -- skipping"
    continue
  fi

  start=$(date +%s)
  python -u experiments/train.py --method "$method" --model "$model" "${COMMON[@]}"
  status=$?
  mins=$(( ($(date +%s) - start) / 60 ))

  if [ $status -ne 0 ]; then
    echo "  FAILED after ${mins}m (exit $status) -- continuing with the next run"
    echo "$method $model exit=$status" >> "$OUT/failures.txt"
  else
    echo "  done in ${mins}m"
  fi
done

echo
echo "=============================================================="
echo "Runs finished. Results:"
echo "  python analysis/aggregate_runs.py --metrics $METRICS --out figures/ \\"
echo "      --metric f1_binary_pos1_LEGACY"
echo
echo "f1_binary_pos1_LEGACY is the conference metric, so the new table lines up"
echo "with the published one. Drop --metric for the corrected macro average."
[ -f "$OUT/failures.txt" ] && echo && echo "Some runs failed:" && cat "$OUT/failures.txt"
echo "=============================================================="
