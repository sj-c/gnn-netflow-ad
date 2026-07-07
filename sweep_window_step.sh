#!/usr/bin/env bash
#
# Sweep window-size x step-size combinations for train_and_infer.py and collect
# each run's comparison table into results_infer/window_step/.
#
# Grid:  window in {500, 1000, 2000},  step in {window, window/2}
# Fixed: --epochs 200 --patience 20
#
# Every combo trains into its own output/checkpoint/log dir so runs never clobber
# each other, then its comparison.csv is copied out with a window/step-tagged name.
# --resume is passed through, so re-running the sweep skips combos already done.
#
# Usage:
#   ./sweep_window_step.sh                          # full sweep, all datasets
#   WINDOWS="500 1000" ./sweep_window_step.sh
#   ./sweep_window_step.sh --datasets NF-ToN-IoT-v3 # extra args pass through
set -euo pipefail

cd "$(dirname "$0")"

# --- sweep grid (override via env) ----------------------------------------
WINDOWS="${WINDOWS:- 1000 2000}"
EPOCHS="${EPOCHS:-200}"
PATIENCE="${PATIENCE:-20}"
PYTHON="${PYTHON:-python}"

# --- layout ---------------------------------------------------------------
RESULTS_ROOT="results_infer/window_step"
RUNS_DIR="$RESULTS_ROOT/runs"
CKPT_ROOT="$RESULTS_ROOT/checkpoints"
LOG_ROOT="$RESULTS_ROOT/logs"
SUMMARY_CSV="$RESULTS_ROOT/sweep_summary.csv"
mkdir -p "$RESULTS_ROOT" "$RUNS_DIR" "$CKPT_ROOT" "$LOG_ROOT"

# Any extra CLI args (e.g. --datasets ...) are forwarded to train_and_infer.py.
EXTRA_ARGS=("$@")

echo "Window values: $WINDOWS"
echo "Step values:   {window, window/2}"
echo "Epochs:        $EPOCHS   Patience: $PATIENCE"
echo "Results dir:   $RESULTS_ROOT"
echo "Extra args:    ${EXTRA_ARGS[*]:-<none>}"
echo

for win in $WINDOWS; do
  # step in {window, window/2}; dedupe when window/2 == window (never here, but safe)
  for step in "$win" "$((win / 2))"; do
    tag="w${win}_s${step}"
    out_dir="$RUNS_DIR/$tag"
    ckpt_dir="$CKPT_ROOT/$tag"
    log_dir="$LOG_ROOT/$tag"
    mkdir -p "$out_dir" "$ckpt_dir" "$log_dir"

    echo "===== $tag ====="
    "$PYTHON" train_and_infer.py \
      --window-size "$win" \
      --step-size "$step" \
      --epochs "$EPOCHS" \
      --patience "$PATIENCE" \
      --output-dir "$out_dir" \
      --checkpoint-dir "$ckpt_dir" \
      --log-dir "$log_dir" \
      --resume \
      "${EXTRA_ARGS[@]}"

    if [[ -f "$out_dir/comparison.csv" ]]; then
      cp "$out_dir/comparison.csv" "$RESULTS_ROOT/comparison_${tag}.csv"
      echo "saved -> $RESULTS_ROOT/comparison_${tag}.csv"
    else
      echo "WARN: no comparison.csv produced for $tag" >&2
    fi
    echo
  done
done

# --- aggregate every combo's comparison.csv into one ranked summary -------
echo "Building sweep summary -> $SUMMARY_CSV"
"$PYTHON" - "$RUNS_DIR" "$SUMMARY_CSV" <<'PY'
import sys, glob, os
import pandas as pd

runs_dir, summary_csv = sys.argv[1], sys.argv[2]
rows = []
for path in sorted(glob.glob(os.path.join(runs_dir, "*", "comparison.csv"))):
    tag = os.path.basename(os.path.dirname(path))  # w<win>_s<step>
    win = tag.split("_s")[0].replace("w", "")
    step = tag.split("_s")[1]
    df = pd.read_csv(path)
    row = {"tag": tag, "window": int(win), "step": int(step)}
    for metric in ("f1", "precision", "recall", "auc_roc", "pr_auc"):
        if metric in df.columns:
            row[f"mean_{metric}"] = df[metric].mean()
    row["n_datasets"] = len(df)
    rows.append(row)

if not rows:
    print("No comparison.csv files found; nothing to summarize.")
    sys.exit(0)

sort_keys = [k for k in ("mean_f1", "mean_precision", "mean_recall") if k in rows[0]]
summary = pd.DataFrame(rows).sort_values(sort_keys, ascending=False)
summary.to_csv(summary_csv, index=False)
print("\n=== Sweep summary (ranked by mean F1 across datasets) ===")
print(summary.to_string(index=False))
best = summary.iloc[0]
print(
    f"\nBest combo: {best['tag']}  "
    f"mean_f1={best['mean_f1']:.4f} "
    f"mean_precision={best['mean_precision']:.4f} "
    f"mean_recall={best['mean_recall']:.4f}"
)
PY
