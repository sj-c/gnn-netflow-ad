#!/usr/bin/env bash
#
# Sweep model-architecture combinations for train_and_infer.py and collect each
# run's comparison table into results_infer/arch/.
#
# Grid:  hidden-dim in {128, 256, 512}
#        heads      in {2, 4, 8}
#        global-emb-dim in {64, 128, 256}
#
# Every combo trains into its own output/checkpoint/log dir so runs never clobber
# each other, then its comparison.csv is copied out with an arch-tagged name.
# --resume is passed through, so re-running the sweep skips combos already done.
#
# Usage:
#   ./sweep_arch.sh                                 # full 3x3x3 sweep, all datasets
#   HIDDENS="128 256" HEADS="4" GDIMS="128" ./sweep_arch.sh
#   ./sweep_arch.sh --datasets NF-ToN-IoT-v3        # extra args pass through
set -euo pipefail

cd "$(dirname "$0")"

# --- sweep grid (override via env) ----------------------------------------
HIDDENS="${HIDDENS:- 512}"
HEADS="${HEADS:-2 4 8}"
GDIMS="${GDIMS:-64 128 256}"
PYTHON="${PYTHON:-python}"

# --- layout ---------------------------------------------------------------
RESULTS_ROOT="results_infer/arch"
RUNS_DIR="$RESULTS_ROOT/runs"
CKPT_ROOT="$RESULTS_ROOT/checkpoints"
LOG_ROOT="$RESULTS_ROOT/logs"
SUMMARY_CSV="$RESULTS_ROOT/sweep_summary.csv"
mkdir -p "$RESULTS_ROOT" "$RUNS_DIR" "$CKPT_ROOT" "$LOG_ROOT"

# Any extra CLI args (e.g. --datasets ...) are forwarded to train_and_infer.py.
EXTRA_ARGS=("$@")

echo "Hidden dims:   $HIDDENS"
echo "Heads:         $HEADS"
echo "Global emb:    $GDIMS"
echo "Results dir:   $RESULTS_ROOT"
echo "Extra args:    ${EXTRA_ARGS[*]:-<none>}"
echo

for hid in $HIDDENS; do
  for head in $HEADS; do
    for gdim in $GDIMS; do
      tag="h${hid}_nh${head}_g${gdim}"
      out_dir="$RUNS_DIR/$tag"
      ckpt_dir="$CKPT_ROOT/$tag"
      log_dir="$LOG_ROOT/$tag"
      mkdir -p "$out_dir" "$ckpt_dir" "$log_dir"

      echo "===== $tag ====="
      "$PYTHON" train_and_infer.py \
        --hidden-dim "$hid" \
        --heads "$head" \
        --global-emb-dim "$gdim" \
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
done

# --- aggregate every combo's comparison.csv into one ranked summary -------
echo "Building sweep summary -> $SUMMARY_CSV"
"$PYTHON" - "$RUNS_DIR" "$SUMMARY_CSV" <<'PY'
import sys, glob, os
import pandas as pd

runs_dir, summary_csv = sys.argv[1], sys.argv[2]
rows = []
for path in sorted(glob.glob(os.path.join(runs_dir, "*", "comparison.csv"))):
    tag = os.path.basename(os.path.dirname(path))  # h<hid>_nh<heads>_g<gdim>
    hid = tag.split("_nh")[0].replace("h", "")
    head = tag.split("_nh")[1].split("_g")[0]
    gdim = tag.split("_g")[-1]
    df = pd.read_csv(path)
    row = {"tag": tag, "hidden_dim": int(hid), "heads": int(head), "global_emb_dim": int(gdim)}
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
