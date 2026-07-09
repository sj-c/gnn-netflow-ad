#!/usr/bin/env bash
#
# Seed-robustness sweep: default configuration (top-k-protocols=4, baseline
# edge columns, etc. -- whatever train_and_infer.py's defaults are), run 3
# seeds for each of the 4 individually-trained per-dataset models plus 3
# seeds of the --combined pooled model. 4*3 + 1*3 = 15 runs total.
#
# Purpose: earlier single-seed comparisons (individual vs combined, top-k
# sweeps) couldn't tell real effects apart from run-to-run training noise.
# This sweep exists to put mean +/- std around both the per-dataset numbers
# and the individual-vs-combined comparison.
#
# Every run trains into its own output/checkpoint/log dir so runs never
# clobber each other; --resume is passed through so re-running the sweep
# skips combos already done.
#
# Usage:
#   ./sweep_seeds.sh
#   SEEDS="42 43 44 45" ./sweep_seeds.sh
#   ./sweep_seeds.sh --window-size 2000     # extra args pass through to every run
set -euo pipefail

cd "$(dirname "$0")"

# --- sweep grid (override via env) ----------------------------------------
SEEDS="${SEEDS:-42 43 44}"
DATASETS="${DATASETS:-NF-BoT-IoT-v3 NF-UNSW-NB15-v3 NF-ToN-IoT-v3 NF-CICIDS2018-v3}"
PYTHON="${PYTHON:-python}"

# --- layout ---------------------------------------------------------------
RESULTS_ROOT="${RESULTS_ROOT:-results_infer/seed_sweep}"
RUNS_DIR="$RESULTS_ROOT/runs"
CKPT_ROOT="$RESULTS_ROOT/checkpoints"
LOG_ROOT="$RESULTS_ROOT/logs"
SUMMARY_CSV="$RESULTS_ROOT/sweep_summary.csv"
PER_RUN_CSV="$RESULTS_ROOT/per_run.csv"
mkdir -p "$RESULTS_ROOT" "$RUNS_DIR" "$CKPT_ROOT" "$LOG_ROOT"

# Any extra CLI args (e.g. --window-size 2000) are forwarded to every run.
EXTRA_ARGS=("$@")

echo "seeds:         $SEEDS"
echo "datasets:      $DATASETS"
echo "Results dir:   $RESULTS_ROOT"
echo "Extra args:    ${EXTRA_ARGS[*]:-<none>}"
echo

# --- 12 individual per-dataset runs ----------------------------------------
for ds in $DATASETS; do
  for seed in $SEEDS; do
    tag="individual_${ds}_seed${seed}"
    out_dir="$RUNS_DIR/$tag"
    ckpt_dir="$CKPT_ROOT/$tag"
    log_dir="$LOG_ROOT/$tag"
    mkdir -p "$out_dir" "$ckpt_dir" "$log_dir"

    echo "===== $tag ====="
    "$PYTHON" train_and_infer.py \
      --datasets "$ds" \
      --seed "$seed" \
      --output-dir "$out_dir" \
      --checkpoint-dir "$ckpt_dir" \
      --log-dir "$log_dir" \
      --resume \
      "${EXTRA_ARGS[@]}"
    echo
  done
done

# --- 3 combined (pooled) runs ----------------------------------------------
for seed in $SEEDS; do
  tag="combined_seed${seed}"
  out_dir="$RUNS_DIR/$tag"
  ckpt_dir="$CKPT_ROOT/$tag"
  log_dir="$LOG_ROOT/$tag"
  mkdir -p "$out_dir" "$ckpt_dir" "$log_dir"

  echo "===== $tag ====="
  "$PYTHON" train_and_infer.py \
    --combined \
    --datasets "$(echo "$DATASETS" | tr ' ' ',')" \
    --seed "$seed" \
    --output-dir "$out_dir" \
    --checkpoint-dir "$ckpt_dir" \
    --log-dir "$log_dir" \
    --resume \
    "${EXTRA_ARGS[@]}"
  echo
done

# --- aggregate everything into one neat mean +/- std summary ---------------
echo "Building sweep summary -> $SUMMARY_CSV"
"$PYTHON" - "$RUNS_DIR" "$SUMMARY_CSV" "$PER_RUN_CSV" <<'PY'
import sys, glob, os, re
import pandas as pd

runs_dir, summary_csv, per_run_csv = sys.argv[1], sys.argv[2], sys.argv[3]
METRICS = ("auc_roc", "pr_auc", "precision", "recall", "f1", "fpr")

rows = []

# individual_<dataset>_seed<seed>/comparison.csv -> single row for that dataset
for path in sorted(glob.glob(os.path.join(runs_dir, "individual_*", "comparison.csv"))):
    tag = os.path.basename(os.path.dirname(path))
    m = re.fullmatch(r"individual_(.+)_seed(\d+)", tag)
    if not m:
        continue
    dataset, seed = m.group(1), int(m.group(2))
    df = pd.read_csv(path)
    df = df[df["dataset"] == dataset]
    if df.empty:
        continue
    r = df.iloc[0].to_dict()
    rows.append({"model": "individual", "dataset": dataset, "seed": seed,
                 **{k: r.get(k) for k in METRICS}})

# combined_seed<seed>/combined/comparison.csv -> one row per dataset + ALL(pooled)
for path in sorted(glob.glob(os.path.join(runs_dir, "combined_*", "combined", "comparison.csv"))):
    tag = os.path.basename(os.path.dirname(os.path.dirname(path)))
    m = re.fullmatch(r"combined_seed(\d+)", tag)
    if not m:
        continue
    seed = int(m.group(1))
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        rows.append({"model": "combined", "dataset": r["dataset"], "seed": seed,
                      **{k: r.get(k) for k in METRICS}})

if not rows:
    print("No comparison.csv files found; nothing to summarize.")
    sys.exit(0)

per_run = pd.DataFrame(rows)
per_run.to_csv(per_run_csv, index=False)

agg = (per_run.groupby(["dataset", "model"])[list(METRICS)]
       .agg(["mean", "std"]).round(4))
agg.columns = ["_".join(c) for c in agg.columns]
agg["n_seeds"] = per_run.groupby(["dataset", "model"]).size()
agg = agg.reset_index()

# order: each dataset's individual/combined pair together, ALL(pooled) last
dataset_order = ["NF-BoT-IoT-v3", "NF-UNSW-NB15-v3", "NF-ToN-IoT-v3", "NF-CICIDS2018-v3", "ALL(pooled)"]
agg["__order"] = agg["dataset"].apply(lambda d: dataset_order.index(d) if d in dataset_order else len(dataset_order))
agg = agg.sort_values(["__order", "model"]).drop(columns="__order")
agg.to_csv(summary_csv, index=False)

print("\n=== Per-run results (15 runs) ===")
print(per_run.sort_values(["dataset", "model", "seed"]).to_string(index=False))

print("\n=== Mean +/- std across seeds (individual vs combined) ===")
display_cols = ["dataset", "model", "n_seeds"]
for m in METRICS:
    print(f"\n--- {m} ---")
    view = agg[["dataset", "model"]].copy()
    view[m] = agg.apply(lambda r: f"{r[f'{m}_mean']:.4f} +/- {r[f'{m}_std']:.4f}", axis=1)
    print(view.to_string(index=False))

print(f"\nSaved per-run table to {per_run_csv}")
print(f"Saved aggregated summary to {summary_csv}")
PY
