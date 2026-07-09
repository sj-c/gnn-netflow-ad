#!/usr/bin/env bash
#
# Seed-robustness sweep for the --personalised model (M2: single stage, one
# shared encoder + one personalised head per dataset): run 3 seeds of the
# joint training, then report mean +/- std per dataset across seeds.
#
# Every run trains into its own output/checkpoint/log dir so runs never
# clobber each other; --resume is passed through so re-running the sweep
# skips seeds already done.
#
# Usage:
#   ./sweep_seeds_personalised.sh
#   SEEDS="42 43 44 45" ./sweep_seeds_personalised.sh
#   ./sweep_seeds_personalised.sh --window-size 2000   # extra args pass through to every run
set -euo pipefail

cd "$(dirname "$0")"

# --- sweep grid (override via env) ----------------------------------------
SEEDS="${SEEDS:-42 43 44}"
DATASETS="${DATASETS:-NF-BoT-IoT-v3 NF-UNSW-NB15-v3 NF-ToN-IoT-v3 NF-CICIDS2018-v3}"
PYTHON="${PYTHON:-python}"

# --- layout ---------------------------------------------------------------
RESULTS_ROOT="${RESULTS_ROOT:-results_infer/seed_sweep_personalised}"
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

# --- 3 personalised (joint single-stage) runs -------------------------------
for seed in $SEEDS; do
  tag="personalised_seed${seed}"
  out_dir="$RUNS_DIR/$tag"
  ckpt_dir="$CKPT_ROOT/$tag"
  log_dir="$LOG_ROOT/$tag"
  mkdir -p "$out_dir" "$ckpt_dir" "$log_dir"

  echo "===== $tag ====="
  "$PYTHON" train_and_infer.py \
    --personalised \
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

# personalised_seed<seed>/personalised/comparison.csv -> one row per dataset
for path in sorted(glob.glob(os.path.join(runs_dir, "personalised_*", "personalised", "comparison.csv"))):
    tag = os.path.basename(os.path.dirname(os.path.dirname(path)))
    m = re.fullmatch(r"personalised_seed(\d+)", tag)
    if not m:
        continue
    seed = int(m.group(1))
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        rows.append({"model": "personalised", "dataset": r["dataset"], "seed": seed,
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

dataset_order = ["NF-BoT-IoT-v3", "NF-UNSW-NB15-v3", "NF-ToN-IoT-v3", "NF-CICIDS2018-v3"]
agg["__order"] = agg["dataset"].apply(lambda d: dataset_order.index(d) if d in dataset_order else len(dataset_order))
agg = agg.sort_values("__order").drop(columns="__order")
agg.to_csv(summary_csv, index=False)

print("\n=== Per-run results ===")
print(per_run.sort_values(["dataset", "seed"]).to_string(index=False))

print("\n=== Mean +/- std across seeds (personalised) ===")
for m in METRICS:
    print(f"\n--- {m} ---")
    view = agg[["dataset"]].copy()
    view[m] = agg.apply(lambda r: f"{r[f'{m}_mean']:.4f} +/- {r[f'{m}_std']:.4f}", axis=1)
    print(view.to_string(index=False))

print(f"\nSaved per-run table to {per_run_csv}")
print(f"Saved aggregated summary to {summary_csv}")
PY
