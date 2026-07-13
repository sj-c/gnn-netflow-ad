#!/usr/bin/env bash
#
# Train + evaluate the federated Guided GAE on GPU, using Flower's standard
# workflow API (SecAgg+-ready, no early stopping: every run is exactly --rounds
# rounds and the final-round global model is evaluated).
#
# Stage 1: `flwr run federated local-simulation-gpu` -- 4 clients (one per
#   NetFlow dataset) share one GPU (0.25 each). Writes
#   checkpoints_infer/federated.pt (+ checkpoints_infer/federated_private/ for
#   fedbn/fedrep) and results_infer/federated/history.json.
# Stage 2: evaluate_federated.py restores the final global model (+ per-client
#   private parameters) and scores it on every dataset's holdout (per-client
#   calibrated thresholds), writing
#   <OUTPUT_DIR>/seed_<seed>/federated/<dataset>/metrics.json + comparison.csv.
#
# Both stages are repeated once per seed (default 3 seeds). Afterwards the
# per-seed comparison.csv files are aggregated into mean +/- std tables at
# <OUTPUT_DIR>/federated_seed_avg.csv (+ federated_per_run.csv).
#
# Usage:
#   ./run_federated_gpu.sh \
#       --shared_weight=FedAvg|FedProx \      # how shared weights are learned
#       --personalisation=NA|FedBN|FedRep \   # which part stays private per client
#       --output_dir=results_infer \          # metrics root
#       --rounds=150 \                        # communication rounds (always all run)
#       --epoch=1 \                           # local epochs per round
#       --secagg=true|false \                 # SecAgg+ secure aggregation
#       --clip=8.0 \                          # SecAgg+ clipping range
#       [--dp_mode=off|central|local] \       # differential privacy mode
#       [--dp_sigma=1.0] \                    # DP noise multiplier (both modes)
#       [--dp_clip=10.0] \                    # DP L2 update-clipping norm C
#       [--dp_delta=1e-5] \                   # DP delta (fixed by convention)
#       [--mu=0.01] \                         # FedProx proximal term strength
#       [--seeds="42 43 44"] \                # seeds to train+eval, then average
#       [--run-config "key=value ..."]        # extra raw flwr run-config overrides
#
# Values are case-insensitive; every flag is optional (defaults come from
# federated/pyproject.toml [tool.flwr.app.config]). Examples:
#   ./run_federated_gpu.sh                                        # FedAvg, no secagg
#   ./run_federated_gpu.sh --shared_weight=FedProx --personalisation=FedBN \
#       --secagg=true --clip=8.0 --rounds=150 --epoch=1 --output_dir=results_fedprox_fedbn
#   CUDA_VISIBLE_DEVICES=1 ./run_federated_gpu.sh                # pin to GPU 1
#   ./run_federated_gpu.sh --run-config "max-train-rows=20000"   # smoke test
set -euo pipefail

# Wrap the whole body in a function so bash reads the entire script into memory
# before executing. The script lives on NFS and long training runs can outlast
# the file handle; without this, bash reading the next chunk mid-run dies with
# "error reading input file: Stale file handle".
main() {

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
FEDERATION="${FEDERATION:-local-simulation-gpu}"   # GPU-sharing federation
DEVICE="${DEVICE:-cuda}"                            # device for evaluation
OUTPUT_DIR="${OUTPUT_DIR:-results_infer}"           # root for metrics
                                                   # (-> $OUTPUT_DIR/seed_<seed>/federated/...)
SEEDS="${SEEDS:-42 43 44}"                          # seeds to train+eval, then average

# ---- CLI flags -> flwr run-config overrides --------------------------------
lower() { printf '%s' "$1" | tr '[:upper:]' '[:lower:]'; }

RUN_CONFIG=""
EXTRA_RUN_CONFIG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --shared_weight=*|--shared-weight=*)
      v="$(lower "${1#*=}")"
      case "$v" in fedavg|fedprox) ;; *) echo "ERROR: --shared_weight must be FedAvg or FedProx (got '${1#*=}')" >&2; exit 1;; esac
      RUN_CONFIG+=" shared-weight='$v'" ;;
    --personalisation=*|--personalization=*)
      v="$(lower "${1#*=}")"
      case "$v" in na|fedbn|fedrep) ;; *) echo "ERROR: --personalisation must be NA, FedBN or FedRep (got '${1#*=}')" >&2; exit 1;; esac
      RUN_CONFIG+=" personalisation='$v'" ;;
    --output_dir=*|--output-dir=*)
      OUTPUT_DIR="${1#*=}" ;;
    --rounds=*)
      RUN_CONFIG+=" num-server-rounds=${1#*=}" ;;
    --epoch=*|--epochs=*)
      RUN_CONFIG+=" local-epochs=${1#*=}" ;;
    --secagg=*|--secAgg=*|--secAGG=*)
      v="$(lower "${1#*=}")"
      case "$v" in true|false) ;; *) echo "ERROR: --secagg must be true or false (got '${1#*=}')" >&2; exit 1;; esac
      RUN_CONFIG+=" secagg=$v" ;;
    --clip=*)
      RUN_CONFIG+=" clipping-range=${1#*=}" ;;
    --mu=*)
      RUN_CONFIG+=" proximal-mu=${1#*=}" ;;
    --dp_mode=*|--dp-mode=*)
      v="$(lower "${1#*=}")"
      case "$v" in off|central|local) ;; *) echo "ERROR: --dp_mode must be off, central or local (got '${1#*=}')" >&2; exit 1;; esac
      RUN_CONFIG+=" dp-mode='$v'" ;;
    --dp_sigma=*|--dp-sigma=*)
      RUN_CONFIG+=" dp-noise-multiplier=${1#*=}" ;;
    --dp_clip=*|--dp-clip=*)
      RUN_CONFIG+=" dp-clipping-norm=${1#*=}" ;;
    --dp_delta=*|--dp-delta=*)
      RUN_CONFIG+=" dp-delta=${1#*=}" ;;
    --seeds=*)
      SEEDS="${1#*=}" ;;
    --run-config)
      shift
      EXTRA_RUN_CONFIG="$1" ;;
    --run-config=*)
      EXTRA_RUN_CONFIG="${1#*=}" ;;
    *)
      echo "ERROR: unknown flag '$1' (see the usage block at the top of this script)" >&2
      exit 1 ;;
  esac
  shift
done

# Expose the chosen GPU(s) to both the Ray-backed simulation and eval. Defaults
# to GPU 0 unless the caller already set CUDA_VISIBLE_DEVICES.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# `flwr run` copies the app into an isolated build dir under ~/.flwr/apps, so the
# fedgnn package can no longer find the repo-root `src`/`train_and_infer` modules
# by walking up from its own location. The ServerApp/ClientApp are spawned by a
# persistent flower-superlink daemon whose environment predates this script, so
# env vars (and PYTHONPATH) don't reliably reach them. fedgnn.task therefore also
# reads the repo root from ~/.flwr/gnn_repo_root; write it here. (The env var is
# kept as a fast path for when the daemon did inherit our environment.)
export GNN_REPO_ROOT="$PWD"
mkdir -p "$HOME/.flwr"
printf '%s\n' "$PWD" > "$HOME/.flwr/gnn_repo_root"

echo "===== Federated GPU run ====="
echo "  federation:            $FEDERATION"
echo "  CUDA_VISIBLE_DEVICES:  $CUDA_VISIBLE_DEVICES"
echo "  eval device:           $DEVICE"
echo "  output dir:            $OUTPUT_DIR"
echo "  seeds:                 $SEEDS"
echo "  run-config overrides:  ${RUN_CONFIG:-<none>} ${EXTRA_RUN_CONFIG}"
echo

# Force a COLD Flower runtime before each seed. The local-simulation-gpu
# federation uses address ":local:", so `flwr run` starts a background
# flower-superlink and then REUSES it (and its Ray actor pool) across successive
# runs. The client actors cache their loaded+split data, their model, and their
# private (fedbn/fedrep) parameters in module-level dicts, so without a cold
# start the split, init AND private state built for the FIRST seed would be
# silently reused for every later seed (identical per-seed metrics, std = 0).
# Killing the superlink between seeds makes the next `flwr run` spin up a fresh
# runtime with empty caches. (Idempotent: no-op if nothing is running.)
stop_flower_runtime() {
  pkill -f 'flower-superlink' 2>/dev/null || true
  pkill -f 'flower-superexec' 2>/dev/null || true
  pkill -f 'flwr-simulation'  2>/dev/null || true
  pkill -f 'flwr-serverapp'   2>/dev/null || true
  pkill -f 'flwr-clientapp'   2>/dev/null || true
  # wait for the control-API port / actors to be released before the next run
  for _ in $(seq 1 15); do
    pgrep -f 'flower-superlink' >/dev/null 2>&1 || break
    sleep 1
  done
}

# Train + evaluate once per seed. `flwr run` always writes the same fixed
# checkpoints_infer/federated.pt (and federated_private/ files), so training and
# evaluation must stay paired within each seed's iteration (eval consumes the
# checkpoint before the next seed's training overwrites it). Each seed's metrics
# land in their own dir.
for seed in $SEEDS; do
  seed_out="$OUTPUT_DIR/seed_$seed"
  echo "################ seed $seed -> $seed_out ################"

  # --- Stage 0: cold-start the runtime so this seed re-loads + re-splits ---
  echo "----- [0/2] stopping any running Flower runtime (fresh cache) -----"
  stop_flower_runtime
  # stale private params from a previous seed/config must never leak into eval
  rm -rf checkpoints_infer/federated_private

  # --- Stage 1: federated training --------------------------------------
  # Inject `seed=$seed` as a trailing --run-config so it overrides the
  # pyproject default (and any user-supplied seed) for this iteration.
  echo "----- [1/2] flwr run federated ($FEDERATION), seed=$seed -----"
  flwr run federated "$FEDERATION" --stream \
      --run-config "$RUN_CONFIG $EXTRA_RUN_CONFIG seed=$seed"

  # --- Stage 2: evaluate the global model on every holdout --------------
  echo
  echo "----- [2/2] evaluate_federated.py, seed=$seed -----"
  "$PYTHON" federated/evaluate_federated.py --resume --device "$DEVICE" \
      --output-dir "$seed_out"
  echo
done

# --- aggregate the per-seed comparison.csv files into mean +/- std --------
SUMMARY_CSV="$OUTPUT_DIR/federated_seed_avg.csv"
PER_RUN_CSV="$OUTPUT_DIR/federated_per_run.csv"
echo "----- aggregating $(echo "$SEEDS" | wc -w) seeds -> $SUMMARY_CSV -----"
"$PYTHON" - "$OUTPUT_DIR" "$SUMMARY_CSV" "$PER_RUN_CSV" <<'PY'
import sys, glob, os, re
import pandas as pd

output_dir, summary_csv, per_run_csv = sys.argv[1], sys.argv[2], sys.argv[3]
METRICS = ("auc_roc", "pr_auc", "precision", "recall", "f1", "fpr")

rows = []
# <output_dir>/seed_<seed>/federated/comparison.csv -> one row per dataset
for path in sorted(glob.glob(os.path.join(output_dir, "seed_*", "federated", "comparison.csv"))):
    m = re.search(r"seed_([^/]+)", path)
    seed = m.group(1) if m else "?"
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        rows.append({"dataset": r["dataset"], "seed": seed,
                     **{k: r.get(k) for k in METRICS}})

if not rows:
    print("No seed_*/federated/comparison.csv files found; nothing to aggregate.")
    sys.exit(0)

per_run = pd.DataFrame(rows)
per_run.to_csv(per_run_csv, index=False)

agg = (per_run.groupby("dataset")[list(METRICS)].agg(["mean", "std"]).round(4))
agg.columns = ["_".join(c) for c in agg.columns]
agg["n_seeds"] = per_run.groupby("dataset").size()
agg = agg.reset_index()
agg.to_csv(summary_csv, index=False)

print("\n=== Per-run federated results ===")
print(per_run.sort_values(["dataset", "seed"]).to_string(index=False))

print("\n=== Mean +/- std across seeds ===")
for mkey in METRICS:
    view = agg[["dataset"]].copy()
    view[mkey] = agg.apply(lambda r: f"{r[f'{mkey}_mean']:.4f} +/- {r[f'{mkey}_std']:.4f}", axis=1)
    print(f"\n--- {mkey} ---")
    print(view.to_string(index=False))

print(f"\nSaved per-run table to {per_run_csv}")
print(f"Saved aggregated summary to {summary_csv}")
PY

echo
echo "===== Done. Per-seed metrics in $OUTPUT_DIR/seed_<seed>/federated/;"
echo "      averaged summary in $SUMMARY_CSV ====="

}

main "$@"
