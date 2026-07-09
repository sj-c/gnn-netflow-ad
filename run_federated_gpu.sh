#!/usr/bin/env bash
#
# Train + evaluate the federated (Flower / FedAvg) Guided GAE on GPU.
#
# Stage 1: `flwr run federated local-simulation-gpu` -- 4 clients (one per
#   NetFlow dataset) share one GPU (0.25 each). Writes
#   checkpoints_infer/federated.pt and results_infer/federated/history.json.
# Stage 2: evaluate_federated.py restores the best-round global model and scores
#   it on every dataset's holdout (per-client calibrated thresholds), writing
#   results_infer/federated/<dataset>/metrics.json + comparison.csv.
#
# Usage:
#   ./run_federated_gpu.sh                                   # FedAvg (default)
#   CUDA_VISIBLE_DEVICES=1 ./run_federated_gpu.sh           # pin to GPU 1
#   ./run_federated_gpu.sh --run-config "num-server-rounds=3 max-train-rows=20000"
#                                                           # extra args -> flwr run
#   # FedRep (shared encoder + private per-client heads). Edit rounds via
#   # num-server-rounds and per-round local work via fedrep-head/body-epochs:
#   ./run_federated_gpu.sh --run-config \
#       "strategy='fedrep' num-server-rounds=100 fedrep-head-epochs=5 fedrep-body-epochs=1"
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
echo

# --- Stage 1: federated training ------------------------------------------
echo "----- [1/2] flwr run federated ($FEDERATION) -----"
flwr run federated "$FEDERATION" --stream "$@"

# --- Stage 2: evaluate the global model on every holdout ------------------
echo
echo "----- [2/2] evaluate_federated.py -----"
"$PYTHON" federated/evaluate_federated.py --resume --device "$DEVICE"

echo
echo "===== Done. See results_infer/federated/ (per-dataset metrics + comparison.csv) ====="

}

main "$@"
