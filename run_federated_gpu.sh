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
# datasets to EVALUATE (passed to evaluate_federated.py --datasets). Default
# "all" = the trained clients (CLIENT_DATASETS). For leave-one-out set this to
# all four dataset names so the held-out one is scored zero-shot each seed.
EVAL_DATASETS="${EVAL_DATASETS:-all}"

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

# Put Flower's runtime state on LOCAL disk. FLWR_HOME (default ~/.flwr) holds the
# SuperLink's SQLite task DB (local-superlink/state.db) and the copied app build
# dir. $HOME here is NFS, and SQLite on NFS intermittently dies with
# "sqlite3.OperationalError: disk I/O error" once a round fans out to multiple
# clients (NFS advisory locking is unreliable). /tmp is node-local NVMe, so we
# point FLWR_HOME there. This state is ephemeral scratch bookkeeping -- results
# and checkpoints still go to $OUTPUT_DIR / checkpoints_infer on NFS.
export FLWR_HOME="${FLWR_HOME:-/tmp/flwr_$USER}"
mkdir -p "$FLWR_HOME"

# Flower 1.30 moved federation ("SuperLink connection") definitions out of
# pyproject.toml and into the Flower config file at $FLWR_HOME/config.toml.
# Because we relocate FLWR_HOME to local disk (above), Flower would otherwise
# read a bare auto-created default there and fail with "SuperLink connection
# '<federation>' not found". Write the federation definitions into $FLWR_HOME so
# `flwr run federated <federation>` resolves them regardless of where FLWR_HOME
# points. (num-supernodes = number of clients; the -3 variant is used by the
# leave-one-dataset-out sweep, which drops one dataset via FED_HELDOUT.)
cat > "$FLWR_HOME/config.toml" <<'TOML'
[superlink]
default = "local-simulation"

[superlink.supergrid]
address = "supergrid.flower.ai"

[superlink.local]
address = ":local:"

[superlink.local-simulation]
address = ":local:"
options.num-supernodes = 4

# on the GPU machine: 4 clients share one GPU (0.25 each)
[superlink.local-simulation-gpu]
address = ":local:"
options.num-supernodes = 4
options.backend.name = "ray"
options.backend.client-resources.num-cpus = 2
options.backend.client-resources.num-gpus = 0.25

# leave-one-dataset-out: 3 clients (one dataset held out via FED_HELDOUT)
[superlink.local-simulation-gpu-3]
address = ":local:"
options.num-supernodes = 3
options.backend.name = "ray"
options.backend.client-resources.num-cpus = 2
options.backend.client-resources.num-gpus = 0.25
TOML

# `flwr run` copies the app into an isolated build dir under ~/.flwr/apps, so the
# fedgnn package can no longer find the repo-root `src`/`train_and_infer` modules
# by walking up from its own location. The ServerApp/ClientApp are spawned by a
# persistent flower-superlink daemon whose environment predates this script, so
# env vars (and PYTHONPATH) don't reliably reach them. fedgnn.task therefore also
# reads the repo root from ~/.flwr/gnn_repo_root; write it here. (The env var is
# kept as a fast path for when the daemon did inherit our environment.)
# NOTE: fedgnn.task reads Path.home()/.flwr/gnn_repo_root -- a FIXED path that is
# NOT affected by FLWR_HOME above -- so this handshake file always lives under
# $HOME/.flwr regardless of where the SuperLink state was relocated.
export GNN_REPO_ROOT="$PWD"
mkdir -p "$HOME/.flwr"
printf '%s\n' "$PWD" > "$HOME/.flwr/gnn_repo_root"

# Leave-one-dataset-out: the held-out dataset is chosen via FED_HELDOUT, but the
# flwr daemon apps do NOT inherit our environment (same reason as gnn_repo_root
# above), so an env-only FED_HELDOUT never reaches training and NOTHING gets held
# out -- all 4 datasets silently stay clients. Mirror the repo-root handshake:
# write it to ~/.flwr/gnn_fed_heldout, which fedgnn.task reads at import. Write it
# EVERY run (empty string when FED_HELDOUT is unset) so a stale held-out from a
# prior LODO run can never leak into a plain 4-client run.
printf '%s\n' "${FED_HELDOUT:-}" > "$HOME/.flwr/gnn_fed_heldout"

# IID single-dataset control: when FED_SINGLE_DATASET names
# a dataset, the 4 clients become 4 disjoint IID shards of THAT dataset instead of
# the 4 different datasets. Same daemon-immune handshake as gnn_fed_heldout above:
# fedgnn.task reads ~/.flwr/gnn_fed_single at import (env doesn't reach the daemon).
# Written EVERY run (empty when unset) so a stale single-dataset can't leak in.
printf '%s\n' "${FED_SINGLE_DATASET:-}" > "$HOME/.flwr/gnn_fed_single"

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
# Patterns for every process the simulation spawns: the Flower daemons AND the
# Ray backend (raylet/GCS/plasma + the ray::ClientAppActor workers). A crashed or
# interrupted run routinely leaves ONE of these orphaned -- most often the
# flower-superlink daemon or a raylet still pinning the GPU. When the next
# `flwr run` reuses/collides with that orphan the simulation dies inside round 1,
# and because `flwr run` still exits 0 (writing no checkpoint) the seed fails with
# the confusing "wrote no federated.pt" guard below. So we must reap Ray too, not
# just the Flower daemons. NOTE: this assumes exclusive use of the Ray/Flower
# runtime for $USER on this node (the same assumption the flwr-simulation pkill
# already made); a second concurrent Ray job of yours would be killed as well.
_RUNTIME_PATTERNS='flower-superlink|flower-superexec|flwr-simulation|flwr-serverapp|flwr-clientapp|raylet|gcs_server|plasma_store|ray::'

stop_flower_runtime() {
  # graceful, then forced -- a plain SIGTERM sometimes leaves the superlink and
  # raylet alive (observed orphans survive a single pkill), so escalate.
  pkill -f  "$_RUNTIME_PATTERNS" 2>/dev/null || true
  for _ in $(seq 1 10); do
    pgrep -f "$_RUNTIME_PATTERNS" >/dev/null 2>&1 || break
    sleep 1
  done
  pkill -9 -f "$_RUNTIME_PATTERNS" 2>/dev/null || true
  # wait for the control-API port / GPU to actually be released before the next run
  for _ in $(seq 1 15); do
    pgrep -f "$_RUNTIME_PATTERNS" >/dev/null 2>&1 || break
    sleep 1
  done
  # Drop stale Ray session state so the next run starts a fresh head instead of
  # trying to attach to a dead cluster (a corrupt /tmp/ray is another way the
  # raylet fails to come up and round 1 crashes).
  rm -rf /tmp/ray/session_* 2>/dev/null || true
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
  # Delete the previous checkpoint so a crashed/aborted flwr run (e.g. a Ray
  # raylet death mid-simulation, which exits the CLI 0 without writing a new
  # model) cannot silently be evaluated as if it were this run's result. If
  # training does not produce a fresh checkpoint, Stage 2 fails loudly below.
  rm -f checkpoints_infer/federated.pt

  # --- Stage 1: federated training --------------------------------------
  # Inject `seed=$seed` as a trailing --run-config so it overrides the
  # pyproject default (and any user-supplied seed) for this iteration.
  echo "----- [1/2] flwr run federated ($FEDERATION), seed=$seed -----"
  flwr run federated "$FEDERATION" --stream \
      --run-config "$RUN_CONFIG $EXTRA_RUN_CONFIG seed=$seed"

  # flwr run exits 0 even when the simulation crashes; a missing checkpoint is
  # the only reliable signal that training never finished this seed.
  if [ ! -f checkpoints_infer/federated.pt ]; then
    echo "ERROR: flwr run wrote no checkpoints_infer/federated.pt for seed=$seed" \
         "(training crashed -- see the Flower/Ray logs). Failing this seed." >&2
    exit 1
  fi

  # --- Stage 2: evaluate the global model on every holdout --------------
  echo
  echo "----- [2/2] evaluate_federated.py, seed=$seed -----"
  "$PYTHON" federated/evaluate_federated.py --resume --device "$DEVICE" \
      --output-dir "$seed_out" --datasets "$EVAL_DATASETS"
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
