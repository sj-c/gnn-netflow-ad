# GNN NetFlow Anomaly Detection

A graph autoencoder (GAE) trained one-class on **benign** NetFlow traffic; attacks
are flagged at inference as high edge-reconstruction-error flows. Model
(`GATEncoderWithEdgeAttr` + `GlobalEdgeEmbedding` + `DecoderWithGlobalEdge`) lives
in `src/guided_gae_model.py`; preprocessing (`NetflowPreprocessor`, row-windowed
graph construction) in `src/preprocess.py`.

Two ways to train it:

- **Centralised** (`train_and_inferv2.py`) — one process, one or more datasets,
  several sharing/personalisation modes.
- **Federated** (`run_federated_gpu.sh` + `federated/`) — Flower workflow API,
  one client per dataset, optional SecAgg+ secure aggregation and central DP.

## Setup

```bash
conda env create -f environment.yml
conda activate gnn-netflow-ad
```

`environment.yml` pins `torch==2.8.0+cu128` (matches CUDA 12.8, driver 575.x). If
your GPU driver supports a different CUDA version, edit the
`--extra-index-url` line (see comments in the file) before creating the env — CPU-
only and older CUDA builds both work, just slower or without a GPU. `flwr[simulation]`
pulls in Ray, which the federated pipeline needs.

### Data layout

Both scripts expect, for each dataset, a `<name>_train.parquet` and
`<name>_holdout.parquet` (NetFlow v3 schema, 55 columns) in one directory:

```
NF-BoT-IoT-v3_train.parquet     NF-BoT-IoT-v3_holdout.parquet
NF-UNSW-NB15-v3_train.parquet   NF-UNSW-NB15-v3_holdout.parquet
NF-ToN-IoT-v3_train.parquet     NF-ToN-IoT-v3_holdout.parquet
NF-CICIDS2018-v3_train.parquet  NF-CICIDS2018-v3_holdout.parquet
```

By default both scripts look for `./netflow/parquet` then `../netflow/parquet`
relative to the repo root; override with `--data-dir` (centralised) or the
`data-dir` run-config key (federated). Train splits are (almost) benign-only —
UNSW-NB15's has ~1.5% attack contamination, filtered automatically by
`Label==0`. Holdouts are large (tens of millions of rows) and are streamed via
`pyarrow.ParquetFile.iter_batches`, never loaded whole.

## Centralised: `train_and_inferv2.py`

One invocation trains **and** evaluates every selected dataset; there's no
separate `--mode train/infer` flag. Outputs:

- `<checkpoint-dir>/` (default `checkpoints_infer/`) — model weights
- `<output-dir>/` (default `results_infer/`) — `metrics.json` + `comparison.csv` per dataset, plus a top-level summary `comparison.csv`
- `<log-dir>/` (default `logs_infer/`) — per-dataset training logs

```bash
# one model per dataset (default), all 4 datasets
python train_and_inferv2.py

# subset of datasets
python train_and_inferv2.py --datasets NF-UNSW-NB15-v3,NF-ToN-IoT-v3

# skip datasets that already have results_infer/<name>/metrics.json
python train_and_inferv2.py --resume
```

### Modes (mutually exclusive: `--combined` / `--personalised` / `--personalised-frozen`)

| mode | what it trains |
|---|---|
| *(default)* | independent model per dataset, never shared |
| `--combined` | ONE model on the pooled benign training data of every selected dataset (shared scaler/vocabs + shared encoder/decoder); evaluated per-dataset and on all holdouts pooled |
| `--personalised` | shared encoder + a personalised head (decoder + global-edge embedding) per dataset, trained **jointly** in one stage. Known failure mode: joint early stopping on the mean val MSE lets divergent datasets (e.g. UNSW) drag the stop point away from slower-converging ones (ToN-IoT/CICIDS) — prefer `--personalised-frozen` unless you specifically want the joint variant |
| `--personalised-frozen` | two stages: (1) train a `--combined` backbone, (2) freeze its encoder and train a fresh head per dataset with its own early stopping. Fixes the negative-transfer/early-stop issues of `--personalised` while still sharing one encoder |

Domain-specific BatchNorm layers on top of `--combined` or `--personalised-frozen`
(mutually exclusive with each other, requires one of those two base modes):

| flag | effect |
|---|---|
| `--adabn` | re-estimate the encoder's BatchNorm running stats per dataset (forward passes only, no training); affine params stay shared |
| `--fedbn` | freeze the shared backbone and **train** each dataset's own BatchNorm (affine + stats); with `--combined` the decoder stays frozen (BN-only fine-tune), with `--personalised-frozen` a fresh head trains alongside its BN |

```bash
python train_and_inferv2.py --combined
python train_and_inferv2.py --personalised-frozen --clip-grad-norm 1.0   # steadies small heads (e.g. BoT-IoT)
python train_and_inferv2.py --combined --fedbn
python train_and_inferv2.py --personalised-frozen --adabn
```

### Common flags

- Data/features: `--edge-columns baseline|all|<comma list>`, `--log1p-columns baseline|none|all|<comma list>`, `--top-k-ports`, `--top-k-protocols`, `--window-size`, `--step-size`, `--node-dim`
- Training: `--epochs`, `--patience` (0 disables early stopping), `--batch-size`, `--hidden-dim`, `--global-emb-dim`, `--heads`, `--lr`, `--weight-decay`, `--clip-grad-norm`, `--scheduler-factor`, `--scheduler-patience`, `--train-ratio`, `--seed`
- Run control: `--device auto|cuda|cuda:0|cpu`, `--resume`, `--calib-quantiles` (label-free anomaly thresholds from benign-validation error quantiles), `--max-train-rows` / `--max-holdout-rows` (smoke testing; `--max-train-rows 0` disables the cap)

Full list: `python train_and_inferv2.py --help`.

### Smoke test (CPU, seconds)

```bash
python train_and_inferv2.py --datasets NF-UNSW-NB15-v3 \
    --max-train-rows 20000 --max-holdout-rows 20000 \
    --window-size 200 --step-size 200 --epochs 3 --patience 0 \
    --hidden-dim 8 --global-emb-dim 8 --node-dim 8 --device cpu
```

## Federated: `run_federated_gpu.sh`

Flower's standard workflow API, one client per NetFlow dataset (4 silos),
**no early stopping** — every run trains exactly `--rounds` communication rounds
and evaluates the final-round model. Preprocessing is silo-local: each client
fits its own scaler/vocabs (nothing pooled across datasets). Config defaults live
in `federated/pyproject.toml`; see `federated/README.md` for the full protocol
writeup (SecAgg+, central DP, FedRep internals).

```bash
./run_federated_gpu.sh \
    --shared_weight=FedAvg \        # or FedProx (add --mu=0.01)
    --personalisation=NA \          # or FedBN / FedRep
    --output_dir=results_infer \
    --rounds=150 --epoch=1 \
    --secagg=true --clip=8.0        # SecAgg+ secure aggregation, optional
```

Runs training + evaluation once per seed (default `--seeds="42 43 44"`), then
aggregates per-seed `comparison.csv` files into mean±std tables. Per-seed output
lands in `<output_dir>/seed_<seed>/federated/`; averaged summary at
`<output_dir>/federated_seed_avg.csv` (+ `federated_per_run.csv`).

```bash
CUDA_VISIBLE_DEVICES=1 ./run_federated_gpu.sh                    # pin to GPU 1
./run_federated_gpu.sh --seeds="42"                               # single seed
./run_federated_gpu.sh --run-config "max-train-rows=20000"        # smoke test
./run_federated_gpu.sh --shared_weight=FedProx --personalisation=FedBN \
    --secagg=true --clip=8.0 --output_dir=results_fedprox_fedbn
```

### Configuration axes

| axis | flag | values | meaning |
|---|---|---|---|
| shared-weight learning | `--shared_weight` | `FedAvg`, `FedProx` | FedProx adds a proximal term (`--mu`) to each client's loss; aggregation is unchanged |
| personalisation | `--personalisation` | `NA`, `FedBN`, `FedRep` | what stays private per client. `FedBN`: BatchNorm params+stats. `FedRep`: decoder + global-edge-embedding head (only the encoder is aggregated; alternates head-then-body each round) |
| secure aggregation | `--secagg` | `true`, `false` | wraps every round in SecAgg+ — the server only ever sees the masked *sum* of client updates |

Private per-client parameters (fedbn/fedrep) persist to
`checkpoints_infer/federated_private/<dataset>.pt`. Extra knobs not exposed as
flags (central-DP clipping/noise, SecAgg+ share counts, client weighting, model
hyperparameters, `--max-train-rows`, etc.) go through `--run-config "key=value ..."`
— see the commented defaults in `federated/pyproject.toml`.

### Direct `flwr run` (single run, no seed loop / aggregation)

```bash
flwr run federated local-simulation-gpu --stream \
    --run-config "shared-weight='fedprox' personalisation='fedbn' secagg=true clipping-range=8.0"
python federated/evaluate_federated.py --resume
```

### Notes

- Trains on GPU by default (4 clients share one GPU, 0.25 each via Ray); evaluation device is separate (`DEVICE` env var, default `cuda`).
- `run_federated_gpu.sh` kills any leftover Flower/Ray processes and clears `checkpoints_infer/federated_private/` before each seed to guarantee a cold start (Ray actors otherwise cache the first seed's data split/model across "different" seeds).
- If a run dies mid-simulation the script fails loudly (missing `checkpoints_infer/federated.pt`) rather than silently evaluating a stale checkpoint.
- `FLWR_HOME` is relocated to `/tmp/flwr_$USER` (local disk) — Flower's SQLite state DB is unreliable on NFS.
