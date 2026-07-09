# Federated training (Flower / FedAvg)

Federated version of `train_and_infer.py`: each of the 4 NetFlow datasets is one
client (silo). Every round, each client trains the full model (shared encoder +
shared decoder) for **one local epoch** on its own benign windows, the server
FedAvg-aggregates the weights, and the aggregated model is validated on every
client's local benign val split. The server keeps the weights of the best
weighted-mean-val-MSE round and stops early after `patience` rounds without
improvement (cap: `num-server-rounds`, default 150).

Preprocessing is **silo-local**: each client fits its own StandardScaler and
protocol/port vocabs on its own data (nothing pooled). Vocabs are padded to the
fixed top-k size so all clients share identical model shapes; the same one-hot
slot can therefore mean a different port/protocol on different clients.

## Run

```bash
# 1. Train (from the repo root). Writes checkpoints_infer/federated.pt
#    and results_infer/federated/history.json
flwr run federated                          # CPU
flwr run federated local-simulation-gpu     # 4 clients sharing one GPU

# 2. Evaluate the global model on every holdout.
#    Writes results_infer/federated/<dataset>/metrics.json + comparison.csv
python federated/evaluate_federated.py --resume
```

Override any `[tool.flwr.app.config]` value from the CLI, e.g. a smoke test:

```bash
flwr run federated --run-config "num-server-rounds=3 max-train-rows=20000 window-size=200 step-size=200 batch-size=16 hidden-dim=8 global-emb-dim=8 node-dim=8"
```

(`--stream` shows live ServerApp/ClientApp logs.)

## Notes / deviations from the centralised pipeline

- 1 round = 1 local epoch; a fresh AdamW is created each round (constant `lr`,
  no ReduceLROnPlateau — an LR schedule would have to be server-driven).
- FedAvg weights clients by training-window count (`num-examples`); val MSE is
  aggregated weighted by validation edge count, matching a pooled per-element mean.
- Anomaly thresholds are calibrated **per client** at evaluation time (each
  silo's scaler gives it its own reconstruction-error scale), reusing the same
  seed so the exact training-time val split is rebuilt.
- `evaluate_federated.py` reads all model/preprocessing settings from the
  checkpoint's stored run config — no need to repeat them.
