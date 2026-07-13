# Federated training (Flower workflow API, SecAgg+-ready)

Federated version of the Guided GAE: each of the 4 NetFlow datasets is one
client (silo). The app uses Flower's **standard workflow API** — the server runs
`DefaultWorkflow` over a stock `FedAvg`/`FedProx` strategy, and the clients are
plain `NumPyClient`s — so protocol plugins like **SecAgg+ secure aggregation**
drop straight in (`secagg = true` swaps in `SecAggPlusWorkflow` +
`secaggplus_mod`; with it the server only ever sees the masked *sum* of client
updates, never an individual update).

There is **no early stopping and no best-round selection**: training always runs
exactly `num-server-rounds` rounds (each = `local-epochs` local epochs per
client) and the final-round global model is saved. Per-round train/val MSE per
dataset still lands in `results_infer/federated/history.json` for learning
curves.

## Configuration axes

| axis | values | meaning |
|---|---|---|
| `shared-weight` | `fedavg`, `fedprox` | how shared weights are learned. FedProx adds `(mu/2)*\|\|w - w_global\|\|^2` to each client's loss (`proximal-mu`); aggregation is identical to FedAvg, so SecAgg+ works with both. |
| `personalisation` | `na`, `fedbn`, `fedrep` | which part of the model stays **private** on each client. `fedbn`: BatchNorm params + running stats. `fedrep`: the decoder + global-edge-embedding head (only the encoder is aggregated; each round trains head-then-body, `local-epochs` each). Private parameters never cross the wire — each client persists its own to `checkpoints_infer/federated_private/<dataset>.pt` for evaluation. |
| `secagg` | `true`, `false` | wrap every fit round in the SecAgg+ protocol. `clipping-range` (the `--clip` flag) bounds every exchanged value before quantisation; `secagg-num-shares` / `secagg-reconstruction-threshold` control the Shamir key sharing. |

Preprocessing is **silo-local**: each client fits its own StandardScaler and
protocol/port vocabs on its own data (nothing pooled). Vocabs are padded to the
fixed top-k size so all clients share identical model shapes — which also gives
SecAgg+ the uniform parameter vector it needs.

## Run

```bash
# everything (all seeds, train + eval + seed-averaged tables):
./run_federated_gpu.sh \
    --shared_weight=FedAvg \        # or FedProx (add --mu=0.01)
    --personalisation=NA \          # or FedBN / FedRep
    --output_dir=results_infer \
    --rounds=150 --epoch=1 \
    --secagg=true --clip=8.0

# or drive flwr directly (single run, defaults from pyproject.toml):
flwr run federated local-simulation-gpu --stream \
    --run-config "shared-weight='fedprox' personalisation='fedbn' secagg=true clipping-range=8.0"
python federated/evaluate_federated.py --resume
```

Smoke test:

```bash
flwr run federated --stream --run-config "num-server-rounds=3 max-train-rows=20000 window-size=200 step-size=200 batch-size=16 hidden-dim=8 global-emb-dim=8 node-dim=8 secagg=true"
```

## Notes / deviations from the centralised pipeline

- 1 round = `local-epochs` local epochs; a fresh AdamW is created each round
  (constant `lr`, no ReduceLROnPlateau — an LR schedule would have to be
  server-driven).
- Client aggregation weight is `client-weight = "equal"` by default (every
  dataset pulls equally on the shared model); `"examples"` restores classic
  FedAvg weighting. Val MSE is aggregated weighted by validation edge count,
  matching a pooled per-element mean.
- `num_batches_tracked` BatchNorm counters are never exchanged (SecAgg+ clipping
  would corrupt the int64 counters; they are irrelevant with default momentum).
- Under SecAgg+ the aggregated weights pick up small quantisation noise; raise
  `clipping-range` if weights or BN running stats exceed it.
- Anomaly thresholds are calibrated **per client** at evaluation time (each
  silo's scaler gives it its own reconstruction-error scale), reusing the same
  seed so the exact training-time val split is rebuilt.
- `evaluate_federated.py` reads all model/preprocessing settings from the
  checkpoint's stored run config — no need to repeat them.
