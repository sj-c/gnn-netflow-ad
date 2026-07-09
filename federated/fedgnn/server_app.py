"""Flower ServerApp: FedAvg or FedRep over the 4 dataset-clients, with early stopping.

A custom main loop (rather than strategy.start) because model selection mirrors
train_and_infer.fit_model: after every round the aggregated global model is
validated on every client's local benign val split, the server tracks the
weighted-mean val MSE, keeps the best round's weights, and stops early after
`patience` rounds without improvement.

strategy = "fedavg" (default):
    All parameters are averaged into one global model -> checkpoints_infer/federated.pt
    with mode "federated_fedavg" ({"model_state_dict": ...}).

strategy = "fedrep":
    Only the body (encoder) is averaged across clients; each client keeps a private
    head (decoder + global-edge embedding). The body is aggregated by hand here so
    the head ArrayRecord in each reply is never fed to FedAvg's array aggregator.
    Saved with mode "federated_fedrep" ({"body_state_dict": ..., "head_state_dicts":
    {dataset: ...}}). "Rounds" = num-server-rounds; local work per round is set by
    fedrep-head-epochs / fedrep-body-epochs.

Per-round history goes to results_infer/federated/history.json either way.
"""

import json
import time
from logging import INFO

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.common import log
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from fedgnn.task import (
    CLIENT_DATASETS,
    REPO_ROOT,
    body_state_dict,
    build_model,
    edge_feature_dim,
)
from train_and_inferv2 import resolve_edge_columns

app = ServerApp()


def _per_client_metrics(replies, key):
    """Map dataset name -> metric value from individual (pre-aggregation) replies."""
    out = {}
    for msg in replies:
        if msg.has_error():
            continue
        metrics = next(iter(msg.content.metric_records.values()))
        out[CLIENT_DATASETS[int(metrics["partition-id"])]] = float(metrics[key])
    return out


def _build_initial_model(cfg):
    """The server builds the model shell from the run config alone (clients derive
    identical shapes from the same config), so it can seed the initial weights."""
    torch.manual_seed(cfg["seed"])
    edge_columns = resolve_edge_columns(cfg["edge-columns"])
    edge_attr_dim = edge_feature_dim(edge_columns, cfg["top-k-ports"], cfg["top-k-protocols"])
    return build_model(cfg, cfg["node-dim"], edge_attr_dim, torch.device("cpu"))


def _make_strategy():
    n_clients = len(CLIENT_DATASETS)
    return FedAvg(
        fraction_train=1.0,
        fraction_evaluate=1.0,
        min_train_nodes=n_clients,
        min_evaluate_nodes=n_clients,
        min_available_nodes=n_clients,
    )


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = dict(context.run_config)
    if cfg.get("strategy", "fedavg") == "fedrep":
        run_fedrep(grid, cfg)
    else:
        run_fedavg(grid, cfg)


def run_fedavg(grid: Grid, cfg: dict) -> None:
    num_rounds = cfg["num-server-rounds"]
    patience = cfg["patience"]

    model = _build_initial_model(cfg)
    arrays = ArrayRecord(model.state_dict())
    strategy = _make_strategy()

    best_val_mse = float("inf")
    best_round = 0
    best_arrays = arrays
    rounds_no_improve = 0
    rounds_trained = 0
    history = []
    t_start = time.time()

    for server_round in range(1, num_rounds + 1):
        t_round = time.time()

        # --- local training: 1 epoch on every client, then FedAvg ---
        train_replies = list(grid.send_and_receive(
            messages=strategy.configure_train(server_round, arrays, ConfigRecord(), grid),
            timeout=cfg["round-timeout"],
        ))
        agg_arrays, agg_train = strategy.aggregate_train(server_round, train_replies)
        if agg_arrays is None:
            raise RuntimeError(f"round {server_round}: training aggregation failed "
                               "(missing/invalid client replies)")
        arrays = agg_arrays
        train_by_ds = _per_client_metrics(train_replies, "train_mse")

        # --- validate the AGGREGATED model on every client's local val split ---
        evaluate_replies = list(grid.send_and_receive(
            messages=strategy.configure_evaluate(server_round, arrays, ConfigRecord(), grid),
            timeout=cfg["round-timeout"],
        ))
        agg_eval = strategy.aggregate_evaluate(server_round, evaluate_replies)
        if agg_eval is None:
            raise RuntimeError(f"round {server_round}: evaluation aggregation failed")
        val_mse = float(agg_eval["val_mse"])
        val_by_ds = _per_client_metrics(evaluate_replies, "val_mse")

        rounds_trained = server_round
        per_ds = " ".join(f"{n}:train={train_by_ds.get(n, float('nan')):.6f}"
                          f"/val={val_by_ds.get(n, float('nan')):.6f}"
                          for n in CLIENT_DATASETS)
        log(INFO, "round %d/%d mean_val_mse=%.6f %s (%.1fs)",
            server_round, num_rounds, val_mse, per_ds, time.time() - t_round)
        history.append({
            "round": server_round,
            "train_mse": float(agg_train["train_mse"]) if agg_train else None,
            "val_mse": val_mse,
            "train_mse_by_dataset": train_by_ds,
            "val_mse_by_dataset": val_by_ds,
        })

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_round = server_round
            best_arrays = arrays
            rounds_no_improve = 0
        else:
            rounds_no_improve += 1
            if patience > 0 and rounds_no_improve >= patience:
                log(INFO, "early stopping at round %d: no val_mse improvement for %d "
                    "rounds (best=%.6f @ round %d)",
                    server_round, patience, best_val_mse, best_round)
                break

    train_time = time.time() - t_start
    log(INFO, "restoring best global model from round %d (val_mse=%.6f)",
        best_round, best_val_mse)

    _save_checkpoint({
        "model_state_dict": best_arrays.to_torch_state_dict(),
        "mode": "federated_fedavg",
        "datasets": CLIENT_DATASETS,
        "run_config": cfg,
        "best_round": best_round,
        "best_val_mse": best_val_mse,
        "rounds_trained": rounds_trained,
        "train_time_sec": train_time,
    })
    _save_history(best_round, best_val_mse, rounds_trained, history)


def _aggregate_body(replies, weight_key="num-examples"):
    """Weighted mean of the body (encoder) ArrayRecords in the replies, plus the
    latest private head of each client. Done by hand (not FedAvg) so the head
    ArrayRecord present in every reply is read but NEVER averaged into the body."""
    acc = None
    total = 0.0
    heads = {}
    for msg in replies:
        if msg.has_error():
            continue
        content = msg.content
        metrics = next(iter(content.metric_records.values()))
        pid = int(metrics["partition-id"])
        w = float(metrics[weight_key])
        body = content["arrays"].to_torch_state_dict()
        heads[pid] = content["head"].to_torch_state_dict()
        if acc is None:
            acc = {k: v.double() * w for k, v in body.items()}
        else:
            for k, v in body.items():
                acc[k] = acc[k] + v.double() * w
        total += w
    if acc is None or total == 0.0:
        return None, heads
    body_avg = {k: (v / total).to(torch.float32) for k, v in acc.items()}
    return ArrayRecord(body_avg), heads


def run_fedrep(grid: Grid, cfg: dict) -> None:
    num_rounds = cfg["num-server-rounds"]
    patience = cfg["patience"]

    model = _build_initial_model(cfg)
    body_arrays = ArrayRecord(body_state_dict(model))
    strategy = _make_strategy()

    log(INFO, "FedRep: %d rounds | per round: head_epochs<=%d (local early-stop "
        "patience=%d) body_epochs=%d | body=encoder (aggregated), "
        "head=decoder+global_edge_embedding (per-client)",
        num_rounds, cfg["fedrep-head-epochs"], cfg["fedrep-head-patience"],
        cfg["fedrep-body-epochs"])

    best_val_mse = float("inf")
    best_round = 0
    best_body = body_arrays
    best_heads = {}
    heads = {}
    rounds_no_improve = 0
    rounds_trained = 0
    history = []
    t_start = time.time()

    for server_round in range(1, num_rounds + 1):
        t_round = time.time()

        # --- local FedRep training (head-then-body), then average ONLY the body ---
        train_replies = list(grid.send_and_receive(
            messages=strategy.configure_train(server_round, body_arrays, ConfigRecord(), grid),
            timeout=cfg["round-timeout"],
        ))
        agg_body, heads = _aggregate_body(train_replies)
        if agg_body is None:
            raise RuntimeError(f"round {server_round}: body aggregation failed "
                               "(missing/invalid client replies)")
        body_arrays = agg_body
        train_by_ds = _per_client_metrics(train_replies, "train_mse")

        # --- validate global body + each client's private head on its val split ---
        evaluate_replies = list(grid.send_and_receive(
            messages=strategy.configure_evaluate(server_round, body_arrays, ConfigRecord(), grid),
            timeout=cfg["round-timeout"],
        ))
        agg_eval = strategy.aggregate_evaluate(server_round, evaluate_replies)
        if agg_eval is None:
            raise RuntimeError(f"round {server_round}: evaluation aggregation failed")
        val_mse = float(agg_eval["val_mse"])
        val_by_ds = _per_client_metrics(evaluate_replies, "val_mse")

        rounds_trained = server_round
        per_ds = " ".join(f"{n}:train={train_by_ds.get(n, float('nan')):.6f}"
                          f"/val={val_by_ds.get(n, float('nan')):.6f}"
                          for n in CLIENT_DATASETS)
        log(INFO, "round %d/%d mean_val_mse=%.6f %s (%.1fs)",
            server_round, num_rounds, val_mse, per_ds, time.time() - t_round)
        history.append({
            "round": server_round,
            "val_mse": val_mse,
            "train_mse_by_dataset": train_by_ds,
            "val_mse_by_dataset": val_by_ds,
        })

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_round = server_round
            best_body = body_arrays
            # snapshot every client's head as it stood at this best round
            best_heads = {pid: {k: v.clone() for k, v in sd.items()}
                          for pid, sd in heads.items()}
            rounds_no_improve = 0
        else:
            rounds_no_improve += 1
            if patience > 0 and rounds_no_improve >= patience:
                log(INFO, "early stopping at round %d: no val_mse improvement for %d "
                    "rounds (best=%.6f @ round %d)",
                    server_round, patience, best_val_mse, best_round)
                break

    train_time = time.time() - t_start
    log(INFO, "restoring best FedRep model from round %d (val_mse=%.6f): shared body "
        "+ %d per-client heads", best_round, best_val_mse, len(best_heads))

    _save_checkpoint({
        "body_state_dict": best_body.to_torch_state_dict(),
        "head_state_dicts": {CLIENT_DATASETS[pid]: sd for pid, sd in best_heads.items()},
        "mode": "federated_fedrep",
        "datasets": CLIENT_DATASETS,
        "run_config": cfg,
        "best_round": best_round,
        "best_val_mse": best_val_mse,
        "rounds_trained": rounds_trained,
        "train_time_sec": train_time,
    })
    _save_history(best_round, best_val_mse, rounds_trained, history)


def _save_checkpoint(payload):
    ckpt_path = REPO_ROOT / "checkpoints_infer" / "federated.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, ckpt_path)
    log(INFO, "saved best %s checkpoint to %s (train_time=%.1fs)",
        payload["mode"], ckpt_path, payload["train_time_sec"])


def _save_history(best_round, best_val_mse, rounds_trained, history):
    history_path = REPO_ROOT / "results_infer" / "federated" / "history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump({"best_round": best_round, "best_val_mse": best_val_mse,
                   "rounds_trained": rounds_trained, "history": history}, f, indent=2)
    log(INFO, "saved round history to %s", history_path)
