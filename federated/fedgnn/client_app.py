"""Flower ClientApp: one client per NetFlow dataset.

Two strategies, selected by the ``strategy`` run-config key:

FedAvg (strategy="fedavg"):
  train    -- load the global weights, run ONE local epoch (1 round = 1 epoch) on
              this client's benign training windows, reply with the updated weights.
  evaluate -- score the aggregated global model on this client's benign validation
              windows and reply with the local val MSE (weighted by edge count).

FedRep (strategy="fedrep"):
  The model is split into a shared *body* (the encoder) and a private *head*
  (decoder + global-edge embedding), exactly like --personalised-frozen. Only the
  body is aggregated; every client keeps its own head locally across rounds.
  train    -- load the aggregated global body, restore this client's private head,
              then per FedRep's alternating scheme: (A) freeze the body and train
              the head for `fedrep-head-epochs` epochs, (B) freeze the head and
              take `fedrep-body-epochs` epochs on the body. Reply with the updated
              body (aggregated) and the head (stashed by the server for eval, but
              never averaged).
  evaluate -- score the global body + this client's private head on its val split.
"""

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fedgnn.task import (
    CLIENT_DATASETS,
    body_state_dict,
    build_model,
    get_client_data,
    get_device,
    head_state_dict,
)
from train_and_inferv2 import test as val_mse_fn
from train_and_inferv2 import train as train_one_epoch

import torch
import torch.nn.functional as F

app = ClientApp()

# Private per-client FedRep heads, keyed by partition-id. The simulation runs each
# client in a long-lived actor (this is the same assumption that lets task.py cache
# a client's data across rounds), so a client's head persists round to round here
# and never leaves the process except as the copy the server stashes for evaluation.
_client_heads = {}


def _client_setup(msg: Message, context: Context):
    """Build the client's model and return the *incoming* global arrays. Loading is
    left to the caller because FedAvg ships a full state-dict while FedRep ships only
    the body (encoder) sub-state-dict."""
    cfg = dict(context.run_config)
    partition_id = int(context.node_config["partition-id"])
    dataset_name = CLIENT_DATASETS[partition_id]
    data = get_client_data(dataset_name, cfg)
    device = get_device()
    model = build_model(cfg, data["in_channels"], data["edge_attr_dim"], device)
    incoming = msg.content["arrays"].to_torch_state_dict()
    return cfg, partition_id, data, device, model, incoming


def _set_requires_grad(modules, flag):
    for module in modules:
        for p in module.parameters():
            p.requires_grad_(flag)


def _local_epochs(model, optimizer, loader, device, n_epochs, encoder_train,
                  val_loader=None, patience=0):
    """Run up to `n_epochs` reconstruction-MSE epochs, stepping only the params that
    currently require grad. When `encoder_train` is False the encoder is held in
    eval mode so its BatchNorm running stats stay frozen while the body is fixed
    (the head phase); mirrors freeze_encoder's eval-lock in train_and_inferv2.

    Per-round LOCAL early stopping: when `val_loader` is given and `patience` > 0,
    validate after every epoch, keep the best-epoch head weights, and stop early
    once val MSE has not improved for `patience` epochs (the model is left holding
    the best-epoch head). This mirrors --personalised-frozen, whose fresh per-dataset
    head trains with its own early stopping. `n_epochs` is then the epoch cap."""
    best_val = float("inf")
    best_head = None
    epochs_no_improve = 0
    last_loss = 0.0
    for _ in range(max(1, n_epochs)):
        model.train()
        if not encoder_train:
            model.encoder.eval()
        running = 0.0
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            target = data.edge_attr[:, :-1]
            z = model.encode(data.x, data.edge_index, target)
            pred = model.decode(z, data.edge_index, target, data.batch)
            loss = F.mse_loss(pred, target)
            loss.backward()
            optimizer.step()
            running += loss.item()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        last_loss = running / len(loader)

        if val_loader is not None and patience > 0:
            val = val_mse_fn(model, val_loader, device)
            if val < best_val:
                best_val = val
                best_head = {k: v.detach().clone() for k, v in head_state_dict(model).items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

    if best_head is not None:
        model.load_state_dict(best_head, strict=False)  # restore best-epoch head
    return last_loss


@app.train()
def train(msg: Message, context: Context) -> Message:
    cfg, partition_id, data, device, model, incoming = _client_setup(msg, context)
    if cfg.get("strategy", "fedavg") == "fedrep":
        return _train_fedrep(msg, cfg, partition_id, data, device, model, incoming)

    # ---- FedAvg: one local epoch over the whole model, reply with all weights ----
    model.load_state_dict(incoming)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                  weight_decay=cfg["weight-decay"])
    train_mse = train_one_epoch(model, optimizer, data["train_loader"], device)

    model.to("cpu")
    reply = RecordDict({
        "arrays": ArrayRecord(model.state_dict()),
        "metrics": MetricRecord({
            "train_mse": float(train_mse),
            "partition-id": partition_id,
            # FedAvg weight: clients contribute proportionally to their window count
            "num-examples": data["n_train_graphs"],
        }),
    })
    return Message(reply, reply_to=msg)


def _train_fedrep(msg, cfg, partition_id, data, device, model, incoming):
    # global body (encoder) overwrites the encoder; the head is this client's own
    model.load_state_dict(incoming, strict=False)
    cached_head = _client_heads.get(partition_id)
    if cached_head is not None:
        model.load_state_dict(cached_head, strict=False)

    body_modules = (model.encoder,)
    head_modules = (model.decoder, model.global_edge_embedding)
    loader = data["train_loader"]

    # Phase A: freeze the body, train the head to (near) convergence -- with its own
    # per-round early stopping on this client's local val split (fedrep-head-epochs
    # is the epoch cap; fedrep-head-patience=0 disables and runs the full cap).
    _set_requires_grad(body_modules, False)
    _set_requires_grad(head_modules, True)
    head_opt = torch.optim.AdamW(
        [p for m in head_modules for p in m.parameters()],
        lr=cfg["lr"], weight_decay=cfg["weight-decay"])
    _local_epochs(model, head_opt, loader, device, cfg["fedrep-head-epochs"],
                  encoder_train=False, val_loader=data["val_loader"],
                  patience=cfg["fedrep-head-patience"])

    # Phase B: freeze the head, take a few steps on the shared body.
    _set_requires_grad(head_modules, False)
    _set_requires_grad(body_modules, True)
    body_opt = torch.optim.AdamW(
        [p for m in body_modules for p in m.parameters()],
        lr=cfg["lr"], weight_decay=cfg["weight-decay"])
    train_mse = _local_epochs(model, body_opt, loader, device, cfg["fedrep-body-epochs"],
                              encoder_train=True)

    model.to("cpu")
    # persist this client's private head for the next round + reply
    head = {k: v.detach().clone() for k, v in head_state_dict(model).items()}
    _client_heads[partition_id] = head

    reply = RecordDict({
        # only the body is aggregated; the server sees "head" but never averages it
        "arrays": ArrayRecord(body_state_dict(model)),
        "head": ArrayRecord(head),
        "metrics": MetricRecord({
            "train_mse": float(train_mse),
            "partition-id": partition_id,
            "num-examples": data["n_train_graphs"],
        }),
    })
    return Message(reply, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    cfg, partition_id, data, device, model, incoming = _client_setup(msg, context)

    if cfg.get("strategy", "fedavg") == "fedrep":
        model.load_state_dict(incoming, strict=False)  # global body
        cached_head = _client_heads.get(partition_id)
        if cached_head is not None:
            model.load_state_dict(cached_head, strict=False)  # private head
    else:
        model.load_state_dict(incoming)

    val_mse = val_mse_fn(model, data["val_loader"], device)

    reply = RecordDict({
        "metrics": MetricRecord({
            "val_mse": float(val_mse),
            "partition-id": partition_id,
            # weight val MSE by edge count so the aggregate matches a pooled mean
            "num-examples": data["n_val_edges"],
        }),
    })
    return Message(reply, reply_to=msg)
