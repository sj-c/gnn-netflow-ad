"""Flower ClientApp: one client per NetFlow dataset (NumPyClient + SecAgg+ mod).

The client is a plain ``NumPyClient`` so it slots straight into Flower's standard
workflow API -- including the SecAgg+ secure-aggregation protocol, whose client
mod (``secaggplus_mod``) expects exactly the FitRes-shaped replies a NumPyClient
produces. There is no early stopping anywhere: every round is `local-epochs`
epochs of reconstruction-MSE training, and the server keeps the final round.

Two orthogonal run-config switches shape what happens locally:

shared-weight ("fedavg" | "fedprox"):
  fedprox adds the proximal term (mu/2)*||w - w_global||^2 to every batch loss,
  pulling local updates towards the incoming global weights. The server-side
  FedProx strategy injects ``proximal_mu`` into the fit config; aggregation is
  identical to FedAvg, so SecAgg+ works unchanged.

personalisation ("na" | "fedbn" | "fedrep"):
  Decides which state-dict keys are exchanged with the server (see
  fedgnn.task.exchanged_keys). The private remainder lives in this process across
  rounds (each simulated client runs in a long-lived Ray actor) and is persisted
  to checkpoints_infer/federated_private/<dataset>.pt after every fit so
  evaluate_federated.py can pair it with the final global model -- the private
  parameters themselves never cross the wire.
  fedrep trains in the usual alternating fashion each round: (A) freeze the
  shared encoder (body) and train the private head, then (B) freeze the head and
  train the body. Both phases run `local-epochs` epochs.

SecAgg+ (secagg=true):
  ``secaggplus_or_passthrough`` delegates to Flower's secaggplus_mod whenever the
  incoming TRAIN message carries the SecAgg+ config record and passes plain
  messages straight through, so one ClientApp serves both secagg settings.

Differential privacy (dp-mode = "off" | "central" | "local"):
  ``dp_dispatch`` routes TRAIN replies through fixedclipping_mod (central: clip
  here, noise at the server) or LocalDpMod (local: clip AND noise here) based on
  the run config; sigma/C/delta knobs are shared with the server side. With
  dp-log-norms=true, fit() also reports the pre-clip L2 norm of the exchanged
  update ("update_norm") for calibrating dp-clipping-norm.
"""

import logging
import math

import torch
import torch.nn.functional as F
from flwr.client import ClientApp, NumPyClient
from flwr.client.mod import LocalDpMod, fixedclipping_mod, secaggplus_mod
from flwr.common import Context, Message
from flwr.common.secure_aggregation.secaggplus_constants import RECORD_KEY_CONFIGS

from fedgnn.task import (
    CLIENT_DATASETS,
    build_model,
    exchanged_keys,
    get_client_data,
    get_device,
    private_keys,
    private_state_path,
)
from train_and_inferv2 import test as val_mse_fn

_LOGGER = logging.getLogger("fedgnn.client")

# Long-lived per-actor caches (same assumption that lets task.py cache a client's
# data across rounds): the client object keeps the model resident, and the private
# (non-exchanged) parameters survive from one round to the next.
_clients = {}
_private_state = {}


def _run_epochs(model, params, loader, device, epochs, lr, weight_decay,
                prox_mu=0.0, prox_pairs=(), encoder_eval=False):
    """`epochs` reconstruction-MSE epochs over `params` with a fresh AdamW.

    prox_mu/prox_pairs -- FedProx: add (mu/2)*sum ||p - p_global||^2 over the
        given (local_param, global_snapshot) pairs to every batch loss.
    encoder_eval -- hold the encoder in eval mode so its BatchNorm running stats
        stay frozen while only the head trains (FedRep phase A); mirrors
        freeze_encoder's eval-lock in train_and_inferv2.
    """
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    last_loss = 0.0
    for _ in range(epochs):
        model.train()
        if encoder_eval:
            model.encoder.eval()
        running = 0.0
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            target = data.edge_attr[:, :-1]
            z = model.encode(data.x, data.edge_index, target)
            pred = model.decode(z, data.edge_index, target, data.batch)
            loss = F.mse_loss(pred, target)
            if prox_mu > 0.0:
                prox = sum(((p - g) ** 2).sum() for p, g in prox_pairs)
                loss = loss + (prox_mu / 2.0) * prox
            loss.backward()
            optimizer.step()
            running += loss.item()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        last_loss = running / len(loader)
    return last_loss


def _set_requires_grad(modules, flag):
    for module in modules:
        for p in module.parameters():
            p.requires_grad_(flag)


class NetflowClient(NumPyClient):
    """One NetFlow dataset = one client. Exchanges the personalisation-dependent
    subset of the state dict as a flat ndarray list (uniform across clients, which
    is what SecAgg+ masking requires)."""

    def __init__(self, partition_id, cfg):
        self.cfg = cfg
        self.pid = partition_id
        self.dataset = CLIENT_DATASETS[partition_id]
        self.data = get_client_data(self.dataset, cfg)
        self.device = get_device()
        # deterministic model init per (seed, client): the exchanged part is
        # overwritten by the server every round, but a FedRep/FedBN client's
        # private part starts from this init in round 1
        torch.manual_seed(cfg["seed"] + 100 + partition_id)
        self.model = build_model(cfg, self.data["in_channels"],
                                 self.data["edge_attr_dim"], self.device)
        self.personalisation = cfg["personalisation"]
        self.exchanged = exchanged_keys(self.model, self.personalisation)
        self.private = private_keys(self.model, self.personalisation)

    # --- exchanged/private state plumbing ---------------------------------
    def _load(self, parameters):
        """Global exchanged arrays + this client's cached private part -> model."""
        incoming = {k: torch.from_numpy(v.copy())
                    for k, v in zip(self.exchanged, parameters, strict=True)}
        self.model.load_state_dict(incoming, strict=False)
        cached = _private_state.get(self.pid)
        if cached is not None:
            self.model.load_state_dict(cached, strict=False)
        self.model.to(self.device)

    def _exchanged_arrays(self):
        sd = self.model.state_dict()
        return [sd[k].detach().cpu().numpy() for k in self.exchanged]

    def _persist_private(self):
        """Cache the private part for the next round and (for fedbn/fedrep) write
        it to disk for evaluate_federated.py. Written every round, so after the
        final round the file holds the final-round private parameters."""
        sd = self.model.state_dict()
        state = {k: sd[k].detach().cpu().clone() for k in self.private}
        _private_state[self.pid] = state
        if self.personalisation != "na":
            path = private_state_path(self.dataset)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(state, path)

    def _agg_weight(self):
        """Aggregation weight ("num examples"): "equal" (default) gives every
        dataset the same pull on the shared model so the largest one does not
        dominate; "examples" is classic FedAvg weighting."""
        if self.cfg.get("client-weight", "equal") == "equal":
            return 1
        return int(self.data["n_train_graphs"])

    # --- Flower entry points -----------------------------------------------
    def fit(self, parameters, config):
        self._load(parameters)
        prox_mu = float(config.get("proximal_mu", 0.0))
        epochs = int(self.cfg["local-epochs"])
        lr, wd = self.cfg["lr"], self.cfg["weight-decay"]

        # FedProx pulls the trained parameters towards the *incoming* global ones
        prox_pairs = ()
        if prox_mu > 0.0:
            named = dict(self.model.named_parameters())
            prox_pairs = [(named[k], named[k].detach().clone())
                          for k in self.exchanged if k in named]

        if self.personalisation == "fedrep":
            body = (self.model.encoder,)
            head = (self.model.decoder, self.model.global_edge_embedding)
            # Phase A: freeze the shared body, train the private head
            _set_requires_grad(body, False)
            _set_requires_grad(head, True)
            _run_epochs(self.model, [p for m in head for p in m.parameters()],
                        self.data["train_loader"], self.device, epochs, lr, wd,
                        encoder_eval=True)
            # Phase B: freeze the head, train the shared body (prox on the body)
            _set_requires_grad(head, False)
            _set_requires_grad(body, True)
            train_mse = _run_epochs(self.model, [p for m in body for p in m.parameters()],
                                    self.data["train_loader"], self.device, epochs, lr, wd,
                                    prox_mu=prox_mu, prox_pairs=prox_pairs)
            _set_requires_grad(head, True)
        else:
            train_mse = _run_epochs(self.model, self.model.parameters(),
                                    self.data["train_loader"], self.device, epochs, lr, wd,
                                    prox_mu=prox_mu, prox_pairs=prox_pairs)

        self._persist_private()
        metrics = {
            "train_mse": float(train_mse),
            "dataset": self.dataset,
            "n_train_graphs": int(self.data["n_train_graphs"]),
        }
        if self.cfg.get("dp-log-norms", False):
            # pre-clip L2 norm of this round's exchanged update (`parameters` is
            # the untouched incoming global model) -- the calibration signal for
            # dp-clipping-norm; folded into history.json as update_norm/<dataset>
            sq = sum(float(((new.astype("float64") - old.astype("float64")) ** 2).sum())
                     for old, new in zip(parameters, self._exchanged_arrays(), strict=True))
            metrics["update_norm"] = math.sqrt(sq)
        return self._exchanged_arrays(), self._agg_weight(), metrics

    def evaluate(self, parameters, config):
        """Score the aggregated global model (+ this client's private part) on the
        local benign val split; the loss is weighted by edge count so the server
        aggregate matches a pooled per-element mean."""
        self._load(parameters)
        val_mse = val_mse_fn(self.model, self.data["val_loader"], self.device)
        return float(val_mse), int(self.data["n_val_edges"]), {
            "val_mse": float(val_mse),
            "dataset": self.dataset,
        }


def client_fn(context: Context):
    cfg = dict(context.run_config)
    pid = int(context.node_config["partition-id"])
    if pid not in _clients:
        _clients[pid] = NetflowClient(pid, cfg)
    return _clients[pid].to_client()


def secaggplus_or_passthrough(msg: Message, ctxt: Context, call_next) -> Message:
    """Route SecAgg+ protocol messages through Flower's secaggplus_mod and plain
    rounds straight to the app. secaggplus_mod raises a KeyError on TRAIN messages
    without the SecAgg+ config record, so this makes `secagg` a pure run-config
    switch handled by a single ClientApp."""
    if RECORD_KEY_CONFIGS not in msg.content.config_records:
        return call_next(msg, ctxt)
    return secaggplus_mod(msg, ctxt, call_next)


# --- differential privacy (dp-mode run-config switch) ---------------------------
_local_dp_mods = {}


def _local_dp_mod_for(run_config):
    """LocalDpMod parameterised from the same sigma knob central DP uses.

    Flower's LocalDpMod takes a per-round epsilon, not a noise multiplier, so we
    invert the classic Gaussian mechanism (noise_std = sensitivity *
    sqrt(2*ln(1.25/delta)) / eps) at sensitivity = C to get the eps whose noise
    std equals sigma*C. Central and local runs at the same sigma are then matched
    in per-client per-round noise, keeping dp-noise-multiplier the single swept
    axis for both modes. (Cached: the mod is stateless across rounds.)"""
    sigma = float(run_config["dp-noise-multiplier"])
    clip = float(run_config["dp-clipping-norm"])
    delta = float(run_config["dp-delta"])
    key = (sigma, clip, delta)
    if key not in _local_dp_mods:
        epsilon = math.sqrt(2.0 * math.log(1.25 / delta)) / sigma
        _LOGGER.info("local DP: sigma=%s -> per-round epsilon=%.4f (C=%s, delta=%s)",
                     sigma, epsilon, clip, delta)
        _local_dp_mods[key] = LocalDpMod(
            clipping_norm=clip, sensitivity=clip, epsilon=epsilon, delta=delta)
    return _local_dp_mods[key]


def dp_dispatch(msg: Message, ctxt: Context, call_next) -> Message:
    """Apply the client-side DP mod selected by run-config `dp-mode`:
      central -- fixedclipping_mod clips the update to the norm the server wrapper
                 sends along (noise is added server-side to the aggregate)
      local   -- LocalDpMod clips AND noises the update before it leaves the client
    Both mods pass non-TRAIN messages straight through. This mod sits AFTER the
    SecAgg+ mod in the mods list (= closer to the app), so clipping/noising happens
    on the raw update before SecAgg+ masks the reply."""
    mode = str(ctxt.run_config.get("dp-mode", "off"))
    if mode == "central":
        return fixedclipping_mod(msg, ctxt, call_next)
    if mode == "local":
        return _local_dp_mod_for(ctxt.run_config)(msg, ctxt, call_next)
    if mode != "off":
        raise ValueError(f"unknown dp-mode {mode!r}; choices: off, central, local")
    return call_next(msg, ctxt)


app = ClientApp(client_fn=client_fn, mods=[secaggplus_or_passthrough, dp_dispatch])
