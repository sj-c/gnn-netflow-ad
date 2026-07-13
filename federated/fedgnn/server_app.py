"""Flower ServerApp: standard Flower workflow (FedAvg/FedProx), SecAgg+-ready.

Uses Flower's stock workflow API instead of a hand-rolled round loop, which is
what makes SecAgg+ (and other protocol plugins) a drop-in:

    DefaultWorkflow(fit_workflow=SecAggPlusWorkflow(...))   # secagg = true
    DefaultWorkflow()                                       # secagg = false

There is NO early stopping and NO best-round model selection: training always
runs exactly `num-server-rounds` rounds and the final-round global model is
saved. (Server-side model selection would need per-client validation scores to
steer aggregation, which cuts against the secure-aggregation setting anyway.)

Run-config switches:
  shared-weight   -- "fedavg" | "fedprox" (FedProx: server injects proximal_mu
                     into the fit config; clients add the proximal term locally;
                     aggregation itself is identical to FedAvg)
  personalisation -- "na" | "fedbn" | "fedrep": which state-dict subset is
                     exchanged/aggregated (see fedgnn.task.exchanged_keys). The
                     private remainder never reaches the server; clients persist
                     it themselves for evaluation.
  secagg          -- true: wrap every fit round in the SecAgg+ secure-aggregation
                     protocol (the server only ever sees the masked sum of client
                     updates, never an individual update).
  dp-mode         -- "off" | "central" | "local" differential privacy.
                     central: strategy is wrapped in DifferentialPrivacyClientSide-
                     FixedClipping (clients clip to dp-clipping-norm via the
                     fixedclipping_mod client mod; the server adds Gaussian noise
                     to the aggregate). local: handled entirely client-side by
                     LocalDpMod -- no server wrapper (see client_app.py).

Outputs: checkpoints_infer/federated.pt (final global model + run config) and
results_infer/federated/history.json (per-round distributed losses/metrics).
"""

import json
import time
from logging import INFO

import torch
from flwr.common import Context, log, ndarrays_to_parameters
from flwr.server import LegacyContext, ServerConfig
from flwr.server.strategy import (
    DifferentialPrivacyClientSideFixedClipping,
    FedAvg,
    FedProx,
)
from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow
from flwr.server.workflow.constant import MAIN_PARAMS_RECORD
from flwr.serverapp import Grid, ServerApp

from fedgnn.task import (
    CLIENT_DATASETS,
    REPO_ROOT,
    build_model,
    edge_feature_dim,
    exchanged_keys,
)
from train_and_inferv2 import resolve_edge_columns

app = ServerApp()

SHARED_WEIGHT_MODES = ("fedavg", "fedprox")
DP_MODES = ("off", "central", "local")


def _build_initial_model(cfg):
    """The server builds the model shell from the run config alone (clients derive
    identical shapes from the same config), so it can seed the initial weights."""
    torch.manual_seed(cfg["seed"])
    edge_columns = resolve_edge_columns(cfg["edge-columns"])
    edge_attr_dim = edge_feature_dim(edge_columns, cfg["top-k-ports"], cfg["top-k-protocols"])
    return build_model(cfg, cfg["node-dim"], edge_attr_dim, torch.device("cpu"))


def _per_dataset_metrics(*keys):
    """Aggregation fn: fold each client's scalars into '<key>/<dataset>' entries
    so the History keeps a per-dataset learning curve for every given key."""
    def aggregate(results):
        out = {}
        for _, metrics in results:
            ds = metrics.get("dataset")
            if ds is None:
                continue
            for key in keys:
                if key in metrics:
                    out[f"{key}/{ds}"] = float(metrics[key])
        return out
    return aggregate


def _make_strategy(cfg, initial_parameters):
    shared_weight = cfg["shared-weight"]
    if shared_weight not in SHARED_WEIGHT_MODES:
        raise ValueError(f"unknown shared-weight {shared_weight!r}; "
                         f"choices: {SHARED_WEIGHT_MODES}")
    n_clients = len(CLIENT_DATASETS)
    common = dict(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=n_clients,
        min_evaluate_clients=n_clients,
        min_available_clients=n_clients,
        initial_parameters=initial_parameters,
        fit_metrics_aggregation_fn=_per_dataset_metrics("train_mse", "update_norm"),
        evaluate_metrics_aggregation_fn=_per_dataset_metrics("val_mse"),
    )
    if shared_weight == "fedprox":
        strategy = FedProx(proximal_mu=cfg["proximal-mu"], **common)
    else:
        strategy = FedAvg(**common)

    dp_mode = cfg["dp-mode"]
    if dp_mode not in DP_MODES:
        raise ValueError(f"unknown dp-mode {dp_mode!r}; choices: {DP_MODES}")
    if dp_mode == "central":
        # Client-level central DP, client-side clipping variant (the only one that
        # composes with SecAgg+: the server never sees an individual update, so it
        # cannot clip -- clients clip via fixedclipping_mod, the server adds
        # Gaussian noise (std = noise_multiplier*clipping_norm/n) to the mean.
        # "local" needs no server wrapper: clients noise their own updates.
        strategy = DifferentialPrivacyClientSideFixedClipping(
            strategy,
            noise_multiplier=float(cfg["dp-noise-multiplier"]),
            clipping_norm=float(cfg["dp-clipping-norm"]),
            num_sampled_clients=n_clients,
        )
    return strategy


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = dict(context.run_config)
    num_rounds = int(cfg["num-server-rounds"])
    personalisation = cfg["personalisation"]
    secagg = bool(cfg["secagg"])

    # seeded global init; only the exchanged (aggregated) subset crosses the wire
    model = _build_initial_model(cfg)
    keys = exchanged_keys(model, personalisation)
    init_sd = model.state_dict()
    initial_parameters = ndarrays_to_parameters([init_sd[k].numpy() for k in keys])

    strategy = _make_strategy(cfg, initial_parameters)
    legacy_context = LegacyContext(
        context,
        config=ServerConfig(num_rounds=num_rounds, round_timeout=cfg["round-timeout"]),
        strategy=strategy,
    )

    fit_workflow = None
    if secagg:
        fit_workflow = SecAggPlusWorkflow(
            num_shares=cfg["secagg-num-shares"],
            reconstruction_threshold=cfg["secagg-reconstruction-threshold"],
            clipping_range=cfg["clipping-range"],
            quantization_range=cfg["secagg-quantization-range"],
            max_weight=cfg["secagg-max-weight"],
            timeout=cfg["round-timeout"],
        )
        log(INFO, "SecAgg+ enabled: num_shares=%s reconstruction_threshold=%s "
            "clipping_range=%s quantization_range=%s",
            cfg["secagg-num-shares"], cfg["secagg-reconstruction-threshold"],
            cfg["clipping-range"], cfg["secagg-quantization-range"])
    workflow = DefaultWorkflow(fit_workflow=fit_workflow)

    log(INFO, "federated run: shared_weight=%s personalisation=%s secagg=%s "
        "rounds=%d local_epochs=%s (%d exchanged arrays, no early stopping)",
        cfg["shared-weight"], personalisation, secagg, num_rounds,
        cfg["local-epochs"], len(keys))
    if cfg["dp-mode"] != "off":
        log(INFO, "differential privacy: mode=%s noise_multiplier=%s "
            "clipping_norm=%s delta=%s",
            cfg["dp-mode"], cfg["dp-noise-multiplier"],
            cfg["dp-clipping-norm"], cfg["dp-delta"])

    t_start = time.time()
    workflow(grid, legacy_context)
    train_time = time.time() - t_start

    # final-round global model: the workflow keeps the current parameters in the
    # context state under the standard "parameters" record
    final_arrays = legacy_context.state.array_records[MAIN_PARAMS_RECORD]
    final_nds = final_arrays.to_numpy_ndarrays()
    global_sd = {k: torch.from_numpy(v.copy())
                 for k, v in zip(keys, final_nds, strict=True)}

    ckpt_path = REPO_ROOT / "checkpoints_infer" / "federated.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": global_sd,   # exchanged subset only; load strict=False
        "mode": "federated",
        "shared_weight": cfg["shared-weight"],
        "personalisation": personalisation,
        "secagg": secagg,
        "datasets": CLIENT_DATASETS,
        "run_config": cfg,
        "rounds_trained": num_rounds,
        "train_time_sec": train_time,
    }, ckpt_path)
    log(INFO, "saved final-round global model to %s (train_time=%.1fs)",
        ckpt_path, train_time)

    hist = legacy_context.history
    history_path = REPO_ROOT / "results_infer" / "federated" / "history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump({
            "rounds_trained": num_rounds,
            "shared_weight": cfg["shared-weight"],
            "personalisation": personalisation,
            "secagg": secagg,
            # everything a post-hoc accountant needs to map this run to epsilon
            "dp_mode": cfg["dp-mode"],
            "dp_noise_multiplier": cfg["dp-noise-multiplier"],
            "dp_clipping_norm": cfg["dp-clipping-norm"],
            "dp_delta": cfg["dp-delta"],
            "train_time_sec": train_time,
            # [(round, weighted-mean val MSE across clients), ...]
            "val_mse": hist.losses_distributed,
            # {"val_mse/<dataset>": [(round, value), ...], ...}
            "val_mse_by_dataset": hist.metrics_distributed,
            # {"train_mse/<dataset>": [(round, value), ...], ...}
            "train_mse_by_dataset": hist.metrics_distributed_fit,
        }, f, indent=2)
    log(INFO, "saved round history to %s", history_path)
