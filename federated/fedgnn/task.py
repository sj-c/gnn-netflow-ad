"""Shared plumbing for the federated version of train_and_infer.py.

Every client (= one NetFlow dataset) fits its OWN NetflowPreprocessor on its own
benign training rows: no raw data, scaler statistics, or vocabularies ever leave
the client. FedAvg only requires that every client's model has identical tensor
shapes, so the categorical vocabularies are padded to their fixed top-k size with
sentinel values that can never occur in the data (the padded one-hot slots simply
stay zero). The same one-hot slot can therefore mean a different port/protocol on
different clients; that is the price of silo-local preprocessing.
"""

import os
import sys
from pathlib import Path

# `flwr run` copies this package into an isolated build dir under ~/.flwr/apps,
# so walking up from __file__ no longer lands on the repo (and `src` /
# train_and_infer become unimportable). The ServerApp/ClientApp are spawned by a
# long-lived flower-superlink daemon whose environment predates the launcher, so
# neither PYTHONPATH nor an exported GNN_REPO_ROOT reliably reaches them. Resolve
# the repo root in order of decreasing reliability:
#   1. GNN_REPO_ROOT env var        (works when the daemon inherited the launcher env)
#   2. ~/.flwr/gnn_repo_root file   (daemon-immune: read from disk at import time)
#   3. walk up from __file__        (plain source-tree `python -m` / test usage)
_ROOT_STATE_FILE = Path.home() / ".flwr" / "gnn_repo_root"


def _resolve_repo_root():
    env_root = os.environ.get("GNN_REPO_ROOT")
    if env_root:
        return Path(env_root).resolve()
    if _ROOT_STATE_FILE.is_file():
        stored = _ROOT_STATE_FILE.read_text().strip()
        if stored:
            return Path(stored).resolve()
    return Path(__file__).resolve().parents[2]


REPO_ROOT = _resolve_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch_geometric.loader import DataLoader

from src.guided_gae_model import (
    DecoderWithGlobalEdge,
    GAEWithGlobalEdge,
    GATEncoderWithEdgeAttr,
    GlobalEdgeEmbedding,
)
from src.preprocess import CATEGORICAL_EDGE_COLUMNS, NetflowPreprocessor
from train_and_inferv2 import (
    DATASETS,
    read_parquet_head,
    resolve_edge_columns,
    resolve_log1p_columns,
    split_data,
)

# stable global dataset order -- NEVER changes with the federation composition,
# so each dataset's train/val split is reproducible whether or not it is a client
ALL_DATASETS = list(DATASETS)

# partition-id -> dataset; order matches the DATASETS insertion order. For the
# leave-one-dataset-out transfer study the held-out <dataset name> is dropped from
# the federation (the remaining ones become the clients) but can still be
# *evaluated* zero-shot. num-supernodes MUST equal len() below (use the
# local-simulation-gpu-3 federation when one dataset is held out).
#
# The held-out is resolved the SAME daemon-immune way as REPO_ROOT above: the
# flwr-superlink daemon that spawns the training apps does NOT inherit the
# launcher's environment, so an env-only FED_HELDOUT never reaches training and
# NOTHING would be held out (all 4 stay clients). Resolve in order of decreasing
# reliability:
#   1. FED_HELDOUT env var             (works when the daemon inherited the env,
#                                        e.g. the direct-subprocess eval)
#   2. ~/.flwr/gnn_fed_heldout file    (daemon-immune: read from disk at import;
#                                        run_federated_gpu.sh writes it every run,
#                                        empty when no dataset is held out)
_HELDOUT_STATE_FILE = Path.home() / ".flwr" / "gnn_fed_heldout"


def _resolve_heldout():
    env_held = os.environ.get("FED_HELDOUT", "").strip()
    if env_held:
        return env_held
    if _HELDOUT_STATE_FILE.is_file():
        stored = _HELDOUT_STATE_FILE.read_text().strip()
        if stored:
            return stored
    return ""


_HELDOUT = _resolve_heldout()
if _HELDOUT and _HELDOUT not in ALL_DATASETS:
    raise ValueError(f"FED_HELDOUT={_HELDOUT!r} not in {ALL_DATASETS}")

# IID single-dataset control: instead of 4 clients = 4 different datasets
# (extreme non-IID), split ONE dataset into N_IID_SHARDS disjoint IID shards, one
# per client. Isolates how much of a result is driven by client heterogeneity vs
# other factors by removing the non-IID split. Resolved the
# same daemon-immune way as _HELDOUT: env var first, then ~/.flwr/gnn_fed_single
# (run_federated_gpu.sh writes it every run, empty when unset). A client name is
# then "<dataset>#<shard>", e.g. "NF-CICIDS2018-v3#0"; parse_client_name() splits
# it back into (base dataset, shard index) everywhere the base is needed.
N_IID_SHARDS = 4
_SINGLE_STATE_FILE = Path.home() / ".flwr" / "gnn_fed_single"


def _resolve_single():
    env_single = os.environ.get("FED_SINGLE_DATASET", "").strip()
    if env_single:
        return env_single
    if _SINGLE_STATE_FILE.is_file():
        stored = _SINGLE_STATE_FILE.read_text().strip()
        if stored:
            return stored
    return ""


def parse_client_name(name):
    """Split a client name into (base dataset, shard index or None). Plain dataset
    names (non-IID mode) return (name, None); IID shard names "<dataset>#<k>"
    return (dataset, k)."""
    if "#" in name:
        base, shard = name.rsplit("#", 1)
        return base, int(shard)
    return name, None


_SINGLE = _resolve_single()
if _SINGLE:
    if _SINGLE not in ALL_DATASETS:
        raise ValueError(f"FED_SINGLE_DATASET={_SINGLE!r} not in {ALL_DATASETS}")
    if _HELDOUT:
        raise ValueError("FED_SINGLE_DATASET and FED_HELDOUT are mutually exclusive")
    CLIENT_DATASETS = [f"{_SINGLE}#{k}" for k in range(N_IID_SHARDS)]
else:
    CLIENT_DATASETS = [d for d in ALL_DATASETS if d != _HELDOUT]

# per-process cache so a client only loads data + builds graphs once, not every round
_client_data_cache = {}


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_data_dir(data_dir_cfg):
    """Like train_and_infer.resolve_data_dir, but anchored on the repo root because
    the ClientApp/ServerApp processes do not run with the repo as their cwd."""
    if data_dir_cfg:
        path = Path(data_dir_cfg)
        if not path.exists():
            raise FileNotFoundError(f"data-dir {path} does not exist")
        return path
    for candidate in (REPO_ROOT / "netflow" / "parquet", REPO_ROOT.parent / "netflow" / "parquet"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not auto-locate netflow/parquet under {REPO_ROOT} or {REPO_ROOT.parent}; "
        "set data-dir in the run config."
    )


def pad_vocabs(processor):
    """Pad the fitted categorical vocabularies to their fixed top-k length with
    negative sentinels (real protocol/port codes are non-negative), so every
    client's one-hot blocks -- and hence model tensors -- have identical shapes."""
    if processor.protocol_vocab is not None:
        pad = processor.top_k_protocols - len(processor.protocol_vocab)
        processor.protocol_vocab += [-(i + 1) for i in range(pad)]
    if processor.dst_port_vocab is not None:
        pad = processor.top_k_ports - len(processor.dst_port_vocab)
        processor.dst_port_vocab += [-(i + 1) for i in range(pad)]


def edge_feature_dim(edge_columns, top_k_ports, top_k_protocols):
    """Transformed edge-feature dim, mirroring NetflowPreprocessor.feature_names
    with padded vocabs; lets the server build the initial model without any data."""
    dim = len([c for c in edge_columns if c not in CATEGORICAL_EDGE_COLUMNS])
    if "PROTOCOL" in edge_columns:
        dim += top_k_protocols + 1  # vocab + other
    if "L4_SRC_PORT" in edge_columns:
        dim += 3  # wellknown / registered / ephemeral range buckets
    if "L4_DST_PORT" in edge_columns:
        dim += top_k_ports + 2  # vocab + other + ephemeral
    return dim


def build_model(cfg, in_channels, edge_attr_dim, device):
    encoder = GATEncoderWithEdgeAttr(in_channels, cfg["hidden-dim"], edge_attr_dim,
                                     num_heads=cfg["heads"])
    global_edge_embedding = GlobalEdgeEmbedding(edge_attr_dim, cfg["global-emb-dim"])
    decoder = DecoderWithGlobalEdge(cfg["hidden-dim"], edge_attr_dim, cfg["global-emb-dim"])
    return GAEWithGlobalEdge(encoder, decoder, global_edge_embedding).to(device)


# --- shared/private parameter split ------------------------------------------
# The personalisation mode decides which state-dict entries cross the wire and
# get aggregated ("exchanged") and which stay on the client ("private"):
#
#   "na"     -- everything is exchanged (one fully-shared global model).
#   "fedbn"  -- FedBN: BatchNorm parameters AND running stats stay local; every
#               client keeps its own normalisation statistics.
#   "fedrep" -- FedRep: only the encoder (the shared representation) is
#               exchanged; the head (decoder + global-edge embedding) is private,
#               exactly the --personalised-frozen split in train_and_inferv2.py.
#
# `num_batches_tracked` buffers are NEVER exchanged: they are int64 counters, and
# SecAgg+ clips every exchanged value to [-clipping_range, clipping_range] before
# quantising, which would corrupt them. (PyTorch only consults them when BN
# momentum is None; ours uses the default momentum, so keeping them local is
# harmless.)
#
# Every client derives the identical, deterministically-ordered key list from the
# same run config, so the flat ndarray list exchanged with the server is uniform
# across clients -- the property SecAgg+ masking relies on.
ENCODER_PREFIX = "encoder."

PERSONALISATION_MODES = ("na", "fedbn", "fedrep")


def _batchnorm_prefixes(model):
    """State-dict key prefixes of every BatchNorm module in the model."""
    return tuple(
        f"{name}." for name, m in model.named_modules()
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
    )


def exchanged_keys(model, personalisation):
    """Ordered state-dict keys that are sent to / received from the server."""
    if personalisation not in PERSONALISATION_MODES:
        raise ValueError(f"unknown personalisation {personalisation!r}; "
                         f"choices: {PERSONALISATION_MODES}")
    keys = [k for k in model.state_dict() if not k.endswith("num_batches_tracked")]
    if personalisation == "fedbn":
        bn = _batchnorm_prefixes(model)
        keys = [k for k in keys if not k.startswith(bn)]
    elif personalisation == "fedrep":
        keys = [k for k in keys if k.startswith(ENCODER_PREFIX)]
    return keys


def private_keys(model, personalisation):
    """State-dict keys that never leave the client (complement of exchanged_keys)."""
    shared = set(exchanged_keys(model, personalisation))
    return [k for k in model.state_dict() if k not in shared]


def private_state_path(dataset_name):
    """Where a client persists its private (non-aggregated) parameters each round
    so evaluate_federated.py can pair them with the final global model. In the
    simulation all clients share this filesystem; in a real deployment this file
    simply stays on the client, which is where per-client evaluation happens."""
    return REPO_ROOT / "checkpoints_infer" / "federated_private" / f"{dataset_name}.pt"


def load_client_data(dataset_name, cfg):
    """Load one client's benign training split, fit its preprocessor, build the
    windowed graphs, and split train/val. Deterministic given the run config, so
    evaluate_federated.py can rebuild the exact same preprocessor and val split."""
    base_name, shard_idx = parse_client_name(dataset_name)
    edge_columns = resolve_edge_columns(cfg["edge-columns"])
    log1p_columns = resolve_log1p_columns(cfg["log1p-columns"], edge_columns)
    data_dir = resolve_data_dir(cfg["data-dir"])

    df = read_parquet_head(data_dir / DATASETS[base_name][0], cfg["max-train-rows"])

    processor = NetflowPreprocessor(
        df,
        edge_columns=edge_columns,
        node_dim=cfg["node-dim"],
        log1p_columns=log1p_columns,
        top_k_ports=cfg["top-k-ports"],
        top_k_protocols=cfg["top-k-protocols"],
    )
    pad_vocabs(processor)

    graphs, _, _ = processor.construct_graph_list(
        window_size=cfg["window-size"], step_size=cfg["step-size"]
    )
    if not graphs:
        raise RuntimeError(f"no training graphs built for {dataset_name}; "
                           "check data / window-size")

    # IID single-dataset control: keep only this client's disjoint 1/N_IID_SHARDS
    # slice of the windows. The permutation is seeded by cfg["seed"] ALONE (NOT the
    # shard index), so every shard-client draws the SAME shuffle and the contiguous
    # slices are guaranteed disjoint -- an IID partition of one dataset across the
    # clients. No-op in the normal non-IID mode (shard_idx is None).
    if shard_idx is not None:
        gen = torch.Generator().manual_seed(int(cfg["seed"]))
        perm = torch.randperm(len(graphs), generator=gen).tolist()
        n = len(graphs)
        lo = shard_idx * n // N_IID_SHARDS
        hi = (shard_idx + 1) * n // N_IID_SHARDS
        graphs = [graphs[i] for i in perm[lo:hi]]
        if not graphs:
            raise RuntimeError(
                f"IID shard {shard_idx}/{N_IID_SHARDS} of {base_name} is empty "
                f"({n} windows); use fewer shards or more max-train-rows")

    processor.df = None  # scaler/vocabs are fitted; free the raw rows

    edge_attr_dim = processor.feature_dim
    expected = edge_feature_dim(edge_columns, cfg["top-k-ports"], cfg["top-k-protocols"])
    if edge_attr_dim != expected:
        raise RuntimeError(
            f"{dataset_name}: padded edge feature dim {edge_attr_dim} != expected {expected}; "
            "clients would disagree on model shapes"
        )

    # seed right before the split so every process (client actor or the eval
    # script) reproduces the identical train/val partition of the windows. Use
    # the STABLE global index so a dataset's split does not shift when the
    # federation is smaller (leave-one-out), and so a held-out dataset -- absent
    # from CLIENT_DATASETS -- can still be evaluated without an index error.
    # stable, per-client offset so each client's train/val split differs but is
    # reproducible in eval. Non-IID: the dataset's global index. IID shards: the
    # base index times N_IID_SHARDS plus the shard, so the shards get distinct splits.
    base_idx = ALL_DATASETS.index(base_name)
    split_offset = base_idx if shard_idx is None else base_idx * N_IID_SHARDS + shard_idx
    torch.manual_seed(cfg["seed"] + split_offset)
    train_dataset, val_dataset = split_data(graphs, train_ratio=cfg["train-ratio"])

    return {
        "processor": processor,
        "train_loader": DataLoader(train_dataset, batch_size=cfg["batch-size"], shuffle=True),
        "val_loader": DataLoader(val_dataset, batch_size=cfg["batch-size"], shuffle=False),
        "n_train_graphs": len(train_dataset),
        "n_val_edges": int(sum(g.num_edges for g in val_dataset)),
        "in_channels": graphs[0].x.size(1),
        "edge_attr_dim": edge_attr_dim,
    }


def get_client_data(dataset_name, cfg):
    if dataset_name not in _client_data_cache:
        _client_data_cache[dataset_name] = load_client_data(dataset_name, cfg)
    return _client_data_cache[dataset_name]
