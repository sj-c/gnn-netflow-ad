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

# partition-id -> dataset; order matches the DATASETS insertion order
CLIENT_DATASETS = list(DATASETS)

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


# --- FedRep body/head split ------------------------------------------------
# FedRep = a shared representation (the "body", aggregated across clients via
# FedAvg) + a private per-client "head" (never aggregated). We use exactly the
# same split as train_and_inferv2.py's --personalised-frozen: the encoder is the
# shared representation; the decoder + global-edge embedding are the head. Both
# helpers return an ordered state-dict subset; load them back with
# `model.load_state_dict(subset, strict=False)` (loads only the matching keys,
# leaving the other half untouched).
BODY_PREFIX = "encoder."


def body_state_dict(model):
    """The shared-representation (encoder) parameters -- FedRep aggregates these."""
    return {k: v for k, v in model.state_dict().items() if k.startswith(BODY_PREFIX)}


def head_state_dict(model):
    """The private per-client head (decoder + global-edge embedding) -- kept local."""
    return {k: v for k, v in model.state_dict().items() if not k.startswith(BODY_PREFIX)}


def load_client_data(dataset_name, cfg):
    """Load one client's benign training split, fit its preprocessor, build the
    windowed graphs, and split train/val. Deterministic given the run config, so
    evaluate_federated.py can rebuild the exact same preprocessor and val split."""
    edge_columns = resolve_edge_columns(cfg["edge-columns"])
    log1p_columns = resolve_log1p_columns(cfg["log1p-columns"], edge_columns)
    data_dir = resolve_data_dir(cfg["data-dir"])

    df = read_parquet_head(data_dir / DATASETS[dataset_name][0], cfg["max-train-rows"])

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
    processor.df = None  # scaler/vocabs are fitted; free the raw rows

    edge_attr_dim = processor.feature_dim
    expected = edge_feature_dim(edge_columns, cfg["top-k-ports"], cfg["top-k-protocols"])
    if edge_attr_dim != expected:
        raise RuntimeError(
            f"{dataset_name}: padded edge feature dim {edge_attr_dim} != expected {expected}; "
            "clients would disagree on model shapes"
        )

    # seed right before the split so every process (client actor or the eval
    # script) reproduces the identical train/val partition of the windows
    torch.manual_seed(cfg["seed"] + CLIENT_DATASETS.index(dataset_name))
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
