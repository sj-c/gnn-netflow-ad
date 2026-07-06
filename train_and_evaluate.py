# SPDX-License-Identifier: Apache-2.0
"""Train + evaluate the repo's Guided GAE (src/guided_gae_model.py) link-prediction
anomaly detector for each NetFlow v3 dataset in netflow/parquet/.

This follows the original training-testing.ipynb methodology exactly:
  - Encoder: GATEncoderWithEdgeAttr (GATv2Conv + GraphUNet)
  - Global edge context: GlobalEdgeEmbedding
  - Decoder: DecoderWithGlobalEdge -> sigmoid probability that an edge exists
  - Trained one-class on benign flows only, via BCE + negative edge sampling
  - Anomaly score at inference = 1 - P(edge exists); best cutoff chosen by
    Youden's J statistic (max TPR - FPR) on the ROC curve, same as the notebook.

Run `python train_and_evaluate.py --help` for all options.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import random_split
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import negative_sampling

from src.guided_gae_model import (
    DecoderWithGlobalEdge,
    GAEWithGlobalEdge,
    GATEncoderWithEdgeAttr,
    GlobalEdgeEmbedding,
)

# --------------------------------------------------------------------------
# Dataset / schema configuration (NetFlow v3 datasets, all share the same schema)
# --------------------------------------------------------------------------

DATASETS = {
    "NF-BoT-IoT-v3": ("NF-BoT-IoT-v3_train.parquet", "NF-BoT-IoT-v3_holdout.parquet"),
    "NF-UNSW-NB15-v3": ("NF-UNSW-NB15-v3_train.parquet", "NF-UNSW-NB15-v3_holdout.parquet"),
    "NF-ToN-IoT-v3": ("NF-ToN-IoT-v3_train.parquet", "NF-ToN-IoT-v3_holdout.parquet"),
    "NF-CICIDS2018-v3": ("NF-CICIDS2018-v3_train.parquet", "NF-CICIDS2018-v3_holdout.parquet"),
}

SRC_IP_COL = "IPV4_SRC_ADDR"
DST_IP_COL = "IPV4_DST_ADDR"
LABEL_COL = "Label"

BASELINE_EDGE_COLUMNS = ["IN_BYTES", "OUT_BYTES", "FLOW_DURATION_MILLISECONDS"]

ALL_EDGE_COLUMNS = [
    "L7_PROTO", "IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS", "TCP_FLAGS",
    "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS", "FLOW_DURATION_MILLISECONDS",
    "DURATION_IN", "DURATION_OUT", "MIN_TTL", "MAX_TTL", "LONGEST_FLOW_PKT",
    "SHORTEST_FLOW_PKT", "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN",
    "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES",
    "RETRANSMITTED_IN_BYTES", "RETRANSMITTED_IN_PKTS", "RETRANSMITTED_OUT_BYTES",
    "RETRANSMITTED_OUT_PKTS", "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT",
    "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES", "NUM_PKTS_256_TO_512_BYTES",
    "NUM_PKTS_512_TO_1024_BYTES", "NUM_PKTS_1024_TO_1514_BYTES",
    "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT", "ICMP_TYPE", "ICMP_IPV4_TYPE",
    "DNS_QUERY_ID", "DNS_QUERY_TYPE", "DNS_TTL_ANSWER", "FTP_COMMAND_RET_CODE",
    "SRC_TO_DST_IAT_MIN", "SRC_TO_DST_IAT_MAX", "SRC_TO_DST_IAT_AVG", "SRC_TO_DST_IAT_STDDEV",
    "DST_TO_SRC_IAT_MIN", "DST_TO_SRC_IAT_MAX", "DST_TO_SRC_IAT_AVG", "DST_TO_SRC_IAT_STDDEV",
]

# --------------------------------------------------------------------------
# Graph construction
# --------------------------------------------------------------------------

def ip_octets(ip_values):
    """Vectorized IPv4 string -> 4 normalized octet floats, malformed -> zeros."""
    s = pd.Series(ip_values, dtype="string")
    parts = s.str.extract(r"^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$")
    parts = parts.apply(pd.to_numeric, errors="coerce")
    parts = parts.fillna(0.0).clip(upper=255)
    return (parts.to_numpy(dtype=np.float32) / 255.0)


def build_graph(window_df, edge_columns, scaler):
    """One window of flows -> a PyG Data graph (nodes=unique IPs, edges=flows).

    `edge_attr` layout matches src/preprocess.py: scaled edge feature columns
    followed by a final Label column (0=benign, 1=attack; 0 for all rows when
    building benign-only training windows).
    """
    src = window_df[SRC_IP_COL].to_numpy()
    dst = window_df[DST_IP_COL].to_numpy()
    n = len(window_df)
    unique_ips, inverse = np.unique(np.concatenate([src, dst]), return_inverse=True)
    src_idx = inverse[:n]
    dst_idx = inverse[n:]

    edge_index = torch.tensor(np.stack([src_idx, dst_idx]), dtype=torch.long)

    raw_attr = window_df[edge_columns].to_numpy(dtype=np.float64)
    scaled_attr = scaler.transform(raw_attr).astype(np.float32)
    labels = window_df[LABEL_COL].to_numpy(dtype=np.float32) if LABEL_COL in window_df.columns else np.zeros(n, dtype=np.float32)
    edge_attr = torch.tensor(np.concatenate([scaled_attr, labels[:, None]], axis=1), dtype=torch.float32)

    x = torch.tensor(ip_octets(unique_ips), dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(unique_ips))
    return data


def iter_windows(parquet_path, columns, window_size, chunk_rows, label_filter=None, row_limit=None):
    """Stream a parquet file and yield non-overlapping row-windows as DataFrames.

    Reads in `chunk_rows`-sized record batches (bounded memory even on
    multi-GB / tens-of-millions-of-rows files) and slices each accumulated
    buffer into `window_size`-row windows.
    """
    pf = pq.ParquetFile(parquet_path)
    buffer = None
    rows_seen = 0
    for batch in pf.iter_batches(batch_size=chunk_rows, columns=columns):
        df = batch.to_pandas()
        if label_filter is not None:
            df = df[df[LABEL_COL] == label_filter]
        rows_seen += len(df)
        if df.empty:
            if row_limit is not None and rows_seen >= row_limit:
                break
            continue
        buffer = df if buffer is None else pd.concat([buffer, df], ignore_index=True)
        while len(buffer) >= window_size:
            yield buffer.iloc[:window_size].reset_index(drop=True)
            buffer = buffer.iloc[window_size:].reset_index(drop=True)
        if row_limit is not None and rows_seen >= row_limit:
            buffer = None
            break
    if buffer is not None and len(buffer) > 0:
        yield buffer.reset_index(drop=True)


def fit_scaler(parquet_path, edge_columns, chunk_rows, row_limit=None):
    """Fit a StandardScaler on benign (Label==0) edge features, streaming the file."""
    scaler = StandardScaler()
    pf = pq.ParquetFile(parquet_path)
    columns = edge_columns + [LABEL_COL]
    rows_seen = 0
    for batch in pf.iter_batches(batch_size=chunk_rows, columns=columns):
        df = batch.to_pandas()
        df = df[df[LABEL_COL] == 0]
        if not df.empty:
            scaler.partial_fit(df[edge_columns].to_numpy(dtype=np.float64))
        rows_seen += len(df)
        if row_limit is not None and rows_seen >= row_limit:
            break
    return scaler


def build_train_graphs(parquet_path, edge_columns, scaler, window_size, chunk_rows, row_limit=None):
    """Materialize benign-only training windows as a list of PyG Data graphs."""
    columns = [SRC_IP_COL, DST_IP_COL] + edge_columns + [LABEL_COL]
    graphs = []
    for window_df in iter_windows(
        parquet_path, columns, window_size, chunk_rows, label_filter=0, row_limit=row_limit
    ):
        if len(window_df) < 2:
            continue
        data = build_graph(window_df, edge_columns, scaler)
        graphs.append(data)
    return graphs


# --------------------------------------------------------------------------
# Training (BCE + negative edge sampling, matching training-testing.ipynb)
# --------------------------------------------------------------------------

def train_epoch(model, optimizer, loader, device):
    model.train()
    total_loss = 0.0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        z = model.encode(data.x, data.edge_index, data.edge_attr[:, :-1])

        pos_edge_index = data.edge_index
        pos_edge_attr = data.edge_attr[:, :-1]

        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=pos_edge_index.size(1),
            method="sparse",
        )
        neg_edge_attr = torch.zeros(neg_edge_index.size(1), data.edge_attr.size(1) - 1, device=device)

        pos_pred = model.decode(z, pos_edge_index, pos_edge_attr, data.batch)
        neg_pred = model.decode(z, neg_edge_index, neg_edge_attr, data.batch)

        preds = torch.cat([pos_pred, neg_pred], dim=0)
        labels = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)], dim=0)

        loss = F.binary_cross_entropy(preds, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate_epoch(model, loader, device):
    """Self-supervised sanity check on held-out benign windows: can the model
    still tell real (positive) edges from sampled negative ones?"""
    model.eval()
    preds, labels = [], []
    for data in loader:
        data = data.to(device)
        z = model.encode(data.x, data.edge_index, data.edge_attr[:, :-1])

        pos_edge_index = data.edge_index
        pos_edge_attr = data.edge_attr[:, :-1]
        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=pos_edge_index.size(1),
            method="sparse",
        )
        neg_edge_attr = torch.zeros(neg_edge_index.size(1), data.edge_attr.size(1) - 1, device=device)

        pos_pred = model.decode(z, pos_edge_index, pos_edge_attr, data.batch)
        neg_pred = model.decode(z, neg_edge_index, neg_edge_attr, data.batch)

        preds.append(torch.cat([pos_pred, neg_pred], dim=0).cpu())
        labels.append(torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)], dim=0).cpu())

    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    return roc_auc_score(labels, preds), average_precision_score(labels, preds)


def train_model(train_graphs, in_dim, edge_attr_dim, args, device, logger):
    val_size = max(1, int(len(train_graphs) * args.val_ratio)) if len(train_graphs) > 1 else 0
    train_size = len(train_graphs) - val_size
    if val_size > 0:
        train_split, val_split = random_split(train_graphs, [train_size, val_size])
    else:
        train_split, val_split = train_graphs, []

    train_loader = DataLoader(train_split, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_split, batch_size=args.batch_size, shuffle=False) if len(val_split) else None

    encoder = GATEncoderWithEdgeAttr(in_dim, args.hidden_dim, edge_attr_dim, num_heads=args.heads)
    global_edge_embedding = GlobalEdgeEmbedding(edge_attr_dim, args.global_emb_dim)
    decoder = DecoderWithGlobalEdge(args.hidden_dim, edge_attr_dim, args.global_emb_dim)
    model = GAEWithGlobalEdge(encoder, decoder, global_edge_embedding).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.scheduler_tmax)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, optimizer, train_loader, device)
        if val_loader is not None:
            val_auc, val_ap = validate_epoch(model, val_loader, device)
            logger.info(
                f"epoch {epoch}/{args.epochs} bce_loss={train_loss:.6f} "
                f"val_auc={val_auc:.4f} val_ap={val_ap:.4f} ({time.time() - t0:.1f}s)"
            )
        else:
            logger.info(f"epoch {epoch}/{args.epochs} bce_loss={train_loss:.6f} ({time.time() - t0:.1f}s)")
        scheduler.step()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return model


# --------------------------------------------------------------------------
# Inference: real edges only, anomaly score = 1 - P(edge exists)
# --------------------------------------------------------------------------

def run_inference(
    model, parquet_path, edge_columns, scaler, window_size, chunk_rows,
    device, infer_batch_windows, row_limit, logger, log_every_windows=200,
):
    """Stream the holdout file window-by-window, batching a few windows at a
    time into one PyG Batch for the forward pass, without ever materializing
    the whole (potentially tens-of-millions-of-rows) file in memory at once."""
    columns = [SRC_IP_COL, DST_IP_COL] + edge_columns + [LABEL_COL]
    model.eval()

    all_scores, all_labels = [], []
    graph_buffer = []
    windows_done, rows_done = 0, 0
    t0 = time.time()

    def flush():
        if not graph_buffer:
            return
        batch = Batch.from_data_list(graph_buffer).to(device)
        with torch.no_grad():
            edge_attr = batch.edge_attr[:, :-1]
            labels = batch.edge_attr[:, -1]
            z = model.encode(batch.x, batch.edge_index, edge_attr)
            pred = model.decode(z, batch.edge_index, edge_attr, batch.batch)
            score = 1.0 - pred
        all_scores.append(score.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        graph_buffer.clear()

    for window_df in iter_windows(
        parquet_path, columns, window_size, chunk_rows, label_filter=None, row_limit=row_limit
    ):
        if len(window_df) < 2:
            continue
        data = build_graph(window_df, edge_columns, scaler)
        graph_buffer.append(data)
        rows_done += len(window_df)
        windows_done += 1

        if len(graph_buffer) >= infer_batch_windows:
            flush()
        if windows_done % log_every_windows == 0:
            logger.info(f"inference progress: {windows_done} windows, {rows_done} rows, {time.time() - t0:.1f}s elapsed")
    flush()

    y_score = np.concatenate(all_scores) if all_scores else np.array([], dtype=np.float32)
    y_true = np.concatenate(all_labels) if all_labels else np.array([], dtype=np.float32)
    return y_true.astype(np.int64), y_score


def compute_metrics(y_true, y_score):
    """AUC-ROC / PR-AUC are threshold-free. For precision/recall/F1/FPR we pick
    the cutoff that maximizes Youden's J (TPR - FPR) on the ROC curve, same as
    the `compute_metrics` cell in training-testing.ipynb."""
    if len(y_true) == 0:
        raise RuntimeError(
            "0 edges were evaluated during inference (every window was filtered out, e.g. by "
            "--max-holdout-rows cutting off before a full --window-size window completed). "
            "Increase --max-holdout-rows or drop it to run on the full holdout file."
        )
    auc_roc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan")
    pr_auc = average_precision_score(y_true, y_score)

    fpr_curve, tpr_curve, thresholds = roc_curve(y_true, y_score)
    best_idx = int(np.argmax(tpr_curve - fpr_curve))
    threshold = float(thresholds[best_idx])

    y_pred = (y_score >= threshold).astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")

    return {
        "auc_roc": float(auc_roc),
        "pr_auc": float(pr_auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "threshold": threshold,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "n_eval_edges": int(len(y_true)),
    }


# --------------------------------------------------------------------------
# Logging / CLI / orchestration
# --------------------------------------------------------------------------

def setup_logger(name, log_path):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def resolve_data_dir(explicit):
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"--data-dir {path} does not exist")
        return path
    for candidate in (Path("netflow/parquet"), Path("../netflow/parquet")):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not auto-locate netflow/parquet (looked in ./netflow/parquet and "
        "../netflow/parquet). Pass --data-dir explicitly."
    )


def resolve_edge_columns(spec):
    if spec == "baseline":
        return list(BASELINE_EDGE_COLUMNS)
    if spec == "all":
        return list(ALL_EDGE_COLUMNS)
    return [c.strip() for c in spec.split(",") if c.strip()]


def parse_args():
    p = argparse.ArgumentParser(description="Train & evaluate the Guided GAE anomaly detector for NetFlow v3 datasets")
    p.add_argument("--datasets", default="all", help=f"comma-separated subset of {list(DATASETS)} or 'all'")
    p.add_argument("--data-dir", default=None, help="folder containing the *_train.parquet/*_holdout.parquet files")
    p.add_argument("--edge-columns", default="baseline", help="'baseline' (4 cols), 'all' (43 cols), or comma-separated column names")
    p.add_argument("--window-size", type=int, default=1000, help="flows per graph window (matches training-testing.ipynb)")
    p.add_argument("--chunk-rows", type=int, default=500_000, help="parquet streaming read batch size")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256, help="graph-windows per training batch")
    p.add_argument("--infer-batch-windows", type=int, default=32, help="graph-windows per inference batch")
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--global-emb-dim", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--scheduler-tmax", type=int, default=5)
    p.add_argument("--val-ratio", type=float, default=0.2, help="fraction of benign windows held out to monitor overfitting")
    p.add_argument("--node-dim", type=int, default=4, help="node feature dim (IPv4 octets)")
    p.add_argument("--device", default="auto", help="'auto', 'cuda', 'cuda:0', or 'cpu'")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="results")
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--mode", choices=["train", "infer", "both"], default="both")
    p.add_argument("--resume", action="store_true", help="skip datasets whose output already exists")
    p.add_argument("--max-train-rows", type=int, default=None, help="cap rows read from the train split (smoke-testing)")
    p.add_argument("--max-holdout-rows", type=int, default=None, help="cap rows read from the holdout split (smoke-testing)")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = resolve_data_dir(args.data_dir)
    default_edge_columns = resolve_edge_columns(args.edge_columns)

    dataset_names = list(DATASETS) if args.datasets == "all" else [d.strip() for d in args.datasets.split(",")]
    for name in dataset_names:
        if name not in DATASETS:
            raise ValueError(f"Unknown dataset {name!r}; choices: {list(DATASETS)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)

    out_dir, log_dir, ckpt_dir = Path(args.output_dir), Path(args.log_dir), Path(args.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for name in dataset_names:
        train_file, holdout_file = DATASETS[name]
        train_path, holdout_path = data_dir / train_file, data_dir / holdout_file

        dataset_out_dir = out_dir / name
        dataset_out_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = dataset_out_dir / "metrics.json"
        ckpt_path = ckpt_dir / f"{name}.pt"

        logger = setup_logger(name, log_dir / f"{name}.log")
        logger.info(f"===== dataset={name} device={device} mode={args.mode} =====")

        target_marker = metrics_path if args.mode in ("both", "infer") else ckpt_path
        if args.resume and target_marker.exists():
            logger.info(f"{target_marker} already exists, skipping due to --resume")
            if target_marker == metrics_path:
                with open(metrics_path) as f:
                    summary_rows.append(json.load(f))
            continue

        t_start = time.time()
        edge_cols = list(default_edge_columns)

        need_train = args.mode in ("train", "both") or not ckpt_path.exists()
        if need_train:
            logger.info(f"edge_columns={edge_cols}")
            logger.info(f"fitting scaler on benign rows of {train_path}")
            scaler = fit_scaler(train_path, edge_cols, args.chunk_rows, row_limit=args.max_train_rows)

            logger.info("building windowed benign training graphs")
            train_graphs = build_train_graphs(
                train_path, edge_cols, scaler, args.window_size, args.chunk_rows, row_limit=args.max_train_rows
            )
            n_edges = sum(g.edge_index.size(1) for g in train_graphs)
            logger.info(f"built {len(train_graphs)} training graphs ({n_edges} benign edges)")
            if not train_graphs:
                raise RuntimeError(f"no training graphs built for {name}; check data / --window-size")

            model = train_model(train_graphs, args.node_dim, len(edge_cols), args, device, logger)

            train_time = time.time() - t_start
            torch.save({
                "model_state": model.state_dict(),
                "edge_columns": edge_cols,
                "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_,
                "hidden_dim": args.hidden_dim,
                "global_emb_dim": args.global_emb_dim,
                "heads": args.heads,
                "node_dim": args.node_dim,
                "window_size": args.window_size,
            }, ckpt_path)
            logger.info(f"saved checkpoint to {ckpt_path} (train_time={train_time:.1f}s)")
        else:
            logger.info(f"loading existing checkpoint {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            edge_cols = ckpt["edge_columns"]
            scaler = StandardScaler()
            scaler.mean_ = ckpt["scaler_mean"]
            scaler.scale_ = ckpt["scaler_scale"]
            scaler.n_features_in_ = len(edge_cols)

            encoder = GATEncoderWithEdgeAttr(ckpt["node_dim"], ckpt["hidden_dim"], len(edge_cols), num_heads=ckpt["heads"])
            global_edge_embedding = GlobalEdgeEmbedding(len(edge_cols), ckpt["global_emb_dim"])
            decoder = DecoderWithGlobalEdge(ckpt["hidden_dim"], len(edge_cols), ckpt["global_emb_dim"])
            model = GAEWithGlobalEdge(encoder, decoder, global_edge_embedding).to(device)
            model.load_state_dict(ckpt["model_state"])
            train_time = 0.0

        if args.mode in ("infer", "both"):
            logger.info(f"running inference on holdout {holdout_path}")
            t_infer = time.time()
            y_true, y_score = run_inference(
                model, holdout_path, edge_cols, scaler, args.window_size, args.chunk_rows,
                device, args.infer_batch_windows, args.max_holdout_rows, logger,
            )
            infer_time = time.time() - t_infer
            logger.info(f"inference done: {len(y_true)} edges evaluated in {infer_time:.1f}s")

            metrics = compute_metrics(y_true, y_score)
            metrics.update({
                "dataset": name,
                "edge_columns": edge_cols,
                "train_time_sec": train_time,
                "infer_time_sec": infer_time,
            })
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)

            logger.info(
                f"RESULTS {name}: AUC-ROC={metrics['auc_roc']:.4f} PR-AUC={metrics['pr_auc']:.4f} "
                f"Precision={metrics['precision']:.4f} Recall={metrics['recall']:.4f} "
                f"F1={metrics['f1']:.4f} FPR={metrics['fpr']:.4f} (Youden's-J threshold={metrics['threshold']:.4f})"
            )
            summary_rows.append(metrics)

    if summary_rows:
        cols = ["dataset", "auc_roc", "pr_auc", "precision", "recall", "f1", "fpr",
                "threshold", "n_eval_edges", "train_time_sec", "infer_time_sec"]
        summary_df = pd.DataFrame(summary_rows)
        summary_df = summary_df[[c for c in cols if c in summary_df.columns]]
        summary_path = out_dir / "comparison.csv"
        summary_df.to_csv(summary_path, index=False)
        print("\n=== Summary across datasets ===")
        print(summary_df.to_string(index=False))
        print(f"\nSaved comparison table to {summary_path}")


if __name__ == "__main__":
    main()
