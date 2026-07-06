# SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Train + evaluate the repo's Guided GAE (src/guided_gae_model.py) link-prediction
anomaly detector using src/preprocess.py's NetflowPreprocessor, for each NetFlow v3
dataset. Trains one model per dataset (they are never shared/fine-tuned across
datasets), unless --datasets narrows the run to a subset.

Run `python train_and_infer.py --help` for all options.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.guided_gae_model import (
    DecoderWithGlobalEdge,
    GAEWithGlobalEdge,
    GATEncoderWithEdgeAttr,
    GlobalEdgeEmbedding,
)
from src.preprocess import NetflowPreprocessor

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


def train(model, optimizer, data_loader, device):
    model.train()
    total_loss = 0

    for data in tqdm(data_loader, desc='Training'):
        data = data.to(device)
        optimizer.zero_grad()
        target = data.edge_attr[:, :-1]
        # Encode
        z = model.encode(data.x, data.edge_index, target)
        # Reconstruct edge attributes from endpoint embeddings
        pred = model.decode(z, data.edge_index, target, data.batch)
        loss = F.mse_loss(pred, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return total_loss / len(data_loader)


def test(model, data_loader, device):
    """Mean per-element reconstruction MSE on held-out benign graphs."""
    model.eval()
    total_sq_err = 0.0
    total_elements = 0
    with torch.no_grad():
        for data in tqdm(data_loader, desc='Validating'):
            data = data.to(device)
            target = data.edge_attr[:, :-1]
            z = model.encode(data.x, data.edge_index, target)
            pred = model.decode(z, data.edge_index, target, data.batch)
            total_sq_err += F.mse_loss(pred, target, reduction='sum').item()
            total_elements += target.numel()

    return total_sq_err / total_elements


def split_data(data_list, train_ratio=0.8):
    train_size = int(len(data_list) * train_ratio)
    test_size = len(data_list) - train_size
    train_dataset, test_dataset = random_split(data_list, [train_size, test_size])
    return train_dataset, test_dataset


def compute_metrics(data_loader, model, device, roc_plot_path=None):
    all_labels = []
    all_preds = []

    model.eval()
    with torch.no_grad():
        for data in tqdm(data_loader, desc='Inference examples'):
            data = data.to(device)
            target = data.edge_attr[:, :-1]
            z = model.encode(data.x, data.edge_index, target)
            pred = model.decode(z, data.edge_index, target, data.batch)
            # Anomaly score: per-edge reconstruction error (higher = more anomalous)
            err = ((pred - target) ** 2).mean(dim=1)
            all_preds.append(err.float().cpu().numpy())
            all_labels.append(np.asarray(data.edge_attr[:, -1].cpu().numpy()))

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    unique_values, counts = np.unique(all_labels, return_counts=True)
    print(f"Unique values: {unique_values}")
    print(f"Counts: {counts}")

    fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_preds)
    pr_auc = average_precision_score(all_labels, all_preds)

    if roc_plot_path is not None:
        plt.figure()
        plt.plot(fpr, tpr, color='blue', label='ROC curve (area = {:.2f})'.format(roc_auc))
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.savefig(roc_plot_path)
        plt.close()

    youdens_j = tpr - fpr
    idx = np.argmax(youdens_j)
    best_threshold = float(thresholds[idx])
    best_fpr = float(fpr[idx])
    best_tpr = float(tpr[idx])

    y_pred = (all_preds >= best_threshold).astype(int)
    precision = precision_score(all_labels, y_pred, zero_division=0)
    recall = recall_score(all_labels, y_pred, zero_division=0)
    f1 = f1_score(all_labels, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(all_labels, y_pred, labels=[0, 1]).ravel()

    return {
        "auc_roc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": best_fpr,
        "tpr": best_tpr,
        "threshold": best_threshold,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "n_eval_edges": int(len(all_labels)),
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


def resolve_log1p_columns(spec, edge_columns):
    if spec == "none":
        return []
    if spec == "baseline":
        return list(BASELINE_EDGE_COLUMNS)
    if spec == "all":
        return list(edge_columns)
    return [c.strip() for c in spec.split(",") if c.strip()]


def parse_args():
    p = argparse.ArgumentParser(description="Train & evaluate the Guided GAE anomaly detector for NetFlow v3 datasets")
    p.add_argument("--datasets", default="all", help=f"comma-separated subset of {list(DATASETS)} or 'all' (default: train a separate model per dataset)")
    p.add_argument("--data-dir", default=None, help="folder containing the *_train.parquet/*_holdout.parquet files")
    p.add_argument("--edge-columns", default="baseline", help="'baseline' (3 cols), 'all' (43 cols), or comma-separated column names. PROTOCOL/L4_SRC_PORT/L4_DST_PORT are auto one-hot encoded; TOTAL_PKTS (IN_PKTS+OUT_PKTS) and TOTAL_BYTES (IN_BYTES+OUT_BYTES) are derived columns")
    p.add_argument("--log1p-columns", default="none", help="edge columns to log1p-compress before scaling: 'none' (default, raw features), 'baseline', 'all' (= every selected edge column), or comma-separated column names")
    p.add_argument("--top-k-ports", type=int, default=16, help="number of most-frequent non-ephemeral destination ports to one-hot; the rest fall into 'other'/'ephemeral' buckets")
    p.add_argument("--window-size", type=int, default=1000, help="flows per graph window")
    p.add_argument("--step-size", type=int, default=1000, help="row stride between windows")
    p.add_argument("--node-dim", type=int, default=32, help="node embedding dimension")
    p.add_argument("--epochs", type=int, default=50, help="max epochs; early stopping usually ends training sooner")
    p.add_argument("--patience", type=int, default=10, help="stop after this many epochs without val_mse improvement; 0 disables early stopping (always trains the full --epochs, still restores the best-val_mse epoch)")
    p.add_argument("--batch-size", type=int, default=256, help="graph-windows per batch")
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--global-emb-dim", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--scheduler-tmax", type=int, default=None, help="cosine annealing T_max (default: --epochs, so LR decays once over the full run)")
    p.add_argument("--train-ratio", type=float, default=0.8, help="fraction of benign windows used for training vs held-out validation")
    p.add_argument("--device", default="auto", help="'auto', 'cuda', 'cuda:0', or 'cpu'")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="results_infer", help="separate from train_and_evaluate.py's results/ to avoid clobbering it")
    p.add_argument("--log-dir", default="logs_infer")
    p.add_argument("--checkpoint-dir", default="checkpoints_infer")
    p.add_argument("--resume", action="store_true", help="skip datasets whose results/metrics.json already exists")
    p.add_argument("--max-train-rows", type=int, default=1_000_000, help="cap rows read from the train split; pass 0 to disable")
    p.add_argument("--max-holdout-rows", type=int, default=None, help="cap rows read from the holdout split (smoke-testing)")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = resolve_data_dir(args.data_dir)
    edge_columns = resolve_edge_columns(args.edge_columns)
    log1p_columns = resolve_log1p_columns(args.log1p_columns, edge_columns)

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
        roc_path = dataset_out_dir / "roc_curve.png"
        ckpt_path = ckpt_dir / f"{name}.pt"

        logger = setup_logger(name, log_dir / f"{name}.log")
        logger.info(f"===== dataset={name} device={device} epochs={args.epochs} edge_columns={edge_columns} =====")

        if args.resume and metrics_path.exists():
            logger.info(f"{metrics_path} already exists, skipping due to --resume")
            with open(metrics_path) as f:
                summary_rows.append(json.load(f))
            continue

        t_start = time.time()

        logger.info(f"loading train data from {train_path}")
        train_data = pd.read_parquet(train_path)
        if args.max_train_rows and len(train_data) > args.max_train_rows:
            train_data = train_data.head(n=args.max_train_rows)

        logger.info(f"loading holdout data from {holdout_path}")
        test_data = pd.read_parquet(holdout_path)
        if args.max_holdout_rows:
            test_data = test_data.head(n=args.max_holdout_rows)

        processor = NetflowPreprocessor(
            train_data,
            edge_columns=edge_columns,
            node_dim=args.node_dim,
            log1p_columns=log1p_columns,
            top_k_ports=args.top_k_ports,
        )

        logger.info("building windowed graphs from training data")
        benign_graphs, benign_ip_map, benign_data_windows = processor.construct_graph_list(
            window_size=args.window_size, step_size=args.step_size
        )
        logger.info(f"built {len(benign_graphs)} graphs from training data")
        if not benign_graphs:
            raise RuntimeError(f"no training graphs built for {name}; check data / --window-size")

        train_dataset, val_dataset = split_data(benign_graphs, train_ratio=args.train_ratio)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        logger.info("building windowed graphs from holdout data")
        attack_graphs, attack_ip_map, attack_data_windows = processor.construct_graph_list(
            df=test_data, window_size=args.window_size, step_size=args.step_size
        )
        attack_loader = DataLoader(attack_graphs, batch_size=args.batch_size, shuffle=False)

        in_channels = benign_graphs[0].x.size(1)
        # categorical columns expand into one-hot blocks, so the model's edge dim
        # is the transformed feature count, not the raw column count
        edge_attr_dim = processor.feature_dim
        logger.info(f"edge feature dim = {edge_attr_dim} ({len(processor.numeric_columns)} numeric "
                    f"+ {edge_attr_dim - len(processor.numeric_columns)} one-hot categorical)")

        encoder = GATEncoderWithEdgeAttr(in_channels, args.hidden_dim, edge_attr_dim, num_heads=args.heads)
        global_edge_embedding = GlobalEdgeEmbedding(edge_attr_dim, args.global_emb_dim)
        decoder = DecoderWithGlobalEdge(args.hidden_dim, edge_attr_dim, args.global_emb_dim)
        model = GAEWithGlobalEdge(encoder, decoder, global_edge_embedding).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.scheduler_tmax or args.epochs)

        best_val_mse = float("inf")
        best_epoch = 0
        best_state = None
        epochs_no_improve = 0
        epochs_trained = 0

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_loss = train(model, optimizer, train_loader, device)
            val_mse = test(model, val_loader, device)
            scheduler.step()
            epochs_trained = epoch
            logger.info(
                f"epoch {epoch}/{args.epochs} train_mse={train_loss:.6f} "
                f"val_mse={val_mse:.6f} ({time.time() - t0:.1f}s)"
            )
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if args.patience > 0 and epochs_no_improve >= args.patience:
                    logger.info(
                        f"early stopping at epoch {epoch}: no val_mse improvement "
                        f"for {args.patience} epochs (best={best_val_mse:.6f} @ epoch {best_epoch})"
                    )
                    break
            if device.type == "cuda":
                torch.cuda.empty_cache()

        model.load_state_dict(best_state)
        logger.info(f"restored best checkpoint from epoch {best_epoch} (val_mse={best_val_mse:.6f})")
        train_time = time.time() - t_start
        torch.save({
            "model_state": model.state_dict(),
            "edge_columns": edge_columns,
            "numeric_columns": processor.numeric_columns,
            "categorical_columns": processor.categorical_columns,
            "log1p_columns": processor.log1p_columns,
            "protocol_vocab": processor.protocol_vocab,
            "dst_port_vocab": processor.dst_port_vocab,
            "feature_names": processor.feature_names,
            "scaler_mean": processor.edge_scaler.mean_ if processor.edge_scaler is not None else None,
            "scaler_scale": processor.edge_scaler.scale_ if processor.edge_scaler is not None else None,
            "hidden_dim": args.hidden_dim,
            "global_emb_dim": args.global_emb_dim,
            "heads": args.heads,
            "node_dim": args.node_dim,
            "window_size": args.window_size,
            "best_epoch": best_epoch,
            "best_val_mse": best_val_mse,
        }, ckpt_path)
        logger.info(f"saved checkpoint to {ckpt_path} (train_time={train_time:.1f}s)")

        logger.info(f"running inference on holdout {holdout_path}")
        t_infer = time.time()
        metrics = compute_metrics(attack_loader, model, device, roc_plot_path=roc_path)
        infer_time = time.time() - t_infer
        logger.info(f"inference done: {metrics['n_eval_edges']} edges evaluated in {infer_time:.1f}s")

        metrics.update({
            "dataset": name,
            "edge_columns": edge_columns,
            "log1p_columns": log1p_columns,
            "train_time_sec": train_time,
            "infer_time_sec": infer_time,
            "best_epoch": best_epoch,
            "epochs_trained": epochs_trained,
            "best_val_mse": best_val_mse,
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
