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
anomaly detector using src/preprocess.py's NetflowPreprocessor, for the NetFlow v3
datasets. Three modes:

  default         one model per dataset (never shared across datasets); --datasets
                  narrows the run to a subset
  --combined      ONE model trained on the pooled benign training data of every
                  selected dataset (StandardScaler + categorical vocabs fit on the
                  pooled rows), evaluated on each dataset's holdout and on all
                  holdouts pooled
  --personalised  domain adaptation: ONE shared (centralised) encoder and one
                  personalised head (decoder + global-edge embedding) per dataset,
                  trained jointly in a single stage; each holdout is scored by its
                  own dataset's head

Run `python train_and_infer.py --help` for all options.
"""

import argparse
import gc
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
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
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

COMBINED_NAME = "combined"
PERSONALISED_NAME = "personalised"

SRC_IP_COL = "IPV4_SRC_ADDR"
DST_IP_COL = "IPV4_DST_ADDR"
LABEL_COL = "Label"

BASELINE_EDGE_COLUMNS = ["PROTOCOL","L4_SRC_PORT","L4_DST_PORT","MIN_TTL","FLOW_DURATION_MILLISECONDS","TOTAL_PKTS","TOTAL_BYTES"]
BASELINE_LOG1P_COLUMNS = ["FLOW_DURATION_MILLISECONDS","TOTAL_PKTS", "TOTAL_BYTES"]

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


def per_edge_errors(model, data_loader, device, desc='Scoring'):
    """Per-edge reconstruction MSE for every edge in the loader (higher = more anomalous)."""
    model.eval()
    errs = []
    with torch.no_grad():
        for data in tqdm(data_loader, desc=desc):
            data = data.to(device)
            target = data.edge_attr[:, :-1]
            z = model.encode(data.x, data.edge_index, target)
            pred = model.decode(z, data.edge_index, target, data.batch)
            err = ((pred - target) ** 2).mean(dim=1)
            errs.append(err.float().cpu().numpy().astype(np.float32))
    return np.concatenate(errs)


def threshold_metrics(all_labels, all_preds, threshold):
    """Classification metrics at a fixed anomaly-score threshold (score >= threshold -> attack)."""
    y_pred = (all_preds >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(all_labels, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(all_labels, y_pred, zero_division=0)),
        "recall": float(recall_score(all_labels, y_pred, zero_division=0)),
        "f1": float(f1_score(all_labels, y_pred, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "tpr": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def split_data(data_list, train_ratio=0.8):
    train_size = int(len(data_list) * train_ratio)
    test_size = len(data_list) - train_size
    train_dataset, test_dataset = random_split(data_list, [train_size, test_size])
    return train_dataset, test_dataset


def stream_inference(df, processor, model, device, window_size, step_size,
                     batch_size, chunk_windows):
    """Score the holdout without materializing every windowed graph at once.

    ``construct_graph_list`` builds a Data object *and* a copy of every (overlapping)
    window up front, so for a large holdout (e.g. CICIDS) the full graph list can
    exhaust RAM before inference even starts. Instead we walk the rows in chunks of
    ``chunk_windows`` windows, build+score just that slice, keep only the small
    per-edge (error, label) arrays, then free the slice's graphs before the next one.

    Chunks advance by ``chunk_windows * step_size`` rows and are read
    ``window_size - step_size`` rows long so that consecutive chunks reproduce exactly
    the same set of window starts as a single non-chunked pass (no gaps, no overlap).
    """
    all_preds = []
    all_labels = []
    n = len(df)
    chunk_rows = chunk_windows * step_size
    tail = window_size - step_size  # extra rows so the last window in a chunk is whole

    model.eval()
    with torch.no_grad():
        start = 0
        while start + window_size <= n:
            end = min(start + chunk_rows + tail, n)
            chunk_df = df.iloc[start:end]
            graphs, _, _ = processor.construct_graph_list(
                df=chunk_df, window_size=window_size, step_size=step_size
            )
            loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
            for data in loader:
                data = data.to(device)
                target = data.edge_attr[:, :-1]
                z = model.encode(data.x, data.edge_index, target)
                pred = model.decode(z, data.edge_index, target, data.batch)
                # Anomaly score: per-edge reconstruction error (higher = more anomalous)
                err = ((pred - target) ** 2).mean(dim=1)
                all_preds.append(err.float().cpu().numpy().astype(np.float32))
                all_labels.append(data.edge_attr[:, -1].cpu().numpy().astype(np.int8))
            del graphs, loader
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            start += chunk_rows

    return np.concatenate(all_preds), np.concatenate(all_labels)


def metrics_from_scores(all_labels, all_preds, roc_plot_path=None):
    unique_values, counts = np.unique(all_labels, return_counts=True)
    print(f"Unique values: {unique_values}")
    print(f"Counts: {counts}")

    fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)
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

    # Youden's J threshold: label-dependent oracle operating point (upper bound,
    # not achievable in deployment where holdout labels are unavailable)
    youdens_j = tpr - fpr
    idx = np.argmax(youdens_j)
    best_threshold = float(thresholds[idx])

    out = {"auc_roc": float(roc_auc), "pr_auc": float(pr_auc)}
    out.update(threshold_metrics(all_labels, all_preds, best_threshold))
    out["n_eval_edges"] = int(len(all_labels))
    return out


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
        return list(BASELINE_LOG1P_COLUMNS)
    if spec == "all":
        return list(edge_columns)
    return [c.strip() for c in spec.split(",") if c.strip()]


def read_parquet_head(path, max_rows):
    """Read a parquet file, stopping after ~max_rows rows (0/None = whole file).
    Reading batch-wise avoids materializing a multi-GB dataframe just to .head() it."""
    if not max_rows:
        return pd.read_parquet(path)
    pf = pq.ParquetFile(path)
    batches = []
    rows = 0
    for batch in pf.iter_batches():
        batches.append(batch)
        rows += batch.num_rows
        if rows >= max_rows:
            break
    return pa.Table.from_batches(batches).to_pandas().head(n=max_rows)


def load_train_df(train_path, max_train_rows, logger):
    logger.info(f"loading train data from {train_path}")
    return read_parquet_head(train_path, max_train_rows)


def build_model(args, in_channels, edge_attr_dim, device):
    encoder = GATEncoderWithEdgeAttr(in_channels, args.hidden_dim, edge_attr_dim, num_heads=args.heads)
    global_edge_embedding = GlobalEdgeEmbedding(edge_attr_dim, args.global_emb_dim)
    decoder = DecoderWithGlobalEdge(args.hidden_dim, edge_attr_dim, args.global_emb_dim)
    return GAEWithGlobalEdge(encoder, decoder, global_edge_embedding).to(device)


def fit_model(model, train_loader, val_loader, args, device, logger, max_epochs=None, lr=None):
    """Epoch loop with ReduceLROnPlateau + early stopping on val_mse; restores the
    best-val_mse state before returning. Only parameters with requires_grad=True are
    optimized, so callers may freeze submodules (e.g. the shared encoder) beforehand."""
    max_epochs = max_epochs or args.epochs
    lr = lr if lr is not None else args.lr
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.scheduler_factor, patience=args.scheduler_patience
    )

    best_val_mse = float("inf")
    best_epoch = 0
    best_state = None
    epochs_no_improve = 0
    epochs_trained = 0

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        train_loss = train(model, optimizer, train_loader, device)
        val_mse = test(model, val_loader, device)
        scheduler.step(val_mse)
        epochs_trained = epoch
        logger.info(
            f"epoch {epoch}/{max_epochs} train_mse={train_loss:.6f} "
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

    return {
        "best_epoch": best_epoch,
        "best_val_mse": best_val_mse,
        "epochs_trained": epochs_trained,
    }


def fit_joint(models, train_loaders, val_loaders, args, device, logger):
    """Single-stage joint fit for --personalised: every model in ``models`` shares
    ONE encoder object but owns its decoder + global-edge embedding, so stepping a
    dataset's batches updates the centralised encoder and only that dataset's head.
    Each epoch visits the datasets in a fresh random order (the optimizer skips
    parameters whose grad is None, i.e. the other datasets' heads). LR scheduling
    and early stopping act on the mean per-dataset val_mse; the best joint state is
    restored before returning."""
    params = list({id(p): p for m in models.values() for p in m.parameters()}.values())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.scheduler_factor, patience=args.scheduler_patience
    )

    best_val_mse = float("inf")
    best_epoch = 0
    best_states = None
    best_val_by_ds = None
    epochs_no_improve = 0
    epochs_trained = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_losses = {}
        for name in np.random.permutation(list(models)):
            train_losses[name] = train(models[name], optimizer, train_loaders[name], device)
        val_by_ds = {name: test(models[name], val_loaders[name], device) for name in models}
        val_mse = float(np.mean(list(val_by_ds.values())))
        scheduler.step(val_mse)
        epochs_trained = epoch
        per_ds = " ".join(f"{n}:train={train_losses[n]:.6f}/val={val_by_ds[n]:.6f}" for n in models)
        logger.info(
            f"epoch {epoch}/{args.epochs} mean_val_mse={val_mse:.6f} {per_ds} "
            f"({time.time() - t0:.1f}s)"
        )
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch
            best_states = {name: {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
                           for name, m in models.items()}
            best_val_by_ds = dict(val_by_ds)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if args.patience > 0 and epochs_no_improve >= args.patience:
                logger.info(
                    f"early stopping at epoch {epoch}: no mean val_mse improvement "
                    f"for {args.patience} epochs (best={best_val_mse:.6f} @ epoch {best_epoch})"
                )
                break
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for name, m in models.items():
        m.load_state_dict(best_states[name])
    logger.info(f"restored best checkpoint from epoch {best_epoch} (mean val_mse={best_val_mse:.6f})")

    return {
        "best_epoch": best_epoch,
        "best_val_mse": best_val_mse,
        "epochs_trained": epochs_trained,
        "val_mse_by_dataset": best_val_by_ds,
    }


def calibrate_thresholds(model, val_loader, device, calib_quantiles, logger):
    """Label-free threshold calibration: quantiles of per-edge reconstruction error
    on the held-out benign validation windows. The q-quantile is a fixed error
    cutoff that targets an expected FPR of ~(1-q) on future benign traffic."""
    logger.info("calibrating thresholds on benign validation windows")
    benign_errs = per_edge_errors(model, val_loader, device, desc='Calibrating')
    n_calib_edges = int(len(benign_errs))
    calib_thresholds = {q: float(np.quantile(benign_errs, q)) for q in calib_quantiles}
    del benign_errs
    for q, thr in calib_thresholds.items():
        logger.info(f"calibrated threshold @ benign q={q}: {thr:.6f} "
                    f"(expected FPR ~{(1 - q) * 100:.2f}%, from {n_calib_edges} benign edges)")
    return calib_thresholds, n_calib_edges


def train_on_graphs(benign_graphs, processor, args, device, logger, calib_quantiles):
    """Split benign graphs into train/val, train with early stopping, restore the
    best-val_mse epoch, and calibrate benign-quantile thresholds on the val edges.

    Returns (model, info) where info carries best_epoch/best_val_mse/epochs_trained
    plus the calibrated thresholds and the benign edge count they were fit on.
    """
    train_dataset, val_dataset = split_data(benign_graphs, train_ratio=args.train_ratio)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # categorical columns expand into one-hot blocks, so the model's edge dim
    # is the transformed feature count, not the raw column count
    edge_attr_dim = processor.feature_dim
    logger.info(f"edge feature dim = {edge_attr_dim} ({len(processor.numeric_columns)} numeric "
                f"+ {edge_attr_dim - len(processor.numeric_columns)} one-hot categorical)")

    model = build_model(args, benign_graphs[0].x.size(1), edge_attr_dim, device)
    info = fit_model(model, train_loader, val_loader, args, device, logger)

    calib_thresholds, n_calib_edges = calibrate_thresholds(model, val_loader, device, calib_quantiles, logger)
    info["calib_thresholds"] = calib_thresholds
    info["n_calib_edges"] = n_calib_edges
    return model, info


def save_checkpoint(ckpt_path, processor, args, info, model=None, extra=None):
    payload = {
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
        "best_epoch": info["best_epoch"],
        "best_val_mse": info["best_val_mse"],
        "epochs_trained": info.get("epochs_trained"),
        "calib_thresholds": info["calib_thresholds"],
    }
    if model is not None:
        payload["model_state_dict"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if extra:
        payload.update(extra)
    torch.save(payload, ckpt_path)


def run_threshold_sweep(all_labels, all_preds, metrics, calib_thresholds, sweep_path, logger):
    """Threshold sweep: the Youden oracle operating point plus every calibrated
    (label-free) benign-quantile threshold, evaluated on the same holdout scores."""
    sweep_keys = ["threshold", "precision", "recall", "f1", "fpr", "tpr", "tp", "fp", "tn", "fn"]
    sweep_rows = [{"method": "youden_oracle", "quantile": None, "expected_fpr": None,
                   **{k: metrics[k] for k in sweep_keys}}]
    for q, thr in calib_thresholds.items():
        row = threshold_metrics(all_labels, all_preds, thr)
        sweep_rows.append({"method": "benign_quantile", "quantile": q,
                           "expected_fpr": float(1 - q), **row})
        logger.info(
            f"THRESHOLD benign q={q} thr={thr:.6f}: Precision={row['precision']:.4f} "
            f"Recall={row['recall']:.4f} F1={row['f1']:.4f} "
            f"FPR={row['fpr']:.4f} (expected ~{1 - q:.4f})"
        )
    pd.DataFrame(sweep_rows).to_csv(sweep_path, index=False)
    logger.info(f"saved threshold sweep to {sweep_path}")
    return sweep_rows


def evaluate_holdout(holdout_path, processor, model, device, args, calib_thresholds,
                     dataset_out_dir, logger):
    """Score one holdout parquet; writes roc_curve.png + threshold_sweep.csv into
    dataset_out_dir. Returns (metrics, all_preds, all_labels) so combined mode can
    also pool scores across holdouts."""
    dataset_out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"loading holdout data from {holdout_path}")
    test_data = read_parquet_head(holdout_path, args.max_holdout_rows)

    logger.info(f"running inference on holdout {holdout_path}")
    t_infer = time.time()
    # stream the holdout through inference in row-chunks so the full set of
    # windowed graphs never has to sit in RAM at once (avoids OOM on CICIDS)
    all_preds, all_labels = stream_inference(
        test_data, processor, model, device,
        window_size=args.window_size, step_size=args.step_size,
        batch_size=args.batch_size, chunk_windows=args.infer_chunk_windows,
    )
    del test_data
    gc.collect()
    metrics = metrics_from_scores(all_labels, all_preds,
                                  roc_plot_path=dataset_out_dir / "roc_curve.png")
    infer_time = time.time() - t_infer
    logger.info(f"inference done: {metrics['n_eval_edges']} edges evaluated in {infer_time:.1f}s")

    sweep_rows = run_threshold_sweep(all_labels, all_preds, metrics, calib_thresholds,
                                     dataset_out_dir / "threshold_sweep.csv", logger)
    metrics["calibrated_thresholds"] = [r for r in sweep_rows if r["method"] == "benign_quantile"]
    metrics["infer_time_sec"] = infer_time
    return metrics, all_preds, all_labels


def parse_args():
    p = argparse.ArgumentParser(description="Train & evaluate the Guided GAE anomaly detector for NetFlow v3 datasets")
    p.add_argument("--datasets", default="all", help=f"comma-separated subset of {list(DATASETS)} or 'all' (default: train a separate model per dataset)")
    p.add_argument("--combined", action="store_true",
                   help="train ONE model on the pooled benign training data of every selected dataset "
                        "(StandardScaler + categorical vocabs fit on the pooled rows; graph windows never "
                        "span dataset boundaries), then evaluate it on each dataset's holdout and on all "
                        "holdouts pooled. Outputs go under <output-dir>/combined/ and <checkpoint-dir>/combined.pt")
    p.add_argument("--personalised", action="store_true",
                   help="domain adaptation (M2): ONE shared (centralised) encoder plus a personalised head "
                        "(decoder + global-edge embedding) per dataset, all trained jointly in a single "
                        "stage; every epoch runs each dataset's benign windows through its own head, so the "
                        "encoder gets gradients from every dataset while each head only fits its own. "
                        "Thresholds are calibrated per dataset and each holdout is scored by its own head. "
                        "Outputs go under <output-dir>/personalised/ and <checkpoint-dir>/personalised-*.pt")
    p.add_argument("--data-dir", default=None, help="folder containing the *_train.parquet/*_holdout.parquet files")
    p.add_argument("--edge-columns", default="baseline", help="'baseline' (3 cols), 'all' (43 cols), or comma-separated column names. PROTOCOL/L4_SRC_PORT/L4_DST_PORT are auto one-hot encoded; TOTAL_PKTS (IN_PKTS+OUT_PKTS) and TOTAL_BYTES (IN_BYTES+OUT_BYTES) are derived columns")
    p.add_argument("--log1p-columns", default="baseline", help="edge columns to log1p-compress before scaling: 'baseline' (default: FLOW_DURATION_MILLISECONDS/TOTAL_PKTS/TOTAL_BYTES), 'none' (raw features), 'all' (= every selected edge column), or comma-separated column names")
    p.add_argument("--top-k-ports", type=int, default=4, help="number of most-frequent non-ephemeral destination ports to one-hot; the rest fall into 'other'/'ephemeral' buckets")
    p.add_argument("--top-k-protocols", type=int, default=4, help="number of most-frequent protocols to one-hot; the rest fall into the 'other' bucket")
    p.add_argument("--window-size", type=int, default=1000, help="flows per graph window")
    p.add_argument("--step-size", type=int, default=1000, help="row stride between windows")
    p.add_argument("--node-dim", type=int, default=32, help="node embedding dimension")
    p.add_argument("--epochs", type=int, default=150, help="max epochs; early stopping usually ends training sooner")
    p.add_argument("--patience", type=int, default=5, help="stop after this many epochs without val_mse improvement; 0 disables early stopping (always trains the full --epochs, still restores the best-val_mse epoch)")
    p.add_argument("--batch-size", type=int, default=256, help="graph-windows per batch")
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--global-emb-dim", type=int, default=64)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.0001)
    p.add_argument("--weight-decay", type=float, default=0.00, help="AdamW L2 weight decay (default matches AdamW's built-in default)")
    p.add_argument("--scheduler-factor", type=float, default=0.5, help="ReduceLROnPlateau: factor by which the LR is multiplied when val_mse plateaus")
    p.add_argument("--scheduler-patience", type=int, default=3, help="ReduceLROnPlateau: epochs without val_mse improvement before reducing LR")
    p.add_argument("--train-ratio", type=float, default=0.8, help="fraction of benign windows used for training vs held-out validation")
    p.add_argument("--device", default="auto", help="'auto', 'cuda', 'cuda:0', or 'cpu'")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="results_infer", help="separate from train_and_evaluate.py's results/ to avoid clobbering it")
    p.add_argument("--log-dir", default="logs_infer")
    p.add_argument("--checkpoint-dir", default="checkpoints_infer")
    p.add_argument("--calib-quantiles", default="0.90,0.95,0.99,0.995,0.999",
                   help="comma-separated benign-validation error quantiles used as label-free anomaly "
                        "thresholds; each quantile q targets an expected FPR of 1-q on benign traffic")
    p.add_argument("--resume", action="store_true", help="skip datasets whose results/metrics.json already exists")
    p.add_argument("--infer-chunk-windows", type=int, default=1000, help="holdout inference streams this many windowed graphs at a time to bound peak RAM; lower it if inference still OOMs")
    p.add_argument("--max-train-rows", type=int, default=1_000_000, help="cap rows read from the train split (applied per dataset, also under --combined); pass 0 to disable")
    p.add_argument("--max-holdout-rows", type=int, default=None, help="cap rows read from the holdout split (smoke-testing)")
    return p.parse_args()


def run_per_dataset(args, dataset_names, data_dir, edge_columns, log1p_columns,
                    calib_quantiles, device, out_dir, log_dir, ckpt_dir):
    """Original mode: one independent model (and preprocessor) per dataset."""
    summary_rows = []

    for name in dataset_names:
        train_file, holdout_file = DATASETS[name]
        train_path, holdout_path = data_dir / train_file, data_dir / holdout_file

        dataset_out_dir = out_dir / name
        dataset_out_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = dataset_out_dir / "metrics.json"
        ckpt_path = ckpt_dir / f"{name}.pt"

        logger = setup_logger(name, log_dir / f"{name}.log")
        logger.info(f"===== dataset={name} device={device} epochs={args.epochs} edge_columns={edge_columns} =====")

        if args.resume and metrics_path.exists():
            logger.info(f"{metrics_path} already exists, skipping due to --resume")
            with open(metrics_path) as f:
                summary_rows.append(json.load(f))
            continue

        t_start = time.time()

        train_data = load_train_df(train_path, args.max_train_rows, logger)

        processor = NetflowPreprocessor(
            train_data,
            edge_columns=edge_columns,
            node_dim=args.node_dim,
            log1p_columns=log1p_columns,
            top_k_ports=args.top_k_ports,
            top_k_protocols=args.top_k_protocols,
        )

        logger.info("building windowed graphs from training data")
        benign_graphs, benign_ip_map, benign_data_windows = processor.construct_graph_list(
            window_size=args.window_size, step_size=args.step_size
        )
        logger.info(f"built {len(benign_graphs)} graphs from training data")
        if not benign_graphs:
            raise RuntimeError(f"no training graphs built for {name}; check data / --window-size")

        model, info = train_on_graphs(benign_graphs, processor, args, device, logger, calib_quantiles)

        train_time = time.time() - t_start
        save_checkpoint(ckpt_path, processor, args, info, model=model)
        logger.info(f"saved checkpoint to {ckpt_path} (train_time={train_time:.1f}s)")

        # training is complete; free the training-side graphs so they do not
        # coexist in RAM with the holdout inference + metric computation
        del benign_graphs, benign_ip_map, benign_data_windows, train_data
        gc.collect()

        metrics, all_preds, all_labels = evaluate_holdout(
            holdout_path, processor, model, device, args,
            info["calib_thresholds"], dataset_out_dir, logger,
        )
        del all_preds, all_labels

        metrics.update({
            "dataset": name,
            "edge_columns": edge_columns,
            "log1p_columns": log1p_columns,
            "train_time_sec": train_time,
            "best_epoch": info["best_epoch"],
            "epochs_trained": info["epochs_trained"],
            "best_val_mse": info["best_val_mse"],
            "n_calib_edges": info["n_calib_edges"],
        })
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(
            f"RESULTS {name}: AUC-ROC={metrics['auc_roc']:.4f} PR-AUC={metrics['pr_auc']:.4f} "
            f"Precision={metrics['precision']:.4f} Recall={metrics['recall']:.4f} "
            f"F1={metrics['f1']:.4f} FPR={metrics['fpr']:.4f} (Youden's-J threshold={metrics['threshold']:.4f})"
        )
        summary_rows.append(metrics)

    return summary_rows


def run_combined(args, dataset_names, data_dir, edge_columns, log1p_columns,
                 calib_quantiles, device, out_dir, log_dir, ckpt_dir):
    """--combined mode: ONE model trained on the pooled benign training data of every
    selected dataset. The preprocessor (StandardScaler mean/scale, protocol vocab,
    dst-port vocab) is fit on the pooled rows so all datasets share a single feature
    space and a single anomaly-score scale. The model is then evaluated on each
    dataset's holdout separately and on all holdouts pooled."""
    combined_out_dir = out_dir / COMBINED_NAME
    combined_out_dir.mkdir(parents=True, exist_ok=True)
    pooled_metrics_path = combined_out_dir / "metrics.json"
    ckpt_path = ckpt_dir / f"{COMBINED_NAME}.pt"

    logger = setup_logger(COMBINED_NAME, log_dir / f"{COMBINED_NAME}.log")
    logger.info(f"===== combined model over datasets={dataset_names} device={device} "
                f"epochs={args.epochs} edge_columns={edge_columns} =====")

    if args.resume and pooled_metrics_path.exists():
        logger.info(f"{pooled_metrics_path} already exists, skipping due to --resume")
        summary_rows = []
        for name in dataset_names:
            ds_metrics_path = combined_out_dir / name / "metrics.json"
            if ds_metrics_path.exists():
                with open(ds_metrics_path) as f:
                    summary_rows.append(json.load(f))
        with open(pooled_metrics_path) as f:
            summary_rows.append(json.load(f))
        return summary_rows

    t_start = time.time()

    # dropna here to mirror NetflowPreprocessor.__init__ (which fits on df.dropna()),
    # so the per-dataset windows below see exactly the rows the scaler was fit on
    train_dfs = {}
    for name in dataset_names:
        df = load_train_df(data_dir / DATASETS[name][0], args.max_train_rows, logger).dropna()
        train_dfs[name] = df
        logger.info(f"{name}: {len(df)} train rows")

    pooled_df = pd.concat(train_dfs.values(), ignore_index=True)
    logger.info(f"fitting preprocessor (scaler + vocabs) on {len(pooled_df)} pooled rows "
                f"from {len(train_dfs)} datasets")
    processor = NetflowPreprocessor(
        pooled_df,
        edge_columns=edge_columns,
        node_dim=args.node_dim,
        log1p_columns=log1p_columns,
        top_k_ports=args.top_k_ports,
        top_k_protocols=args.top_k_protocols,
    )

    # graph windows must never span a dataset boundary (the row order across
    # datasets is meaningless), so build windows per dataset and pool the graphs
    benign_graphs = []
    for name, df in train_dfs.items():
        graphs, _, _ = processor.construct_graph_list(
            df=df, window_size=args.window_size, step_size=args.step_size
        )
        logger.info(f"built {len(graphs)} graphs from {name}")
        benign_graphs.extend(graphs)
    if not benign_graphs:
        raise RuntimeError("no training graphs built; check data / --window-size")
    logger.info(f"pooled training set: {len(benign_graphs)} graphs")

    del train_dfs, pooled_df
    processor.df = None  # scaler/vocabs are fitted; free the 4x-sized pooled copy
    gc.collect()

    model, info = train_on_graphs(benign_graphs, processor, args, device, logger, calib_quantiles)

    train_time = time.time() - t_start
    save_checkpoint(ckpt_path, processor, args, info, model=model, extra={"datasets": dataset_names})
    logger.info(f"saved checkpoint to {ckpt_path} (train_time={train_time:.1f}s)")

    del benign_graphs
    gc.collect()

    common = {
        "model": COMBINED_NAME,
        "trained_on": dataset_names,
        "edge_columns": edge_columns,
        "log1p_columns": log1p_columns,
        "best_epoch": info["best_epoch"],
        "epochs_trained": info["epochs_trained"],
        "best_val_mse": info["best_val_mse"],
        "n_calib_edges": info["n_calib_edges"],
    }

    summary_rows = []
    pooled_preds, pooled_labels = [], []
    for name in dataset_names:
        holdout_path = data_dir / DATASETS[name][1]
        ds_out_dir = combined_out_dir / name
        metrics, all_preds, all_labels = evaluate_holdout(
            holdout_path, processor, model, device, args,
            info["calib_thresholds"], ds_out_dir, logger,
        )
        pooled_preds.append(all_preds)
        pooled_labels.append(all_labels)

        metrics.update(common)
        metrics["dataset"] = name
        with open(ds_out_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(
            f"RESULTS combined->{name}: AUC-ROC={metrics['auc_roc']:.4f} PR-AUC={metrics['pr_auc']:.4f} "
            f"Precision={metrics['precision']:.4f} Recall={metrics['recall']:.4f} "
            f"F1={metrics['f1']:.4f} FPR={metrics['fpr']:.4f} (Youden's-J threshold={metrics['threshold']:.4f})"
        )
        summary_rows.append(metrics)

    # pooled operating point across every holdout: one model + one scaler means the
    # anomaly scores from all datasets live on the same scale and can share thresholds
    all_preds = np.concatenate(pooled_preds)
    all_labels = np.concatenate(pooled_labels)
    del pooled_preds, pooled_labels
    gc.collect()
    metrics = metrics_from_scores(all_labels, all_preds,
                                  roc_plot_path=combined_out_dir / "roc_curve.png")
    sweep_rows = run_threshold_sweep(all_labels, all_preds, metrics, info["calib_thresholds"],
                                     combined_out_dir / "threshold_sweep.csv", logger)
    metrics["calibrated_thresholds"] = [r for r in sweep_rows if r["method"] == "benign_quantile"]
    metrics.update(common)
    metrics["dataset"] = "ALL(pooled)"
    metrics["train_time_sec"] = train_time
    metrics["infer_time_sec"] = float(sum(r["infer_time_sec"] for r in summary_rows))
    with open(pooled_metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(
        f"RESULTS combined pooled holdouts: AUC-ROC={metrics['auc_roc']:.4f} "
        f"PR-AUC={metrics['pr_auc']:.4f} Precision={metrics['precision']:.4f} "
        f"Recall={metrics['recall']:.4f} F1={metrics['f1']:.4f} FPR={metrics['fpr']:.4f} "
        f"(Youden's-J threshold={metrics['threshold']:.4f})"
    )
    summary_rows.append(metrics)
    return summary_rows


def run_personalised(args, dataset_names, data_dir, edge_columns, log1p_columns,
                     calib_quantiles, device, out_dir, log_dir, ckpt_dir):
    """--personalised mode (M2, domain adaptation, single stage).

    ONE centralised encoder is shared by every selected dataset while each dataset
    owns a personalised head (decoder + global-edge embedding). Everything trains
    jointly in a single stage: every epoch runs each dataset's benign windows
    through the shared encoder + that dataset's own head, so the encoder receives
    gradients from every dataset while each head only ever fits its own. The pooled
    scaler/vocabs (as in --combined) keep all datasets in one feature space.
    Thresholds are calibrated per dataset because each head has its own
    reconstruction-error scale, and every holdout is scored by its own head."""
    per_out_dir = out_dir / PERSONALISED_NAME
    per_out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(PERSONALISED_NAME, log_dir / f"{PERSONALISED_NAME}.log")
    logger.info(f"===== personalised mode (single stage: shared encoder + per-dataset heads) over "
                f"datasets={dataset_names} device={device} epochs={args.epochs} "
                f"edge_columns={edge_columns} =====")

    # --resume: datasets whose personalised metrics already exist are skipped entirely
    summary_rows = []
    pending = []
    for name in dataset_names:
        metrics_path = per_out_dir / name / "metrics.json"
        if args.resume and metrics_path.exists():
            logger.info(f"{metrics_path} already exists, skipping {name} due to --resume")
            with open(metrics_path) as f:
                summary_rows.append(json.load(f))
        else:
            pending.append(name)
    if not pending:
        return summary_rows

    # checkpoint resume: the encoder trains jointly with every head, so training can
    # only be skipped when EVERY pending dataset already has a saved head checkpoint
    resumed = {}
    if args.resume:
        loaded = {}
        for name in pending:
            head_ckpt_path = ckpt_dir / f"{PERSONALISED_NAME}-{name}.pt"
            if head_ckpt_path.exists():
                ckpt = torch.load(head_ckpt_path, map_location="cpu", weights_only=False)
                if "model_state_dict" in ckpt:
                    loaded[name] = ckpt
        if loaded and len(loaded) == len(pending):
            logger.info("all pending head checkpoints found; skipping joint training due to --resume")
            resumed = loaded
        elif loaded:
            logger.info("only some pending head checkpoints found; joint training is "
                        "all-or-nothing, so retraining from scratch")

    t_start = time.time()

    # joint training needs every selected dataset's data; a checkpoint-only resume
    # needs just one df to instantiate the preprocessor, whose scaler/vocabs are
    # then overwritten from the checkpoint below
    load_names = dataset_names if not resumed else pending[:1]
    train_dfs = {}
    for name in load_names:
        df = load_train_df(data_dir / DATASETS[name][0], args.max_train_rows, logger).dropna()
        train_dfs[name] = df
        logger.info(f"{name}: {len(df)} train rows")

    pooled_df = pd.concat(train_dfs.values(), ignore_index=True)
    logger.info(f"fitting preprocessor (scaler + vocabs) on {len(pooled_df)} pooled rows "
                f"from {len(train_dfs)} datasets")
    processor = NetflowPreprocessor(
        pooled_df,
        edge_columns=edge_columns,
        node_dim=args.node_dim,
        log1p_columns=log1p_columns,
        top_k_ports=args.top_k_ports,
        top_k_protocols=args.top_k_protocols,
    )
    if resumed:
        # holdout graphs must live in the exact feature space the heads were trained
        # in, so restore the fitted vocabs + scaler instead of this run's re-fit
        ref = next(iter(resumed.values()))
        if ref["protocol_vocab"] is not None:
            processor.protocol_vocab = list(ref["protocol_vocab"])
        if ref["dst_port_vocab"] is not None:
            processor.dst_port_vocab = list(ref["dst_port_vocab"])
        if ref["scaler_mean"] is not None and processor.edge_scaler is not None:
            processor.edge_scaler.mean_ = np.asarray(ref["scaler_mean"])
            processor.edge_scaler.scale_ = np.asarray(ref["scaler_scale"])

    if not resumed:
        # windows never span a dataset boundary; each dataset keeps its own loaders
        # because its head must only ever see its own benign windows
        train_loaders, val_loaders = {}, {}
        in_channels = None
        for name, df in train_dfs.items():
            graphs, _, _ = processor.construct_graph_list(
                df=df, window_size=args.window_size, step_size=args.step_size
            )
            logger.info(f"built {len(graphs)} graphs from {name}")
            if not graphs:
                raise RuntimeError(f"no training graphs built for {name}; check data / --window-size")
            in_channels = graphs[0].x.size(1)
            train_dataset, val_dataset = split_data(graphs, train_ratio=args.train_ratio)
            train_loaders[name] = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            val_loaders[name] = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        del train_dfs, pooled_df
        processor.df = None  # scaler/vocabs are fitted; free the pooled fit copy
        gc.collect()

        edge_attr_dim = processor.feature_dim
        logger.info(f"edge feature dim = {edge_attr_dim} ({len(processor.numeric_columns)} numeric "
                    f"+ {edge_attr_dim - len(processor.numeric_columns)} one-hot categorical)")

        # one model per dataset but all sharing the SAME encoder object: stepping any
        # dataset's batches updates the centralised encoder + only that dataset's head
        models = {}
        for name in dataset_names:
            m = build_model(args, in_channels, edge_attr_dim, device)
            if models:
                m.encoder = next(iter(models.values())).encoder
            models[name] = m
        n_shared = sum(p.numel() for p in next(iter(models.values())).encoder.parameters())
        n_head = sum(p.numel() for p in models[dataset_names[0]].parameters()) - n_shared
        logger.info(f"shared encoder params: {n_shared}; per-dataset head params: {n_head} "
                    f"(decoder + global-edge embedding) x {len(models)} heads")

        info = fit_joint(models, train_loaders, val_loaders, args, device, logger)
        train_time = time.time() - t_start

        # each head has its own error scale, so thresholds are calibrated per dataset
        # on that dataset's own benign validation edges
        calib_by_ds, n_calib_by_ds = {}, {}
        for name in dataset_names:
            logger.info(f"calibrating thresholds for {name}")
            calib_by_ds[name], n_calib_by_ds[name] = calibrate_thresholds(
                models[name], val_loaders[name], device, calib_quantiles, logger)

        for name in dataset_names:
            head_info = dict(info)
            head_info["calib_thresholds"] = calib_by_ds[name]
            head_ckpt_path = ckpt_dir / f"{PERSONALISED_NAME}-{name}.pt"
            save_checkpoint(head_ckpt_path, processor, args, head_info, model=models[name],
                            extra={"dataset": name, "stage": "joint-single-stage",
                                   "datasets": dataset_names,
                                   "n_calib_edges": n_calib_by_ds[name],
                                   "val_mse_by_dataset": info["val_mse_by_dataset"]})
            logger.info(f"saved personalised checkpoint to {head_ckpt_path}")

        del train_loaders, val_loaders
        gc.collect()
    else:
        del train_dfs, pooled_df
        processor.df = None
        gc.collect()

        models, calib_by_ds, n_calib_by_ds = {}, {}, {}
        for name, ckpt in resumed.items():
            m = build_model(args, ckpt.get("node_dim", args.node_dim), processor.feature_dim, device)
            m.load_state_dict(ckpt["model_state_dict"])
            models[name] = m
            calib_by_ds[name] = ckpt["calib_thresholds"]
            n_calib_by_ds[name] = ckpt.get("n_calib_edges")
        ref = next(iter(resumed.values()))
        info = {
            "best_epoch": ref.get("best_epoch"),
            "best_val_mse": ref.get("best_val_mse"),
            "epochs_trained": ref.get("epochs_trained"),
            "val_mse_by_dataset": ref.get("val_mse_by_dataset"),
        }
        train_time = 0.0

    for name in pending:
        model = models[name]
        ds_out_dir = per_out_dir / name
        metrics, all_preds, all_labels = evaluate_holdout(
            data_dir / DATASETS[name][1], processor, model, device, args,
            calib_by_ds[name], ds_out_dir, logger,
        )
        del all_preds, all_labels
        gc.collect()

        metrics.update({
            "dataset": name,
            "mode": "personalised_joint",
            "trained_on": dataset_names,
            "edge_columns": edge_columns,
            "log1p_columns": log1p_columns,
            # single stage: one joint run trains the encoder and every head together
            "train_time_sec": train_time,
            "best_epoch": info["best_epoch"],
            "epochs_trained": info["epochs_trained"],
            "best_val_mse": info["best_val_mse"],  # mean across datasets
            "val_mse": (info.get("val_mse_by_dataset") or {}).get(name),
            "n_calib_edges": n_calib_by_ds[name],
        })
        with open(ds_out_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(
            f"RESULTS personalised->{name}: AUC-ROC={metrics['auc_roc']:.4f} PR-AUC={metrics['pr_auc']:.4f} "
            f"Precision={metrics['precision']:.4f} Recall={metrics['recall']:.4f} "
            f"F1={metrics['f1']:.4f} FPR={metrics['fpr']:.4f} (Youden's-J threshold={metrics['threshold']:.4f})"
        )
        summary_rows.append(metrics)

    # resumed + freshly trained rows arrive out of order; restore the canonical order
    summary_rows.sort(key=lambda r: dataset_names.index(r["dataset"]))
    return summary_rows


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = resolve_data_dir(args.data_dir)
    edge_columns = resolve_edge_columns(args.edge_columns)
    log1p_columns = resolve_log1p_columns(args.log1p_columns, edge_columns)

    calib_quantiles = sorted(float(q) for q in args.calib_quantiles.split(",") if q.strip())
    if any(not (0.0 < q < 1.0) for q in calib_quantiles):
        raise ValueError(f"--calib-quantiles must be in (0, 1), got {calib_quantiles}")

    dataset_names = list(DATASETS) if args.datasets == "all" else [d.strip() for d in args.datasets.split(",")]
    for name in dataset_names:
        if name not in DATASETS:
            raise ValueError(f"Unknown dataset {name!r}; choices: {list(DATASETS)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)

    out_dir, log_dir, ckpt_dir = Path(args.output_dir), Path(args.log_dir), Path(args.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if args.combined and args.personalised:
        raise ValueError("--combined and --personalised are mutually exclusive; pick one mode")

    # each mode keeps its comparison table inside its own folder so it never
    # clobbers another mode's comparison.csv
    if args.personalised:
        summary_rows = run_personalised(args, dataset_names, data_dir, edge_columns, log1p_columns,
                                        calib_quantiles, device, out_dir, log_dir, ckpt_dir)
        summary_path = out_dir / PERSONALISED_NAME / "comparison.csv"
    elif args.combined:
        summary_rows = run_combined(args, dataset_names, data_dir, edge_columns, log1p_columns,
                                    calib_quantiles, device, out_dir, log_dir, ckpt_dir)
        summary_path = out_dir / COMBINED_NAME / "comparison.csv"
    else:
        summary_rows = run_per_dataset(args, dataset_names, data_dir, edge_columns, log1p_columns,
                                       calib_quantiles, device, out_dir, log_dir, ckpt_dir)
        summary_path = out_dir / "comparison.csv"

    if summary_rows:
        cols = ["dataset", "auc_roc", "pr_auc", "precision", "recall", "f1", "fpr",
                "threshold", "n_eval_edges", "train_time_sec", "infer_time_sec"]
        summary_df = pd.DataFrame(summary_rows)
        summary_df = summary_df[[c for c in cols if c in summary_df.columns]]
        summary_df.to_csv(summary_path, index=False)
        print("\n=== Summary across datasets ===")
        print(summary_df.to_string(index=False))
        print(f"\nSaved comparison table to {summary_path}")


if __name__ == "__main__":
    main()
