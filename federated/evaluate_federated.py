"""Score the federated (FedAvg) global model on every dataset's holdout.

Run AFTER `flwr run federated` has written checkpoints_infer/federated.pt.
For each dataset this rebuilds the client's silo-local preprocessing and val
split exactly as during training (same run config + seed), calibrates label-free
benign-quantile thresholds on that client's validation edges, then streams the
holdout through the global model. Outputs mirror the other modes:
results_infer/federated/<dataset>/metrics.json + roc_curve.png +
threshold_sweep.csv, and results_infer/federated/comparison.csv.

Usage: python federated/evaluate_federated.py [--datasets ...] [--resume] ...
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

FEDERATED_DIR = Path(__file__).resolve().parent
if str(FEDERATED_DIR) not in sys.path:
    sys.path.insert(0, str(FEDERATED_DIR))

from fedgnn.task import CLIENT_DATASETS, REPO_ROOT, build_model, load_client_data

import pandas as pd
import torch

from train_and_infer import (
    DATASETS,
    calibrate_thresholds,
    evaluate_holdout,
    setup_logger,
)
from fedgnn.task import resolve_data_dir

FEDERATED_NAME = "federated"


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate the FedAvg global model on the holdouts")
    p.add_argument("--checkpoint", default=str(REPO_ROOT / "checkpoints_infer" / "federated.pt"))
    p.add_argument("--datasets", default="all",
                   help=f"comma-separated subset of {CLIENT_DATASETS} or 'all'")
    p.add_argument("--output-dir", default=str(REPO_ROOT / "results_infer"))
    p.add_argument("--log-dir", default=str(REPO_ROOT / "logs_infer"))
    p.add_argument("--device", default="auto", help="'auto', 'cuda', 'cuda:0', or 'cpu'")
    p.add_argument("--calib-quantiles", default="0.90,0.95,0.99,0.995,0.999")
    p.add_argument("--infer-chunk-windows", type=int, default=1000)
    p.add_argument("--max-holdout-rows", type=int, default=None,
                   help="cap rows read from the holdout split (smoke-testing)")
    p.add_argument("--batch-size", type=int, default=None,
                   help="inference batch size; defaults to the training batch size")
    p.add_argument("--resume", action="store_true",
                   help="skip datasets whose federated metrics.json already exists")
    return p.parse_args()


def main():
    args = parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["run_config"]
    mode = ckpt.get("mode", "federated_fedavg")

    dataset_names = CLIENT_DATASETS if args.datasets == "all" else \
        [d.strip() for d in args.datasets.split(",")]
    for name in dataset_names:
        if name not in DATASETS:
            raise ValueError(f"Unknown dataset {name!r}; choices: {CLIENT_DATASETS}")

    calib_quantiles = sorted(float(q) for q in args.calib_quantiles.split(",") if q.strip())
    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    data_dir = resolve_data_dir(cfg["data-dir"])

    out_dir = Path(args.output_dir) / FEDERATED_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(FEDERATED_NAME, log_dir / f"{FEDERATED_NAME}.log")
    logger.info(f"===== federated evaluation: checkpoint={args.checkpoint} "
                f"(best_round={ckpt['best_round']}/{ckpt['rounds_trained']}, "
                f"best_val_mse={ckpt['best_val_mse']:.6f}) device={device} =====")

    # evaluate_holdout reads the streaming/window settings off an args-style object
    eval_args = argparse.Namespace(
        window_size=cfg["window-size"],
        step_size=cfg["step-size"],
        batch_size=args.batch_size or cfg["batch-size"],
        max_holdout_rows=args.max_holdout_rows,
        infer_chunk_windows=args.infer_chunk_windows,
    )

    summary_rows = []
    for name in dataset_names:
        ds_out_dir = out_dir / name
        metrics_path = ds_out_dir / "metrics.json"
        if args.resume and metrics_path.exists():
            logger.info(f"{metrics_path} already exists, skipping {name} due to --resume")
            with open(metrics_path) as f:
                summary_rows.append(json.load(f))
            continue

        t_start = time.time()
        logger.info(f"rebuilding {name}'s silo-local preprocessing + val split")
        data = load_client_data(name, cfg)

        model = build_model(cfg, data["in_channels"], data["edge_attr_dim"], device)
        if mode == "federated_fedrep":
            # shared body (encoder) + this client's private head (decoder + global-edge)
            model.load_state_dict(ckpt["body_state_dict"], strict=False)
            head_sd = ckpt["head_state_dicts"].get(name)
            if head_sd is None:
                raise KeyError(f"no FedRep head saved for {name}; "
                               f"have {list(ckpt['head_state_dicts'])}")
            model.load_state_dict(head_sd, strict=False)
        else:
            model.load_state_dict(ckpt["model_state_dict"])

        # thresholds are per client: each silo's scaler gives it its own error scale
        calib_thresholds, n_calib_edges = calibrate_thresholds(
            model, data["val_loader"], device, calib_quantiles, logger)

        metrics, all_preds, all_labels = evaluate_holdout(
            data_dir / DATASETS[name][1], data["processor"], model, device, eval_args,
            calib_thresholds, ds_out_dir, logger,
        )
        del all_preds, all_labels, data
        gc.collect()

        metrics.update({
            "dataset": name,
            "mode": mode,
            "trained_on": CLIENT_DATASETS,
            "edge_columns": cfg["edge-columns"],
            "log1p_columns": cfg["log1p-columns"],
            "best_round": ckpt["best_round"],
            "rounds_trained": ckpt["rounds_trained"],
            "best_val_mse": ckpt["best_val_mse"],  # weighted mean across clients
            "train_time_sec": ckpt["train_time_sec"],
            "n_calib_edges": n_calib_edges,
            "eval_time_sec": time.time() - t_start,
        })
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(
            f"RESULTS federated->{name}: AUC-ROC={metrics['auc_roc']:.4f} "
            f"PR-AUC={metrics['pr_auc']:.4f} Precision={metrics['precision']:.4f} "
            f"Recall={metrics['recall']:.4f} F1={metrics['f1']:.4f} "
            f"FPR={metrics['fpr']:.4f} (Youden's-J threshold={metrics['threshold']:.4f})"
        )
        summary_rows.append(metrics)

    if summary_rows:
        cols = ["dataset", "auc_roc", "pr_auc", "precision", "recall", "f1", "fpr",
                "threshold", "n_eval_edges", "best_round", "train_time_sec", "infer_time_sec"]
        summary_df = pd.DataFrame(summary_rows)
        summary_df = summary_df[[c for c in cols if c in summary_df.columns]]
        summary_path = out_dir / "comparison.csv"
        summary_df.to_csv(summary_path, index=False)
        print("\n=== Federated summary across datasets ===")
        print(summary_df.to_string(index=False))
        print(f"\nSaved comparison table to {summary_path}")


if __name__ == "__main__":
    main()
