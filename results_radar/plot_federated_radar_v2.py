#!/usr/bin/env python3
"""Radar chart per federated method (shared-weight x personalisation grid),
each axis = a metric averaged over the 4 NetFlow datasets. Mirror of
plot_federated_radar.py, pointed at the current results/ layout (the
2026-07-13 run_federated_gpu.sh rewrite: shared-weight=fedavg/fedprox x
personalisation=na/fedbn/fedrep). SecAgg+ variants are excluded (ambiguous/
stale run-dir naming, user call 2026-07-13). Recomputes averages from
results/<method>/seed_*/federated/<dataset>/metrics.json. FPR is shown as
Specificity (1-FPR) so that outward = better on every axis. Writes
results_radar/federated_radar_v2.png (per-method, dataset-averaged) and
results_radar/federated_radar_v2_by_dataset.png (per-dataset breakdown, one
polygon per method)."""
import json
import glob
import os
import collections

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless / no display
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # repo root

DATASETS = ["NF-CICIDS2018-v3", "NF-UNSW-NB15-v3", "NF-BoT-IoT-v3", "NF-ToN-IoT-v3"]
DS_SHORT = {"NF-CICIDS2018-v3": "CICIDS", "NF-UNSW-NB15-v3": "UNSW",
            "NF-BoT-IoT-v3": "BoT", "NF-ToN-IoT-v3": "ToN"}

# method label -> glob of per-seed run dirs (each holding <dataset>/metrics.json)
# SecAgg+ runs (results/fedAvgRepSecAgg, results/fedAveBNSecAgg) deliberately
# excluded: their directory names disagree with the --personalisation flag
# actually used in federated_runs.sh, so identity is ambiguous.
CONFIGS = {
    "FedAvg + NA":     "results/fedAvgNA/seed_*/federated",
    "FedAvg + FedBN":  "results/fedAvgBN/seed_*/federated",
    "FedAvg + FedRep": "results/fedAvgRep/seed_*/federated",
    "FedProx + NA":    "results/fedProxNA/seed_*/federated",
    "FedProx + FedBN": "results/fedProxBN/seed_*/federated",
    "FedProx + FedRep": "results/fedProxRep/seed_*/federated",
}

AXES = [
    ("auc_roc",     "ROC-AUC"),
    ("recall",      "Recall"),
    ("precision",   "Precision"),
    ("f1",          "F1"),
    ("specificity", "Specificity\n(1-FPR)"),
]
RAW_METRICS = ["auc_roc", "recall", "precision", "f1", "fpr"]


def per_dataset_means(base):
    """Mean over seeds for each dataset. Returns {dataset: {metric: mean}}."""
    per_ds = {}
    n_seeds = 0
    for ds in DATASETS:
        vals = collections.defaultdict(list)
        for d in sorted(glob.glob(os.path.join(ROOT, base))):
            f = os.path.join(d, ds, "metrics.json")
            if os.path.exists(f):
                j = json.load(open(f))
                for m in RAW_METRICS:
                    vals[m].append(j[m])
        n_seeds = max(n_seeds, max((len(v) for v in vals.values()), default=0))
        d = {m: float(np.mean(vals[m])) for m in RAW_METRICS}
        d["specificity"] = 1.0 - d["fpr"]
        per_ds[ds] = d
    return per_ds, n_seeds


def model_average(per_ds):
    """Mean over the 4 datasets."""
    avg = {m: float(np.mean([per_ds[ds][m] for ds in DATASETS])) for m in RAW_METRICS}
    avg["specificity"] = 1.0 - avg["fpr"]
    return avg


KEYS = [k for k, _ in AXES]
LABELS = [lbl for _, lbl in AXES]
ANGLES = np.linspace(0, 2 * np.pi, len(KEYS), endpoint=False).tolist()
ANGLES_C = ANGLES + ANGLES[:1]
five = lambda a: np.mean([a[k] for k in KEYS])


def style_ax(ax, labelsize=9.5):
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1)
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0], labels=["", ".4", "", ".8", ""],
                  angle=90, fontsize=7, color="#8a8983")
    ax.set_xticks(ANGLES)
    ax.set_xticklabels(LABELS, fontsize=labelsize, color="#333333")
    ax.tick_params(axis="x", pad=8)
    ax.grid(color="#d9d9d4", linewidth=0.8)
    ax.spines["polar"].set_color("#c9c9c2")


def polygon(ax, avg, color, label=None, value_labels=False):
    vals = [avg[k] for k in KEYS] + [avg[KEYS[0]]]
    ax.plot(ANGLES_C, vals, color=color, linewidth=2, solid_joinstyle="round", label=label)
    ax.fill(ANGLES_C, vals, color=color, alpha=0.16)
    ax.scatter(ANGLES, [avg[k] for k in KEYS], s=22, color=color, zorder=5)
    if value_labels:
        for ang, k in zip(ANGLES, KEYS):
            ax.text(ang, min(avg[k] + 0.13, 1.14), f"{avg[k]:.2f}", ha="center",
                    va="center", fontsize=8, fontweight="bold", color="#222222")


def main():
    data = {name: per_dataset_means(base) for name, base in CONFIGS.items()}
    averages = {name: (model_average(pd), n) for name, (pd, n) in data.items()}
    best = max(averages, key=lambda n: five(averages[n][0]))

    # ---- Figure 1: per-method, dataset-averaged --------------------------
    ACCENT, BEST = "#2a78d6", "#008300"
    ncol = 3
    nrow = int(np.ceil(len(averages) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 5.3 * nrow),
                             subplot_kw=dict(polar=True))
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, (avg, n_seeds)) in zip(axes, averages.items()):
        is_best = name == best
        color = BEST if is_best else ACCENT
        style_ax(ax)
        polygon(ax, avg, color, value_labels=True)
        seed_note = "1 seed" if n_seeds <= 1 else f"{n_seeds} seeds"
        ax.set_title(("★ " if is_best else "") + name, fontsize=11.5,
                     fontweight="bold", color=(BEST if is_best else "#111111"), pad=22)
        ax.text(0.5, -0.14, f"5-metric mean {five(avg):.3f}  ·  {seed_note}",
                transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color="#666666")

    for ax in axes[len(averages):]:
        ax.axis("off")

    fig.suptitle("Federated methods — metrics averaged across the 4 NetFlow datasets",
                 fontsize=15, fontweight="bold", y=0.995)
    fig.text(0.5, 0.965,
             "shared-weight (FedAvg/FedProx) x personalisation (NA/FedBN/FedRep), no SecAgg+. "
             "Each axis is the metric averaged over CICIDS / UNSW / BoT-IoT / ToN-IoT. "
             "FPR shown as Specificity so further out = better. ★ = best 5-metric mean.",
             ha="center", fontsize=10, color="#555555")
    fig.tight_layout(rect=[0, 0.01, 1, 0.94], h_pad=6.0)
    out1 = os.path.join(HERE, "federated_radar_v2.png")
    fig.savefig(out1, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out1}")

    # ---- Figure 2: per-dataset breakdown, 6 method polygons ----------------
    fig, axes = plt.subplots(1, 4, figsize=(4.6 * 4, 6.0), subplot_kw=dict(polar=True))
    axes = axes.ravel()
    METHOD_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#eb6834", "#4a3aa7", "#008300"]
    for ax, ds in zip(axes, DATASETS):
        style_ax(ax, labelsize=8.5)
        for (name, (per_ds, _)), color in zip(data.items(), METHOD_COLORS):
            polygon(ax, per_ds[ds], color, label=name)
        ax.set_title(DS_SHORT[ds], fontsize=12, fontweight="bold", pad=20)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=9.5,
               frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Federated methods compared per dataset (mean over seeds; outward = better)",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0.08, 1, 0.94])
    out2 = os.path.join(HERE, "federated_radar_v2_by_dataset.png")
    fig.savefig(out2, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out2}")
    print(f"best (5-metric mean): {best} = {five(averages[best][0]):.3f}")


if __name__ == "__main__":
    main()
