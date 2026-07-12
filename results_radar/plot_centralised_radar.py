#!/usr/bin/env python3
"""Radar chart per centralised model, each axis = a metric averaged over the 4
NetFlow datasets. Recomputes averages from results_infer/*/metrics.json so the
figure stays in sync with the sweeps. FPR is shown as Specificity (1-FPR) so that
outward = better on every axis. Writes results_radar/centralised_radar.png."""
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

# model label -> glob of per-seed run dirs (each holding <dataset>/metrics.json)
CONFIGS = {
    "Combined":                    "results_infer/seed_sweep/runs/combined_seed*/combined",
    "Personalised (single-stage)": "results_infer/personalised/personalised",
    "Personalised-frozen":         "results_infer/seed_sweep_personalised_frozen/runs/personalised-frozen_seed*/personalised-frozen",
    "AdaBN + Combined":            "results_infer/seed_sweep_batchnorm/runs/seed*/combined-adabn",
    "AdaBN + Personalised-frozen": "results_infer/seed_sweep_batchnorm/runs/seed*/personalised-frozen-adabn",
    "FedBN + Combined":            "results_infer/seed_sweep_batchnorm/runs/seed*/combined-fedbn",
    "FedBN + Personalised-frozen": "results_infer/seed_sweep_batchnorm/runs/seed*/personalised-frozen-fedbn",
}

# axis key in metrics.json, display label, whether it is already "higher=better"
AXES = [
    ("auc_roc",     "ROC-AUC"),
    ("recall",      "Recall"),
    ("precision",   "Precision"),
    ("f1",          "F1"),
    ("specificity", "Specificity\n(1-FPR)"),
]
RAW_METRICS = ["auc_roc", "recall", "precision", "f1", "fpr"]


def model_averages(base):
    """Mean over seeds per dataset, then mean over datasets. Returns dict + n_seeds."""
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
        per_ds[ds] = {m: np.mean(vals[m]) for m in RAW_METRICS}
    avg = {m: float(np.mean([per_ds[ds][m] for ds in DATASETS])) for m in RAW_METRICS}
    avg["specificity"] = 1.0 - avg["fpr"]
    return avg, n_seeds


def main():
    results = {name: model_averages(base) for name, base in CONFIGS.items()}
    five = lambda a: np.mean([a[k] for k, _ in AXES])
    best = max(results, key=lambda n: five(results[n][0]))

    keys = [k for k, _ in AXES]
    labels = [lbl for _, lbl in AXES]
    n = len(keys)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # close the loop

    ACCENT, ACCENT_F = "#2a78d6", "#2a78d6"
    BEST, BEST_F = "#008300", "#008300"

    ncol = 4
    nrow = int(np.ceil(len(results) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 5.1 * nrow),
                             subplot_kw=dict(polar=True))
    axes = np.atleast_1d(axes).ravel()

    for ax, (name, (avg, n_seeds)) in zip(axes, results.items()):
        is_best = name == best
        color = BEST if is_best else ACCENT
        vals = [avg[k] for k in keys]
        vals += vals[:1]

        ax.set_theta_offset(np.pi / 2)   # first axis at top
        ax.set_theta_direction(-1)       # clockwise
        ax.set_ylim(0, 1)
        ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0],
                      labels=["", ".4", "", ".8", ""],
                      angle=90, fontsize=7, color="#8a8983")
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=9.5, color="#333333")
        ax.tick_params(axis="x", pad=8)
        ax.grid(color="#d9d9d4", linewidth=0.8)
        ax.spines["polar"].set_color("#c9c9c2")

        ax.plot(angles, vals, color=color, linewidth=2, solid_joinstyle="round")
        ax.fill(angles, vals, color=color, alpha=0.16)
        ax.scatter(angles[:-1], vals[:-1], s=22, color=color, zorder=5)

        # per-vertex value labels just outside the point
        for ang, k in zip(angles[:-1], keys):
            ax.text(ang, min(avg[k] + 0.13, 1.14), f"{avg[k]:.2f}",
                    ha="center", va="center", fontsize=8, fontweight="bold",
                    color="#222222")

        seed_note = "1 seed" if n_seeds <= 1 else f"{n_seeds} seeds"
        title = ("★ " if is_best else "") + name
        ax.set_title(title, fontsize=11.5, fontweight="bold",
                     color=(BEST if is_best else "#111111"), pad=22)
        ax.text(0.5, -0.13, f"5-metric mean {five(avg):.3f}  ·  {seed_note}",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=8.5, color="#666666")

    for ax in axes[len(results):]:
        ax.axis("off")

    fig.suptitle("Centralised models — metrics averaged across the 4 NetFlow datasets",
                 fontsize=15, fontweight="bold", y=0.995)
    fig.text(0.5, 0.955,
             "Each axis is the metric averaged over CICIDS / UNSW / BoT-IoT / ToN-IoT. "
             "FPR shown as Specificity so further out = better. ★ = best 5-metric mean.",
             ha="center", fontsize=10, color="#555555")

    fig.tight_layout(rect=[0, 0.01, 1, 0.945], h_pad=6.0)
    out = os.path.join(HERE, "centralised_radar.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")
    print(f"best (5-metric mean): {best} = {five(results[best][0]):.3f}")


if __name__ == "__main__":
    main()
