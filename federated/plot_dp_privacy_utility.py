"""Privacy-utility curve for the central-DP sigma sweep.

Reads every results/dp_sweep/sigma_<s>/federated_per_run.csv, converts each
noise multiplier sigma into an exact client-level (epsilon, delta=1e-5) via the
analytic Gaussian accountant (Balle & Wang 2018), and plots utility (AUC-ROC,
y-axis) against epsilon (x-axis, log scale) with sigma labelled on a top axis.

Privacy accounting: every round is a full-participation Gaussian mechanism with
noise multiplier sigma (aggregate noise std sigma*C/n against a one-client
sensitivity of C/n on the unweighted mean). T-fold composition of Gaussians is
itself Gaussian with GDP parameter mu = sqrt(T)/sigma, so epsilon is solved
exactly from  delta = Phi(-eps/mu + mu/2) - e^eps * Phi(-eps/mu - mu/2)
(evaluated in log space -- e^eps overflows for the small sigmas swept here).

The sigma=0 run has epsilon = infinity; it is drawn as the no-DP baseline
(dashed horizontal line), not as a point on the epsilon axis.

Usage:
    python federated/plot_dp_privacy_utility.py
Outputs (next to the sweep):
    results/dp_sweep/privacy_utility.png
    results/dp_sweep/privacy_utility.csv   (one row per sigma: epsilon + AUCs)
"""

import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm

REPO_ROOT = Path(__file__).resolve().parents[1]
SWEEP_DIR = REPO_ROOT / "results" / "dp_sweep"

ROUNDS = 50      # every sweep run trained exactly this many rounds
DELTA = 1e-5


def gaussian_delta(eps, mu):
    """delta(eps) of a Gaussian mechanism with GDP parameter mu, in log space:
    delta = Phi(-eps/mu + mu/2) - e^eps * Phi(-eps/mu - mu/2)."""
    log_a = norm.logcdf(-eps / mu + mu / 2)
    log_b = eps + norm.logcdf(-eps / mu - mu / 2)
    return math.exp(log_a) * -math.expm1(log_b - log_a)


def epsilon_for(sigma, rounds=ROUNDS, delta=DELTA):
    """Exact client-level epsilon after `rounds` full-participation Gaussian
    rounds with noise multiplier sigma (bisection on the analytic accountant)."""
    if sigma <= 0:
        return math.inf
    mu = math.sqrt(rounds) / sigma
    lo, hi = 0.0, 1.0
    while gaussian_delta(hi, mu) > delta:
        hi *= 2.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if gaussian_delta(mid, mu) > delta:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def load_sweep():
    """One row per (sigma, dataset): AUC mean/std over seeds, plus a 'mean'
    pseudo-dataset = per-seed average over the 4 datasets (so its std reflects
    seed-to-seed variation of the headline number, not dataset spread)."""
    rows = []
    for run_dir in sorted(SWEEP_DIR.glob("sigma_*")):
        m = re.fullmatch(r"sigma_([0-9.]+)", run_dir.name)
        if not m:
            continue
        sigma = float(m.group(1))
        per_run = pd.read_csv(run_dir / "federated_per_run.csv")
        by_ds = per_run.groupby("dataset")["auc_roc"].agg(["mean", "std"])
        for ds, r in by_ds.iterrows():
            rows.append({"sigma": sigma, "dataset": ds,
                         "auc_mean": r["mean"], "auc_std": r["std"]})
        per_seed = per_run.groupby("seed")["auc_roc"].mean()
        rows.append({"sigma": sigma, "dataset": "mean",
                     "auc_mean": per_seed.mean(), "auc_std": per_seed.std()})
        per_seed_med = per_run.groupby("seed")["auc_roc"].median()
        rows.append({"sigma": sigma, "dataset": "median",
                     "auc_mean": per_seed_med.mean(), "auc_std": per_seed_med.std()})
    df = pd.DataFrame(rows)
    df["epsilon"] = df["sigma"].map(epsilon_for)
    return df.sort_values(["dataset", "sigma"]).reset_index(drop=True)


def main():
    df = load_sweep()

    wide = df.pivot(index="sigma", columns="dataset", values="auc_mean").round(4)
    wide.insert(0, "epsilon", [round(epsilon_for(s), 2) for s in wide.index])
    csv_path = SWEEP_DIR / "privacy_utility.csv"
    wide.to_csv(csv_path)
    print(wide.to_string())

    dp = df[df["sigma"] > 0]
    base = df[df["sigma"] == 0].set_index("dataset")["auc_mean"]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    mean = dp[dp["dataset"] == "mean"].sort_values("epsilon")
    median = dp[dp["dataset"] == "median"].sort_values("epsilon")
    ax.plot(mean["epsilon"], mean["auc_mean"], "o-", color="black",
            lw=2.0, ms=5, label="mean of 4 datasets", zorder=5)
    ax.plot(median["epsilon"], median["auc_mean"], "s--", color="tab:blue",
            lw=1.6, ms=4.5, label="median of 4 datasets", zorder=4)
    if "mean" in base:
        ax.axhline(base["mean"], color="black", ls=":", lw=1.2, alpha=0.7,
                   label=f"no-DP baseline, mean (AUC {base['mean']:.3f})")

    ax.set_ylim(bottom=0.5)
    ax.set_xscale("log")
    ax.set_xlabel(rf"client-level $\varepsilon$ at $\delta$={DELTA:g} "
                  f"({ROUNDS} rounds, analytic Gaussian accountant)")
    ax.set_ylabel("AUC-ROC (utility)")
    ax.set_title("Privacy-utility trade-off: central DP, server-side clipping "
                 "(C=0.13, FedBN, 4 clients)")
    ax.grid(True, which="both", alpha=0.25)
    # stronger privacy is to the LEFT (smaller epsilon); annotate sigma on top
    sec = ax.secondary_xaxis("top")
    sec.set_xscale("log")
    sec.set_xticks(mean["epsilon"].tolist())
    sec.set_xticklabels([f"{s:g}" for s in mean["sigma"]], fontsize=8)
    sec.set_xlabel(r"noise multiplier $\sigma$", fontsize=9)
    sec.tick_params(axis="x", which="minor", top=False, labeltop=False)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()

    png_path = SWEEP_DIR / "privacy_utility.png"
    fig.savefig(png_path, dpi=200)
    print(f"\nwrote {png_path}\nwrote {csv_path}")


if __name__ == "__main__":
    main()
