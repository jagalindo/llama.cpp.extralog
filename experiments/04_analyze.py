#!/usr/bin/env python3
"""
04_analyze.py — Visualize activation importance and quantization decisions.

Produces plots:
  1. Per-layer outlier fraction over time (from activations.csv)
     → shows which layers have the most salient activations
  2. Channel importance distribution per weight
     → histogram of importance scores with threshold line
  3. Block-32 heatmap for k-quant decisions
     → 2D grid: layer × block, coloured by mean importance
  4. Domain vs baseline comparison (if both datasets available)
     → scatter plot: baseline importance vs domain importance per channel

Usage:
  python experiments/04_analyze.py [--data-dir experiments/data] [--out-dir experiments/plots]
"""

import argparse
import sys
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")  # no display needed
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
except ImportError:
    sys.exit("pip install pandas numpy matplotlib")


# ── helpers ─────────────────────────────────────────────────────────────────

def load(path_stem: Path) -> "pd.DataFrame":
    for suffix in (".parquet", ".csv"):
        p = path_stem.with_suffix(suffix)
        if p.exists():
            return pd.read_parquet(p) if suffix == ".parquet" else pd.read_csv(p)
    raise FileNotFoundError(f"{path_stem}.[parquet|csv] not found")


def save_fig(fig: "plt.Figure", out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def extract_layer(name: str) -> int:
    """Extract layer index from 'blk.7.attn_qkv.weight' → 7."""
    parts = name.split(".")
    for p in parts:
        if p.isdigit():
            return int(p)
    return -1


# ── plot 1: outlier fraction over time ──────────────────────────────────────

def plot_outlier_over_time(data_dir: Path, label: str, out_dir: Path) -> None:
    csv = data_dir / label / "activations.csv"
    if not csv.exists():
        print(f"  Skipping plot 1 ({label}): {csv} not found.")
        return

    df = pd.read_csv(csv)
    df["layer"] = df["weight_name"].apply(extract_layer)
    df["weight_type"] = df["weight_name"].apply(lambda n: n.split(".")[-2] + "." + n.split(".")[-1])

    fig, ax = plt.subplots(figsize=(12, 5))
    for wtype, grp in df.groupby("weight_type"):
        ax.plot(grp["call_id"], grp["outlier_frac"], alpha=0.6, label=wtype, linewidth=0.8)
    ax.set_xlabel("MUL_MAT call index")
    ax.set_ylabel("Outlier channel fraction\n(channels > mean + 3σ)")
    ax.set_title(f"[{label}] Outlier activation fraction over time")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    save_fig(fig, out_dir, f"01_outlier_over_time_{label}")


# ── plot 2: channel importance distribution ──────────────────────────────────

def plot_importance_hist(scores: "pd.DataFrame", out_dir: Path) -> None:
    weight_names = scores["weight_name"].unique()
    n = len(weight_names)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3))
    axes = np.array(axes).flatten()

    for i, wname in enumerate(sorted(weight_names)):
        ax = axes[i]
        grp = scores[scores["weight_name"] == wname]["importance"]
        thresh = grp.mean() + 2.0 * grp.std()
        ax.hist(grp, bins=60, color="steelblue", alpha=0.7, edgecolor="none")
        ax.axvline(thresh, color="red", linewidth=1.5, label=f"threshold={thresh:.3f}")
        ax.set_title(wname.split(".")[-3] + "." + wname.split(".")[-2], fontsize=8)
        ax.set_xlabel("importance", fontsize=7)
        ax.legend(fontsize=6)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Channel importance distribution (domain activations)", fontsize=11)
    fig.tight_layout()
    save_fig(fig, out_dir, "02_channel_importance_hist")


# ── plot 3: block-32 heatmap ─────────────────────────────────────────────────

def plot_block_heatmap(scores: "pd.DataFrame", out_dir: Path) -> None:
    scores = scores.copy()
    scores["layer"]      = scores["weight_name"].apply(extract_layer)
    scores["weight_type"] = scores["weight_name"].apply(
        lambda n: ".".join(n.split(".")[-2:]))
    scores["block32"]    = scores["channel_idx"] // 32

    for wtype, grp in scores.groupby("weight_type"):
        pivot = grp.pivot_table(
            index="layer", columns="block32",
            values="importance", aggfunc="mean")

        if pivot.empty:
            continue

        fig, ax = plt.subplots(figsize=(min(24, pivot.shape[1] * 0.12 + 2),
                                        max(4, pivot.shape[0] * 0.4)))
        im = ax.imshow(pivot.values, aspect="auto", cmap="hot",
                       interpolation="nearest", origin="lower")
        plt.colorbar(im, ax=ax, label="mean importance")
        ax.set_xlabel("Block index (32 channels per block)")
        ax.set_ylabel("Layer")
        ax.set_title(f"{wtype} — per-block importance heatmap")
        ax.set_yticks(range(pivot.shape[0]))
        ax.set_yticklabels(pivot.index, fontsize=6)
        save_fig(fig, out_dir, f"03_block_heatmap_{wtype.replace('.', '_')}")


# ── plot 4: domain vs baseline scatter ───────────────────────────────────────

def plot_domain_vs_baseline(data_dir: Path, out_dir: Path) -> None:
    try:
        domain   = load(data_dir / "importance_raw_domain")
        baseline = load(data_dir / "importance_raw_baseline")
    except FileNotFoundError as e:
        print(f"  Skipping plot 4: {e}")
        return

    merged = domain.merge(
        baseline[["weight_name", "channel_idx", "mean_abs_mean"]],
        on=["weight_name", "channel_idx"], how="inner",
        suffixes=("_domain", "_baseline"))

    weight_names = merged["weight_name"].unique()
    n = len(weight_names)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 4))
    axes = np.array(axes).flatten()

    for i, wname in enumerate(sorted(weight_names)):
        ax = axes[i]
        grp = merged[merged["weight_name"] == wname]
        ax.scatter(grp["mean_abs_mean_baseline"], grp["mean_abs_mean_domain"],
                   alpha=0.3, s=2, c="steelblue")
        # Diagonal = same importance in both
        lim = max(grp["mean_abs_mean_baseline"].max(), grp["mean_abs_mean_domain"].max())
        ax.plot([0, lim], [0, lim], "r--", linewidth=0.8, label="equal importance")
        ax.set_xlabel("baseline mean_abs", fontsize=7)
        ax.set_ylabel("domain mean_abs",   fontsize=7)
        ax.set_title(".".join(wname.split(".")[-3:]), fontsize=8)
        # Points above the diagonal are domain-specific.

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Domain vs baseline channel importance\n"
                 "(above diagonal = domain-specific, below = general)", fontsize=11)
    fig.tight_layout()
    save_fig(fig, out_dir, "04_domain_vs_baseline_scatter")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir",  default="experiments/data")
    parser.add_argument("--out-dir",   default="experiments/plots")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)

    print("=== Plot 1: outlier fraction over time ===")
    plot_outlier_over_time(data_dir, "domain",   out_dir)
    plot_outlier_over_time(data_dir, "baseline", out_dir)

    print("\n=== Plot 2: channel importance histograms ===")
    scores = load(data_dir / "importance_scores")
    plot_importance_hist(scores, out_dir)

    print("\n=== Plot 3: per-block-32 heatmaps ===")
    plot_block_heatmap(scores, out_dir)

    print("\n=== Plot 4: domain vs baseline scatter ===")
    plot_domain_vs_baseline(data_dir, out_dir)

    print(f"\nAll plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
