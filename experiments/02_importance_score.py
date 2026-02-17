#!/usr/bin/env python3
"""
02_importance_score.py — Score each channel and k-quant block for preservation.

Reads:  data/importance_raw_domain.{parquet,csv}
        data/importance_raw_baseline.{parquet,csv}   (optional)
Writes: data/importance_scores.{parquet,csv}
        data/keepmask_per_row.json
        data/keepmask_per_block32.json

Importance metric (inspired by AWQ):
  importance(ch) = mean_abs_mean(ch)        (domain activation magnitude)

If baseline data is available, compute domain-specific importance:
  importance(ch) = mean_abs_mean_domain(ch) - mean_abs_mean_baseline(ch)
  This isolates channels that are specifically important for your domain,
  not just always-active channels.

Two keepmask formats are generated:

  per_row (for Q8_0 / per-row quantization):
    Channels above the threshold → entire row kept at higher precision.
    "row" here = input channel index = column of the weight matrix.

  per_block_32 (for k-quants Q4_K_M, Q5_K_S, etc.):
    k-quants quantize in blocks of 32 consecutive input channels.
    If any channel in a block exceeds the threshold, the whole block is marked.
    Block index = channel_idx // 32.

Usage:
  python experiments/02_importance_score.py [options]

Options:
  --threshold-sigma  N   keep channels > mean + N*std  (default: 2.0)
  --top-k-fraction   F   alternatively keep top F fraction of channels (0.0–1.0)
  --data-dir         DIR default: experiments/data
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
except ImportError:
    sys.exit("Install dependencies first:  pip install pandas numpy")


def load(path_stem: Path) -> "pd.DataFrame":
    """Try parquet first, then CSV."""
    for suffix in (".parquet", ".csv"):
        p = path_stem.with_suffix(suffix)
        if p.exists():
            print(f"  Loading {p} ...")
            return pd.read_parquet(p) if suffix == ".parquet" else pd.read_csv(p)
    raise FileNotFoundError(f"Neither {path_stem}.parquet nor .csv found.")


def save(df: "pd.DataFrame", path_stem: Path) -> None:
    try:
        out = path_stem.with_suffix(".parquet")
        df.to_parquet(out, index=False)
    except Exception:
        out = path_stem.with_suffix(".csv")
        df.to_csv(out, index=False)
    print(f"  Saved → {out}  ({out.stat().st_size // 1024} KB)")


def compute_importance(domain: "pd.DataFrame",
                       baseline: "pd.DataFrame | None") -> "pd.DataFrame":
    df = domain.copy()
    if baseline is not None:
        merged = df.merge(
            baseline[["weight_name", "channel_idx", "mean_abs_mean"]],
            on=["weight_name", "channel_idx"],
            how="left",
            suffixes=("", "_base"),
        )
        merged["mean_abs_mean_base"] = merged["mean_abs_mean_base"].fillna(0)
        df["importance"] = (merged["mean_abs_mean"] - merged["mean_abs_mean_base"]).clip(lower=0)
    else:
        df["importance"] = df["mean_abs_mean"]
    return df


def build_keepmask(df: "pd.DataFrame", threshold_sigma: float,
                   top_k_fraction: float) -> dict:
    """
    Returns two dicts:
      per_row[weight_name]     = sorted list of channel indices to preserve
      per_block32[weight_name] = sorted list of block indices to preserve
    """
    per_row: dict[str, list[int]] = {}
    per_block32: dict[str, list[int]] = {}

    for weight_name, grp in df.groupby("weight_name"):
        importance = grp["importance"].values
        channel_idx = grp["channel_idx"].values

        if top_k_fraction > 0.0:
            k = max(1, int(len(importance) * top_k_fraction))
            thresh_val = np.partition(importance, -k)[-k]
        else:
            thresh_val = importance.mean() + threshold_sigma * importance.std()

        keep_channels = sorted(channel_idx[importance >= thresh_val].tolist())
        keep_blocks   = sorted(set(int(ch) // 32 for ch in keep_channels))

        per_row[weight_name]     = keep_channels
        per_block32[weight_name] = keep_blocks

        keep_frac = len(keep_channels) / max(1, len(importance))
        print(f"  {weight_name:50s}  keep {len(keep_channels):5d}/{len(importance):5d} "
              f"channels ({keep_frac:.1%})  |  {len(keep_blocks):4d} blocks")

    return {"per_row": per_row, "per_block_32": per_block32}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threshold-sigma", type=float, default=2.0,
                        help="Sigma multiplier for threshold (default: 2.0)")
    parser.add_argument("--top-k-fraction", type=float, default=0.0,
                        help="Keep top-K fraction instead of sigma threshold. "
                             "0.0 = use sigma threshold.")
    parser.add_argument("--data-dir", default="experiments/data")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    print("Loading importance data ...")
    domain_df   = load(data_dir / "importance_raw_domain")
    baseline_df = None
    try:
        baseline_df = load(data_dir / "importance_raw_baseline")
        print("  Baseline data found — computing domain-specific importance.")
    except FileNotFoundError:
        print("  No baseline data — using raw domain activation magnitudes.")

    print("\nComputing importance scores ...")
    scored = compute_importance(domain_df, baseline_df)
    save(scored, data_dir / "importance_scores")

    print(f"\nBuilding keepmasks  (sigma={args.threshold_sigma}, "
          f"top_k={args.top_k_fraction if args.top_k_fraction > 0 else 'off'}) ...")
    masks = build_keepmask(scored, args.threshold_sigma, args.top_k_fraction)

    # Summary statistics
    total_ch, kept_ch = 0, 0
    total_b32, kept_b32 = 0, 0
    for wname in masks["per_row"]:
        grp = scored[scored["weight_name"] == wname]
        n_ch = len(grp)
        n_b32 = (n_ch + 31) // 32
        total_ch  += n_ch;  kept_ch  += len(masks["per_row"][wname])
        total_b32 += n_b32; kept_b32 += len(masks["per_block_32"][wname])

    print(f"\nSummary:")
    print(f"  per-row  : {kept_ch:,} / {total_ch:,} channels kept  "
          f"({kept_ch/max(1,total_ch):.1%})")
    print(f"  per-block: {kept_b32:,} / {total_b32:,} blocks kept  "
          f"({kept_b32/max(1,total_b32):.1%})")

    out_row   = data_dir / "keepmask_per_row.json"
    out_block = data_dir / "keepmask_per_block32.json"
    with open(out_row,   "w") as f: json.dump(masks["per_row"],      f, indent=2)
    with open(out_block, "w") as f: json.dump(masks["per_block_32"], f, indent=2)
    print(f"\n  Saved → {out_row}")
    print(f"  Saved → {out_block}")
    print("\nDone.  Next step: python experiments/03_quant_strategy.py")


if __name__ == "__main__":
    main()
