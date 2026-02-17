#!/usr/bin/env python3
"""
01_aggregate.py — Aggregate per-channel activation stats across all logged calls.

Reads:  data/{domain,baseline}/act_channels.csv
Writes: data/importance_raw.parquet  (or .csv if parquet unavailable)
        data/importance_raw_baseline.parquet

For each (weight_name, channel_idx) pair the output contains:
  - mean_abs_mean   : mean of per-call mean_abs across all logged calls
  - mean_abs_max    : max  of per-call mean_abs across all logged calls
  - mean_abs_std    : std  of per-call mean_abs (stability across calls)
  - max_abs_max     : global max activation seen for this channel
  - call_count      : number of logged calls contributing to this channel

Usage:
  python experiments/01_aggregate.py [--data-dir experiments/data]
"""

import argparse
import sys
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
except ImportError:
    sys.exit("Install dependencies first:  pip install pandas numpy")

# Optional: use parquet for faster I/O on large files.
try:
    import pyarrow  # noqa: F401
    USE_PARQUET = True
except ImportError:
    USE_PARQUET = False
    print("pyarrow not found — writing CSV instead of parquet (slower for large files).")


def save(df: "pd.DataFrame", path: Path) -> None:
    if USE_PARQUET:
        out = path.with_suffix(".parquet")
        df.to_parquet(out, index=False)
    else:
        out = path.with_suffix(".csv")
        df.to_csv(out, index=False)
    print(f"  Saved {len(df):,} rows → {out}  ({out.stat().st_size // 1024} KB)")


def aggregate_channels(csv_path: Path) -> "pd.DataFrame":
    print(f"Reading {csv_path} ...")
    # Stream-read in chunks to handle large files.
    chunks = []
    for chunk in pd.read_csv(csv_path, chunksize=500_000):
        agg = (
            chunk
            .groupby(["weight_name", "channel_idx"], sort=False)
            .agg(
                mean_abs_sum=("mean_abs", "sum"),
                mean_abs_sq_sum=("mean_abs", lambda x: (x**2).sum()),
                max_abs_max=("max_abs", "max"),
                call_count=("call_id", "count"),
            )
            .reset_index()
        )
        chunks.append(agg)

    combined = pd.concat(chunks, ignore_index=True)

    # Re-aggregate across chunks.
    final = (
        combined
        .groupby(["weight_name", "channel_idx"], sort=False)
        .agg(
            mean_abs_sum=("mean_abs_sum", "sum"),
            mean_abs_sq_sum=("mean_abs_sq_sum", "sum"),
            max_abs_max=("max_abs_max", "max"),
            call_count=("call_count", "sum"),
        )
        .reset_index()
    )

    n = final["call_count"]
    s = final["mean_abs_sum"]
    s2 = final["mean_abs_sq_sum"]

    final["mean_abs_mean"] = s / n
    final["mean_abs_std"]  = np.sqrt(np.maximum(0, s2 / n - (s / n) ** 2))
    final["max_abs_max"]   = final["max_abs_max"]

    return final[["weight_name", "channel_idx",
                  "mean_abs_mean", "mean_abs_std", "max_abs_max", "call_count"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="experiments/data",
                        help="Root directory produced by 00_run_prompts.sh")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    for label in ("domain", "baseline"):
        csv = data_dir / label / "act_channels.csv"
        if not csv.exists():
            print(f"WARNING: {csv} not found — skipping {label}.")
            continue
        df = aggregate_channels(csv)
        out = data_dir / f"importance_raw_{label}"
        save(df, out)

    print("\nDone.  Next step: python experiments/02_importance_score.py")


if __name__ == "__main__":
    main()
