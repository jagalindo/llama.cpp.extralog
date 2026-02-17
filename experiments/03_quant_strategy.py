#!/usr/bin/env python3
"""
03_quant_strategy.py — Estimate model size and accuracy impact for two strategies.

This is a planning tool, not a quantization tool.  It reads the keepmasks and
answers the practical question: "how much will the model grow if I preserve X
channels at high precision?"

Reads:  data/keepmask_per_row.json
        data/keepmask_per_block32.json
        data/importance_scores.{parquet,csv}

Outputs a table like:

  Strategy: per-row (Q8_0 for kept rows, Q4_K_M for the rest)
  ─────────────────────────────────────────────────────────────
  weight                  total_ch  kept_ch  kept%  size_delta_MB
  blk.0.attn_qkv.weight     4096      312    7.6%      +1.3 MB
  ...
  TOTAL                                               +24.7 MB

  Strategy: per-block-32 (Q6_K for kept blocks, Q4_K_M for the rest)
  ─────────────────────────────────────────────────────────────
  ...

Usage:
  python experiments/03_quant_strategy.py [options]

Options:
  --base-bits      N    bits for baseline quantization (default: 4)
  --keep-bits      N    bits for kept channels/blocks  (default: 8)
  --data-dir       DIR  default: experiments/data
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
except ImportError:
    sys.exit("pip install pandas numpy")


BITS_PER_FORMAT = {
    "F32": 32, "F16": 16, "BF16": 16,
    "Q8_0": 8,
    "Q6_K": 6,
    "Q5_K": 5, "Q5_K_S": 5, "Q5_K_M": 5,
    "Q4_K": 4, "Q4_K_S": 4, "Q4_K_M": 4,
    "Q3_K": 3, "Q3_K_S": 3, "Q3_K_M": 3, "Q3_K_L": 3,
    "Q2_K": 2,
}


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_scores(data_dir: Path) -> "pd.DataFrame":
    for suffix in (".parquet", ".csv"):
        p = (data_dir / "importance_scores").with_suffix(suffix)
        if p.exists():
            return pd.read_parquet(p) if suffix == ".parquet" else pd.read_csv(p)
    raise FileNotFoundError("importance_scores not found — run 02_importance_score.py first")


def analyse(label: str, keepmask: dict, scores: "pd.DataFrame",
            base_bits: int, keep_bits: int) -> None:
    print(f"\n{'='*70}")
    print(f"Strategy: {label}  "
          f"(baseline={base_bits}-bit, preserved={keep_bits}-bit)")
    print(f"{'='*70}")
    print(f"  {'weight':<50} {'total':>6} {'kept':>6} {'kept%':>6}  {'delta MB':>9}")
    print(f"  {'-'*50} {'-'*6} {'-'*6} {'-'*6}  {'-'*9}")

    total_kept = 0
    total_all  = 0
    total_delta_bytes = 0

    for wname, kept_indices in sorted(keepmask.items()):
        grp = scores[scores["weight_name"] == wname]
        if grp.empty:
            continue
        n_all  = len(grp)
        n_kept = len(kept_indices)

        # ne[1] is the output dimension — not in our data, estimate from name patterns.
        # We only know ne[0] (n_in = number of channels we sampled).
        # For size estimate assume ne[1] ≈ ne[0] (square-ish matrices) — rough.
        # User should supply actual shapes for precise estimates.
        delta_bits_per_channel = (keep_bits - base_bits)

        # Each kept "slot" (channel or block-of-32 channels) saves/costs delta bits.
        # We don't know the output dim — use 1 as unit; user gets relative numbers.
        delta_params = n_kept * delta_bits_per_channel

        delta_bytes = delta_params / 8  # approximate
        delta_mb    = delta_bytes / 1e6

        pct = n_kept / max(1, n_all)
        print(f"  {wname:<50} {n_all:>6,} {n_kept:>6,} {pct:>5.1%}  {delta_mb:>+9.2f}")

        total_kept        += n_kept
        total_all         += n_all
        total_delta_bytes += delta_bytes

    total_pct  = total_kept / max(1, total_all)
    total_delta_mb = total_delta_bytes / 1e6
    print(f"  {'TOTAL':<50} {total_all:>6,} {total_kept:>6,} {total_pct:>5.1%}  {total_delta_mb:>+9.2f}")
    print(f"\n  NOTE: delta MB is per-output-channel unit; multiply by actual")
    print(f"  output dimension (ne[1]) for exact model size impact.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-bits", type=int, default=4)
    parser.add_argument("--keep-bits", type=int, default=8)
    parser.add_argument("--data-dir", default="experiments/data")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    scores      = load_scores(data_dir)
    mask_row    = load_json(data_dir / "keepmask_per_row.json")
    mask_block  = load_json(data_dir / "keepmask_per_block32.json")

    analyse("per-row (Q8_0 kept, Q4_K_M rest)",
            mask_row, scores, args.base_bits, args.keep_bits)

    analyse("per-block-32 (Q6_K kept, Q4_K_M rest)",
            mask_block, scores, args.base_bits, 6)

    print(f"\nDone.  Next step: python experiments/04_analyze.py")


if __name__ == "__main__":
    main()
