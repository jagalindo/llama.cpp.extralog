# Activation-guided quantization experiments

Goal: identify which weight channels are most activated for a specific domain,
then keep those channels at higher precision during quantization.

## Quick start

```bash
# 1. Install Python deps
pip install -r experiments/requirements.txt

# 2. Build the instrumented binary
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-cli -j$(nproc)

# 3. Set paths
export MODEL=/path/to/model.gguf
export LLAMA_CLI=./build/bin/llama-cli

# 4. Edit your domain prompts
$EDITOR experiments/prompts/domain_prompts.txt

# 5. Collect data (runs all prompts, appends to CSVs in experiments/data/)
./experiments/00_run_prompts.sh both

# 6. Aggregate per-channel stats across all calls
python experiments/01_aggregate.py

# 7. Score each channel and build keepmasks
python experiments/02_importance_score.py

# 8. Estimate size impact of each strategy
python experiments/03_quant_strategy.py

# 9. Visualize
python experiments/04_analyze.py
```

## Output files

| File | Description |
|------|-------------|
| `data/{domain,baseline}/activations.csv` | One row per MUL_MAT call per tracked weight. Lightweight monitoring. |
| `data/{domain,baseline}/act_channels.csv` | One row per input channel per sampled call. The core importance data. |
| `data/importance_raw_{domain,baseline}.parquet` | Aggregated per-(weight, channel) stats. |
| `data/importance_scores.parquet` | Final importance score per channel (domain − baseline). |
| `data/keepmask_per_row.json` | Channels to preserve for per-row quantization (Q8_0). |
| `data/keepmask_per_block32.json` | Blocks to preserve for k-quant strategies. |
| `plots/` | All visualizations. |

## Quantization strategies being compared

### Strategy A: per-row  (target: Q8_0 vs Q4_K_M)
- Quantization granularity = one row = all `n_in` input channels of a weight row.
- Any row containing a salient channel → entire row kept at Q8_0 (8-bit).
- Simpler to implement; coarser; larger size overhead.

### Strategy B: per-block-32  (target: Q6_K vs Q4_K_M)
- k-quants (Q4_K_M, Q5_K_S, …) quantize in blocks of 32 consecutive channels.
- Any block containing a salient channel → that block kept at Q6_K (6-bit).
- Finer granularity; smaller overhead; requires modifying llama.cpp's quantize tool.

## Key parameters

| Parameter | Where | Default | Effect |
|-----------|-------|---------|--------|
| `BILLAUD_CHANNEL_SAMPLES` | `ggml.c` | 5 | Number of MUL_MAT calls logged per weight in act_channels.csv. Increase for more coverage, decrease to save disk. |
| `--threshold-sigma` | `02_importance_score.py` | 2.0 | Channels above mean+N×std are kept. Lower = keep more, higher = keep fewer. |
| `--top-k-fraction` | `02_importance_score.py` | off | Alternative: keep top K% of channels by importance. |

## Expected data volumes (7B model, 10 × 128-token prompts)

| File | Rows | Size |
|------|------|------|
| `activations.csv` | ~300K | ~20 MB |
| `act_channels.csv` | ~20M | ~1.2 GB |

Reduce `BILLAUD_CHANNEL_SAMPLES` (e.g. to 2) to cut `act_channels.csv` by 60%.
