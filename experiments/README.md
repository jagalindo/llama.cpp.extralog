# Activation-guided mixed-precision quantization — experiment pipeline

## Table of contents

1. [Scientific objective](#1-scientific-objective)
2. [How it works — the theory](#2-how-it-works--the-theory)
3. [What is instrumented in the C code](#3-what-is-instrumented-in-the-c-code)
4. [Prerequisites](#4-prerequisites)
5. [Compiling the instrumented llama.cpp](#5-compiling-the-instrumented-llamacpp)
6. [Getting a model](#6-getting-a-model)
7. [Preparing your prompts](#7-preparing-your-prompts)
8. [Running the experiments — step by step](#8-running-the-experiments--step-by-step)
9. [Understanding the output files](#9-understanding-the-output-files)
10. [Quantization strategies](#10-quantization-strategies)
11. [Tunable parameters](#11-tunable-parameters)
12. [Expected data volumes](#12-expected-data-volumes)
13. [Interpreting the plots](#13-interpreting-the-plots)
14. [What comes next](#14-what-comes-next)
15. [Troubleshooting](#15-troubleshooting)

---

## 1. Scientific objective

Standard post-training quantization (PTQ) treats all weights equally: every
matrix is reduced from 16-bit floating point to 4-bit integers using the same
method, regardless of how important each weight is for the task you care about.

**The hypothesis being tested here:**

> For a specific domain (e.g. medical, legal, code), only a subset of weight
> channels is heavily activated during inference on domain-relevant inputs.
> Keeping those channels at higher precision while aggressively quantizing the
> rest should preserve domain-specific performance better than uniform
> quantization at the same average bit-width.

This approach is inspired by:

- **AWQ** (Activation-aware Weight Quantization, Lin et al. 2023): scales
  salient channels before quantization using per-channel activation statistics.
- **LLM.int8()** (Dettmers et al. 2022): uses mixed-precision matmul, keeping
  ~0.1% of channels that carry activation outliers in FP16.
- **GPTQ** (Frantar et al. 2022): uses second-order Hessian information to
  find quantization-resistant parameters.

The key difference of our approach: the importance criterion is **domain
specific**. We compare activations produced by domain prompts against a general
baseline, isolating channels that are uniquely important for your target domain.

---

## 2. How it works — the theory

### Why activations, not weights?

The output of a linear layer is `y = W · x`. The quantization error in `W`
propagates proportionally to the magnitude of `x`. A weight column with a large
quantization error but a small activation `x` contributes little to the output
error. Conversely, a small weight column with a large activation `x` is critical.

The importance of input channel `j` for layer `W` is therefore:

```
importance(j) = E[ |x_j| ]   over all domain inputs
```

where the expectation is taken across all tokens generated from your domain
prompts.

### Domain-specific importance

To find channels that matter *specifically for your domain* (not just channels
that are always active), we subtract the baseline importance:

```
domain_importance(j) = E_domain[ |x_j| ] − E_baseline[ |x_j| ]
```

Channels with high `domain_importance` are the ones your domain uniquely relies
on. These should be preserved at higher precision.

### Two quantization granularities

The experiment measures importance at two granularities, because different
quantization formats work at different block sizes:

| Granularity | Block size | Target format | Trade-off |
|-------------|-----------|---------------|-----------|
| **per-row** | 1 channel | Q8_0 for salient rows | Coarser; easy to implement; larger size overhead |
| **per-block-32** | 32 channels | Q6_K for salient blocks | Finer; matches k-quant internals; smaller overhead |

---

## 3. What is instrumented in the C code

Three modifications were made to this fork of llama.cpp:

### `ggml/src/ggml.c` — the logging functions

**`BILLAUD_log_mulmat_activations(tensor)`**
Called before every `MUL_MAT` execution. It reads `src[1]` (the F32 input
activation tensor) and computes per-input-channel statistics:

- `mean_abs` — mean of |x| across all tokens in this call (the importance signal)
- `max_abs` — peak activation for this channel
- `std_abs` — standard deviation (signals how stable the channel's activation is)

It writes to two files:
- `activations.csv` — one aggregate row per MUL_MAT call (always written,
  lightweight).
- `act_channels.csv` — one row per input channel per call, written only for
  the first `BILLAUD_CHANNEL_SAMPLES` (default: 5) calls per weight tensor.

**`BILLAUD_weight_repartition(tensor)`**
Logs statistics about the *weight values* themselves (distributions, histograms)
to `weights.csv`. This is separate from activation logging and was the original
instrumentation.

### `ggml/src/ggml-cpu/ggml-cpu.c` — the hook point

`BILLAUD_log_mulmat_activations` is called at the start of `GGML_OP_MUL_MAT`,
guarded by `params->ith == 0` so that only thread 0 logs — avoiding duplicate
rows from the thread pool.

### Tracked weight names

Only the following weight patterns are logged (the main transformer weights):

```
attn_norm.weight    attn_qkv.weight    attn_output.weight
ffn_norm.weight     ffn_up.weight      ffn_down.weight
output_norm.weight  output.weight
```

These cover all major matmul operations in a standard transformer block.

---

## 4. Prerequisites

### System dependencies

```bash
# Debian / Ubuntu
sudo apt install cmake build-essential git python3 python3-pip

# macOS (Homebrew)
brew install cmake python3
```

CMake ≥ 3.14 and a C/C++ compiler with C11 and C++17 support are required.
GCC ≥ 9 or Clang ≥ 10 work fine.

### Python dependencies

```bash
pip install -r experiments/requirements.txt
```

The `requirements.txt` contains:

| Package | Purpose |
|---------|---------|
| `pandas` | CSV/parquet loading and aggregation |
| `numpy` | Channel statistics and threshold computation |
| `matplotlib` | All plots |
| `pyarrow` | Parquet support (optional but strongly recommended for large files) |

---

## 5. Compiling the instrumented llama.cpp

All commands run from the **repository root** (`llama.cpp.extralog/`).

### Standard CPU build

```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE=ON          # optimize for your CPU (recommended)

cmake --build build --target llama-cli -j$(nproc)
```

The binary will be at `build/bin/llama-cli`.

### Build with CUDA (if you have an NVIDIA GPU)

```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON

cmake --build build --target llama-cli -j$(nproc)
```

> **Note:** The activation logging runs entirely on the CPU regardless of
> whether GPU offload is used, because the F32 activation tensors are read
> before the GPU kernel executes.

### Build with Metal (Apple Silicon)

```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_METAL=ON

cmake --build build --target llama-cli -j$(nproc)
```

### Verify the build

```bash
./build/bin/llama-cli --version
```

---

## 6. Getting a model

The experiment works with any GGUF model. A good starting point for
experimentation is a 7B model (manageable size, fast inference):

```bash
mkdir -p models

# Example: download Llama-3.2-3B-Instruct in Q4_K_M
# (replace with your preferred model and source)
huggingface-cli download \
    bartowski/Llama-3.2-3B-Instruct-GGUF \
    --include "Llama-3.2-3B-Instruct-Q4_K_M.gguf" \
    --local-dir models/
```

Or use any `.gguf` file you already have. The experiment is format-agnostic —
it captures activations during inference regardless of the quantization format
of the model file.

---

## 7. Preparing your prompts

Two prompt files live in `experiments/prompts/`:

### `domain_prompts.txt`

Edit this with prompts representative of your **target domain**. These are the
inputs that will drive which channels are identified as important.

Guidelines:
- Use 20–100 diverse prompts for a stable importance estimate. Fewer than 10
  gives noisy results; more than 200 provides diminishing returns.
- Prompts should reflect the *kinds of questions* your model will answer in
  production, not necessarily the exact same prompts.
- Avoid very short prompts (< 20 tokens) — too few tokens means poor channel
  coverage.

```
# Good domain prompts for a medical LLM
Explain the mechanism of action of metformin in type 2 diabetes.
What are the diagnostic criteria for major depressive disorder per DSM-5?
Describe the pharmacokinetics of warfarin and its clinical implications.
```

### `baseline_prompts.txt`

General-purpose prompts covering topics *outside* your domain. These are used
to compute domain-specific importance:

```
importance_domain_specific(ch) = domain_importance(ch) − baseline_importance(ch)
```

Channels that are always active (across both domain and baseline) are not
domain-specific and can be quantized normally.

The existing `baseline_prompts.txt` contains general knowledge questions that
work well as a baseline for most domains.

---

## 8. Running the experiments — step by step

All commands assume you are in the **repository root**.

### Step 0 — set environment variables

```bash
export MODEL=./models/your-model.gguf
export LLAMA_CLI=./build/bin/llama-cli

# Optional overrides:
export N_PREDICT=128    # tokens generated per prompt (more = slower but better coverage)
export N_CTX=512        # context window size
export OUT_DIR=./experiments/data   # where CSVs are stored
```

### Step 1 — collect activation data

```bash
./experiments/00_run_prompts.sh both
```

This runs every prompt from both `domain_prompts.txt` and `baseline_prompts.txt`
through `llama-cli`, collecting four CSV files per run mode into
`experiments/data/domain/` and `experiments/data/baseline/`:

| File | Written by | Content |
|------|-----------|---------|
| `activations.csv` | C logging | Aggregate stats per MUL_MAT call |
| `act_channels.csv` | C logging | Per-channel stats (first 5 calls/weight) |
| `weights.csv` | C logging | Weight value distributions |
| `count.csv` | C logging | All tensor operations (debugging) |

You can also run only one mode:

```bash
./experiments/00_run_prompts.sh domain    # domain prompts only
./experiments/00_run_prompts.sh baseline  # baseline only
```

Expected duration: 2–10 minutes for 10 prompts on a CPU, depending on model
size and `N_PREDICT`.

### Step 2 — aggregate per-channel stats

```bash
python experiments/01_aggregate.py
```

Reads `act_channels.csv` for both domain and baseline. Groups all rows by
`(weight_name, channel_idx)` and computes:

- `mean_abs_mean` — mean importance across all logged calls
- `mean_abs_std` — stability of importance across calls
- `max_abs_max` — peak activation ever seen for this channel
- `call_count` — how many calls contributed data

Writes:
- `experiments/data/importance_raw_domain.parquet`
- `experiments/data/importance_raw_baseline.parquet`

### Step 3 — compute importance scores and build keepmasks

```bash
python experiments/02_importance_score.py
```

Computes domain-specific importance for each channel:

```
importance(ch) = mean_abs_mean_domain(ch) − mean_abs_mean_baseline(ch)
```

Then applies a threshold (`mean + 2σ` by default) to mark channels that must
be preserved at high precision.

Key options:

```bash
# Use a tighter threshold (keep fewer channels, smaller model, more quality loss)
python experiments/02_importance_score.py --threshold-sigma 3.0

# Use a looser threshold (keep more channels, safer)
python experiments/02_importance_score.py --threshold-sigma 1.5

# Keep the top 5% of channels by importance score
python experiments/02_importance_score.py --top-k-fraction 0.05
```

Writes:
- `experiments/data/importance_scores.parquet` — importance score per channel
- `experiments/data/keepmask_per_row.json` — for per-row quantization
- `experiments/data/keepmask_per_block32.json` — for k-quant block strategy

**keepmask format:**

```json
{
  "blk.0.attn_qkv.weight": [14, 231, 891, 1204, ...],
  "blk.1.attn_qkv.weight": [7, 44, 308, ...]
}
```

Each value is a channel index (per-row) or block index (per-block-32) to
preserve at higher precision.

### Step 4 — estimate model size impact

```bash
python experiments/03_quant_strategy.py
```

Reads the keepmasks and prints a table showing, for each weight tensor, how
many channels/blocks would be kept and the approximate size delta relative to
uniform quantization. This lets you compare the two strategies before actually
quantizing anything.

Example output:

```
Strategy: per-row (Q8_0 kept, Q4_K_M rest)
weight                                  total   kept   kept%   delta MB
blk.0.attn_qkv.weight                  4096     312   7.6%     +1.3 MB
blk.0.attn_output.weight               4096     198   4.8%     +0.8 MB
...
TOTAL                                          2841   6.1%    +24.7 MB

Strategy: per-block-32 (Q6_K kept, Q4_K_M rest)
...
TOTAL                                           89   5.4%     +9.2 MB
```

Options:

```bash
python experiments/03_quant_strategy.py --base-bits 4 --keep-bits 8
python experiments/03_quant_strategy.py --base-bits 4 --keep-bits 16
```

### Step 5 — visualize

```bash
python experiments/04_analyze.py
```

Saves four sets of plots to `experiments/plots/`:

1. **Outlier fraction over time** — which layers have salient activations and
   when.
2. **Channel importance histograms** — distribution of importance scores per
   weight, with threshold line.
3. **Per-block-32 heatmaps** — 2D grid of layer × block coloured by importance;
   shows spatial structure of domain-specific activations.
4. **Domain vs baseline scatter** — one point per channel, x = baseline
   importance, y = domain importance. Points above the diagonal are
   domain-specific.

Options:

```bash
python experiments/04_analyze.py --data-dir experiments/data --out-dir experiments/plots
```

---

## 9. Understanding the output files

### `activations.csv`

```
weight_name,             call_id, n_in, n_tokens, mean_abs, max_abs, std_abs, outlier_frac
blk.0.attn_qkv.weight,  0,       4096, 128,      0.0412,  0.8741,  0.0231,  0.0078
```

| Column | Meaning |
|--------|---------|
| `weight_name` | Name of the weight tensor, includes layer index (e.g. `blk.7.ffn_up.weight`) |
| `call_id` | Sequential index of this MUL_MAT call for this weight (resets per process) |
| `n_in` | Number of input channels (ne[0] of the weight) |
| `n_tokens` | Number of tokens in this batch |
| `mean_abs` | Mean of per-channel mean_abs — the "temperature" of this layer |
| `max_abs` | Peak activation seen across all channels and tokens |
| `std_abs` | Standard deviation of per-channel mean_abs — spread of channel importance |
| `outlier_frac` | Fraction of channels with mean_abs > mean + 3σ |

### `act_channels.csv`

```
weight_name,            call_id, channel_idx, mean_abs, max_abs, std_abs
blk.0.attn_qkv.weight, 0,       0,           0.0212,  0.4312,  0.0181
blk.0.attn_qkv.weight, 0,       1,           0.2891,  4.1200,  0.3412   ← outlier
```

One row per input channel per sampled call. This is the raw data that drives
all quantization decisions. `channel_idx` corresponds to the column index of
the weight matrix — the dimension that will be kept or quantized.

### `keepmask_per_row.json`

```json
{
  "blk.0.attn_qkv.weight":    [14, 231, 891, 2048, 3012],
  "blk.0.attn_output.weight": [7, 44, 308],
  ...
}
```

For each weight, the list of **input channel indices** to preserve at higher
precision. Channels not listed are quantized at the base precision.

### `keepmask_per_block32.json`

```json
{
  "blk.0.attn_qkv.weight":    [0, 7, 27, 64, 95],
  ...
}
```

Same concept but at block granularity. `block_idx = channel_idx // 32`. A
block is marked if *any* channel in it exceeded the importance threshold.

---

## 10. Quantization strategies

### Strategy A — per-row (simpler, coarser)

**Target formats:** salient rows at Q8_0 (8-bit), rest at Q4_K_M (4-bit).

In llama.cpp, `Q8_0` quantizes each row of the weight matrix independently
with a single scale factor. To preserve a channel at Q8_0, you keep the entire
row containing that channel at Q8_0 and quantize all other rows at Q4_K_M.

**Pros:**
- Conceptually simple: row granularity maps directly to llama.cpp's existing
  quantization primitives.
- No changes needed to dequantization kernels.

**Cons:**
- Coarse: even one important channel forces the whole row (128–1024 weights) to
  be kept at 8-bit.
- Larger size overhead than needed.

### Strategy B — per-block-32 (finer, matches k-quants)

**Target formats:** salient blocks at Q6_K, rest at Q4_K_M.

K-quant formats (Q4_K_M, Q5_K_S, etc.) quantize in "super-blocks" of 256
elements, internally divided into 8 sub-blocks of 32 elements each. Our
block-32 granularity aligns with these sub-blocks.

By keeping a 32-channel block at Q6_K instead of Q4_K_M, we add only 2 bits
per weight in that block, for a much smaller size penalty than Q8_0.

**Pros:**
- Finer granularity: only the specific 32-channel group containing important
  channels is affected.
- Smaller model size overhead.
- Natural fit with k-quant's internal structure.

**Cons:**
- Requires modifications to llama.cpp's `llama-quantize` tool to produce a
  mixed-format weight file.
- More complex implementation.

### Choosing a strategy for your first experiment

Start with **per-row** to validate the concept end-to-end, because it requires
no changes to the quantization tool. Once you confirm that domain-important
channels are correctly identified, move to per-block-32 for production use.

---

## 11. Tunable parameters

### C-level (compile-time, in `ggml/src/ggml.c`)

| Constant | Default | Effect |
|----------|---------|--------|
| `BILLAUD_CHANNEL_SAMPLES` | `5` | Number of MUL_MAT calls per weight that write to `act_channels.csv`. Increase for better coverage, decrease to save disk space. Recompile to change. |
| `BILLAUD_MAX_TRACKED` | `512` | Maximum number of distinct weight tensors tracked. Sufficient for any model up to 70B. |
| `BILLAUD_MAX_CHANNELS` | `16384` | Maximum input channels per weight. Covers 70B models. |

To change `BILLAUD_CHANNEL_SAMPLES`:

```c
// ggml/src/ggml.c, line ~7676
#define BILLAUD_CHANNEL_SAMPLES    10   // was 5
```

Then recompile:

```bash
cmake --build build --target llama-cli -j$(nproc)
```

### Python-level (runtime, passed as arguments)

| Script | Option | Default | Effect |
|--------|--------|---------|--------|
| `02_importance_score.py` | `--threshold-sigma` | `2.0` | Channels above `mean + N×std` are marked for preservation. Lower → more channels kept. |
| `02_importance_score.py` | `--top-k-fraction` | off | Alternative to sigma: keep the top K fraction of channels by importance score. |
| `03_quant_strategy.py` | `--base-bits` | `4` | Assumed bit-width of the base quantization format. |
| `03_quant_strategy.py` | `--keep-bits` | `8` | Bit-width used for preserved channels. |

### Shell-level (environment variables for `00_run_prompts.sh`)

| Variable | Default | Effect |
|----------|---------|--------|
| `MODEL` | `./models/model.gguf` | Path to the GGUF model |
| `LLAMA_CLI` | `./build/bin/llama-cli` | Path to the instrumented binary |
| `N_PREDICT` | `128` | Tokens generated per prompt. More tokens = better channel coverage, slower collection. |
| `N_CTX` | `512` | Context window size |
| `OUT_DIR` | `./experiments/data` | Root directory for all output CSVs |

---

## 12. Expected data volumes

Reference figures for a 7B model (32 layers, 4096 hidden dim), 10 prompts of
128 tokens each:

| File | Approx. rows | Approx. size |
|------|-------------|-------------|
| `activations.csv` | 300 K | 20 MB |
| `act_channels.csv` | 20 M | 1.2 GB |
| `weights.csv` | 2 K | 200 KB |
| `count.csv` | 800 K | 40 MB |
| `importance_raw_*.parquet` | 2 M | 60 MB |
| `importance_scores.parquet` | 2 M | 30 MB |

Scaling factors:
- `act_channels.csv` scales linearly with `BILLAUD_CHANNEL_SAMPLES` and number
  of prompts.
- For a 13B model (hidden dim 5120), multiply row counts by ~1.25×.
- For a 70B model (hidden dim 8192, 80 layers), expect ~8× the volume.

To reduce disk usage without losing coverage, lower `BILLAUD_CHANNEL_SAMPLES`
to 2 or 3. The importance signal stabilises quickly if prompts are diverse.

---

## 13. Interpreting the plots

### `01_outlier_over_time_{domain,baseline}.png`

Shows `outlier_frac` (y-axis) vs MUL_MAT call index (x-axis), one line per
weight type. High outlier fraction = that layer has channels with very large
activations relative to the mean.

**What to look for:** Lines that are consistently higher in the domain plot
than in the baseline plot identify layers that are disproportionately activated
by your domain. These are the most important layers to preserve.

### `02_channel_importance_hist.png`

Histogram of importance scores per weight tensor. The red vertical line is the
threshold used to select channels for preservation.

**What to look for:** A bimodal distribution (a large mass near zero and a small
tail of high values) is the ideal signal — it means a small fraction of
channels is genuinely important and the rest can be safely quantized. A flat
or unimodal distribution suggests your domain prompts are not sufficiently
distinctive from the baseline.

### `03_block_heatmap_{weight_type}.png`

2D heatmap: rows are transformer layers, columns are 32-channel blocks,
colour is mean importance. Brighter = more important for your domain.

**What to look for:** Bright columns that appear consistently across many
layers indicate input channels that are structurally important for your domain.
Bright rows that only appear in specific layers may indicate that those layers
specialise in domain-relevant computation.

### `04_domain_vs_baseline_scatter.png`

Scatter plot: x = baseline importance, y = domain importance, one point per
channel. Points on the diagonal are channels that are equally active for domain
and general inputs. Points above the diagonal are **domain-specific channels**.

**What to look for:** Points far above the diagonal with high y-values are your
most valuable channels to preserve. Points on the diagonal with high x-values
are always-active channels that do not need special treatment.

---

## 14. What comes next

This pipeline produces the keepmasks. The next engineering step — not yet
implemented — is to consume those masks during quantization.

### For per-row (Q8_0 strategy)

Modify `tools/quantize/quantize.cpp` in llama.cpp to:
1. Load `keepmask_per_row.json`.
2. For each weight tensor, check which rows contain a preserved channel.
3. Quantize those rows with `GGML_TYPE_Q8_0` instead of the target type.

The key function to modify is `llama_model_quantize_internal()` in
`src/llama.cpp`.

### For per-block-32 (k-quant strategy)

This requires changes at the kernel level:
1. Add a new mixed quantization type (or re-use existing types per block).
2. Modify the k-quant packing functions in `ggml/src/ggml-quants.c` to accept
   a per-block precision mask.
3. Update dequantization accordingly so inference remains correct.

### Validation workflow (once quantization is implemented)

```
1. Collect activation data on domain prompts        (00_run_prompts.sh)
2. Build keepmask                                   (01..02 scripts)
3. Quantize model with keepmask applied             (modified llama-quantize)
4. Benchmark perplexity on domain eval set          (llama-perplexity)
5. Compare vs. uniform Q4_K_M perplexity            (baseline)
6. Adjust threshold-sigma and repeat from step 2
```

---

## 15. Troubleshooting

### `activations.csv` is empty after running prompts

- Confirm the binary was rebuilt after the C changes: check that
  `BILLAUD_log_mulmat_activations` appears in the binary with
  `nm build/bin/llama-cli | grep BILLAUD`.
- The CSVs are written to the **current working directory** when `llama-cli`
  runs. `00_run_prompts.sh` handles this automatically by `cd`-ing to
  `experiments/data/{label}` before running the binary. If you run `llama-cli`
  manually, run it from your desired output directory.

### `act_channels.csv` is very small (few rows)

- Only the first `BILLAUD_CHANNEL_SAMPLES` (default: 5) MUL_MAT calls per
  weight write to `act_channels.csv`. If you have few prompts or short prompts,
  some weights may not reach 5 calls.
- Increase `N_PREDICT` or add more prompts.

### Python script complains about missing parquet file

- Run the scripts in order: `01 → 02 → 03 → 04`. Each depends on outputs of
  the previous one.
- If `pyarrow` is not installed, the scripts fall back to CSV. Ensure both
  `.parquet` and `.csv` variants are not mixed between runs.

### `01_aggregate.py` runs out of memory

- `act_channels.csv` can be several GB. The script uses chunked reading
  (500K rows at a time) to avoid loading it all at once.
- If it still OOMs, reduce `BILLAUD_CHANNEL_SAMPLES` and re-run the data
  collection step.

### Importance scores look the same for all channels

- This usually means the domain and baseline prompts are too similar. Make your
  domain prompts more specific and ensure the baseline prompts are genuinely
  general.
- Also check that the model is actually processing the prompts (confirm
  `activations.csv` has rows with non-zero `mean_abs`).

### Build errors after modifying `BILLAUD_CHANNEL_SAMPLES`

```bash
cmake --build build --target llama-cli -j$(nproc) --clean-first
```

A full rebuild ensures the constant change propagates correctly.
