#!/usr/bin/env bash
# =============================================================================
# 00_run_prompts.sh — Collect activation data for a set of prompts.
#
# Usage:
#   ./experiments/00_run_prompts.sh [domain|baseline|both]
#
# Environment variables:
#   MODEL        path to the .gguf model file
#   LLAMA_CLI    path to the llama-cli binary
#   N_PREDICT    tokens to generate per prompt (default: 128)
#   N_CTX        context size (default: 512)
#   OUT_DIR      where to store the collected CSVs (default: experiments/data/)
#
# After this script, you should have in $OUT_DIR:
#   domain/activations.csv    act_channels.csv    weights.csv    count.csv
#   baseline/activations.csv  act_channels.csv    weights.csv    count.csv
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

MODEL="${MODEL:-${REPO_DIR}/models/model.gguf}"
LLAMA_CLI="${LLAMA_CLI:-${REPO_DIR}/build/bin/llama-cli}"
N_PREDICT="${N_PREDICT:-128}"
N_CTX="${N_CTX:-512}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/data}"
MODE="${1:-both}"

# Validate
if [[ ! -f "$MODEL" ]]; then
    echo "ERROR: Model not found at $MODEL"
    echo "Set the MODEL env variable, e.g.: MODEL=/path/to/model.gguf $0"
    exit 1
fi
if [[ ! -x "$LLAMA_CLI" ]]; then
    echo "ERROR: llama-cli not found or not executable at $LLAMA_CLI"
    echo "Build first: cmake --build build --target llama-cli"
    exit 1
fi

run_prompts() {
    local label="$1"
    local prompts_file="$2"
    local out_subdir="${OUT_DIR}/${label}"

    mkdir -p "$out_subdir"

    # CSVs are written to the CWD by the C code — run from the output directory.
    cd "$out_subdir"
    # Remove stale CSV files from a previous run so we start fresh.
    rm -f activations.csv act_channels.csv weights.csv count.csv

    local count=0
    while IFS= read -r prompt; do
        # Skip comments and blank lines.
        [[ "$prompt" =~ ^[[:space:]]*# ]] && continue
        [[ -z "${prompt// }" ]] && continue

        count=$((count + 1))
        printf "[%s prompt %d] %.70s...\n" "$label" "$count" "$prompt"

        "$LLAMA_CLI" \
            --model "$MODEL" \
            --prompt "$prompt" \
            --n-predict "$N_PREDICT" \
            --ctx-size "$N_CTX" \
            --no-display-prompt \
            --log-disable \
            2>/dev/null || true   # continue even if a single prompt fails

    done < "$prompts_file"

    cd "$REPO_DIR"
    echo "[$label] Finished $count prompts. Output in $out_subdir"
    if [[ -f "${out_subdir}/activations.csv" ]]; then
        local rows
        rows=$(wc -l < "${out_subdir}/activations.csv")
        echo "  activations.csv  : $rows rows"
    fi
    if [[ -f "${out_subdir}/act_channels.csv" ]]; then
        local rows
        rows=$(wc -l < "${out_subdir}/act_channels.csv")
        echo "  act_channels.csv : $rows rows"
    fi
}

echo "=== Activation data collection ==="
echo "Model     : $MODEL"
echo "llama-cli : $LLAMA_CLI"
echo "N_PREDICT : $N_PREDICT"
echo "N_CTX     : $N_CTX"
echo "Output    : $OUT_DIR"
echo ""

if [[ "$MODE" == "domain" || "$MODE" == "both" ]]; then
    run_prompts "domain"   "${SCRIPT_DIR}/prompts/domain_prompts.txt"
fi
if [[ "$MODE" == "baseline" || "$MODE" == "both" ]]; then
    run_prompts "baseline" "${SCRIPT_DIR}/prompts/baseline_prompts.txt"
fi

echo ""
echo "=== Done. Next step: python experiments/01_aggregate.py ==="
