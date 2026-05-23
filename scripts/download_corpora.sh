#!/usr/bin/env bash
# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
#
# Download the public training corpora used by RAIN's Phase-1 pretrain
# and Tier-2 LLM-parity benchmarks. All files land under data/corpora/
# (gitignored). Re-runnable; skips files already present.

set -euo pipefail

CORPUS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/corpora"
mkdir -p "$CORPUS_DIR"

fetch() {
    local url="$1"; local out="$2"
    if [[ -s "$CORPUS_DIR/$out" ]]; then
        echo "skip  $out (already present, $(wc -c <"$CORPUS_DIR/$out") bytes)"
        return
    fi
    echo "fetch $out from $url"
    curl -sS -L -o "$CORPUS_DIR/$out" "$url"
    echo "  ok  $out ($(wc -c <"$CORPUS_DIR/$out") bytes)"
}

# Tiny Shakespeare -- L1 benchmark corpus + Phase-1 smoke pretrain
fetch \
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt" \
    "tiny_shakespeare.txt"

# WikiText-2 raw (v1) -- L2 benchmark corpus + Phase-1 v0.5 pretrain
# Public mirror; the HF datasets URL is also fine but requires the datasets lib.
fetch \
    "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/train-00000-of-00001.parquet" \
    "wikitext2_train.parquet"

echo
echo "corpora ready under $CORPUS_DIR"
ls -la "$CORPUS_DIR"
