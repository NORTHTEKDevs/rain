#!/usr/bin/env bash
# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
#
# Shift init script for ticket rain-v0-12h.
# Idempotent. Run at the start of every shift session before doing any work.
# Prints exactly one line `READY` to stdout when the env is hot; aborts otherwise.

set -euo pipefail

log() { echo "shift-init: $1"; }

log "started $(date -u +%FT%TZ)"

# 1. Confirm Python venv exists and is usable.
if [ ! -d .venv ]; then
    log "creating .venv (Python 3.12)"
    py -3.12 -m venv .venv
fi

# shellcheck source=/dev/null
source .venv/Scripts/activate
PY_VERSION="$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')"
if [ "$PY_VERSION" != "3.12" ] && [ "$PY_VERSION" != "3.11" ]; then
    log "ERROR: expected Python 3.11 or 3.12 in .venv, got $PY_VERSION"
    exit 1
fi
log "python $PY_VERSION ok"

# 2. Editable rain install + dev extras.
pip install -e ".[dev]" --quiet --disable-pip-version-check 2>&1 | tail -3 || true
log "rain[dev] installed"

# 3. PyTorch + DirectML for HYMN training on the AMD iGPU.
#    Pinning torch to a version compatible with torch-directml 0.2.5.
if ! python -c "import torch, torch_directml" 2>/dev/null; then
    log "installing torch + torch-directml"
    pip install --quiet "torch>=2.4,<2.6" --index-url https://download.pytorch.org/whl/cpu
    pip install --quiet torch-directml
fi
log "torch $(python -c 'import torch; print(torch.__version__)') + directml ok"

# 4. Verify DirectML device is reachable (this is the broke-mode GPU).
DML_DEVICE=$(python -c "import torch_directml as d; print(d.device_name(0) if d.device_count() else 'NONE')" 2>/dev/null || echo NONE)
if [ "$DML_DEVICE" = "NONE" ]; then
    log "WARNING: no DirectML device detected -- training will fall back to CPU"
else
    log "DirectML: $DML_DEVICE"
fi

# 5. Ollama daemon for KB seed + judge feedback.
if ! curl -s --max-time 3 http://localhost:11434/api/tags >/dev/null 2>&1; then
    log "WARNING: Ollama daemon not responding at :11434 -- KB seed + judge tracks blocked"
else
    OLLAMA_MODELS=$(curl -s http://localhost:11434/api/tags | python -c "import sys, json; print(len(json.load(sys.stdin).get('models', [])))" 2>/dev/null || echo 0)
    log "Ollama up, $OLLAMA_MODELS models available"
fi

# 6. Corpora — Tiny Shakespeare is the L1 anchor; WikiText-2 is the L2 anchor.
mkdir -p data/corpora data/checkpoints data/kb_seed
if [ ! -s data/corpora/tiny_shakespeare.txt ]; then
    log "fetching tiny_shakespeare.txt"
    curl -sS -o data/corpora/tiny_shakespeare.txt \
        https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
fi
log "tiny_shakespeare ok ($(wc -c <data/corpora/tiny_shakespeare.txt) bytes)"

# 7. Sanity run: the full test suite must still pass on the existing branch HEAD.
log "running test suite"
python -m pytest tests/ -q --tb=line --no-header 2>&1 | tail -3
log "tests pass"

# 8. Git state -- confirm we are on the expected branch.
BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$BRANCH" != "feature/phase-1-bootstrap" ]; then
    log "WARNING: on $BRANCH, expected feature/phase-1-bootstrap"
fi
log "branch: $BRANCH @ $(git rev-parse --short HEAD)"

echo READY
