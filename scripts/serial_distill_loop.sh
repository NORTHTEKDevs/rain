#!/usr/bin/env bash
# Serial Ollama distillation through 15 remaining domains.
# Each domain saves incrementally so kills mid-run don't lose work.
# Total wall: ~2.5 hours at 6s/example * 100 ex * 15 domains.

set -e

PER_DOMAIN=${1:-100}
DOMAINS=(code math regulated science history geography business technology medicine philosophy art music food travel sports)

echo "=== Serial distillation: ${#DOMAINS[@]} domains x $PER_DOMAIN each ==="
echo

for D in "${DOMAINS[@]}"; do
    OUT="data/distill/v1_${D}.jsonl"
    if [ -f "$OUT" ]; then
        N=$(wc -l < "$OUT")
        if [ "$N" -ge "$PER_DOMAIN" ]; then
            echo "[skip] $D already has $N examples"
            continue
        fi
    fi
    echo "[run]  $D -> $OUT"
    .venv/Scripts/python -u scripts/run_ollama_distill.py \
        --model llama3.2:3b \
        --n "$PER_DOMAIN" \
        --domain "$D" \
        --out "$OUT" \
        --report "data/distill/v1_${D}_report.md" \
        2>&1 | tail -3
    echo "[done] $D = $(wc -l < $OUT) examples"
    echo
done

echo
echo "=== ALL DOMAINS COMPLETE ==="
echo
echo "Final corpus:"
wc -l data/distill/v1_*.jsonl
