#!/usr/bin/env bash
set -euo pipefail
PY="${PY:-/home/tfbao/new/HY-MT/.venv/bin/python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/output_macbert_large_crf"
mkdir -p "$OUT"
cd "$HERE"

echo "=== MacBERT-large + CRF on PD-1998 ===" | tee "$OUT/train.log"
date | tee -a "$OUT/train.log"

$PY -u train.py \
    --model_path ./macbert-large \
    --train_jsonl ./data/cws.jsonl \
    --dev_jsonl ./data/cws_dev.jsonl \
    --output_dir "$OUT" \
    --epochs 3 \
    --batch_size 32 \
    --max_chars 254 \
    --bert_lr 2e-5 \
    --crf_lr 5e-4 \
    --warmup_ratio 0.1 \
    --log_every 50 \
    --eval_dev_limit 2000 \
    2>&1 | tee -a "$OUT/train.log"

echo "" | tee -a "$OUT/train.log"
echo "=== Full PD-06 dev eval (best.pt) ===" | tee -a "$OUT/train.log"
$PY -u eval.py --ckpt "$OUT/best.pt" --dev_jsonl ./data/cws_dev.jsonl 2>&1 | tee -a "$OUT/train.log"

echo "" | tee -a "$OUT/train.log"
echo "=== Full PD-06 dev eval (final.pt) ===" | tee -a "$OUT/train.log"
$PY -u eval.py --ckpt "$OUT/final.pt" --dev_jsonl ./data/cws_dev.jsonl 2>&1 | tee -a "$OUT/train.log"

echo "DONE $(date)" | tee -a "$OUT/train.log"
