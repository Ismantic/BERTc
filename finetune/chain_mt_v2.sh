#!/usr/bin/env bash
# 等 distill 完 → 串行跑 v6 / v4 / MacBERT / RoBERTa MT(bs=64 单 GPU 全力)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
PY=/home/tfbao/.venv/bin/python

DISTILL_PID="${DISTILL_PID:-0}"
LOG=logs/chain_mt_v2.log
mkdir -p logs
echo "=== Chain MT v2 start $(date) ===" | tee "$LOG"

if [ "$DISTILL_PID" != "0" ]; then
  echo "Waiting for distill PID $DISTILL_PID..." | tee -a "$LOG"
  while ps -p "$DISTILL_PID" > /dev/null 2>&1; do sleep 30; done
  echo "distill done at $(date)" | tee -a "$LOG"
  tail -3 logs/distill_mt.log | tee -a "$LOG"
fi

run_mt () {
  local TAG="$1"; local MODEL_PATH="$2"; local EPOCHS="$3"; local BATCH="$4"
  local OUT="output_${TAG}_mt_crf"
  rm -rf "$OUT"; mkdir -p "$OUT"
  echo "=== START $TAG MT bs=$BATCH at $(date) ===" | tee -a "$LOG"
  $PY -u train_mt.py \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUT" \
    --epochs "$EPOCHS" --batch_size "$BATCH" \
    --bert_lr 2e-5 --head_lr 5e-4 \
    --warmup_ratio 0.1 \
    --alpha_pos 0.5 --beta_ner 0.5 \
    --log_every 100 --eval_dev_limit 2000 \
    > "$OUT/train.log" 2>&1
  echo "=== DONE $TAG MT at $(date) ===" | tee -a "$LOG"
  grep -E "dev: cws_F1|Best score" "$OUT/train.log" | tail -10 | tee -a "$LOG"
}

run_mt "v6"      "/home/tfbao/Shiyu/Summer/BERT/bert_train_v6_mid" 3 64
run_mt "v4"      "/home/tfbao/Shiyu/Summer/BERT/bert_train_v4_mid" 3 64
run_mt "macbert" "./macbert-large"                                  3 24
run_mt "roberta" "./roberta-wwm-ext"                                3 64

echo "=== ALL DONE $(date) ===" | tee -a "$LOG"
