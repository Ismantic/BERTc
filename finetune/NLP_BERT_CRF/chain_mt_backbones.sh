#!/usr/bin/env bash
# Wait v6 MT done, then run v4 MT, then MacBERT MT, then RoBERTa MT
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
PY=/home/tfbao/.venv/bin/python

V6_MT_PID="${V6_MT_PID:-0}"
LOG=logs/chain_mt.log
mkdir -p logs
echo "=== Chain MT backbones start $(date) ===" | tee "$LOG"

# 1) wait v6 MT
if [ "$V6_MT_PID" != "0" ]; then
  echo "Waiting for v6 MT PID $V6_MT_PID..." | tee -a "$LOG"
  while ps -p "$V6_MT_PID" > /dev/null 2>&1; do sleep 30; done
  echo "v6 MT done at $(date)" | tee -a "$LOG"
fi

run_mt () {
  local TAG="$1"
  local MODEL_PATH="$2"
  local EPOCHS="$3"
  local BATCH="$4"
  local OUT="output_${TAG}_mt_crf"
  mkdir -p "$OUT"
  echo "=== START $TAG MT at $(date) ===" | tee -a "$LOG"
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

# 2) v4 MT (BERT-mid 165M)
run_mt "v4" "/home/tfbao/Shiyu/Summer/BERT/bert_train_v4_mid" 3 32

# 3) MacBERT MT (326M, 较慢)
run_mt "macbert" "./macbert-large" 3 16

# 4) RoBERTa-wwm-ext MT (base 102M)
run_mt "roberta" "./roberta-wwm-ext" 3 32

echo "=== ALL DONE $(date) ===" | tee -a "$LOG"
