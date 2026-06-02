#!/usr/bin/env bash
# 等 chain_mt_v2 完后串行跑 v6 MT 改进系列 M1-M4。
# 每个实验完后用 eval_mt_on_ckpt.py 测 full dev raw+clean,append 到 experiments.tsv。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
PY=/home/tfbao/.venv/bin/python
TSV=experiments.tsv
LOG=logs/super_mt_chain.log
mkdir -p logs
echo "=== super_mt_chain start $(date) ===" | tee "$LOG"

PREV_PID="${PREV_PID:-0}"
if [ "$PREV_PID" != "0" ]; then
  echo "Waiting for chain v2 PID $PREV_PID..." | tee -a "$LOG"
  while ps -p "$PREV_PID" > /dev/null 2>&1; do sleep 60; done
  echo "chain v2 done at $(date)" | tee -a "$LOG"
fi

V6=/home/tfbao/Shiyu/Summer/BERT/bert_train_v6_mid

eval_and_log () {
  local TAG="$1"; local OUT="$2"; local EXTRAS="$3"; local EPOCHS="$4"; local BS="$5"
  echo "=== EVAL $TAG raw + clean ===" | tee -a "$LOG"
  R=$($PY -u eval_mt_on_ckpt.py --ckpt "$OUT/best.pt" --model_path "$V6" \
      --cws_dev ./data/cws_dev.pd98.jsonl --pos_dev ./data/pos_dev.pd98.jsonl --ner_dev ./data/ner_dev.pd98.jsonl \
      --batch_size 32 2>/dev/null | grep -E "CWS|POS per-word|NER")
  echo "$R" | tee -a "$LOG"
  CWS_RAW=$(echo "$R" | grep "CWS"  | awk '{print $4}')
  POS_RAW=$(echo "$R" | grep "POS per-word" | awk '{print $4}')
  NER_RAW=$(echo "$R" | grep "NER"  | awk '{print $4}')
  RC=$($PY -u eval_mt_on_ckpt.py --ckpt "$OUT/best.pt" --model_path "$V6" \
      --cws_dev ./data/cws_dev.pd98.cleanjudge.jsonl --pos_dev ./data/pos_dev.pd98.jsonl --ner_dev ./data/ner_dev.pd98.jsonl \
      --batch_size 32 2>/dev/null | grep "CWS")
  echo "clean: $RC" | tee -a "$LOG"
  CWS_CLEAN=$(echo "$RC" | awk '{print $4}')
  printf "%s\tBERT-mid\tMT\tmulti\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tauto\n" \
    "$TAG" "$EPOCHS" "$BS" "$EXTRAS" "$CWS_RAW" "$CWS_CLEAN" "$POS_RAW" "$NER_RAW" >> "$TSV"
}

run_mt () {
  local TAG="$1"; local EPOCHS="$2"; local BS="$3"; shift 3
  local OUT="output_${TAG}_crf"
  rm -rf "$OUT"; mkdir -p "$OUT"
  echo "=== START $TAG at $(date) ===" | tee -a "$LOG"
  $PY -u train_mt.py \
    --model_path "$V6" \
    --output_dir "$OUT" \
    --epochs "$EPOCHS" --batch_size "$BS" \
    --bert_lr 2e-5 --head_lr 5e-4 \
    --warmup_ratio 0.1 \
    --log_every 100 --eval_dev_limit 2000 \
    "$@" \
    > "$OUT/train.log" 2>&1
  echo "=== DONE $TAG at $(date) ===" | tee -a "$LOG"
  grep -E "dev: cws_F1|Best" "$OUT/train.log" | tail -8 | tee -a "$LOG"
  eval_and_log "$TAG" "$OUT" "$*" "$EPOCHS" "$BS"
}

# M1: v6 MT + FGM
run_mt "v6_mt_fgm" 3 64 --alpha_pos 0.5 --beta_ner 0.5 --fgm --fgm_eps 1.0

# M2: v6 MT + alpha_pos=2(POS 权重 4x)
run_mt "v6_mt_a2" 3 64 --alpha_pos 2.0 --beta_ner 0.5

# M3: v6 MT + LTP distill 1M(弱监督混训,3ep)
run_mt "v6_mt_distill" 3 64 --alpha_pos 0.5 --beta_ner 0.5 \
  --distill_jsonl ./data/mt.ltp_distill_100w.jsonl

# M4: v6 MT + FGM + α=2 + distill(终极组合,5ep)
run_mt "v6_mt_full" 5 64 --alpha_pos 2.0 --beta_ner 0.5 \
  --fgm --fgm_eps 1.0 \
  --distill_jsonl ./data/mt.ltp_distill_100w.jsonl

echo "=== ALL DONE $(date) ===" | tee -a "$LOG"
tail -20 "$TSV" | tee -a "$LOG"
