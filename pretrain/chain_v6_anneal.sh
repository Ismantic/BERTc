#!/usr/bin/env bash
# 等 super_mt_chain 完 + train_v6_anneal.pt ready → 启 v6 anneal CPT。
# CPT 完后 GPU 不闲:立刻起 v6.5 MT fine-tune。
set -euo pipefail
HERE=/home/tfbao/Shiyu/Summer/BERT
NLP=$HERE/NLP_BERT_CRF
PY=/home/tfbao/.venv/bin/python
LOG=/home/tfbao/Shiyu/Summer/BERT/logs_v6_anneal.log
mkdir -p $(dirname $LOG)

SUPER_PID="${SUPER_PID:-0}"
PRETOK_PID="${PRETOK_PID:-0}"

echo "=== chain_v6_anneal start $(date) ===" | tee $LOG

if [ "$SUPER_PID" != "0" ]; then
  echo "Waiting super_mt_chain PID $SUPER_PID..." | tee -a $LOG
  while ps -p "$SUPER_PID" > /dev/null 2>&1; do sleep 60; done
  echo "super_mt_chain done $(date)" | tee -a $LOG
fi

if [ "$PRETOK_PID" != "0" ]; then
  echo "Waiting pretokenize PID $PRETOK_PID..." | tee -a $LOG
  while ps -p "$PRETOK_PID" > /dev/null 2>&1; do sleep 30; done
  echo "pretokenize done $(date)" | tee -a $LOG
fi

# 确认 train_v6_anneal.pt 就绪
if [ ! -f /home/tfbao/Shiyu/data/train_v6_anneal.pt ]; then
  echo "ERROR: train_v6_anneal.pt 缺失" | tee -a $LOG
  exit 1
fi
if [ ! -f /home/tfbao/Shiyu/data/train_v6_anneal.pt.wid ]; then
  echo "ERROR: train_v6_anneal.pt.wid 缺失" | tee -a $LOG
  exit 1
fi
ls -lh /home/tfbao/Shiyu/data/train_v6_anneal.pt /home/tfbao/Shiyu/data/train_v6_anneal.pt.wid 2>&1 | tee -a $LOG

cd $HERE
echo "=== START v6 anneal CPT $(date) ===" | tee -a $LOG

# 3B token / (eff_bs 1024 × 512) = 5859 step
# 用 cosine + 较低 lr(anneal phase)
$PY -u train_bert_mlm.py \
    --model_path $HERE/bert_train_v6_mid \
    --train_data /home/tfbao/Shiyu/data/train_v6_anneal.pt \
    --output_dir $HERE/bert_train_v6_5_mid \
    --wwm \
    --word_ids_data /home/tfbao/Shiyu/data/train_v6_anneal.pt.wid \
    --max_seq_length 512 \
    --batch_size 128 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --compile \
    --max_steps 5859 \
    --warmup_steps 300 \
    --lr 2e-5 \
    --min_lr_ratio 0.1 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --mlm_prob 0.15 \
    --save_steps 1000 \
    --logging_steps 50 \
    2>&1 | tee -a $LOG

cp $HERE/bert_train_v6_mid/piece.model    $HERE/bert_train_v6_5_mid/piece.model
cp $HERE/bert_train_v6_mid/mask_token_id.txt $HERE/bert_train_v6_5_mid/mask_token_id.txt
cp $HERE/bert_train_v6_mid/dict.txt $HERE/bert_train_v6_5_mid/dict.txt 2>/dev/null || true

echo "=== v6 anneal CPT DONE $(date) ===" | tee -a $LOG

# 衔接 fine-tune,GPU 不闲
cd $NLP
OUT=$NLP/output_v65_mt_crf
rm -rf $OUT
mkdir -p $OUT
echo "=== START v6.5 MT fine-tune $(date) ===" | tee -a $LOG
$PY -u train_mt.py \
    --model_path $HERE/bert_train_v6_5_mid \
    --output_dir $OUT \
    --epochs 3 --batch_size 64 \
    --bert_lr 2e-5 --head_lr 5e-4 \
    --warmup_ratio 0.1 \
    --alpha_pos 2.0 --beta_ner 0.5 \
    --log_every 100 --eval_dev_limit 2000 \
    2>&1 | tee -a $LOG

echo "=== v6.5 MT fine-tune DONE $(date) ===" | tee -a $LOG
grep -E "Epoch|↑" $OUT/train.log | tail -10 | tee -a $LOG
