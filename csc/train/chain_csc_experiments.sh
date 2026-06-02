#!/usr/bin/env bash
# CSC 多 backbone / 多配置链式训练
# 用法: V2_PID=<v2 pid> bash chain_csc_experiments.sh
# (若 V2_PID 不传,直接从 v3 开始)
set -uo pipefail

PY=/home/tfbao/.venv/bin/python
CSC=/home/tfbao/Shiyu/BERTc/csc
TRAIN_PKL=$CSC/data/sighan_wang271k_pairs.pkl
LOG=$CSC/chain_experiments.log

echo "=== chain start $(date) ===" | tee $LOG

# 等 v2(若指定)
V2_PID="${V2_PID:-0}"
if [ "$V2_PID" != "0" ]; then
  echo "Waiting v2 PID $V2_PID..." | tee -a $LOG
  while ps -p "$V2_PID" > /dev/null 2>&1; do sleep 30; done
  echo "v2 done $(date)" | tee -a $LOG
fi

# ===== v3: RoBERTa-wwm-ext 10ep batch32 =====
OUT=$CSC/output_roberta_csc_v3
echo "=== START v3 RoBERTa-wwm-ext (10ep batch32) $(date) ===" | tee -a $LOG
mkdir -p $OUT
$PY -u $CSC/train/train_csc_hf.py \
    --backbone_path /home/tfbao/Shiyu/Summer/BERT/NLP_BERT_CRF/roberta-wwm-ext \
    --train_pkl $TRAIN_PKL \
    --output_dir $OUT \
    --epochs 10 --batch_size 32 --lr 5e-5 \
    --warmup_ratio 0.1 --max_len 128 \
    --det_weight 0.3 --threshold 0.7 \
    --log_every 400 \
    > $OUT/train.log 2>&1
echo "=== v3 RoBERTa DONE $(date) ===" | tee -a $LOG
grep -E "Epoch.*SIGHAN|Best|Total" $OUT/train.log | tail -5 | tee -a $LOG

# ===== v4: BERTc batch64 10ep(扩展 v1 best config)=====
OUT=$CSC/output_v7_csc_v4_batch64_10ep
echo "=== START v4 BERTc batch64 10ep $(date) ===" | tee -a $LOG
mkdir -p $OUT
$PY -u $CSC/train/train_csc.py \
    --backbone_path /home/tfbao/Shiyu/Summer/BERT/bert_train_v7_mid \
    --train_pkl $TRAIN_PKL \
    --output_dir $OUT \
    --epochs 10 --batch_size 64 --lr 5e-5 \
    --warmup_ratio 0.1 --max_len 128 \
    --det_weight 0.3 --threshold 0.7 \
    --log_every 200 \
    > $OUT/train.log 2>&1
echo "=== v4 BERTc batch64 10ep DONE $(date) ===" | tee -a $LOG
grep -E "Epoch.*SIGHAN|Best|Total" $OUT/train.log | tail -5 | tee -a $LOG

# ===== v5: BERTc det_weight=0.5 batch64 5ep =====
OUT=$CSC/output_v7_csc_v5_det05
echo "=== START v5 BERTc det_weight=0.5 batch64 5ep $(date) ===" | tee -a $LOG
mkdir -p $OUT
$PY -u $CSC/train/train_csc.py \
    --backbone_path /home/tfbao/Shiyu/Summer/BERT/bert_train_v7_mid \
    --train_pkl $TRAIN_PKL \
    --output_dir $OUT \
    --epochs 5 --batch_size 64 --lr 5e-5 \
    --warmup_ratio 0.1 --max_len 128 \
    --det_weight 0.5 --threshold 0.7 \
    --log_every 200 \
    > $OUT/train.log 2>&1
echo "=== v5 BERTc det0.5 DONE $(date) ===" | tee -a $LOG
grep -E "Epoch.*SIGHAN|Best|Total" $OUT/train.log | tail -5 | tee -a $LOG

# ===== v6: MacBERT-large 5ep batch16 =====
OUT=$CSC/output_macbert_large_csc_v6
echo "=== START v6 MacBERT-large 5ep batch16 $(date) ===" | tee -a $LOG
mkdir -p $OUT
$PY -u $CSC/train/train_csc_hf.py \
    --backbone_path /home/tfbao/Shiyu/Summer/BERT/NLP_BERT_CRF/macbert-large \
    --train_pkl $TRAIN_PKL \
    --output_dir $OUT \
    --epochs 5 --batch_size 16 --lr 3e-5 \
    --warmup_ratio 0.1 --max_len 128 \
    --det_weight 0.3 --threshold 0.7 \
    --log_every 400 \
    > $OUT/train.log 2>&1
echo "=== v6 MacBERT-large DONE $(date) ===" | tee -a $LOG
grep -E "Epoch.*SIGHAN|Best|Total" $OUT/train.log | tail -5 | tee -a $LOG

echo "=== ALL DONE $(date) ===" | tee -a $LOG
