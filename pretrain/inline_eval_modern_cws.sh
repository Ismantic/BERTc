#!/usr/bin/env bash
# Modern BERTc inline_eval — 用 CWS fine-tune 评估 backbone 进展
# 用法: bash inline_eval_modern_cws.sh <ckpt_dir>
set -uo pipefail

CKPT="${1:?usage: $0 <ckpt_path>}"
PY=/home/tfbao/.venv/bin/python
SCRIPT=/home/tfbao/Shiyu/BERTc/pretrain/modern_bertc/eval_modern_cws.py
TRACK=/home/tfbao/Shiyu/BERTc/pretrain/inline_track.tsv

STEP=$(echo "$CKPT" | grep -oE 'checkpoint-[0-9]+' | grep -oE '[0-9]+' || echo "NA")
echo "[modern_eval] $(date)  ckpt=$CKPT  step=$STEP"

# 限制单卡 fine-tune 跑得快(用 50K train + 2K dev,1ep,batch 64)
$PY -u $SCRIPT \
    --ckpt "$CKPT" \
    --train_jsonl /home/tfbao/Shiyu/Summer/BERT/NLP_BERT_CRF/data/cws.pd98.clean.jsonl \
    --dev_jsonl /home/tfbao/Shiyu/Summer/BERT/NLP_BERT_CRF/data/cws_dev.pd98.clean.jsonl \
    --track_tsv "$TRACK" \
    --max_train 50000 --max_dev 2000 \
    --epochs 1 --batch_size 64 --lr 5e-5 || {
        echo "[modern_eval] FAILED on $CKPT" | tee -a "$TRACK"
        exit 0
    }
echo "[modern_eval] done"
