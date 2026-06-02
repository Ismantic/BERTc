#!/usr/bin/env bash
# Modern BERTc inline_eval CSC
set -uo pipefail

CKPT="${1:?usage: $0 <ckpt_path>}"
PY=/home/tfbao/.venv/bin/python
SCRIPT=/home/tfbao/Shiyu/BERTc/pretrain/modern_bertc/eval_modern_csc.py
TRACK=/home/tfbao/Shiyu/BERTc/pretrain/inline_track_csc.tsv

STEP=$(echo "$CKPT" | grep -oE 'checkpoint-[0-9]+' | grep -oE '[0-9]+' || echo "NA")
echo "[modern_csc_eval] $(date)  ckpt=$CKPT  step=$STEP"

$PY -u $SCRIPT \
    --ckpt "$CKPT" \
    --track_tsv "$TRACK" \
    --max_train 50000 \
    --epochs 1 --batch_size 64 --lr 5e-5 || {
        echo "[modern_csc_eval] FAILED on $CKPT" | tee -a "$TRACK"
        exit 0
    }
echo "[modern_csc_eval] done"
