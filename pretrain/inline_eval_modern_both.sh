#!/usr/bin/env bash
# Modern BERTc inline_eval — 同时跑 CWS + CSC fine-tune,看 backbone 进展
# 用法: bash inline_eval_modern_both.sh <ckpt_dir>
set -uo pipefail

CKPT="${1:?usage: $0 <ckpt_path>}"

echo "============================================================"
echo "[modern_both_eval] $(date)  ckpt=$CKPT"
echo "============================================================"

# 1) CWS
bash /home/tfbao/Shiyu/BERTc/pretrain/inline_eval_modern_cws.sh "$CKPT"

# 2) CSC
echo ""
echo "------ CSC ------"
bash /home/tfbao/Shiyu/BERTc/pretrain/inline_eval_modern_csc.sh "$CKPT"

echo ""
echo "[modern_both_eval] both done"
