#!/usr/bin/env bash
# Modern BERTc v4-Large 全流程:语料 → 预训练 → MT / CSC 微调。
#
#   bash prepare/run_v4_large.sh data        # 下载语料 + 加工
#   bash prepare/run_v4_large.sh pretokenize # 语料 → chunk
#   bash prepare/run_v4_large.sh pretrain    # 预训练(3-5 天,单卡 4090)
#   bash prepare/run_v4_large.sh finetune    # MT + CSC 微调
#   bash prepare/run_v4_large.sh all
#
# 架构 24L/1024H/2752I/16h ≈ 316M,即 HF 上的 Ismantic/BERTc-315M。
# 单卡,没有 DDP。eff_batch = 16 × 256 = 4096,17.65B token / (4096×512) ≈ 8500 步。
set -euo pipefail

PY=/home/tfbao/.venv/bin/python
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CORPUS=${CORPUS:-$ROOT/prepare/corpus/v4.pt}
BACKBONE=${BACKBONE:-$ROOT/prepare/output/v4_large}
DATASETS=$ROOT/prepare/datasets
TARGET_TOKENS=${TARGET_TOKENS:-20000000000}

log() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

step_data() {
    log "下载语料"
    $PY data/download.py --pretrain
    $PY data/download.py --finetune
    log "加工"
    $PY data/process.py --all
    $PY data/process_cws.py
    $PY data/process_csc.py --verify
    log "预编码下游数据集"
    $PY -m prepare.build_mt
    $PY -m prepare.build_csc
}

step_pretokenize() {
    log "语料 → chunk(目标 ${TARGET_TOKENS} token)"
    $PY -m prepare.pretokenize --output "$CORPUS" \
        --target_tokens "$TARGET_TOKENS" --num_workers 14
}

step_pretrain() {
    log "预训练 v4-Large"
    # recipe 与 v4-Large 实跑一致:固定 15% MLM、关 EMA、grad clip 0.5、
    # warmup 6%(Cramming 推荐),LR 8e-4 → 8e-5 单周期余弦
    $PY -m src.pretrain \
        --train_data "$CORPUS" \
        --output_dir "$BACKBONE" \
        --vocab_size 12536 --pad_token_id 12531 --mask_token_id 12535 \
        --hidden_size 1024 --num_layers 24 --num_heads 16 \
        --intermediate_size 2752 --max_position 1024 \
        --max_seq_length 512 --batch_size 16 --gradient_accumulation_steps 256 \
        --max_steps 8500 --warmup_steps 510 \
        --lr 8e-4 --min_lr 8e-5 --weight_decay 0.01 \
        --beta1 0.9 --beta2 0.95 --eps 1e-6 --max_grad_norm 0.5 \
        --mlm_low 0.15 --mlm_high 0.15 --wwm --no_ema \
        --save_steps 1500 --logging_steps 10 \
        2>&1 | tee -a "$BACKBONE/train.log"
}

step_finetune() {
    local ckpt="${CKPT:-$BACKBONE/checkpoint-8500}"
    [[ -d "$ckpt" ]] || { echo "没有 $ckpt,先跑 pretrain"; exit 1; }

    log "MT 微调(CWS + POS + NER)"
    $PY -m src.finetune_mt \
        --ckpt_dir "$ckpt" \
        --train_data "$DATASETS/mt_train.pt" --dev_data "$DATASETS/mt_dev.pt" \
        --output_dir "$ROOT/prepare/output/mt_v4_large" \
        --epochs 5 --batch_size 64 --bert_lr 2e-5 --head_lr 5e-4 \
        --alpha_pos 2.0 --beta_ner 0.5 --fgm --fgm_eps 1.0

    log "CSC 微调"
    $PY -m src.finetune_csc \
        --ckpt_dir "$ckpt" \
        --train_data "$DATASETS/csc_train.pt" --test_data "$DATASETS/csc_test.pt" \
        --output_dir "$ROOT/prepare/output/csc_v4_large" \
        --epochs 10 --batch_size 32 --lr 3e-5 --warmup_ratio 0.1 \
        --det_weight 0.3 --threshold 0.7
}

case "${1:-all}" in
    data)         step_data ;;
    pretokenize)  step_pretokenize ;;
    pretrain)     step_pretrain ;;
    finetune)     step_finetune ;;
    all)          step_data; step_pretokenize; step_pretrain; step_finetune ;;
    *)            echo "用法: $0 [all|data|pretokenize|pretrain|finetune]"; exit 1 ;;
esac
