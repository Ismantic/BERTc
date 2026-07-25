#!/usr/bin/env bash
# Modern BERTc 全流程:语料 → 预训练 → MT / CSC 微调。
#
#   bash prepare/run.sh data                # 下载语料 + 加工 + 预编码下游数据集
#   bash prepare/run.sh pretokenize         # 语料 → 定长 chunk
#   bash prepare/run.sh pretrain            # 预训练(单张 4090 约 3-5 天)
#   bash prepare/run.sh finetune            # MT + CSC 微调
#   bash prepare/run.sh all
#
# 规格用 SIZE 选,默认 large:
#   SIZE=large bash prepare/run.sh pretrain    24L/1024H ≈ 315M(Ismantic/BERTc-315M)
#   SIZE=mid   bash prepare/run.sh pretrain    12L/1024H ≈ 165M(Ismantic/BERTc-165M)
#
# 微调可以跳过预训练,直接用 HF 上的骨干:
#   huggingface-cli download Ismantic/BERTc-315M --local-dir models/BERTc-315M
#   CKPT=models/BERTc-315M bash prepare/run.sh finetune
#
# 两个规格的预训练配方**完全相同**,只差层数和 batch 切分方式
# (有效 batch 都是 4096:large 16×256,mid 32×128)。
# CSC 微调配方不同:mid 用 b64 lr5e-5 5ep,large 用 b32 lr3e-5 10ep ——
# large 5 epoch 严重欠训,这是调 315M 时最大的发现。
set -euo pipefail

PY=${BERTC_PYTHON:-/home/tfbao/.venv/bin/python}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SIZE=${SIZE:-large}
case "$SIZE" in
    large) LAYERS=24; BATCH=16; ACCUM=256; NAME=BERTc-315M
           CSC_EPOCHS=10; CSC_BATCH=32; CSC_LR=3e-5 ;;
    mid)   LAYERS=12; BATCH=32; ACCUM=128; NAME=BERTc-165M
           CSC_EPOCHS=5;  CSC_BATCH=64; CSC_LR=5e-5 ;;
    *)     echo "SIZE 只能是 large 或 mid,收到 $SIZE"; exit 1 ;;
esac

CORPUS=${CORPUS:-$ROOT/prepare/corpus/v4.pt}
OUT=${OUT:-$ROOT/prepare/output}
BACKBONE=$OUT/$NAME
DATASETS=$ROOT/prepare/datasets
TARGET_TOKENS=${TARGET_TOKENS:-20000000000}

log() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

step_data() {
    log "下载"
    $PY data/download.py --pretrain
    $PY data/download.py --finetune
    log "加工"
    $PY data/process.py --all
    $PY data/process_cws.py
    $PY data/process_csc.py
    log "预编码下游数据集"
    $PY -m prepare.build_mt
    $PY -m prepare.build_csc
}

step_pretokenize() {
    log "语料 → chunk(目标 $TARGET_TOKENS token)"
    $PY -m prepare.pretokenize --output "$CORPUS" \
        --target_tokens "$TARGET_TOKENS" --num_workers 14
}

step_pretrain() {
    log "预训练 $NAME(${LAYERS}L/1024H,有效 batch $(( BATCH * ACCUM )))"
    mkdir -p "$BACKBONE"
    # 固定 15% MLM、关 EMA、grad clip 0.5、warmup 6%(Cramming 推荐)、
    # LR 8e-4 → 8e-5 单周期余弦。17.65B / (4096×512) ≈ 8421,取 8500 步。
    $PY -m src.pretrain \
        --train_data "$CORPUS" \
        --output_dir "$BACKBONE" \
        --vocab_size 12536 --pad_token_id 12531 --mask_token_id 12535 \
        --hidden_size 1024 --num_layers "$LAYERS" --num_heads 16 \
        --intermediate_size 2752 --max_position 1024 \
        --max_seq_length 512 --batch_size "$BATCH" \
        --gradient_accumulation_steps "$ACCUM" \
        --max_steps 8500 --warmup_steps 510 \
        --lr 8e-4 --min_lr 8e-5 --weight_decay 0.01 \
        --beta1 0.9 --beta2 0.95 --eps 1e-6 --max_grad_norm 0.5 \
        --accum_warmup_frac 0.05 --accum_min 1 \
        --mlm_low 0.15 --mlm_high 0.15 --mlm_warmup_frac 0.05 \
        --wwm --no_ema --damp_gamma 0.0 --n_cycles 1 \
        --save_steps 1500 --logging_steps 10 \
        2>&1 | tee -a "$BACKBONE/train.log"
}

step_finetune() {
    local ckpt="${CKPT:-$BACKBONE/checkpoint-8500}"
    [[ -d "$ckpt" ]] || {
        echo "没有 $ckpt。先跑 pretrain,或者用 HF 上的骨干:"
        echo "  huggingface-cli download Ismantic/$NAME --local-dir models/$NAME"
        echo "  CKPT=models/$NAME bash prepare/run.sh finetune"
        exit 1
    }

    log "MT 微调($NAME:分词 + 词性 + 实体)"
    $PY -m src.finetune_mt \
        --ckpt_dir "$ckpt" \
        --train_data "$DATASETS/mt_train.pt" --dev_data "$DATASETS/mt_dev.pt" \
        --output_dir "$OUT/mt_$SIZE" \
        --epochs 5 --batch_size 64 --bert_lr 2e-5 --head_lr 5e-4 \
        --alpha_pos 2.0 --beta_ner 0.5 --fgm --fgm_eps 1.0 \
        --dev_limit 2000

    log "CSC 微调($NAME:${CSC_EPOCHS} epoch,batch $CSC_BATCH,lr $CSC_LR)"
    $PY -m src.finetune_csc \
        --ckpt_dir "$ckpt" \
        --train_data "$DATASETS/csc_train.pt" --test_data "$DATASETS/csc_test.pt" \
        --output_dir "$OUT/csc_$SIZE" \
        --epochs "$CSC_EPOCHS" --batch_size "$CSC_BATCH" --lr "$CSC_LR" \
        --warmup_ratio 0.1 --det_weight 0.3 --threshold 0.7
}

case "${1:-all}" in
    data)         step_data ;;
    pretokenize)  step_pretokenize ;;
    pretrain)     step_pretrain ;;
    finetune)     step_finetune ;;
    all)          step_data; step_pretokenize; step_pretrain; step_finetune ;;
    *)            echo "用法: [SIZE=large|mid] $0 [all|data|pretokenize|pretrain|finetune]"
                  exit 1 ;;
esac
