#!/usr/bin/env bash
# Modern BERTc v4 — Mid(~165M),对齐 v6.5/v7 mid 参数量做对照实验
#
# 目的: 同 params (165M, 12L/1024H/16h) 下,Modern recipe vs v6.5 BERT-classic
#       recipe 的真实差距。Cramming 对齐方向砍掉 v3 自加的 2 项。
#
# 架构: 12L / 1024H / 2752I / 16h ≈ 165M params
#       L=12, H=1024, heads=16 跟 v6.5/v7 mid 全对齐
#       I=2752 = GeGLU 等效 BERT 8H/3(参数/FLOPs 跟 BERT 4096 等效)
#
# 数据: data3/train_v3.pt 全部 17.65B token(复用 v3 pretokenize)
#
# Recipe 跟 v3 比的改动:
#   砍 1: Dynamic MLM 15→30→15  →  固定 15%(mlm_high=mlm_low=0.15)
#   砍 2: EMA decay=0.999       →  无(--no_ema)
#   留:  Cross-doc 隔离(.seg 文件已 ready,flex_attention block_mask)
#        StableAdamW β2=0.95、Damped Cosine、Megatron init、GeGLU、
#        ScaledSinusoidal、简化 head、bias-free、Dropout 0、
#        grad_clip 0.5、batch_warmup、6% warmup
#
# GPU: 1× 4090,bf16,batch_size 32(跟 v3 同,模型比 v3 base 略大但可承受)

set -euo pipefail

ROOT=/home/tfbao/Shiyu/BERTc/pretrain/modern_bertc
DATA=$ROOT/data3/train_v3.pt
OUTPUT=$ROOT/output_v4_mid
PY=/home/tfbao/.venv/bin/python

[[ -f "$DATA" ]] || { echo "ERROR: $DATA 不存在"; exit 1; }
[[ -f "$DATA.wid" ]] || { echo "ERROR: $DATA.wid 缺失"; exit 1; }
[[ -f "$DATA.seg" ]] || { echo "ERROR: $DATA.seg 缺失"; exit 1; }

mkdir -p "$OUTPUT"

# eff_batch = 32 × 128 = 4096(跟 v3 一致)
# 17.65B / (4096 × 512) = ~8421 → 8500 step(用 17.4B / 17.65B)
# warmup 6%(Cramming 推荐)→ 510 steps
BATCH=32
ACCUM=128
MAX_STEPS=8500
WARMUP=510
SAVE_STEPS=1500  # 5 中间 ckpt(1.5k/3k/4.5k/6k/7.5k)+ final 8.5k
LOG_STEPS=10

echo "=== Modern BERTc v4-Mid 训练 $(date) ==="
echo "  data: $DATA (17.65B tokens 全用)"
echo "  output: $OUTPUT"
echo "  架构: 12L/1024H/2752I/16h ≈ 165M(对齐 v6.5 mid)"
echo "  batch: $BATCH × accum $ACCUM = eff $(( BATCH * ACCUM ))"
echo "  max_steps: $MAX_STEPS  warmup: $WARMUP  save_steps: $SAVE_STEPS"
echo "  recipe diff vs v3: 砍 Dynamic MLM + 砍 EMA(其余跟 v3 一致)"

cd $ROOT && $PY -u train_modern.py \
    --train_data "$DATA" \
    --output_dir "$OUTPUT" \
    --vocab_size 12536 \
    --pad_token_id 12531 \
    --mask_token_id 12535 \
    --hidden_size 1024 \
    --num_layers 12 \
    --num_heads 16 \
    --intermediate_size 2752 \
    --max_position 1024 \
    --pe_theta 10000.0 \
    --layer_norm_eps 1e-5 \
    --embed_dropout 0.0 \
    --mlp_dropout 0.0 \
    --attn_out_dropout 0.0 \
    --max_seq_length 512 \
    --batch_size $BATCH \
    --gradient_accumulation_steps $ACCUM \
    --max_steps $MAX_STEPS \
    --warmup_steps $WARMUP \
    --lr 8e-4 \
    --min_lr 8e-5 \
    --weight_decay 0.01 \
    --beta1 0.9 \
    --beta2 0.95 \
    --eps 1e-6 \
    --max_grad_norm 0.5 \
    --accum_warmup_frac 0.05 \
    --accum_min 1 \
    --mlm_low 0.15 \
    --mlm_high 0.15 \
    --mlm_warmup_frac 0.05 \
    --damp_gamma 0.0 \
    --n_cycles 1 \
    --no_ema \
    --save_steps $SAVE_STEPS \
    --logging_steps $LOG_STEPS \
    --wwm \
    --inline_eval_cmd "bash $ROOT/../inline_eval_modern_both.sh {ckpt}" \
    2>&1 | tee -a "$OUTPUT/train.log"
