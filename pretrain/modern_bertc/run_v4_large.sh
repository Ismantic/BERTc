#!/usr/bin/env bash
# Modern BERTc v4 — Large(~350M),用全部 17.65B 数据
#
# 架构: 28L / 1024H / 2624I / 16h(ModernBERT-large 对齐),~350M params
# 数据: data3/train_v3.pt 全部 17.65B token(复用 v3 pretokenize 结果)
# Recipe: 跟 v3 完全一致(StableAdamW + Damped Cosine + Dynamic MLM + EMA +
#                       cross-doc + ScaledSinusoidal + 简化 head + grad_clip 0.5)
# GPU: 1× 4090,bf16,batch_size 16(从 v3 的 32 减半防显存爆)

set -euo pipefail

ROOT=/home/tfbao/Shiyu/BERTc/pretrain/modern_bertc
DATA=$ROOT/data3/train_v3.pt
OUTPUT=$ROOT/output_v4_large
PY=/home/tfbao/.venv/bin/python

[[ -f "$DATA" ]] || { echo "ERROR: $DATA 不存在"; exit 1; }
[[ -f "$DATA.wid" ]] || { echo "ERROR: $DATA.wid 缺失"; exit 1; }
[[ -f "$DATA.seg" ]] || { echo "ERROR: $DATA.seg 缺失"; exit 1; }

mkdir -p "$OUTPUT"

# eff_batch = 16 × 256 = 4096(跟 v3 一致)
# 17.65B / (4096 × 512) = ~8421 → 8500 step(用 17.4B / 17.65B)
# warmup 6%(Cramming 推荐)→ 510 steps
BATCH=16
ACCUM=256
MAX_STEPS=8500
WARMUP=510
SAVE_STEPS=1500  # 5 中间 ckpt(1.5k/3k/4.5k/6k/7.5k)+ final 8.5k
LOG_STEPS=10

echo "=== Modern BERTc v4-Large 训练 $(date) ==="
echo "  data: $DATA (17.65B tokens 全用)"
echo "  output: $OUTPUT"
echo "  架构: 28L/1024H/2624I/16h ≈ 350M"
echo "  batch: $BATCH × accum $ACCUM = eff $(( BATCH * ACCUM ))"
echo "  max_steps: $MAX_STEPS  warmup: $WARMUP  save_steps: $SAVE_STEPS"

cd $ROOT && $PY -u train_modern.py \
    --train_data "$DATA" \
    --output_dir "$OUTPUT" \
    --vocab_size 12536 \
    --pad_token_id 12531 \
    --mask_token_id 12535 \
    --hidden_size 1024 \
    --num_layers 28 \
    --num_heads 16 \
    --intermediate_size 2624 \
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
    --mlm_high 0.30 \
    --mlm_warmup_frac 0.05 \
    --damp_gamma 0.0 \
    --n_cycles 1 \
    --save_steps $SAVE_STEPS \
    --logging_steps $LOG_STEPS \
    --wwm \
    --inline_eval_cmd "bash $ROOT/../inline_eval_modern_both.sh {ckpt}" \
    2>&1 | tee -a "$OUTPUT/train.log"
