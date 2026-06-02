#!/usr/bin/env bash
# Modern BERTc v3 — release-aligned + Chinese ModernBERT 经验 + cross-doc 隔离
#
# 架构: 22L/768H/1152I/12h (120.6M params)
# Norm: LayerNorm(no bias) + embed_norm + skip_first_prenorm + Megatron init
# Optim: StableAdamW(β1=0.9, β2=0.95, wd=0.01, eps=1e-6, filter_bias_norm_wd)
# LR: Damped cosine, peak 8e-4 → min 8e-5
# MLM: Dynamic curriculum 15%→30%→15%(蚂蚁论文 Section 3.2)
# Attention: flex_attention(torch.compiled)+ block-diag mask via seg_ids
# GPU: 1× 4090 24GB,bf16
#
# 数据: 14.6B token,seqlen 512,WWM + cross-doc 隔离

set -euo pipefail

ROOT=/home/tfbao/Shiyu/BERTc/pretrain/modern_bertc
DATA=$ROOT/data3/train_v3.pt
OUTPUT=$ROOT/output_v3
PY=/home/tfbao/.venv/bin/python

# 检查数据存在
[[ -f "$DATA" ]] || { echo "ERROR: $DATA 不存在,等 v3 pretokenize 完成"; exit 1; }
[[ -f "$DATA.wid" ]] || { echo "ERROR: $DATA.wid 缺失"; exit 1; }
[[ -f "$DATA.seg" ]] || { echo "ERROR: $DATA.seg 缺失"; exit 1; }

mkdir -p "$OUTPUT"

# Cramming-aligned: eff_batch 4096(跟 Cramming/ModernBERT/RoBERTa 同 order)
# eff = 32 × 128 = 4096
# 14.6B / (4096 × 512) = ~7000 optim steps
# warmup 6%(Cramming Section 4.3 推荐,大 batch 需要更多 warmup)→ 420 steps
BATCH=32
ACCUM=128
MAX_STEPS=7000
WARMUP=420
SAVE_STEPS=1500  # 5 个 ckpt(1.5k/3k/4.5k/6k + final)— inline_eval 总耗时 ≤2h
LOG_STEPS=10     # 大 batch 下 step 少,频繁 log 才看得清曲线

echo "=== Modern BERTc v3 训练 $(date) ==="
echo "  data: $DATA"
echo "  output: $OUTPUT"
echo "  batch: $BATCH × accum $ACCUM = eff $(( BATCH * ACCUM ))"
echo "  max_steps: $MAX_STEPS  warmup: $WARMUP  save_steps: $SAVE_STEPS"

cd $ROOT && $PY -u train_modern.py \
    --train_data "$DATA" \
    --output_dir "$OUTPUT" \
    --vocab_size 12536 \
    --pad_token_id 12531 \
    --mask_token_id 12535 \
    --hidden_size 768 \
    --num_layers 22 \
    --num_heads 12 \
    --intermediate_size 1152 \
    --max_position 1024 \
    --pe_theta 10000.0 \
    --layer_norm_eps 1e-5 \
    --embed_dropout 0.0 \
    --mlp_dropout 0.0 \
    --attn_out_dropout 0.1 \
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
    --accum_warmup_frac 1.0 \
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
