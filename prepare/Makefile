# BERTc 编排层:装依赖 → 预编码 → 预训练 → 微调
#
#   make deps                clone + 编译 PieceTokenizer / Wapic
#   make deps-wapic-data     只 clone Wapic 拿 PD-1998 语料(微调够用)
#   make datasets            下游数据集预编码(MT + CSC)
#   make corpus              预训练语料切块(约 4.6 小时,产出 180G)
#   make pretrain            预训练(单卡 4090:mid 约 2 天,large 约 4 天)
#   make finetune            MT + CSC 微调(约 6.6 小时)
#   make status              看每一步的产物在不在
#
# 规格用 SIZE 选,默认 mid:
#   make pretrain SIZE=large      24L/1024H ≈ 315M
#   make pretrain SIZE=mid        12L/1024H ≈ 165M
#
# 微调可以跳过预训练,直接用 HF 上的骨干:
#   huggingface-cli download Ismantic/BERTc-315M --local-dir models/BERTc-315M
#   make finetune CKPT=models/BERTc-315M
#
# corpus 和 datasets 是**文件目标** —— 产物在就不会重跑。要强制重来先删产物。

PY      ?= /home/tfbao/.venv/bin/python
HERE    := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
ROOT    := $(patsubst %/,%,$(dir $(HERE)))

SIZE    ?= mid
CORPUS  ?= $(HERE)/corpus/v4.pt
OUT     ?= $(HERE)/output
DATASETS := $(HERE)/datasets
TARGET_TOKENS ?= 20000000000
WORKERS ?= 14

# 两个已发布规格。预训练配方完全相同,只差层数和 batch 切分
# (有效 batch 都是 4096)。CSC 微调配方不同 —— large 5 epoch 严重欠训。
ifeq ($(SIZE),large)
  LAYERS := 24
  BATCH  := 16
  ACCUM  := 256
  NAME   := BERTc-315M
  CSC_EPOCHS := 10
  CSC_BATCH  := 32
  CSC_LR     := 3e-5
else ifeq ($(SIZE),mid)
  LAYERS := 12
  BATCH  := 32
  ACCUM  := 128
  NAME   := BERTc-165M
  CSC_EPOCHS := 5
  CSC_BATCH  := 64
  CSC_LR     := 5e-5
else
  $(error SIZE 只能是 large 或 mid,收到 $(SIZE))
endif

BACKBONE := $(OUT)/$(NAME)
CKPT     ?= $(BACKBONE)/checkpoint-8500

.PHONY: help status deps deps-wapic-data datasets corpus pretrain finetune \
        finetune-mt finetune-csc

help:
	@awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' $(HERE)/Makefile

status:
	@printf '规格      SIZE=%s → %s(%sL,batch %s×%s)\n' \
	    "$(SIZE)" "$(NAME)" "$(LAYERS)" "$(BATCH)" "$(ACCUM)"
	@printf '\n依赖\n'
	@$(PY) -c "import piece_tokenizer,os;print('  ✓ piece_tokenizer',os.path.dirname(piece_tokenizer.__file__))" 2>/dev/null || echo "  ✗ piece_tokenizer 未装 → make deps"
	@$(PY) -c "import wapic;print('  ✓ wapic')" 2>/dev/null || echo "  ✗ wapic 未装 → make deps"
	@printf '\n产物\n'
	@for f in "$(DATASETS)/mt_train.pt" "$(DATASETS)/mt_dev.pt" \
	          "$(DATASETS)/csc_train.pt" "$(DATASETS)/csc_test.pt" \
	          "$(CORPUS)" "$(CKPT)/config.json"; do \
	    if [ -e "$$f" ]; then printf '  ✓ %s\n' "$$f"; \
	    else printf '  ✗ %s\n' "$$f"; fi; done

# ---------------------------------------------------------------- 依赖

deps:
	bash $(HERE)/install_deps.sh

# 只 clone Wapic 拿 PD-1998 语料,不编译不下模型。给只做微调的人用。
deps-wapic-data:
	bash $(HERE)/install_deps.sh wapic-data

# ---------------------------------------------------------------- 下游数据集

datasets: $(DATASETS)/mt_train.pt $(DATASETS)/csc_train.pt

$(DATASETS)/mt_train.pt $(DATASETS)/mt_dev.pt:
	cd $(ROOT) && $(PY) -m prepare.build_mt

$(DATASETS)/csc_train.pt $(DATASETS)/csc_test.pt:
	cd $(ROOT) && $(PY) -m prepare.build_csc

# ---------------------------------------------------------------- 预训练语料

corpus: $(CORPUS)

# 约 4.6 小时、产出 180G(.pt 80G + .wid 80G + .seg 20G),所以做成文件目标 ——
# 产物在就不重跑
$(CORPUS):
	cd $(ROOT) && $(PY) -m prepare.pretokenize --output $(CORPUS) \
	    --target_tokens $(TARGET_TOKENS) --num_workers $(WORKERS)

# ---------------------------------------------------------------- 预训练

# recipe:StableAdamW + Damped Cosine(8e-4 → 8e-5)+ 固定 15% 整词掩码 +
# flex_attention 跨文档隔离 + grad clip 0.5 + warmup 6%(Cramming 推荐)。
# 20B / (4096×512) ≈ 9537,取 8500 步 ≈ 17.4B token。
pretrain: $(CORPUS)
	@mkdir -p $(BACKBONE)
	cd $(ROOT) && $(PY) -m src.pretrain \
	    --train_data $(CORPUS) \
	    --output_dir $(BACKBONE) \
	    --vocab_size 12536 --pad_token_id 12531 --mask_token_id 12535 \
	    --hidden_size 1024 --num_layers $(LAYERS) --num_heads 16 \
	    --intermediate_size 2752 --max_position 1024 \
	    --max_seq_length 512 --batch_size $(BATCH) \
	    --gradient_accumulation_steps $(ACCUM) \
	    --max_steps 8500 --warmup_steps 510 \
	    --lr 8e-4 --min_lr 8e-5 --weight_decay 0.01 \
	    --beta1 0.9 --beta2 0.95 --eps 1e-6 --max_grad_norm 0.5 \
	    --accum_warmup_frac 0.05 --accum_min 1 \
	    --mlm_low 0.15 --mlm_high 0.15 --mlm_warmup_frac 0.05 \
	    --wwm --no_ema --damp_gamma 0.0 --n_cycles 1 \
	    --save_steps 1500 --logging_steps 10 \
	    2>&1 | tee -a $(BACKBONE)/train.log

# ---------------------------------------------------------------- 微调

finetune: finetune-mt finetune-csc

check-ckpt:
	@[ -e "$(CKPT)/config.json" ] || { \
	    echo "没有 $(CKPT)。先 make pretrain,或者用 HF 上的骨干:"; \
	    echo "  huggingface-cli download Ismantic/$(NAME) --local-dir models/$(NAME)"; \
	    echo "  make finetune CKPT=models/$(NAME)"; exit 1; }

# 骨干 lr 2e-5 / 头 lr 5e-4:骨干已训好,大 lr 会冲坏;头是随机初始化的要大 lr。
# α_pos=2.0 是因为词性 loss 量级天然小,不加权只占 1.6% 梯度学不动。
# --dev_limit 2000 与报告指标的口径一致。
finetune-mt: check-ckpt $(DATASETS)/mt_train.pt
	cd $(ROOT) && $(PY) -m src.finetune_mt \
	    --ckpt_dir $(CKPT) \
	    --train_data $(DATASETS)/mt_train.pt --dev_data $(DATASETS)/mt_dev.pt \
	    --output_dir $(OUT)/mt_$(SIZE) \
	    --epochs 5 --batch_size 64 --bert_lr 2e-5 --head_lr 5e-4 \
	    --alpha_pos 2.0 --beta_ner 0.5 --fgm --fgm_eps 1.0 \
	    --dev_limit 2000

finetune-csc: check-ckpt $(DATASETS)/csc_train.pt
	cd $(ROOT) && $(PY) -m src.finetune_csc \
	    --ckpt_dir $(CKPT) \
	    --train_data $(DATASETS)/csc_train.pt --test_data $(DATASETS)/csc_test.pt \
	    --output_dir $(OUT)/csc_$(SIZE) \
	    --epochs $(CSC_EPOCHS) --batch_size $(CSC_BATCH) --lr $(CSC_LR) \
	    --warmup_ratio 0.1 --det_weight 0.3 --threshold 0.7

.PHONY: check-ckpt
