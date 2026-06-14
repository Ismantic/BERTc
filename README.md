# BERTc

**BERTc** (Chinese / Continue) — char-level Chinese BERT, **from-scratch** 预训,目标:
**多任务 fine-tune 全面超过 hfl/MacBERT-large(326M)和 RoBERTa-wwm-ext(102M)** 在 PD-1998 上的 CWS / POS / NER + SIGHAN-15 CSC 表现。

## 当前 SOTA(v4-Large Modern BERTc,2026-06-14)

### Backbone:Modern BERTc v4-Large
- **316M params**,24L / 1024H / 2752I / 16h
- **17.65B token** char-level Wapic-WWM 预训(全 corpus shuffle)
- Recipe:Cramming(ScaledSinusoidal PE / 简化 MLM head / bias-free / Dropout 0 / grad_clip 0.5)
  + ModernBERT(GeGLU / pre-norm / Megatron init / StableAdamW β2=0.95)
  + Damped Cosine LR + 固定 15% WWM MLM + cross-doc 隔离
- 详见 `pretrain/modern_bertc/`

### MT joint(CWS+POS+NER)+ FGM 5ep

| 模型 | size | CWS | POS | NER | score(cws+0.3·pos+0.2·ner)|
|---|---|---|---|---|---|
| **BERTc v4-Large + FGM**(SOTA) | **316M** | **0.9840** | **0.9800** | 0.9660 | **1.4712** |
| BERTc v4-Mid + FGM | 165M | 0.9836 | 0.9753 | 0.9632 | 1.4689 |
| BERTc v6.5 + FGM | 165M | 0.9807 | 0.9719 | 0.9568 | 1.4636 |
| MacBERT-large(no-FGM 3ep)| 326M | **0.9856** | 0.9629 | **0.9664** | 1.4677 |
| RoBERTa-wwm-ext(no-FGM 3ep)| 102M | 0.9828 | 0.9562 | 0.9629 | 1.4623 |

- **综合 score:v4-Large 超 MacBERT-L +0.0035**(0.24% 相对)
- **POS 单项 0.9800 全网最高**(超 MacBERT-L +1.71 pp)
- CWS 单项落后 MacBERT-L 0.0016(MacBERT 326M 优势 + 我们 FGM 5ep 后期过拟合)
- NER 持平 MacBERT-L

### CSC(SIGHAN-15 sentence F1,pycorrector 口径)

| 模型 | size | recipe | F1 | P | R |
|---|---|---|---|---|---|
| **BERTc v4-Large**(SOTA)| **316M** | b32 lr3e-5 10ep tied | **0.8346** | 0.9407 | 0.7480 |
| MacBERT4CSC(开源 baseline)| 110M | — | 0.8314 | 0.9274 | 0.7534 |
| MacBERT-large CSC | 326M | b32 lr2e-5 5ep | 0.8309 | 0.9302 | 0.7507 |
| BERTc v4-Mid | 165M | b64 lr5e-5 5ep tied | 0.8308 | 0.9516 | 0.7373 |
| RoBERTa-wwm CSC | 110M | b32 lr5e-5 10ep | 0.7970 | 0.8907 | 0.7212 |

**v4-Large +0.0037 超 MacBERT-L 326M / 半参数下打平 MacBERT4CSC**。

详细对比 + 8-run ablation 见 `finetune/sota/README.md`。

## 仓库结构

```
BERTc/
├── pretrain/
│   ├── modern_bertc/    Modern BERTc(v3 → v4-Mid → v4-Large 主线)
│   └── (legacy v6/v6.5/v7 stage)
├── finetune/
│   ├── NLP_BERT_CRF/    MT joint(CWS+POS+NER + FGM)
│   ├── sota/            SOTA ckpts + 详细 README
│   └── backbones/       归档 backbone ckpt
└── csc/
    ├── train/           train_csc_modern.py(Modern backbone CSC,cor_head tied)
    ├── eval/            pycorrector 口径 SIGHAN-15 eval
    ├── data/            SIGHAN13/14/15 + Wang271K 训练 pairs
    └── baseline/        MacBERT4CSC reference 复现
```

## 关键设计

### 1. char-level piece tokenizer
`--method sentencepiece --cn-dict no`:中文每字 1 piece,英文 BPE multi-char subword(5381 个),
空格 normalize 为 `▁`(id 5687)。vocab 12536。

### 2. Modern BERTc(v3+ 起)
**架构**(model.py):
- ScaledSinusoidal PE(Cramming-style learnable scale,无 RoPE)
- LayerNorm no-bias + pre-norm + skip_first_prenorm
- GeGLU MLP(intermediate = 8H/3 等效 BERT-classic 4H)
- 简化 MLM head:`logits = h @ embed.weight.T`,无 Dense/LN/GeLU/bias
- Megatron init(残差层 W 缩 1/√(2L))
- 全无 Dropout,grad_clip 0.5

**训练 recipe**(train_modern.py):
- StableAdamW(β2=0.95,wd=0.01)+ filter bias/norm
- Damped Cosine LR(8e-4 → 8e-5)
- 固定 15% WWM MLM,cross-doc 隔离(flex_attention block_mask)
- batch_warmup:grad_accum 1→256 前 5% step
- eff_batch 4096,8500 step ≈ 17.4B tokens

### 3. Wapic-WWM 退火数据(`pretrain/pretokenize_v6_anneal.py`)

```
raw text
  ↓ re.sub(r'\s+', ' ').strip()       # 跟 SentencePiece 内部 normalize 对齐
normalized text
  ├─→ piece.encode_as_ids → token sequence(BPE + ▁ 自然出现)
  └─→ split(' ') → segments
        ├─ 全英文 → 整体作 1 word
        ├─ 含中文 → Wapic CRF 切词,每词独立 word_id
        └─ 单空格 → 1 word_id(SEP)
  ↓ piece char-span 对齐取 word_id
(token_id int32, word_id int64) → 双 memmap
```

**不变量:** token sequence 严格等于 `piece.encode_as_ids(text)`;word_id 仅承载 WWM 信息。

### 4. MT fine-tune
joint loss = cws + **α · pos** + β · ner;默认 α=0.5 时 POS 权重只占 1.6% → 学不到。
**α=2** 让 POS +0.02 持平 MacBERT MT,再加 FGM eps=1.0 + 5ep 得到当前 SOTA。

### 5. CSC fine-tune(2 head + tied cor_head)
- Correction head: **必须 weight-tied 到 `bert.embed.weight`**(Modern BERTc 简化 MLM head 已让 hidden 对齐 embed,fresh Linear 会废掉对齐 → F1 差 -0.05)
- Detection head: Linear → 1(binary, focal BCE)
- Loss = 0.3 × focal(det) + CE(cor)
- v4-Large best recipe:10 ep + b32 + lr 3e-5

## Related Projects (ecosystem)

```
PieceTokenizer  ── 基础 SentencePiece-based char/BPE tokenizer
       │
       ├── Summer       Qwen3 ReTok(把 Qwen3 BBPE 词表换成 piece 词表)
       │     │
       │     └── Interpreter  SOTA 中英翻译 model(承接 Summer)
       │
       ├── BERTc(本仓)  char-level 中文 BERT 预训 + MT/CSC fine-tune
       │     ↑↓ 互补
       └── Wapic        独立 C++ CRF 中文分词工具
             ├→ 给 BERTc 提供 WWM 切词工具
             └← 接受 BERTc fine-tuned model 的蒸馏语料(扩充 Wapic 训练数据)
```

依赖关系:
- **PieceTokenizer** 是所有 4 个项目的基础 tokenizer 依赖
- **Summer → Interpreter** 是 ReTok → 下游翻译的接力
- **BERTc ↔ Wapic** 互补:工具(Wapic→BERTc)+ 蒸馏数据(BERTc→Wapic)

## License

Apache 2.0
