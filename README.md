# BERTc

**BERTc** (Chinese / Continue) — char-level BERT-mid (165M, 12L/1024) **from-scratch** 中文 BERT,目标:
**多任务 fine-tune 全面超过 hfl/MacBERT-large(326M)和 RoBERTa-wwm-ext(102M)** 在 PD-1998 上的 CWS / POS / NER 表现。

## 当前状态(v6 fine-tune)

| 模型 | size | CWS clean F1 | POS per-word | NER micro F1 |
|---|---|---|---|---|
| LTP/base1 zero-shot | 100M | 0.9783 | 0.9945 | 0.8917 |
| RoBERTa-wwm-ext MT | 102M | 0.9839 | 0.9562 | 0.9629 |
| MacBERT-large MT | 326M | **0.9856** | 0.9629 | **0.9664** |
| **BERTc (v6) MT base** | 165M | 0.9784 | 0.9441 | 0.9375 |
| **BERTc (v6) MT + α=2** | 165M | 0.9781 | **0.9638** (≈ MacBERT) | 0.9488 |

POS 已持平 326M MacBERT(α=2 关键 trick),CWS / NER 还差,主要是 backbone 质量。

下一步 **v6.5** = v6 + 3B L3-mix Wapic-WWM 退火 CPT,预期收窄 CWS / NER gap。

## 仓库结构

```
BERTc/
├── pretrain/    char-level MLM 预训练 + 退火 CPT
└── finetune/    CWS / POS / NER 单任务 + 多任务 fine-tune
```

详细脚本一览见各子目录。

## 关键设计

### 1. char-level piece tokenizer
`--method sentencepiece --cn-dict no`:中文每字 1 piece,英文 BPE multi-char subword(5381 个),
空格 normalize 为 `▁`(id 5687)。

### 2. Wapic-WWM 退火数据(`pretrain/pretokenize_v6_anneal.py`)

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

### 3. MT fine-tune
joint loss = cws + **α · pos** + β · ner;默认 α=0.5 时 POS 权重只占 1.6% → 学不到。
**α=2** 让 POS +0.02 持平 MacBERT MT。

## License

Apache 2.0
