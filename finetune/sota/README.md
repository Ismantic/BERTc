# BERTc SOTA 检查点(永久归档)

SOTA + 次好的**硬链接**(同 inode;Summer/BERT/ 已删但 BERTc 这端 inode 引用仍在,数据未释放)。已加 `.gitignore`(.pt 不进 git)。

## 当前 SOTA(2026-06-07 更新,Modern BERTc v4-Mid 165M)

| 文件 | 任务 | 指标 | 用于 |
|---|---|---|---|
| `sota_mt_v4mid_fgm_5ep_best.pt` | **MT joint(CWS/POS/NER)** | **score 1.4689**(cws **0.9836** / pos **0.9753** / ner **0.9632**)| 联合切词 + 词性 + 实体 |
| `sota_csc_v4mid_5ep_best.pt` | **CSC(SIGHAN-15)** | **F1 0.8308**(P 0.9516 R 0.7373 acc 0.8416)| 中文文本纠错 |
| `sota_cws_v6_fgm_5ep_best.pt` | **单 CWS** | clean F1 0.9819 | 纯切词任务(未更新)|

## 旧 SOTA(保留对照)

| 文件 | 任务 | 指标 | 备注 |
|---|---|---|---|
| `sota_mt_v65_fgm_5ep_best.pt` | MT joint | score 1.4636 | v6.5 165M,**被 v4-Mid 超 +0.0053** |

## 次好

| 文件 | backbone | 指标(dev) | 用途 |
|---|---|---|---|
| `secondbest_mt_v65_3ep_best.pt` | v6.5(**无 FGM**) | cws 0.9773 / pos 0.9644 / ner 0.9503 / score 1.4567 | FGM 增益对照(+0.0069)|
| `secondbest_mt_macbert_3ep_best.pt` | MacBERT-large(326M) | cws **0.9856** / pos 0.9629 / ner **0.9664** | baseline ceiling |

## 训练配置

### MT joint(v4-Mid + FGM 5ep)— **新 SOTA**
- backbone:`pretrain/modern_bertc/output_v4_mid/checkpoint-8500`(Modern BERTc 165M,12L/1024H/2752I/16h)
- fine-tune:`pretrain/modern_bertc/train_modern_mt.py --alpha_pos 2.0 --beta_ner 0.5 --fgm --fgm_eps 1.0 --epochs 5 --batch_size 64 --bert_lr 2e-5 --head_lr 5e-4 --warmup_ratio 0.1`
- dev 演化:1.4611 → 1.4663 → 1.4677 → 1.4674 → **1.4689**(ep5 final 反弹)

### CSC(v4-Mid 5ep)— **新 SOTA**(打平 MacBERT-L 326M @ 半参数)
- backbone:同上(v4-Mid 165M)
- fine-tune:`csc/train/train_csc_modern.py --epochs 5 --batch_size 64 --lr 5e-5 --warmup_ratio 0.1 --det_weight 0.3 --threshold 0.7`
- **关键**:`cor_head` 必须 weight-tied 到 `bert.embed.weight`,见下 lesson
- dev 演化:0.7488 → 0.7920 → **0.8111** → 0.8056 → **0.8308**(ep5 反弹超 ep3)

### MT joint(v6.5 + FGM 5ep)— 旧 SOTA
- backbone:`bert_train_v6_5_mid`(v6 + 3B L3-mix anneal CPT)
- fine-tune:`train_mt.py` 同上但用 HF AutoModel 加载
- dev 演化:1.4443 → 1.4568 → 1.4605 → 1.4624 → **1.4636**

### 单 CWS(v6 + FGM 5ep)— 未更新
- backbone:`bert_train_v6_mid`(v4 + 5B WWM CPT)
- fine-tune:CWS single-task + FGM eps=1.0,5 epoch

## ⚠️ Modern BERTc CSC fine-tune 关键 lesson(2026-06-07)

**Cor_head 必须 weight-tied 到 backbone embed**,否则 CSC F1 ~ -0.05。

**Why**:Modern BERTc 用 Cramming 简化 MLM head(`logits = h @ embed.weight.T`,无 Dense/LN/GeLU),
预训完 `last_hidden_state` 已经被优化成"直接乘 embed.weight 出 vocab logits"。CSC 用 fresh `nn.Linear(H, V)`
作为 cor_head 会废掉这个对齐 — 实测 v4-Mid v1(fresh head)= 0.7802,v3(tied head)= **0.8308**,
差 -0.0506 pp。

**How**:
```python
self.cor_head = nn.Linear(H, vocab_size, bias=False)
self.cor_head.weight = self.bert.embed.weight   # weight tying
```

**何时适用**:任何用 Modern BERTc(简化 MLM head)做 token-level vocab prediction 的 fine-tune
(CSC、CTC、MLM-style 推理)都应 tied。MT/CWS/NER 等用 CRF/Linear-tag head 不受影响。

## 加载示例

```python
import torch
# v4-Mid MT
ckpt = torch.load("sota_mt_v4mid_fgm_5ep_best.pt", map_location="cpu")
# state_dict: bert.*, cws_crf.*, pos_head.*, ner_crf.*
# v4-Mid CSC(tied)
ckpt = torch.load("sota_csc_v4mid_5ep_best.pt", map_location="cpu")
# state_dict: bert.*, cor_head.weight(==bert.embed.weight), det_head.*
```

## 对比 baseline(同 PD-1998 dev + SIGHAN-15 test)

### MT joint(score = cws + 0.3·pos + 0.2·ner)

| 模型 | size | CWS | POS | NER | score |
|---|---|---|---|---|---|
| **BERTc v4-Mid + FGM(新 SOTA)** | **165M** | **0.9836** | **0.9753** ✓ | 0.9632 | **1.4689** |
| BERTc v6.5 + FGM(旧 SOTA)| 165M | 0.9807 | 0.9719 | 0.9568 | 1.4636 |
| MacBERT-large MT(no FGM 3ep)| 326M | **0.9856** | 0.9629 | **0.9664** | 1.4677 |
| RoBERTa-wwm-ext MT(no FGM 3ep)| 102M | 0.9828 | 0.9562 | 0.9629 | 1.4623 |

**v4-Mid 综合 score 全部超 baselines**,**POS 单项最高**(0.9753),CWS/NER 次于 MacBERT-large 326M(参数 2× 差距下)。

### CSC(SIGHAN-15 test sentence F1,pycorrector 口径)

| 模型 | size | recipe | F1 | P | R |
|---|---|---|---|---|---|
| **BERTc v4-Mid(tied,新 SOTA)** | **165M** | 5ep b64 lr5e-5 | **0.8308** | 0.9516 | 0.7373 |
| MacBERT-large CSC | 326M | 5ep b32 lr2e-5 | **0.8309** | 0.9302 | 0.7507 |
| MacBERT4CSC(开源,pycorrector 复现)| 110M | — | 0.8314 | 0.9274 | 0.7534 |
| RoBERTa-wwm CSC v3 | 110M | 10ep b32 lr5e-5 | 0.7970 | 0.8907 | 0.7212 |
| BERTc v7 CSC v1 | 165M | 5ep b64 lr5e-5 | 0.7994 | 0.9519 | 0.6890 |
| BERTc v4-Mid CSC v1(无 tied)| 165M | 同上但 fresh head | 0.7802 | 0.9231 | 0.6756 |

**v4-Mid tied 持平 MacBERT-large(326M),半参数下达 MacBERT4CSC baseline 同水平**。
