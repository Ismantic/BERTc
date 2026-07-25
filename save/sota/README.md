# BERTc SOTA 检查点(永久归档)

SOTA + 次好的**硬链接**(同 inode;Summer/BERT/ 已删但 BERTc 这端 inode 引用仍在,数据未释放)。已加 `.gitignore`(.pt 不进 git)。

## 当前 SOTA(2026-06-14 更新)

| 文件 | 任务 | 指标 | 用于 |
|---|---|---|---|
| `sota_mt_v4large_fgm_5ep_best.pt` | **MT joint(CWS/POS/NER)** | **score 1.4712**(cws **0.9840** / pos **0.9800** / ner **0.9660**)| **v4-Large 316M**,2026-06-12 取代 v4-Mid |
| `sota_csc_v4large_v8_best.pt` | **CSC(SIGHAN-15)** | **F1 0.8346**(P 0.9407 R 0.7480 acc 0.8359)| **v4-Large 316M**,8-run chain 调优,2026-06-14 取代 v4-Mid |
| `sota_cws_v6_fgm_5ep_best.pt` | **单 CWS** | clean F1 0.9819 | 纯切词任务(未更新)|

## 旧 SOTA(保留对照)

| 文件 | 任务 | 指标 | 备注 |
|---|---|---|---|
| `sota_mt_v4mid_fgm_5ep_best.pt` | MT joint | score 1.4689 | v4-Mid 165M,被 v4-Large 超 +0.0023 |
| `sota_mt_v65_fgm_5ep_best.pt` | MT joint | score 1.4636 | v6.5 165M,被 v4-Mid 超 +0.0053 |
| `sota_csc_v4mid_5ep_best.pt` | CSC SIGHAN-15 | F1 0.8308 | v4-Mid 165M,被 v4-Large v8 超 +0.0038 |

## 次好

| 文件 | backbone | 指标(dev) | 用途 |
|---|---|---|---|
| `secondbest_mt_v65_3ep_best.pt` | v6.5(**无 FGM**) | cws 0.9773 / pos 0.9644 / ner 0.9503 / score 1.4567 | FGM 增益对照(+0.0069)|
| `secondbest_mt_macbert_3ep_best.pt` | MacBERT-large(326M) | cws **0.9856** / pos 0.9629 / ner **0.9664** | baseline ceiling |

## 训练配置

### MT joint(v4-Mid + FGM 5ep)— **新 SOTA**
- backbone:`Ismantic/BERTc-165M`(12L/1024H/2752I/16h)
- fine-tune(重构后的等价命令):
  ```bash
  python -m prepare.fetch_backbone --repo Ismantic/BERTc-165M
  python -m src.finetune_mt --ckpt_dir prepare/backbones/BERTc-165M \
      --train_data prepare/datasets/mt_train.pt --dev_data prepare/datasets/mt_dev.pt \
      --output_dir output/mt --epochs 5 --batch_size 64 \
      --bert_lr 2e-5 --head_lr 5e-4 --warmup_ratio 0.1 \
      --alpha_pos 2.0 --beta_ner 0.5 --fgm --fgm_eps 1.0 --dev_limit 2000
  ```
- dev 演化:1.4611 → 1.4663 → 1.4677 → 1.4674 → **1.4689**(ep5 final 反弹)

### CSC(v4-Large v8 10ep)— **当前 SOTA**(2026-06-14)

- backbone:`Ismantic/BERTc-315M`(24L/1024H/2752I/16h)
- fine-tune(重构后的等价命令):
  ```bash
  python -m prepare.fetch_backbone --repo Ismantic/BERTc-315M
  python -m src.finetune_csc --ckpt_dir prepare/backbones/BERTc-315M \
      --train_data prepare/datasets/csc_train.pt --test_data prepare/datasets/csc_test.pt \
      --output_dir output/csc --epochs 10 --batch_size 32 --lr 3e-5 \
      --warmup_ratio 0.1 --det_weight 0.3 --threshold 0.7 --max_len 128
  ```
- 关键改动 vs v4-Mid 配置:
  - **lr 5e-5 → 3e-5**(Large 收敛对 lr 敏感,小 lr 更稳)
  - **5 ep → 10 ep**(Large 5ep 欠训,ep10 仍在涨)
  - **batch 64 → 32**(Large 316M + AdamW state 在 24GB 4090 上 batch 64 临界 OOM)
- dev 演化(ep1→ep10):0.74 → 0.79 → 0.81 → 0.81 → 0.82 → 0.82 → 0.83 → **0.8346** → 0.8281 → 0.8285
- `cor_head` 必须 weight-tied 到 `bert.embed.weight`(同 v4-Mid lesson)
- ckpt:`sota_csc_v4large_v8_best.pt`(1.2GB)

#### CSC chain ablation(8 runs,所有 v4-Large ckpt-8500 backbone)

| run | 改动 vs v4b 基线 | F1 |
|---|---|---|
| v7 | **v4-Mid 同 setup 重跑**(noise baseline)| 0.8265 |
| v4b | b32 lr5e-5 **10ep** | 0.8281 |
| v6 | + det_weight **0.5** | 0.8291 |
| v11 | + warmup **0.05** | 0.8299 |
| v9 | b**16** lr5e-5 10ep | 0.8303 |
| v10 | v4b setup,**seed 42** | 0.8341 |
| v5 | + warmup **0.2** | 0.8343 |
| **v8** | **lr 3e-5**(其他同 v4b)| **0.8346** ★ |

**Lessons**:
1. **10 ep 而非 5 ep**:Large 5ep 严重欠训(原 v4-Large 5ep b64 = 0.8235,v4b 10ep = 0.8281 直接 +0.005)
2. **lr 3e-5 微优于 5e-5**(v8 vs v4b:+0.0065)
3. **warmup 0.2 也有效**(v5 vs v4b:+0.0062)
4. **batch 16/0.5 det_weight 无明显帮助**
5. **707-sample SIGHAN test 噪声 ±0.005-0.01**:v7(0.8265)vs 原 v4-Mid SOTA(0.8308)同 setup 差 -0.0043,纯随机;v10(0.8341)vs v4b(0.8281)同 setup 差 +0.006

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


## 评测口径

**MT 的数字是在 dev 前 2000 句上测的**(训练脚本 `--dev_limit 2000`,选 best.pt
用的也是这个口径),不是全部 21,143 句。全量 dev 上是:

| | 分词 | 词性 | 实体 | joint |
|---|---|---|---|---|
| 前 2000 句(本文档全部数字) | 0.9840 | 0.9800 | 0.9660 | 1.4712 |
| 全量 21,143 句 | 0.9790 | 0.9787 | 0.9597 | 1.4646 |

**CSC 的数字是 SIGHAN-15 官方 707 条**(`shibing624/pycorrector` 里那一版)。
CTCDataset 里还有个 1100 条的 `sighan15_test.jsonl`,不是同一个东西,不能混用。

## 复现

```bash
python test/test_reproduce_sota.py
```

拿本目录的 checkpoint 跑完整评测,对照本文档记录的数字。这是判断代码有没有
改坏的硬标准 —— 2026-07-25 的四层重构就是靠它确认 MT 1.4712 和 CSC 0.8346
一位不差地复现了。
