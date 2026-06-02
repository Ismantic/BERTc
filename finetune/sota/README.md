# BERTc SOTA 检查点(永久归档)

SOTA + 次好的**硬链接**(同 inode;Summer/BERT/ 已删但 BERTc 这端 inode 引用仍在,数据未释放)。已加 `.gitignore`(.pt 不进 git)。

## 最好

| 文件 | 任务 | 指标 | 用于 |
|---|---|---|---|
| `sota_mt_v65_fgm_5ep_best.pt` | **MT joint(CWS/POS/NER)** | **score 1.4636**(cws **0.9807** / pos **0.9719** / ner **0.9568**)| 联合切词 + 词性 + 实体 |
| `sota_cws_v6_fgm_5ep_best.pt` | **单 CWS** | **clean F1 0.9819** | 纯切词任务 |

## 次好(MT 对照)

| 文件 | backbone | 指标(dev) | 用途 |
|---|---|---|---|
| `secondbest_mt_v65_3ep_best.pt` | v6.5(**无 FGM**)| 见 train log | 对照 SOTA,验证 FGM 增益 |
| `secondbest_mt_macbert_3ep_best.pt` | MacBERT-large(326M) | cws **0.9856** / pos 0.9629 / ner **0.9664** | baseline ceiling(cws/ner 比 SOTA 高,pos 低)|

CWS 单任务次好未单独 hardlink(若需对照,从 `inline_track.tsv` 推断)。

## 训练配置

### MT joint(v6.5 + FGM 5ep)
- backbone:`bert_train_v6_5_mid`(v6 + 3B L3-mix anneal CPT)
- fine-tune:`train_mt.py --alpha_pos 2.0 --beta_ner 0.5 --fgm --fgm_eps 1.0 --epochs 5 --batch_size 64 --bert_lr 2e-5 --head_lr 5e-4 --warmup_ratio 0.1`
- dev 演化(单调上升):1.4443 → 1.4568 → 1.4605 → 1.4624 → **1.4636**
- 原路径(已删):`Summer/BERT/NLP_BERT_CRF/output_v65_mt_fgm_crf/best.pt`

### 单 CWS(v6 + FGM 5ep)
- backbone:`bert_train_v6_mid`(v4 + 5B WWM CPT,corpus 重叠)
- fine-tune:CWS single-task + FGM eps=1.0,5 epoch
- 原路径(已删):`Summer/BERT/NLP_BERT_CRF/output_v6_fgm5_crf/best.pt`

## 加载示例

```python
import torch
ckpt = torch.load("sota_mt_v65_fgm_5ep_best.pt", map_location="cpu")
# state_dict 内含:bert.*, cws_crf.*, pos_head.*, ner_crf.*
```

## 对比 baseline(同 PD-1998 dev)

| 模型 | size | CWS | POS | NER |
|---|---|---|---|---|
| RoBERTa-wwm-ext MT | 102M | 0.9839 | 0.9562 | 0.9629 |
| MacBERT-large MT | 326M | 0.9856 | 0.9629 | 0.9664 |
| **BERTc v6.5+FGM MT** | **165M** | **0.9807** | **0.9719** ✓ | **0.9568** |
| **BERTc v6+FGM 单 CWS** | **165M** | **0.9819** | — | — |

**POS 全胜**(165M vs 326M MacBERT,+0.009);**CWS/NER 与 RoBERTa 差 0.003-0.006**。v7 退火完了再跑一轮 MT+FGM 目标:全面超 RoBERTa。
