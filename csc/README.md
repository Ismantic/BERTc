# BERTc-CSC:中文文本纠错(Chinese Spelling Correction)

**BERTc 第 2 个核心下游任务**(继 CWS/POS/NER MT 之后),长期投入。

## Why CSC?

- **天花板更高**:不像 CWS/POS/NER 0.98 接近饱和,CSC SIGHAN-15 SOTA ~ 0.83 还有 10+ pt 空间
- **商业价值高**:输入法、ASR 后处理、办公校对等真实场景
- **BERT-class 真正强项**:Matthew Honnibal 论证的"BERT > GPT-4 预测任务"案例之一
- **跟 BERTc 现有 char-level + Wapic-WWM 完全契合**

## 任务范围

- **狭义 CSC**:同长度替换(同音 / 形似字),SIGHAN-15 标准 benchmark
- **广义 CTC**(后期扩展):语法纠错 / 不等长(暂不做,LLM 主场)
- **ASR 后处理 niche**(中长期):同音错误后处理,可加 pinyin embedding 增强

## 目录结构

```
csc/
├── data/
│   ├── raw/
│   │   ├── sighan/             # SIGHAN 13/14/15 train + test
│   │   └── shibing624/         # 14 个 sub-dataset (CSCD-NS, lemon-*, ec-*, ...)
│   └── test/
│       └── sighan2015_test_official.tsv   # pycorrector 官方 707 样本,权威 baseline
├── eval/
│   ├── eval_ctc.py             # 自定义评估框架(char + sentence)
│   └── eval_pycorrector_baseline.py  # 跟 pycorrector 完全对齐的官方 eval
├── baseline/
│   ├── run_macbert4csc_cpu.py  # MacBERT4CSC reference 实测
│   └── macbert4csc_ckpt → ...  # MacBERT4CSC ckpt 符号链接
├── train/                       # (TBD) BERTc-CSC 训练脚本
└── docs/                        # 实验记录、对比表
```

## Baseline:MacBERT4CSC(实测复现)

**pycorrector 官方 SIGHAN-15 test(707 样本)+ 官方 eval**:

| 指标 | 实测(我们复现)| 模型卡报的(过时)|
|---|---|---|
| **Sentence F1** | **0.8314** | 0.7789 |
| Sentence Acc | 0.8388 | (未给) |
| Sentence Precision | 0.9274 | 0.8264 |
| Sentence Recall | 0.7534 | 0.7366 |
| 推理速度 | 54 samples/s @ CPU | (未报) |

**结论**:**真实可复现 baseline 是 0.8314 sentence F1**(不是模型卡 0.7789)。模型卡数字是早期 ckpt,后来未更新。

## 架构(MacBERT4CSC 风格,我们模仿)

**训练时**:**双 head**
- Backbone: BertModel(或 BERTc 165M)
- Correction head: Linear → vocab(MLM 头)
- Detection head: Linear → 1(binary,FocalLoss)
- Loss = 0.3 × focal(det) + CE(cor)

**推理时**(发布 ckpt 实际只用单 head):
- 用 correction MLM argmax + softmax threshold=0.7 过滤
- 低置信度位置保留原字

## BERTc-CSC v1 目标

| 指标 | MacBERT4CSC | **BERTc-CSC v1 目标** | **进取目标** |
|---|---|---|---|
| Sentence F1 | 0.8314 | ≥ 0.83 | ≥ 0.85 |
| Char F1 (paper 口径) | 0.8991 | ≥ 0.89 | ≥ 0.91 |
| 推理 QPS (CPU) | 54 | ≥ 50 | ≥ 100 |
| 模型大小 | 110M | **165M** | 同 |

**至少持平 MacBERT4CSC,争取超过**。

## 训练数据

| 来源 | 量 | 用途 |
|---|---|---|
| SIGHAN 13/14/15 train | ~6K | 经典 + 跟 paper 对齐 |
| shibing624 cscd_ns | 35K | 真实手误 |
| shibing624 medical_csc | 39K | 医疗 |
| shibing624 lemon-* | ~20K | 多领域 |
| shibing624 ec-* | ~8K | 法律 / 医学 / 政府 |
| **小计**(本地) | **~110K** | |
| **+ SIGHAN+Wang271K** | +270K | 跟 MacBERT4CSC 训练完全对齐(需下载) |
| **总计** | **~380K** | 训 5 epoch |

## Roadmap

### Phase 1:**v1 MVP**(2 周)
- ✅ baseline 测完(已完成)
- 🔄 下载 SIGHAN+Wang271K 完整数据(待)
- 🔄 写 `train_csc.py`(BERTc backbone + dual head,~250 行)
- 🔄 训 5 epoch on 4090(~3-5h)
- 🔄 SIGHAN-15 test 对比 MacBERT4CSC
- **目标**:Sentence F1 ≥ 0.83

### Phase 2:**HF + pycorrector 集成**(1 周)
- HF 上传 `BERTc-CSC-165M`
- pycorrector 提 PR 加 BERTcCorrector(跟 MacBertCorrector 同接口)
- 写中文 blog(知乎 / 微信公众号)

### Phase 3:**领域 + ASR niche**(后期)
- 细分领域 fine-tune(医疗 / 法律 / 通用)
- ASR 同音纠错增强(加 pinyin embedding,niche No.1 候选)

## 评估约定(铁律)

1. **所有报数必须用 `pycorrector/data/sighan2015_test.tsv` 707 样本**(不是其他衍生版)
2. **eval 完全照搬 pycorrector 的 `eval_model_batch` 逻辑**(TP/FP/FN/TN 4 类)
3. **Inference 参数**:max_len=128, batch=32, threshold=0.7(跟 pycorrector 默认一致)
4. **报数包含**:Sent Acc / Sent P / Sent R / Sent F1 + Char P/R/F1
5. **不要用模型卡数字作 baseline** — 用实测代码跑出来的 0.8314

## Files of interest

- `eval/eval_pycorrector_baseline.py` — 跟 pycorrector 对齐的官方 eval pipeline
- `baseline/run_macbert4csc_cpu.py` — MacBERT4CSC 实测脚本
- `baseline/macbert4csc_ckpt/` — MacBERT4CSC 模型 ckpt 链接

## 更新日志

- **2026-05-31** 立项,baseline 测完(0.8314 F1),目录就位,等明天 v7 完后写 train_csc.py
