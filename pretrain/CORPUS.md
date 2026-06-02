# BERTc 预训练 corpus 详单

记录每个 backbone version 训练用了什么 corpus、用了多少,以便复现 + 排查 corpus overlap 问题。

## 总览

| Version | 阶段 | corpus | 总 token | 路径 |
|---|---|---|---|---|
| v3 | from-scratch | cn_v1.txt(5 源) | 5B | (early) |
| v4 | from-scratch | **cn_v2.txt(5 源,30GB)** | **8B** shuffled | `train_8B_v4.pt` |
| v5 | CPT(WWM)| 自标注 CWS-v1 | — | (实验,弃) |
| v6 | CPT(WWM)| **同 v4 corpus(seed=44,~60% 重叠)** | 5B | `train_8B_v6.pt`(实际名)|
| v6.5 | anneal CPT(WWM)| **L3-mix(zh-QA + zh-Multi + en-QA + en-Multi)** | 3B | `train_v6_anneal.pt` |
| **v7** | **anneal CPT(WWM)** | **SkyPile 残部 + FineWeb-Edu 残部 + CnnDM + Wikipedia_en** | **6B** | `train_v7_anneal.pt` |

---

## v4 详情 — `cn_v2.txt`(30GB)

由 `build_cn_corpus_30g.py` 拼接(每行 1 doc,空白折叠 + 200K 字截断)。
encode 阶段用 `encode_char_data.py` 全 corpus shuffle(seed=42),取 8B token。

| # | 源 | 路径 | 用量 | 备注 |
|---|---|---|---|---|
| 1 | **Wikipedia_CN** | `/home/tfbao/a6000/Wikipedia_cn_json_files/wiki-cn-*.json` | **全 501MB,3 files** | jsonl |
| 2 | **PeopleDaily.sentences** | `/home/tfbao/Shiyu/Data/data/PeopleDaily.sentences.txt` | **全 5.0GB** | 每行 1 句 |
| 3 | **CCI3-HQ** | `/home/tfbao/a6000/Summer-data/CCI3-HQ/data/part_*.jsonl` | **全 4.9GB,5 files** | jsonl |
| 4 | **SkyPile** | `/home/tfbao/a6000/SkyPile/*.parquet` | **9GB cap,实际用 00000-00008 共 9 个 parquet**(11.4GB text)| sorted glob,顺序读到 cap |
| 5 | **Chinese-FineWeb-Edu** | `/home/tfbao/a6000/Summer-data/Chinese-FineWeb-Edu-V2.2/4_5/*.parquet` | **11GB cap,实际用 idx 0-793 共 794 个 parquet** | sorted glob |

总计 ~30GB text → 64,637,409 行 → seed=42 shuffle → 取前 8B token = `train_8B_v4.pt`

### 训练 hparam
```
bert_init_v3_mid (init) → train_bert_mlm.py
bs 128 × accum 8 (eff 1024) × max_seq 512 × 15258 step
lr 3e-4, warmup 800, min_lr_ratio 0.1, weight_decay 0.01
mlm_prob 0.15(无 WWM,纯随机 15%)
```

### 评估
- CWS clean F1 0.9800(单 CWS,3ep)
- MT base CWS 0.9779 / POS 0.9452 / NER 0.9512
- MT + α=2: POS 0.9638(MacBERT 持平)

---

## v6 详情(**与 v4 corpus 重叠 ~60%**)

`build_cn_corpus_30g.py` + 不同 seed(44)→ shuffle → 5B token。

由于 corpus 完全同源,且 v4 8B 已用 8/64M = ~12% 行,v6 5B 用 5/64M = ~8%,**两者重叠期望 ~60%**(独立 sample 同源 corpus)。

→ v6 fine-tune 数对 v4 几乎完全持平(±0.0005),验证了 corpus 重叠 = 收益 ~0 的判断。
**结论:之后 anneal/CPT 必须换全新 corpus,不能再 re-shuffle v4 源。**

---

## v6.5 详情 — L3-mix(已弃,用作历史记录)

由 `pretokenize_v6_anneal.py` 直接 parquet → Wapic-WWM → memmap,3B token。

源(各 config 前 N parquet 顺序读):
- `data/ultrafineweb_zh_l3/qa`: 20 parquet
- `data/ultrafineweb_zh_l3/multi_style`: 20 parquet
- `data/ultrafineweb_en_l3/qa`: 10 parquet
- `data/ultrafineweb_en_l3/multi_style`: 10 parquet

实测 token 占比 ~ 中 36% / 英 64%(英文 doc 平均 3.5x 中文长)。

### 问题
- QA 子集 30-50% 是「问题:/答案:」模板,污染 fine-tune
- 退火数据中文比例偏低(下游任务全中文)
- → v6.5 vs v6 fine-tune 数提升极小(cws -0.0005,pos +0.0006,ner +0.0017)

---

## v7 详情(本次)— 多源全新融合

由 `pretokenize_v7_anneal.py` 直接源 → Wapic-WWM → memmap,**6B token**,中:英 = 6:4。

**核心原则:**
- 全部 corpus **v4 没见过的部分**(避免 v6 重蹈覆辙)
- 中:英 = 6:4(下游 PD-1998 全中文,中文为主)
- 多源融合(SkyPile + FineWeb-Edu + CnnDM + Wikipedia_en),避免单源偏向

| 类别 | 源 | 路径 + 范围 | est token | 备注 |
|---|---|---|---|---|
| 中文 | **SkyPile**(残部)| `/home/tfbao/a6000/SkyPile/000{09..13}.parquet`(5 parquet)| **~1.8B** | v4 用了 00000-00008 |
| 中文 | **Chinese-FineWeb-Edu**(残部)| `…/4_5/*.parquet` idx 794~1193(400 parquet)| **~1.8B** | v4 cap 在 idx 793 |
| 英文 | **CnnDM**(全)| `/home/tfbao/Shiyu/Data/data/CnnDailyMail.documents.txt` | **~1.3B** | v4 未用 |
| 英文 | **Wikipedia_en**(前 3 file)| `/home/tfbao/a6000/Wikipedia_en_json_files/wiki-20231101-en-00000{0,1,2}.json` | **~1.1B** | v4 未用 |
| **合计** | — | — | **~6B** | **中:英 = 6:4** |

### Interleave 策略
- 中:英 = 18:10 doc-level weighted round-robin → token ratio ~ 6:4
- 中文里 SkyPile:FineWeb-Edu 各 1 doc 轮流 → ~50:50
- 英文里 CnnDM:Wikipedia_en 各 1 doc 轮流 → ~50:50

### 训练 hparam(预定)
```
v4_mid (backbone) → train_bert_mlm.py
bs 128 × accum 8 (eff 1024) × max_seq 512 × ~5859 step ÷ 6/3 = ~11700 step
lr 2e-5(退火 lr),warmup 300,min_lr_ratio 0.1,weight_decay 0.01
mlm_prob 0.15,**WWM via Wapic 切词**
```

### 残量预算(下次 v8 可用)
- SkyPile 还剩 **28 parquet**(00014-00041),~10B token
- Chinese-FineWeb-Edu 还剩 **8551 parquet**(1194-9744),~38B token
- Wikipedia_en 还剩 **62 file**,~6B token
- CnnDM 已用完(只 1 文件)

---

## 复现说明

### v4 复现
```bash
# 1. 拼 cn_v2.txt
python /home/tfbao/Shiyu/Summer/BERT/build_cn_corpus_30g.py
# 2. encode + shuffle (seed=42, 8B token)
python /home/tfbao/Shiyu/Summer/BERT/encode_char_data.py \
    --corpus /home/tfbao/Shiyu/Summer/tokenizer_corpus/cn_v2.txt \
    --tokenizer_model bert_init_v3_mid/piece.model \
    --output train_8B_v4.pt --seq_len 512 \
    --total_tokens 8000000000 --shuffle_docs --seed 42
# 3. train
bash /home/tfbao/Shiyu/Summer/BERT/launch_v4_train_after_encode.sh
```

### v7 复现
```bash
# 1. pretokenize (CPU only, ~75min on 16-core)
python /home/tfbao/Shiyu/data/pretokenize_v7_anneal.py \
    --target_tokens 6000000000 --num_workers 14
# 输出: train_v7_anneal.pt + .wid
# 2. anneal CPT + MT base + MT FGM(单 GPU,~28h)
PRETOK_PID=<pid> bash /home/tfbao/Shiyu/Summer/BERT/chain_v7.sh
```

---

## SOTA 路径(2026-05-31 当前 BERTc 最高)

**BERTc v6.5 + α=2 + FGM 5ep — joint MT(CWS/POS/NER)三冠**:

```
v4 (8B from-scratch)
  → v6 (5B WWM CPT,corpus 同源 收益 0,but checkpoint 用)
  → v6.5 (3B L3-mix anneal CPT)
  → MT fine-tune: train_mt.py --alpha_pos 2.0 --beta_ner 0.5 --fgm --fgm_eps 1.0 --epochs 5 --batch_size 64 --bert_lr 2e-5 --head_lr 5e-4 --warmup_ratio 0.1
```

**dev best.pt(score 1.4636,ep5):**
| 指标 | BERTc v6.5+FGM | RoBERTa-wwm-ext MT(102M)| MacBERT-large MT(326M)|
|---|---|---|---|
| CWS clean | 0.9807 | 0.9839 | 0.9856 |
| POS per-word | **0.9719** ✓+0.016 vs RoBERTa | 0.9562 | 0.9629 |
| NER micro | 0.9568 | 0.9629 | 0.9664 |

**checkpoint 位置**: `/home/tfbao/Shiyu/Summer/BERT/NLP_BERT_CRF/output_v65_mt_fgm_crf/best.pt`

**ep1-5 单调上升**:1.4443 → 1.4568 → 1.4605 → 1.4624 → **1.4636**(无过拟合)

**关键 trick(可复用)**:
1. **α=2**(POS loss 权重 ×4 默认 0.5):POS 从 0.96 → 0.97
2. **FGM eps=1.0**(对 word_embeddings adversarial):cws/ner +0.005 ~ +0.013
3. **5 ep**(而非 3 ep):5ep > 3ep 持续涨(ep1 1.44 → ep5 1.46)

---

## 更新日志

- **2026-05-31 12:40** v6.5 + FGM 完(score 1.4636,POS 超 RoBERTa/MacBERT)
- **2026-05-31 12:57** v7 CPT 起步,~28h ETA
- **2026-05-31** 首版,记录 v4 / v6 / v6.5 已用,v7 plan 锁定
