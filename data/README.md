# data/ — 数据获取与加工

BERTc 用到的全部数据源的下载与加工代码。**只放代码,数据落在 `BERTC_DATA_ROOT`。**

所有 part 数默认值 = **v4-Large 实跑用量**(即 HF 上 `Ismantic/BERTc-315M` 的
训练输入),读自 `pretrain/modern_bertc/data3/pretok_v3.log`,不是脚本默认值 ——
两者当年对不上,是这次重构要修的问题之一。

## 快速开始

```bash
python data/download.py --list        # 看每个源解析到哪、缺不缺
python data/download.py --pretrain    # 下预训练语料
python data/download.py --finetune    # 下下游任务数据
python data/process.py --all          # PeopleDaily / CnnDailyMail → documents.txt
python data/process_cws.py            # PD-1998 → cws/pos/ner jsonl
python data/process_csc.py            # CSC 各源 → all_pairs.rebuilt.pkl
```

## 落地路径

| 环境变量 | 默认 | 作用 |
|---|---|---|
| `BERTC_DATA_ROOT` | `/home/tfbao/a6000` | 语料落地根 |
| `BERTC_DERIVED_ROOT` | `$BERTC_DATA_ROOT/derived` | 加工产物 |
| `BERTC_CSC_RAW` | `csc/data/raw` | CSC 原始源 |
| `HF_ENDPOINT` | `https://hf-mirror.com` | 置空走官方源 |

默认值指向**现有语料位置**,所以现在这台机器上 250G 语料一行都不用重下 ——
`source.Source.dir()` 会优先命中已存在的历史目录,只有真缺的才走标准布局。
换新机器时改 `BERTC_DATA_ROOT` 即可从零拉取。

注意:HF 走 hf-mirror **必须清代理**,而 GitHub 反过来**必须走代理**。
`download.py` 在模块加载时存下代理配置,只在 GitHub 那条路径临时恢复。

## 数据源

### 预训练语料(v4-Large 实跑 17.65B token,zh:en = 6:4)

| 源 | HF repo | 默认用量 | 加工 |
|---|---|---|---|
| SkyPile | `Skywork/SkyPile-150B` | 21 parquet(盘上 42) | 无 |
| CCI3-HQ | `BAAI/CCI3-HQ` | 5 jsonl | 无 |
| Chinese-FineWeb-Edu | `opencsg/Fineweb-Edu-Chinese-V2.2` | `4_5/` 前 500 parquet(全 9745) | 无 |
| finewiki zh | `HuggingFaceFW/finewiki` | `data/zhwiki/` 全 5 parquet | 无 |
| finewiki en | `HuggingFaceFW/finewiki` | `data/enwiki/` 全 15 parquet | 无 |
| PeopleDaily | `Papersnake/people_daily_news` | 全量 | `process.py` |
| CnnDailyMail | `abisee/cnn_dailymail` | 全量 | `process.py` |

finewiki 取代 v4-Large 当年用的 2023-11 Wikipedia json dump(zh 3 files / en 25
files)。新源是 parquet,`pretokenize_modern.py` 已有的 parquet reader 可直接复用。

### 下游任务

| 源 | 来源 | 产物 |
|---|---|---|
| PD-1998 | `chenhui-bupt/PeopleDaily1998` 的 `199801.zip` | `cws/pos/ner{,_dev}.pd98.jsonl` |
| CSC 训练对 | `csc/data/raw/` 下 75 个文件 | `all_pairs.pkl` |

`199801.zip` 名字骗人,里面其实是 **199801.txt ~ 199806.txt 六个月**。
train = 前 5 月(102,739 行),dev = 199806(21,143 行)。

## 可复现性状态

- **PD-1998 ✓ 完全可复现**。从 GitHub 重下 → `process_cws.py`,产出的 6 份 jsonl
  与 MT SOTA(`sota_mt_v4large_fgm_5ep_best.pt`,joint 1.4712)实际训练输入
  **逐字节相同**。
- **CSC ✓ 基本可复现(99.96%)**。`all_pairs.pkl`(826,097 对)当年是临时拼的,
  生成代码从未进过 git。`process_csc.py` 用逐文件集合比对把配方反推了出来:

  1. 扫 raw/ 全部源,含两个易漏的 `.jsonl.gz`(`CTC2021/train_large_v2` 10.2 万对、
     `Wang271k/data` 26.8 万对)
  2. **等长过滤** `len(src)==len(tgt)` —— 原 pkl 里不等长的有 0 个;
     CTC2021 等长部分 100% 入选、不等长部分 0% 入选,是决定性证据
  3. 整文件排除 `NLPCC2023/grammar/`(HSK 10.4 万 + MuCGEC,语法纠错不是同音形似
     字替换)、`lemon_v2/`(与 `CTCDataset/lemon/` 同源的另一版本)、`val_bak`

  结果:重建 826,205 对 vs 原 826,097 对,双向重合 **99.96%**(各差 400 上下,
  是空白与全半角规范化的边角)。

  它仍默认写 `all_pairs.rebuilt.pkl`,**不覆盖** SOTA 实际用的 `all_pairs.pkl` ——
  后者作为既成事实保留。`--verify` 可随时复看比对。

## 文件

```
source.py                数据源注册表 + 路径解析(所有脚本共用)
download.py              HF / GitHub 下载,带 part 数截断与续传
process.py               PeopleDaily(1946-2025 全文)/ CnnDailyMail → documents.txt
process_cws.py           PD-1998 PFR 标注 → cws/pos/ner jsonl(MT 微调用)
process_csc.py           CSC 各源 → (错句, 正句) pkl
```

`process.py` 和 `process_cws.py` 用的是**两份不同的人民日报**:前者是
`Papersnake/people_daily_news`(1946-2025 全文,喂预训练),后者是 PD-1998
(1998 年 1-6 月的 PFR 标注语料,喂 MT 微调)。`process_cws.py` 同时产出
cws / pos / ner 三种 jsonl —— MT 是这三个任务的联合训练,共用一份解析。

`process_cws.py` 是 `c959c17` 死代码清理时误删的
`finetune/NLP_BERT_CRF/build_pd1998_jsonl.py`,解析逻辑原样恢复,仅改路径默认值。
