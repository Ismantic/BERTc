# data/ — 数据获取与加工

BERTc 用到的全部数据源的下载与加工代码。**只放代码,数据落在 gitignore 的
`data/downloads/` 和 `data/derived/`。**

`source.py` 是数据源注册表 —— 每个源的公开出处、用量、用途都在那里,是唯一的
真相来源。**不允许指向本机既有目录**:一旦允许"本地有就跳过下载",别人克隆
下来跑不通,而自己永远发现不了。

## 快速开始

```bash
python data/download.py --list        # 每个源的状态
python data/download.py --pretrain    # 预训练语料(约 90GB)
python data/download.py --finetune    # 下游任务数据(约 1.5GB)

python data/process.py --all          # PeopleDaily / CnnDailyMail → documents.txt
python data/process_cws.py            # PD-1998 → cws/pos/ner jsonl
python data/process_csc.py            # CSC 各源 → 句对 pkl
```

## 路径

| 环境变量 | 默认 | 作用 |
|---|---|---|
| `BERTC_DATA_ROOT` | `data/downloads/` | 下载落地 |
| `BERTC_DERIVED_ROOT` | `data/derived/` | 加工产物 |
| `HF_ENDPOINT` | `https://hf-mirror.com` | 置空走官方源 |

注意:HF 走 hf-mirror **必须清代理**,GitHub 反过来**必须走代理**。
`download.py` 在模块加载时存下代理配置,只在 GitHub 那条路径恢复。

## 数据源

### 预训练语料(v4-Large 实跑 17.65B token,中英 6:4)

| 源 | 出处 | 默认用量 | 加工 |
|---|---|---|---|
| SkyPile | HF `Skywork/SkyPile-150B` | 21 parquet | 无 |
| CCI3-HQ | HF `BAAI/CCI3-HQ` | 5 jsonl | 无 |
| Chinese-FineWeb-Edu | HF `opencsg/Fineweb-Edu-Chinese-V2.2` | `4_5/` 前 500(全量 9745) | 无 |
| finewiki zh | HF `HuggingFaceFW/finewiki` | `data/zhwiki/` 全 5 | 无 |
| finewiki en | HF `HuggingFaceFW/finewiki` | `data/enwiki/` 全 15 | 无 |
| PeopleDaily | HF `Papersnake/people_daily_news` | 全量 | `process.py` |
| CnnDailyMail | HF `abisee/cnn_dailymail` | 全量 | `process.py` |

`n_parts` 的默认值是 **v4-Large 实跑用量**,不是拍脑袋定的。这几个源在 HF 上
都是几百 GB 全量,无参数 `snapshot_download` 会拖垮磁盘。

### 下游任务

| 源 | 出处 | 提供 |
|---|---|---|
| PD-1998 | GitHub `chenhui-bupt/PeopleDaily1998` | 分词 / 词性 / 实体标注 |
| CTCDataset | GitHub `zejunwang1/CTCDataset` | CSC 主力:CCTC / CTC2021 / MCSCSet / ECSpell / lemon / cscd-ns / sighan / yacsc / Wang271k |
| MCSCSet | GitHub `yzhihao/MCSCSet` | 医疗 CSC 专家标注 199,763 条 |
| Wang271K | HF `shibing624/CSC` | CSC 标准训练集 |
| chinese_text_correction | HF `shibing624/chinese_text_correction` | 14 个 tsv |
| SIGHAN-15 测试集 | GitHub `shibing624/pycorrector` | **官方 707 条**,CSC 的权威基准 |

PD-1998 的压缩包名叫 `199801.zip`,里面其实是 **199801.txt ~ 199806.txt
六个月**。train = 前 5 月(102,739 句),dev = 199806(21,143 句)。

CSC 的 `sighan15_test` 是 707 条那一版。CTCDataset 里也有个
`sighan15_test.jsonl`,那是 1100 条的另一个版本,**不能混用**。

## 可复现性

- **PD-1998 完全可复现**。从 GitHub 重下 → `process_cws.py`,产出的 6 份
  jsonl 与 MT SOTA 的实际训练输入**逐字节相同**。
- **CSC 99.99%**。`all_pairs.pkl` 的原始生成代码从未进过 git,配方是用逐文件
  集合比对反推出来的,三条规则:
  1. 扫全部源,含两个易漏的 `.jsonl.gz`(`CTC2021/train_large_v2` 10.2 万对、
     `Wang271k/data` 26.8 万对)
  2. **等长过滤** `len(src)==len(tgt)` —— 原 pkl 里不等长的有 0 个。
     CTC2021 的等长部分 100% 入选、不等长部分 0% 入选,是决定性证据
  3. 整文件排除 `NLPCC2023/grammar/`(HSK + MuCGEC,语法纠错不是同音形似字
     替换)、`val_bak`、SIGHAN-15 测试集

  从公网重建得 826,200 对,原 pkl 是 826,097 对,差 103 对(0.012%),
  各文件贡献量逐项吻合。

## 文件

```
source.py         数据源注册表 + 路径解析(所有脚本共用)
download.py       四种源:hf / hf-snapshot / github-file / github-repo
process.py        PeopleDaily(1946-2025 全文)/ CnnDailyMail → documents.txt
process_cws.py    PD-1998 PFR 标注 → cws/pos/ner jsonl(MT 微调用)
process_csc.py    CSC 各源 → (错句, 正句) pkl
```

`process.py` 和 `process_cws.py` 用的是**两份不同的人民日报**:前者是
`Papersnake/people_daily_news`(1946-2025 全文,喂预训练),后者是 PD-1998
(1998 年 1-6 月的 PFR 标注语料,喂 MT 微调)。`process_cws.py` 同时产出
cws / pos / ner 三种 jsonl —— MT 是这三个任务的联合训练,共用一份解析。
