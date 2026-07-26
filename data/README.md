# data/

下载原始语料与标注数据,加工成统一格式。只放代码,数据落在 gitignore 的
`data/downloads/` 和 `data/derived/`。

```bash
make -C data help       # 全部命令
make -C data status     # 每个源下了没
```

`source.py` 是数据源注册表,记录每个源的公开出处、用量和用途。它不接受
「本机已有的目录」这类配置:允许本地跳过下载,别人克隆下来就跑不通,
而本机不会报错。

## 预训练语料

| 源 | 出处 | 默认用量 | 语言 |
|---|---|---|---|
| SkyPile | HF `Skywork/SkyPile-150B` | 21 jsonl(全量 437) | 中 |
| CCI3-HQ | HF `BAAI/CCI3-HQ` | 5 jsonl | 中 |
| Chinese-FineWeb-Edu | HF `opencsg/Fineweb-Edu-Chinese-V2.2` | `4_5/` 前 500(全量 9745) | 中 |
| FineWiki zh | HF `HuggingFaceFW/finewiki` | `data/zhwiki/` 全 5 | 中 |
| FineWiki en | HF `HuggingFaceFW/finewiki` | `data/enwiki/` 全 15 | 英 |
| PeopleDaily | HF `Papersnake/people_daily_news` | 全量(1946–2025) | 中 |
| CnnDailyMail | HF `abisee/cnn_dailymail` | 全量 | 英 |

合计约 79 GB,是已发布模型的实跑用量。`n_parts` 控制取几个文件。这几个源在
HF 上都是几百 GB 全量(SkyPile 665 GB、Chinese-FineWeb-Edu 9745 个 parquet),
**不要无参数 `snapshot_download`**。

改 `n_parts` 后重跑 `make -C data download-pretrain`,已下载的会跳过。

中英按文档数配比而非 token 数,原因见
[`docs/WHY.md`](../docs/WHY.md#中英语料按文档数配比不是按-token-数)。

## 下游数据

| 源 | 出处 | 提供 |
|---|---|---|
| PD-1998 | Wapic 仓库自带 `data/PeopleDaily1998.zip` | CWS / POS / NER 标注 |
| CTCDataset | GitHub `zejunwang1/CTCDataset` | SIGHAN 13/14/15 训练集(只取这 3 个 jsonl) |
| Wang271K | HF `shibing624/CSC` | CSC 标准训练集(只取 `train.json`) |
| SIGHAN-15 测试集 | GitHub `shibing624/pycorrector` | 官方 707 条 |

两点需要注意:

- PD-1998 随 Wapic 一起 clone,不单独下载。没 clone 过用
  `bash prepare/install_deps.sh wapic-data`(只 clone,不编译)。解压出的
  目录名叫 `199801`,内容是 **199801.txt ~ 199806.txt 六个月**。
  train = 前 5 月(102,739 句),dev = 199806(21,143 句)
- **SIGHAN-15 测试集有两个版本。** 这里用 707 条那版(PyCorrector 口径)。
  CTCDataset 里的 `sighan15_test.jsonl` 是 1100 条的另一版,两者不能混用

`process.py` 和 `process_cws.py` 用的是两份不同的人民日报:前者是 1946–2025
全文(喂预训练),后者是 1998 年 1–6 月的 PFR 标注语料(喂 MT 微调)。

## CSC 配方

`make -C data process-csc` 按四个文件合并,顺序即去重优先级:

```
wang271k/train.json                       248,782
CTCDataset/sighan/sighan13_train.jsonl        350
CTCDataset/sighan/sighan14_train.jsonl        506
CTCDataset/sighan/sighan15_train.jsonl        337
                                    去重后 249,975 对
```

已发布的 CSC 模型(F1 0.8388)用的就是这份。

**等长过滤** `len(src) == len(tgt)`:CSC 是同音 / 形似字的等长替换,含增删的
语法错误进来只会干扰。

同时把 SIGHAN-15 官方 707 条从下载目录原样拷到 `derived/csc/`,评测直接读那份。

PD-1998 完全可复现:重新 clone Wapic → `process_cws.py`,产出的 6 份 jsonl
与已发布 MT 模型的实际训练输入逐字节相同。Wapic 里那份 zip 与上游
`chenhui-bupt/PeopleDaily1998` 的 `199801.zip` 也逐字节相同(SHA256
`17474bbf…`),换源不影响任何结果。

## 代理

HF 走 hf-mirror 需要清代理,GitHub 需要走代理。`download.py` 在模块加载时
存下代理配置,只在 GitHub 那条路径恢复。`HF_ENDPOINT=` 置空走 HF 官方源。

## 文件

```
source.py       数据源注册表 + 路径解析(所有脚本共用)
download.py     四种源:hf / hf-snapshot / github-file / github-repo
process.py      PeopleDaily / CnnDailyMail → documents.txt(一行一篇)
process_cws.py  PD-1998 PFR 标注 → cws / pos / ner 三种 jsonl
process_csc.py  CSC 各源 → (错句, 正句) pkl
```

落地位置可用 `BERTC_DATA_ROOT` / `BERTC_DERIVED_ROOT` 改到别的盘。
