# data/ —— 语料从哪来

第一层要解决的问题:**训一个 BERT 需要几十 GB 文本,这些文本从哪弄、
怎么弄才不会把磁盘撑爆。**

教程里通常一句 `load_dataset("wikipedia")` 带过。真做的时候麻烦的是:
公开语料动辄几百 GB 全量,你只需要其中一小部分;不同源的格式各不相同;
而且几个月后你得能说清楚「当初到底用了哪些数据」。

```bash
make -C data help       # 全部命令
make -C data status     # 每个源下了没
```

## 一条原则:不允许指向本地目录

`source.py` 是数据源注册表 —— 每个源的公开出处、用量、用途都在那里,
是唯一的真相来源。

**它不允许配置「本机已有的目录」。** 这个诱惑很大:数据已经在盘上了,加个
「本地有就跳过下载」的分支能省几小时。但那样一来,别人克隆下来跑不通,
而你自己**永远发现不了** —— 因为你的机器上一直是通的。

代价是重装环境要重新下载。收益是「只依赖 HF + GitHub」这句话是真的,
而且随时可验证。

## 预训练语料:怎么从几百 GB 里按需取

| 源 | 出处 | 默认用量 | 语言 |
|---|---|---|---|
| SkyPile | HF `Skywork/SkyPile-150B` | 21 jsonl(全量 437) | 中 |
| CCI3-HQ | HF `BAAI/CCI3-HQ` | 5 jsonl | 中 |
| Chinese-FineWeb-Edu | HF `opencsg/Fineweb-Edu-Chinese-V2.2` | `4_5/` 前 500(全量 9745) | 中 |
| finewiki zh | HF `HuggingFaceFW/finewiki` | `data/zhwiki/` 全 5 | 中 |
| finewiki en | HF `HuggingFaceFW/finewiki` | `data/enwiki/` 全 15 | 英 |
| PeopleDaily | HF `Papersnake/people_daily_news` | 全量(1946–2025) | 中 |
| CnnDailyMail | HF `abisee/cnn_dailymail` | 全量 | 英 |

关键是那个 `n_parts` 字段 —— **按文件个数取,而不是整个仓库拉下来**。
默认值是已发布模型的实跑用量(合计约 79 GB),不是拍脑袋定的。

`snapshot_download` 不带参数会把几百 GB 全量拖下来。这几个源里 SkyPile 全量
665 GB,Chinese-FineWeb-Edu 有 9745 个 parquet —— 拉全量既没必要也放不下。

要加量就改 `source.py` 里的 `n_parts`,重跑 `make -C data download-pretrain`,
已下载的会跳过。

**中英是按文档数配比的,不是按 token 数** —— 原因见
[`docs/WHY.md`](../docs/WHY.md#中英语料按文档数配比不是按-token-数)。

## 下游任务数据

| 源 | 出处 | 提供 |
|---|---|---|
| PD-1998 | GitHub `chenhui-bupt/PeopleDaily1998` | 分词 / 词性 / 实体标注 |
| CTCDataset | GitHub `zejunwang1/CTCDataset` | CSC 主力:CCTC / CTC2021 / MCSCSet / ECSpell / lemon / cscd-ns / sighan / yacsc / Wang271k |
| MCSCSet | GitHub `yzhihao/MCSCSet` | 医疗 CSC 专家标注 199,763 条 |
| Wang271K | HF `shibing624/CSC` | CSC 标准训练集 |
| chinese_text_correction | HF `shibing624/chinese_text_correction` | 14 个 tsv |
| SIGHAN-15 测试集 | GitHub `shibing624/pycorrector` | **官方 707 条**,CSC 的权威基准 |

两个会踩的坑:

- PD-1998 的压缩包名叫 `199801.zip`,里面其实是 **199801.txt ~ 199806.txt
  六个月**。train = 前 5 月(102,739 句),dev = 199806(21,143 句)
- **SIGHAN-15 测试集有两个版本。** 用的是 707 条那一版(pycorrector 口径,
  也是文献里通行的)。CTCDataset 里另有一个 `sighan15_test.jsonl` 是 1100 条,
  **不能混用** —— 混了数字就没法跟任何人比

还有一个容易看错的地方:`process.py` 和 `process_cws.py` 用的是**两份不同的
人民日报**。前者是 1946–2025 全文(喂预训练),后者是 1998 年 1–6 月的 PFR
标注语料(喂 MT 微调)。同名不同物。

## CSC 的训练配方是反推出来的

这段值得单独说,因为它是**真实项目里常见、教程里从不提**的一种情况:
模型已经发布了,而当初到底用哪些数据训的,没记下来。

线索只有原始训练日志里的一个数字:**249,978 对**。做法是穷举各个源的组合,
去凑这个数 —— 最后定位到 `wang271k/train` + `sighan 13/14/15 的 train`,
去重后 249,975 对,配方吻合。

现在这个配方叫 `sighan_wang271k`,是 `make -C data process-csc` 的默认值。

另有一个 `all` 配方扫全部源得 82.6 万对(`make -C data process-csc-all`),
数据多 3.3 倍、训练时间也是 3.3 倍,效果**还没验证过**。它的三条合并规则:

1. 扫全部源,含两个易漏的 `.jsonl.gz`(`CTC2021/train_large_v2` 10.2 万对、
   `Wang271k/data` 26.8 万对)
2. **等长过滤** `len(src) == len(tgt)`。CSC 是同音 / 形似字的等长替换,
   含增删的语法错误进来只会干扰
3. 整文件排除 `NLPCC2023/grammar/`(HSK + MuCGEC,语法纠错)、`val_bak`、
   以及 SIGHAN-15 测试集(**绝不能混进训练**)

PD-1998 那边是完全可复现的:从 GitHub 重下 → `process_cws.py`,产出的 6 份
jsonl 与已发布 MT 模型的实际训练输入**逐字节相同**。

## 一个环境上的矛盾

HF 走 hf-mirror **必须清代理**,GitHub 反过来**必须走代理**。

`download.py` 在模块加载时把代理配置存下来,只在 GitHub 那条路径恢复。
如果你在别的网络环境下,`HF_ENDPOINT=` 置空可以走 HF 官方源。

## 文件

```
source.py       数据源注册表 + 路径解析(所有脚本共用)
download.py     四种源:hf / hf-snapshot / github-file / github-repo
process.py      PeopleDaily / CnnDailyMail → documents.txt(一行一篇)
process_cws.py  PD-1998 PFR 标注 → cws / pos / ner 三种 jsonl
process_csc.py  CSC 各源 → (错句, 正句) pkl
```

数据落在 gitignore 的 `data/downloads/` 和 `data/derived/`,可以用
`BERTC_DATA_ROOT` / `BERTC_DERIVED_ROOT` 改到别的盘。

`process_cws.py` 一次产出 cws / pos / ner 三种标注 —— MT 是这三个任务的联合
训练,共用一份解析。
