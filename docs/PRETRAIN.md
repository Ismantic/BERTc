# 从零预训练 BERTc

在自己的机器上把 BERTc 从随机初始化训出来。单张 4090,**mid(165M)约 2 天,
large(315M)约 4 天**,外加准备阶段约 8 小时。

只想在下游任务上微调的话不用看这篇 —— 直接拉现成骨干,见
[`FINETUNE.md`](FINETUNE.md),几小时就能出结果。

- [先算账](#先算账)
- [准备依赖](#准备依赖)
- [下载语料](#下载语料)
- [加工成纯文本](#加工成纯文本)
- [切成定长块](#切成定长块)
- [开始训练](#开始训练)
- [训练配方为什么是这样](#训练配方为什么是这样)
- [盯什么](#盯什么)
- [常见问题](#常见问题)

## 先算账

| 阶段 | 时间 | 磁盘 |
|---|---|---|
| 下载语料 | 视带宽,79 GB | 79 GB |
| 加工 documents.txt | 约 20 分钟 | +9 GB |
| 切成定长块 | **约 4.6 小时**(14 核) | **+168 GB** |
| 预训练 mid | 约 2 天 | +2 GB |
| 预训练 large | 约 4 天 | +4 GB |

**峰值要 260 GB 空余**。语料那 168 GB 是大头:20B token 存成
`.pt` 80 GB + `.wid` 80 GB + `.seg` 20 GB。

显存:mid 用 batch 32、large 用 batch 16,24 GB 卡都跑得下(有效 batch 都是
4096,靠梯度累积凑)。

## 准备依赖

```bash
make -C prepare deps
```

clone 并编译两个 C++ 依赖:

- **PieceTokenizer** —— 字级分词器,同时提供词表(`BERTc-Tokenizer.pt`)
- **Wapic** —— CRF 中文分词器,用来标整词掩码的词边界

需要 `cmake`、C++17 编译器、`git`。装完会自动比对词表行为,输出 `校验通过`
才算成功。**这一步不能跳** —— 编码行为一旦变了,词表就和已发布模型对不上,
而代码不会报错。

## 下载语料

```bash
make -C data download-pretrain
make -C data status              # 看下齐了没
```

七个源,全部来自 Hugging Face:

| 源 | 用量 | 语言 |
|---|---|---|
| SkyPile | 21 个 jsonl(全量 437 个 / 665 GB) | 中 |
| CCI3-HQ | 5 个 jsonl | 中 |
| Chinese-FineWeb-Edu | `4_5/` 前 500 个 parquet(全量 9745) | 中 |
| FineWiki zhwiki | 全 5 个 parquet | 中 |
| PeopleDaily | 全量(1946–2025) | 中 |
| FineWiki enwiki | 全 15 个 parquet | 英 |
| CnnDailyMail | 全量 | 英 |

用量默认值是 v4-Large 实跑时的量,不是随手定的 —— 这几个源在 HF 上都是几百
GB 全量,**不要无参数 `snapshot_download`**。想加量改
`data/source.py` 里的 `n_parts`。

下载走 `hf-mirror.com` 且**必须清代理**(脚本内部已处理)。走官方源:
`make -C data download-pretrain HF_ENDPOINT=`。

## 加工成纯文本

```bash
make -C data process-docs
```

只有 PeopleDaily 和 CnnDailyMail 需要这一步 —— 它们是 jsonl.gz / parquet,
要转成「一行一篇、title 和 content 合并」的纯文本。其余四个源本来就是
parquet / jsonl,切块时直接读。

产出 `PeopleDaily.documents.txt`(5.4 GB)和 `CnnDailyMail.documents.txt`
(3.8 GB,93.6 万篇)。约 20 分钟。

## 切成定长块

```bash
make -C prepare corpus
```

这一步把七个源的文本变成模型能直接喂的定长块。**约 4.6 小时,产出 168 GB。**

做三件事:

```
1. 编码       文本 → token id
2. 标词边界   Wapic 切词 → word id(同一个词的字共享一个 id)
3. 打包       拼接 → 切成 512 长的块,记录每个 token 属于哪篇文档
```

产出三个平行文件,逐 token 对齐:

| 文件 | 类型 | 用途 |
|---|---|---|
| `v4.pt` | int32 | 模型输入 |
| `v4.pt.wid` | int32 | **整词掩码** —— 遮"沁县"要两个字一起遮 |
| `v4.pt.seg` | uint8 | **跨文档隔离** —— 一个块里可能有好几篇文章,attention 不该跨过去 |

实测产出:

```
39,065,008 chunk × 512 = 20.00B token,来自 15,533,176 篇文档
中英按文档数 18:10 加权轮询,token 比约 6:4
```

**中英是按文档数配比的,不是按 token 数** —— 英文文档平均比中文长得多,
18:10 的文档比才换来 6:4 的 token 比。

`corpus` 是 Make 的**文件目标**,产物在就不重跑。要强制重来先删
`prepare/corpus/`。

### 验证产出

切完检查一下,免得白训两天:

```python
import sys; sys.path.insert(0, ".")
from src.data import PackedMLMDataset
ds = PackedMLMDataset("prepare/corpus/v4.pt",
                      word_ids_path="prepare/corpus/v4.pt.wid",
                      seg_ids_path="prepare/corpus/v4.pt.seg")
ids, wid, seg = ds[0]
print(len(ds), ds.seq_len)
print(wid[:16])   # 同一个词的字要共享 id,且非降序
print(seg[:16])   # chunk 内从 0 起按出现顺序编号
```

**最该看的是词长分布**。正常的中文分词大致是:

```
1 字 54%   2 字 38%   3 字 5.7%   ≥4 字 2.6%
```

如果**全是 1**,说明 `word_starts` 没生效、整词掩码退化成了逐字掩码 ——
而训练照样跑、loss 照样降,不会有任何报错。这是最值得单独确认的一项。

## 开始训练

```bash
make -C prepare pretrain SIZE=mid      # 165M,约 2 天
make -C prepare pretrain SIZE=large    # 315M,约 4 天
```

两个规格的**配方完全相同**,只差三处:

| | mid | large |
|---|---|---|
| 层数 | 12L | 24L |
| batch × 累积 | 32 × 128 | 16 × 256 |
| 有效 batch | 4096 | 4096 |

隐层都是 1024,中间层 2752,16 个头。

日志长这样:

```
step 170/8500 | loss 5.7886 | mlm_acc 0.1595 | lr 0.000267 | mlm_p 0.150 | accum 52
```

每 1500 步存一次 checkpoint(共 6 个),可以用 `--inline_eval_cmd` 在每次存盘
后自动跑下游探针。

## 训练配方为什么是这样

```
StableAdamW  β=(0.9, 0.95)  wd=0.01  eps=1e-6
Damped Cosine LR  8e-4 → 8e-5,warmup 510 步(6%)
固定 15% 整词掩码
梯度累积从 1 线性爬到峰值(前 5% 步)
grad clip 0.5
8500 步 × 4096 × 512 ≈ 17.4B token
```

| 选择 | 理由 |
|---|---|
| **StableAdamW** 而非 AdamW | 每个张量按 RMS 裁剪有效学习率,bf16 下明显更稳。bias correction 折进 beta,不单独修正 |
| **固定 15% 掩码** | 动态 curriculum(15→30→15)在 v3 试过,v4-Mid 的消融显示固定 15% 就拿到了双 SOTA。代码还在(`src/masking.py`),默认关 |
| **梯度累积爬升** | 等价于 batch size warmup(Cramming 和 ModernBERT 都用)。前期小 batch 多更新,后期大 batch 稳梯度 |
| **grad clip 0.5** | 比常见的 1.0 更紧。深模型早期容易出尖峰 |
| **warmup 6%** | Cramming 的推荐值 |
| **不用 EMA** | v3 用过,v4 关掉了 —— 对 8500 步这个长度收益不明显,还多占一份显存 |
| **跨文档隔离** | 一个 512 块里常有多篇文章。不隔离的话模型会学到跨文章的伪关联 |

架构侧的选择见 [`README.md`](../README.md#架构与配方)。

## 盯什么

**头 200 步**就能判断链路对不对:

```
✓ 参数量对得上           mid 164.6M / large 316.5M
✓ 优化器分组             no-decay 只有两万多(bias 和 norm),不是几百万
✓ mlm_p 恒定 0.150       动态 curriculum 确实关了
✓ accum 在爬             4 → 7 → 10 …… 爬向 128 或 256
✓ lr 在爬                线性升到 8e-4,510 步到顶
✓ loss 下降              11.9 → 5.8 左右(200 步内)
```

**之后主要看两个数**:

- `loss` 应该持续下降,中途有小幅波动正常
- `mlm_acc` 是掩码位置的预测准确率,比 loss 直观。头几百步能到 0.15 左右

发散的信号是 loss 突然跳到 10 以上或变 NaN。真发生了就降 lr 重来 ——
`--init_from_ckpt` 配 `--resume_step` 能从最近的 checkpoint 接着跑,
LR / 掩码率 / 累积调度会自动续上(优化器动量不恢复,会重建)。

## 常见问题

**磁盘不够**

`prepare/corpus/` 那 168 GB 是硬需求。可以调小 `--target_tokens`:

```bash
make -C prepare corpus TARGET_TOKENS=10000000000   # 10B,约 84 GB
```

同时要把训练步数减半(`src/pretrain.py --max_steps`),否则会重复读同一份
数据。语料量和步数的关系:`步数 × 有效batch × 512 ≈ token 数`。

**`CUDA out of memory`**

调小 `--batch_size` 并按比例调大 `--gradient_accumulation_steps`,保持乘积
是 4096。比如 large 在显存更小的卡上可以用 8 × 512。

**切块特别慢**

`--num_workers` 默认 14。如果同时在训练别的东西,CPU 会被抢 —— 实测
10 workers 和 14 workers 都是 1.2M token/秒,瓶颈在磁盘不在 CPU。

**`torch.from_file` 报 Cannot allocate memory**

必须用 `shared=True`(`src/data.py` 里已经是了)。`shared=False` 要一份私有
副本,80 GB 语料直接失败。

**训练中途想换语料**

不行。`.wid` 和 `.seg` 与 `.pt` 是逐 token 对齐的,换一个就全对不上。要换
就重新切块。

**想复现已发布的模型**

已发布的 `Ismantic/BERTc-165M` / `BERTc-315M` 用的是 17.65B token 的语料,
且分词器是 2026-07 之前的版本 —— 当前 Wapic 的切词跟那时不同(PD-1998 dev
上 26.1% 的句子至少有一处词边界差异),所以**逐位复现做不到**。词表没变,
架构没变,配方没变,能复现的是同等水平的模型,不是同一个模型。
