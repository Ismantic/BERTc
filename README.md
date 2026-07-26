# BERTc

**一份能读完的、从零训练中文 BERT 的完整实现。**

从下载语料开始,到发布上 Hugging Face 结束,中间每一步都是可运行的代码,
而不是伪代码或者「调用某个库的某个函数」。

核心约束:**模型和训练代码只依赖 PyTorch**。

| | 行数 | 通常的做法 |
|---|---|---|
| [`src/model.py`](src/model.py) | 344 | `from transformers import BertModel` |
| [`src/pretrain.py`](src/pretrain.py) | 337 | `Trainer(...)` |
| [`src/data.py`](src/data.py) | 250 | `datasets.load_dataset()` |
| [`src/evaluate.py`](src/evaluate.py) | 218 | `seqeval` |
| [`src/crf.py`](src/crf.py) | 192 | `pip install torchcrf` |
| [`src/optim.py`](src/optim.py) | 152 | `pip install optimi` |
| [`src/masking.py`](src/masking.py) | 109 | `DataCollatorForWholeWordMask` |
| [`src/checkpoint.py`](src/checkpoint.py) | 85 | `pip install safetensors` |
| **合计** | **2,159** | |

两千行,一个周末能读完。读完之后你知道的不只是「BERT 有 12 层 Transformer」,
而是**梯度累积怎么和学习率调度对齐**、**整词掩码怎么在打包语料里实现**、
**CRF 的前向算法为什么能在对数空间里做**、**safetensors 的二进制格式是哪三段**。

它不是教学玩具 —— 按这份代码训出来的模型在 PD-1998 和 SIGHAN-15 上
[超过了 MacBERT-large](#它是能用的)。这一点很重要:**教学代码只有真能跑出
可用的模型,里面的每个选择才有说服力。**

## 从哪开始读

按数据流走一遍,每一站解决一个具体问题:

| | 读什么 | 解决的问题 |
|---|---|---|
| 1 | [`data/`](data/) | 语料从哪来,几百 GB 的公开数据集怎么按需取用 |
| 2 | [`prepare/`](prepare/) | 文本怎么变成张量:字→id、词边界、定长打包 |
| 3 | [`src/model.py`](src/model.py) | Transformer 编码器本体,以及 2026 年的架构选择 |
| 4 | [`src/pretrain.py`](src/pretrain.py) | 预训练循环:掩码、累积、调度、存盘 |
| 5 | [`src/finetune_mt.py`](src/finetune_mt.py) | 微调:三个任务共享一个骨干 |
| 6 | [`save/`](save/) | 训完之后:导出、自包含推理代码、发布 |

想动手就直接跳到 [自己训一个](#自己训一个)。想先看模型能干什么,
[直接用](#直接用) 里有下载即跑的例子。

配套文档:

- [`docs/PRETRAIN.md`](docs/PRETRAIN.md) —— 预训练全流程,含每步耗时和磁盘占用
- [`docs/FINETUNE.md`](docs/FINETUNE.md) —— 微调全流程,几小时能出结果
- [`docs/WHY.md`](docs/WHY.md) —— **踩过才知道的部分**:那些不写出来就会
  静默出错的地方

## 直接用

```python
from mt_model import BERTcForMT
BERTcForMT.from_pretrained(".").predict("中国科学院计算技术研究所在北京")
# words: 中国 / 科学院 / 计算技术 / 研究所 / 在 / 北京
# pos:   ns  n  n  n  p  ns
# ner:   [机构名] 中国科学院计算技术研究所   [地名] 北京
```

```python
from csc_model import BERTcForCSC
BERTcForCSC.from_pretrained(".").correct("他平时喜欢锻练身体")
# 他平时喜欢锻炼身体
```

| Hugging Face | 参数 | 任务 |
|---|---|---|
| [`Ismantic/BERTc-315M`](https://huggingface.co/Ismantic/BERTc-315M) · [`-165M`](https://huggingface.co/Ismantic/BERTc-165M) | 315M / 165M | 骨干,可继续微调 |
| [`Ismantic/BERTc-315M-MT`](https://huggingface.co/Ismantic/BERTc-315M-MT) · [`-165M-MT`](https://huggingface.co/Ismantic/BERTc-165M-MT) | — | 分词 + 词性 + 实体 |
| [`Ismantic/BERTc-315M-CSC`](https://huggingface.co/Ismantic/BERTc-315M-CSC) · [`-165M-CSC`](https://huggingface.co/Ismantic/BERTc-165M-CSC) | — | 拼写纠错 |

```bash
huggingface-cli download Ismantic/BERTc-315M-MT --local-dir BERTc-MT
pip install git+https://github.com/Ismantic/PieceTokenizer
cd BERTc-MT && python example_decode.py
```

每个仓库自带推理代码和示例,除 PyTorch 和 PieceTokenizer 外无其他依赖。
仓库里还有两个交互式脚本:

```bash
python -m save.cws        # 分词 + 词性 + 实体
python -m save.csc        # 拼写纠错,会告诉你"模型知道这里有错但选不出字"
```

## 自己训一个

**只依赖 Hugging Face 和 GitHub** —— 语料、标注数据、词表、C++ 依赖全部从
公网获取,不需要任何本地既有文件。这条是实测过的:把本地数据全部删掉之后
重新拉取重建,微调结果复现到位。

```bash
make -C prepare deps        # clone + 编译 PieceTokenizer / Wapic
make -C data status         # 数据源下了没
make -C prepare status      # 每一步产物在不在
```

| | 时间 | 从哪开始 | 教程 |
|---|---|---|---|
| **微调** | 几小时 | HF 上的骨干 | [`docs/FINETUNE.md`](docs/FINETUNE.md) |
| **预训练** | 2–4 天 + 8 小时准备 | 随机初始化 | [`docs/PRETRAIN.md`](docs/PRETRAIN.md) |

先跑微调 —— 几小时就有反馈,而且能验证整条链路(数据、词表、模型、评测)
是通的。确认无误再投两天去跑预训练。

单张 RTX 4090(24GB,bf16),**没有多卡代码路径**。这是刻意的:DDP 会让
训练循环里多出一层包装,而这一层跟「BERT 怎么训」无关。

## 它是能用的

### 分词 + 词性 + 实体(PD-1998,FGM 5 epoch)

| 模型 | 参数 | 分词 | 词性 | 实体 | joint |
|---|---|---|---|---|---|
| **BERTc-315M + FGM** | 315M | 0.9840 | **0.9800** | 0.9660 | **1.4712** |
| BERTc-165M + FGM | 165M | 0.9836 | 0.9753 | 0.9632 | 1.4689 |
| MacBERT-large | 326M | **0.9856** | 0.9629 | **0.9664** | 1.4677 |
| RoBERTa-wwm-ext | 102M | 0.9828 | 0.9562 | 0.9629 | 1.4623 |

joint = 分词 F1 + 0.3 × 词性准确率 + 0.2 × 实体 F1。指标在 dev 前 2000 句上测
(与训练时选 best.pt 的口径一致);全量 21,143 句上是 1.4646。

### 拼写纠错(SIGHAN-15 官方 707 条,pycorrector 口径)

| 模型 | 参数 | F1 | P | R |
|---|---|---|---|---|
| **BERTc-315M** | 315M | **0.8346** | 0.9396 | 0.7507 |
| MacBERT4CSC | 110M | 0.8314 | 0.9274 | 0.7534 |
| MacBERT-large | 326M | 0.8309 | 0.9302 | 0.7507 |
| BERTc-165M | 165M | 0.8308 | 0.9516 | 0.7373 |

消融见 [`save/sota/README.md`](save/sota/README.md)。

**这些数字的用处是当证据,不是当卖点。** 排行榜上比这高的模型有的是。它证明的
是:这两千行代码里没有藏着「其实少了一步」的问题 —— 少一步,数字就掉下来了。

### 复现情况

| | 记录 | 复现 | |
|---|---|---|---|
| 评测(用已发布 checkpoint) | MT 1.4712 / CSC 0.8346 | 一位不差 | ✓ |
| MT 重新训练 | 1.4712 | 1.4705 | −0.0007 |
| CSC 重新训练 | 0.8346 | 0.8316 | −0.0030 |

`python test/test_reproduce_sota.py` 跑第一行。后两行是重训,差距小于训练自身的
波动(CSC 单轮实测波动可达 ±0.02)。

预训练**逐位复现做不到** —— Wapic 的分词在 2026-07 变过,词边界与当初不同
(PD-1998 dev 上 26.1% 的句子至少有一处差异)。词表、架构、配方都没变,能复现
的是同等水平的模型,不是同一个模型。

## 架构

24L / 1024H / 2752I / 16 heads(315M),或 12L 的 165M 版。词表 12536 ——
**字级**:中文一字一 piece,英文走 BPE 子词。

- **ScaledSinusoidal 位置编码**(Cramming):只在嵌入层算一次,attention 里
  零开销。短序列上比 RoPE 划算 —— RoPE 的收益被 5–10% 速度损失抵消
- GeGLU 前馈、LayerNorm 无 bias、pre-norm 且首层跳过、全部 Linear 无 bias
- **简化 MLM 头**:`logits = h @ embed.weightᵀ`,没有 Dense / LN / GeLU / bias
- Megatron 初始化(残差支路 ×1/√2L);全程无 dropout

预训练:StableAdamW(β₂=0.95)+ Damped Cosine LR(8e-4 → 8e-5)+ 固定 15%
整词掩码 + `flex_attention` 跨文档隔离。有效 batch 4096,8500 步 ≈ 17.4B token。

每一条为什么这么选,见 [`docs/WHY.md`](docs/WHY.md)。

## 仓库结构

四层,按数据流切。每层一个 README,讲这层在解决什么问题:

```
data/       下载原始语料与标注数据,加工成统一格式   make -C data
prepare/    编排:词表、预编码、语料切块、调训练     make -C prepare
src/        模型定义 + 训练循环(只依赖 torch)
save/       导出 HF 发布包、上传、交互式体验脚本
```

两条约束,是这个项目能当教材的原因,改代码时别破坏:

- **`src/` 只依赖 torch**。CRF、StableAdamW、LR 调度、整词掩码、safetensors
  读取都是自己实现的,memmap 用 `torch.from_file`。装个 transformers 会让
  这两千行缩到三百行 —— 也就没什么可读的了。
- **`src/` 不碰文本**。分词、字→id、标签构造全在 `prepare/`,`src/` 只读
  预编码好的 id。所以 PieceTokenizer 也不是 `src/` 的依赖,而边界一清晰,
  「张量长什么样」就成了可以单独讲的一件事。

另有 `deps/`(clone 的 C++ 依赖)、`docs/`、`test/`。

## 环境

Python 3.14 + torch 2.11,单张 24GB 卡。两个 C++ 依赖由
`make -C prepare deps` 自动 clone 并编译(需要 `cmake` 和 C++17 编译器):

- [PieceTokenizer](https://github.com/Ismantic/PieceTokenizer) —— 字级分词器,
  同时提供词表
- [Wapic](https://github.com/Ismantic/Wapic) —— CRF 中文分词器,标整词掩码的
  词边界(只有预训练用得到)

这两个是 C++ 写的,没法用纯 torch 替代,也不该替代 —— 它们解决的是中文
分词问题,不是深度学习问题。

## 许可

Apache-2.0。训练语料各自的许可见对应数据集卡。
