# BERTc

字级中文 Modern BERT,从零预训练,纯 PyTorch 实现。

包含从下载语料到发布上 Hugging Face 的完整流程:数据获取与加工、词表与预编码、
预训练、微调、导出发布。模型与训练代码(`src/`)只依赖 torch —— CRF、优化器、
LR 调度、整词掩码、safetensors 读写都是自己实现的。

已发布六个 HF 仓库,两个规格各三个任务。

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

## 模型

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
仓库里另有两个交互式脚本:

```bash
python -m save.cws        # 分词 + 词性 + 实体
python -m save.csc        # 拼写纠错
```

## 表现

分词 + 词性 + 实体(PD-1998,FGM 5 epoch):

| 模型 | 参数 | 分词 | 词性 | 实体 | joint |
|---|---|---|---|---|---|
| **BERTc-315M + FGM** | 315M | 0.9840 | **0.9800** | 0.9660 | **1.4712** |
| BERTc-165M + FGM | 165M | 0.9836 | 0.9753 | 0.9632 | 1.4689 |
| MacBERT-large | 326M | **0.9856** | 0.9629 | **0.9664** | 1.4677 |
| RoBERTa-wwm-ext | 102M | 0.9828 | 0.9562 | 0.9629 | 1.4623 |

joint = 分词 F1 + 0.3 × 词性准确率 + 0.2 × 实体 F1。指标在 dev 前 2000 句上测
(与训练时选 best.pt 的口径一致);全量 21,143 句上是 1.4646。

拼写纠错(SIGHAN-15 官方 707 条,pycorrector 口径):

| 模型 | 参数 | F1 | P | R |
|---|---|---|---|---|
| **BERTc-315M** | 315M | **0.8346** | 0.9396 | 0.7507 |
| MacBERT4CSC | 110M | 0.8314 | 0.9274 | 0.7534 |
| MacBERT-large | 326M | 0.8309 | 0.9302 | 0.7507 |
| BERTc-165M | 165M | 0.8308 | 0.9516 | 0.7373 |

消融见 [`save/sota/README.md`](save/sota/README.md)。

## 架构

24L / 1024H / 2752I / 16 heads(315M),或 12L 的 165M 版。词表 12536 ——
字级:中文一字一 piece,英文走 BPE 子词。

- ScaledSinusoidal 位置编码(Cramming):只在嵌入层算一次,attention 里零开销
- GeGLU 前馈、LayerNorm 无 bias、pre-norm 且首层跳过、全部 Linear 无 bias
- 简化 MLM 头:`logits = h @ embed.weightᵀ`,没有 Dense / LN / GeLU / bias
- Megatron 初始化(残差支路 ×1/√2L);全程无 dropout;输入输出嵌入绑定

预训练:StableAdamW(β₂=0.95)+ Damped Cosine LR(8e-4 → 8e-5)+ 固定 15%
整词掩码 + `flex_attention` 跨文档隔离。有效 batch 4096,8500 步 ≈ 17.4B token。

每条选择的理由见 [`docs/WHY.md`](docs/WHY.md)。

## 仓库结构

四层,按数据流切:

```
data/       下载原始语料与标注数据,加工成统一格式   make -C data
prepare/    词表、预编码、语料切块、调训练           make -C prepare
src/        模型定义 + 训练循环
save/       导出 HF 发布包、上传、交互式脚本
```

两条分层约束:

- **`src/` 只依赖 torch。** CRF、StableAdamW、LR 调度、整词掩码、safetensors
  读取都是自己实现的,memmap 用 `torch.from_file`
- **`src/` 不碰文本。** 分词、字→id、标签构造全在 `prepare/`,`src/` 只读
  预编码好的 id。所以 PieceTokenizer 不是 `src/` 的依赖

每层一个 README 说明这层在做什么。另有 `deps/`(clone 的 C++ 依赖)、
`docs/`、`test/`。

## 自己训

语料、标注数据、词表、C++ 依赖全部从 Hugging Face 和 GitHub 获取,
不需要任何本地既有文件。

```bash
make -C prepare deps        # clone + 编译 PieceTokenizer / Wapic
make -C data status         # 数据源下了没
make -C prepare status      # 每一步产物在不在
```

| | 时间 | 从哪开始 | 教程 |
|---|---|---|---|
| 微调 | 几小时 | HF 上的骨干 | [`docs/FINETUNE.md`](docs/FINETUNE.md) |
| 预训练 | 2–4 天 + 8 小时准备 | 随机初始化 | [`docs/PRETRAIN.md`](docs/PRETRAIN.md) |

建议先跑微调 —— 几小时就有反馈,而且能验证整条链路(数据、词表、模型、评测)
是通的。

### 复现情况

| | 记录 | 复现 | |
|---|---|---|---|
| 评测(用已发布 checkpoint) | MT 1.4712 / CSC 0.8346 | 一位不差 | ✓ |
| MT 重新训练 | 1.4712 | 1.4705 | −0.0007 |
| CSC 重新训练 | 0.8346 | 0.8316 | −0.0030 |

`python test/test_reproduce_sota.py` 跑第一行。后两行是重训,差距小于训练自身的
波动(CSC 单轮实测波动可达 ±0.02)。

预训练逐位复现做不到 —— Wapic 的分词在 2026-07 变过,词边界与当初不同
(PD-1998 dev 上 26.1% 的句子至少有一处差异)。词表、架构、配方都没变,能复现
的是同等水平的模型,不是同一个模型。

## 文档

| | |
|---|---|
| [`docs/PRETRAIN.md`](docs/PRETRAIN.md) | 预训练全流程,含每步耗时和磁盘占用 |
| [`docs/FINETUNE.md`](docs/FINETUNE.md) | 微调全流程 |
| [`docs/WHY.md`](docs/WHY.md) | 各处设计选择的理由,以及改错了不报错的地方 |
| [`docs/TODO.md`](docs/TODO.md) | 想做没做的实验 |
| [`src/README.md`](src/README.md) | `src/` 各模块的导读 |

## 环境

Python 3.14 + torch 2.11,单张 RTX 4090(24GB,bf16),没有多卡代码路径。

两个 C++ 依赖由 `make -C prepare deps` 自动 clone 并编译(需要 `cmake` 和
C++17 编译器):

- [PieceTokenizer](https://github.com/Ismantic/PieceTokenizer) —— 字级分词器,
  同时提供词表
- [Wapic](https://github.com/Ismantic/Wapic) —— CRF 中文分词器,标整词掩码的
  词边界,只有预训练用得到

## 许可

Apache-2.0。训练语料各自的许可见对应数据集卡。
