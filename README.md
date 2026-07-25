# BERTc

字级中文 Modern BERT,**从零预训练**。纯 PyTorch 实现,不依赖 transformers。

目标是在 PD-1998 的分词 / 词性 / 命名实体和 SIGHAN-15 拼写纠错上,全面超过
`hfl/chinese-macbert-large`(326M)和 `hfl/chinese-roberta-wwm-ext`(102M)。

**整个仓库只依赖 Hugging Face 和 GitHub** —— 语料、标注数据、词表、分词器
全部可以从公网重新获取,不需要任何本地既有文件。

## 已发布模型

| Hugging Face | 参数 | 说明 |
|---|---|---|
| [`Ismantic/BERTc-315M`](https://huggingface.co/Ismantic/BERTc-315M) | 315M | 骨干,24L/1024H/2752I/16h |
| [`Ismantic/BERTc-165M`](https://huggingface.co/Ismantic/BERTc-165M) | 165M | 骨干,12L/1024H/2752I/16h |
| `Ismantic/BERTc-315M-MT` / `-165M-MT` | — | 分词 + 词性 + 实体 |
| `Ismantic/BERTc-315M-CSC` / `-165M-CSC` | — | 拼写纠错 |

想在自己的任务上微调,看 [`docs/FINETUNE.md`](docs/FINETUNE.md) —— 从 HF 拉骨干
开始,不需要预训练。

## 表现

### 分词 + 词性 + 实体(PD-1998 联合微调,FGM 5 epoch)

| 模型 | 参数 | 分词 | 词性 | 实体 | joint score |
|---|---|---|---|---|---|
| **BERTc-315M + FGM** | **315M** | **0.9840** | **0.9800** | 0.9660 | **1.4712** |
| BERTc-165M + FGM | 165M | 0.9836 | 0.9753 | 0.9632 | 1.4689 |
| BERTc v6.5 + FGM(旧) | 165M | 0.9807 | 0.9719 | 0.9568 | 1.4636 |
| MacBERT-large(3ep 无 FGM) | 326M | **0.9856** | 0.9629 | **0.9664** | 1.4677 |
| RoBERTa-wwm-ext(3ep 无 FGM) | 102M | 0.9828 | 0.9562 | 0.9629 | 1.4623 |

score = 分词 F1 + 0.3 × 词性准确率 + 0.2 × 实体 F1。综合超 MacBERT-large
**+0.0035**,词性单项 **+1.71 pp** 是最大来源;分词落后 0.0016,实体基本持平。

指标在 dev 集**前 2000 句**上测(与训练时选 best.pt 的口径一致)。
全量 21,143 句上是 0.9790 / 0.9787 / 0.9597 / 1.4646。

### 拼写纠错(SIGHAN-15 官方 707 条,pycorrector 口径)

| 模型 | 参数 | 配方 | F1 | P | R |
|---|---|---|---|---|---|
| **BERTc-315M** | **315M** | b32 lr3e-5 10ep,cor 头绑词嵌入 | **0.8346** | 0.9396 | 0.7507 |
| MacBERT4CSC(开源基线) | 110M | — | 0.8314 | 0.9274 | 0.7534 |
| MacBERT-large | 326M | b32 lr2e-5 5ep | 0.8309 | 0.9302 | 0.7507 |
| BERTc-165M | 165M | b64 lr5e-5 5ep | 0.8308 | 0.9516 | 0.7373 |
| RoBERTa-wwm | 110M | b32 lr5e-5 10ep | 0.7970 | 0.8907 | 0.7212 |

8 组消融见 [`save/sota/README.md`](save/sota/README.md)。

## 架构与配方

**骨干**(`src/model.py`):

- 24L / 1024H / 2752I / 16 heads,词表 12536(字级 SentencePiece:中文一字一
  piece,英文走 BPE 子词,空格归一成 `▁`)
- **ScaledSinusoidal 位置编码**(Cramming):只在嵌入层算一次,attention 里零开销。
  短序列上比 RoPE 划算 —— RoPE 的收益被 5-10% 速度损失抵消
- GeGLU 前馈、LayerNorm 无 bias、pre-norm + 首层跳过 pre-norm
- **简化 MLM 头**:`logits = h @ embed.weightᵀ`,没有 Dense / LN / GeLU / bias
- Megatron 初始化(残差支路 ×1/√2L);全程无 dropout

**预训练**(`src/pretrain.py`):StableAdamW(β₂=0.95)+ Damped Cosine LR
(8e-4 → 8e-5)+ 固定 15% 整词掩码 + `flex_attention` 跨文档隔离。
有效 batch 4096,8500 步 ≈ 17.4B token。grad_accum 在前 5% 步从 1 线性升到 256。

**三个来之不易的结论**:

- **词性 loss 要加权**。joint loss = cws + α·pos + β·ner,α 用默认 0.5 时词性
  只占 1.6% 的梯度,学不动;**α=2** 让词性 +0.02 追平 MacBERT。再加 FGM ε=1.0
  和 5 epoch 才有现在的成绩。
- **CSC 的纠错头必须与词嵌入绑权重**。预训练的 MLM 头就是 `h @ embed.weightᵀ`,
  预训完 h 已经和嵌入空间对齐;换一个随机初始化的 Linear 会废掉这层对齐,
  F1 差 0.05。
- **Large 模型 5 epoch 严重欠训**,10 epoch 才到位。

## 仓库结构

四层,按数据流切:

```
data/       下载原始语料与标注数据,加工成统一格式
src/        模型定义 + 训练循环。**只依赖 torch**
prepare/    编排层:词表、预编码、语料切块、调训练
save/       导出 HF 发布包并上传
```

两条关键约束:

- **`src/` 只依赖 torch**。CRF、StableAdamW、LR 调度都是自己实现的(替掉
  torchcrf / optimi / transformers),memmap 用 `torch.from_file` 而非 numpy。
- **`src/` 不碰文本**。分词、字→id、标签构造全在 `prepare/` 完成,`src/` 只读
  预编码好的 id,所以 PieceTokenizer 也不是它的依赖。

另有 `deps/`(clone 下来的 C++ 依赖)、`docs/`、`test/`。

## 上手

```bash
bash prepare/install_deps.sh        # clone + 编译 PieceTokenizer / Wapic
python data/download.py --list      # 看数据源状态
```

三条链:

```bash
# 下游数据 → 预编码
python data/download.py --finetune
python data/process_cws.py && python data/process_csc.py
python -m prepare.build_mt && python -m prepare.build_csc

# 从 HF 拉骨干做微调(不用预训练)
huggingface-cli download Ismantic/BERTc-315M --local-dir models/BERTc-315M
CKPT=models/BERTc-315M bash prepare/run_v4_large.sh finetune

# 从零预训练(单张 4090 约 3-5 天)
python data/download.py --pretrain && python data/process.py --all
python -m prepare.pretokenize --output prepare/corpus/v4.pt
bash prepare/run_v4_large.sh pretrain
```

细节见各层的 README 和 [`docs/FINETUNE.md`](docs/FINETUNE.md)。

## 环境

单张 RTX 4090(24GB,bf16),没有多卡代码路径。Python 3.14 + torch 2.11。

## 相关项目

```
PieceTokenizer ── 字级 SentencePiece 分词器,提供 BERTc 的词表
Wapic          ── 独立的 C++ CRF 中文分词器,给 BERTc 提供整词掩码的词边界
Summer         ── Qwen3 ReTok(把 BBPE 词表换成 piece 词表)
Interpreter    ── 中英翻译模型,承接 Summer
```

BERTc 与 Wapic 互补:Wapic 给 BERTc 提供切词工具,BERTc 微调出的模型反过来
可以蒸馏语料扩充 Wapic 的训练数据。

## 许可

Apache-2.0。训练语料各自的许可见对应数据集卡。
