# src/

模型定义与训练循环,只依赖 torch。

```
model.py        344   Transformer 编码器 + MLM 头
pretrain.py     337   预训练循环
finetune_mt.py  264   CWS + POS + NER 联合微调
data.py         250   memmap 语料 + 变长数据集
evaluate.py     218   F1 / 准确率,以及 CSC 的 PyCorrector 口径
finetune_csc.py 194   Correction 微调
crf.py          192   线性链 CRF
optim.py        152   StableAdamW
masking.py      109   整词掩码
checkpoint.py    85   safetensors 读写
```

## 阅读顺序

`pretrain.py` 把所有部件串起来,先读会被细节干扰。按下面的顺序从小模块入手。

**`checkpoint.py`** —— safetensors 的格式是三段:8 字节小端 uint64 表示头长度、
一段 JSON 头、剩下是裸张量数据。

**`optim.py`** —— StableAdamW 在 AdamW 上加了一件事:按每个张量的梯度 RMS
裁剪有效学习率。

```python
rms = grad.pow(2).div_(exp_avg_sq.maximum(eps_sq)).mean().sqrt()
eff_lr = lr / max(1.0, rms)
```

梯度相对于二阶动量估计偏大时按比例降低学习率,这是 bf16 下更稳定的原因。
bias correction 折进 β 的计算,不单独修正。

**`model.py`** —— 按 forward 的顺序读:嵌入 → N 层 → MLM 头。三处与标准
BERT 不同:

- ScaledSinusoidal 位置编码在嵌入层加完,attention 里没有位置相关的代码
- GeGLU 用两个投影矩阵 + 门控,所以中间层是 2752 而不是 4096(2/3 宽度补偿)
- MLM 头只有 `logits = h @ embed.weight.T`,原版 BERT 这里有
  Dense + LayerNorm + GeLU + bias

`flex_attention` 那部分可以先跳过。它处理的是「一个块里有多篇文档」的隔离
问题,与 Transformer 本身无关。

**`masking.py`** —— 整词掩码。语料打包成定长块,一个块横跨多篇文档、
数百个词,因此给每个 token 一个 word id(同一个词共享),按 word id 分组遮。

**word id 生成错误时,整词掩码会退化成逐字掩码,且不报错。** 见
[`docs/WHY.md`](../docs/WHY.md#整词掩码退化成逐字掩码)。

**`crf.py`** —— CWS 和 NER 用 BIO 标注,相邻标签有硬约束(`I-` 不能跟在 `O`
后面),CRF 把约束建进模型。两个算法:前向算法算配分函数,在对数空间里做,
用 `logsumexp` 代替求和以免下溢;Viterbi 解码最优路径,结构相同,
`logsumexp` 换成 `max` 并记住选择。

**`data.py`** —— 预训练语料是 80 GB 的 int32 数组,用
`torch.from_file(shared=True)` 做 memmap,按页惰性加载。`shared=False` 会尝试
拷贝私有副本,直接 `Cannot allocate memory`。

微调数据是变长的,用扁平数组 + offsets 存,格式定义在文件头的模块文档里。

**`pretrain.py`** —— 主循环里有三个调度同时跑,需要互相对齐:

```
学习率     warmup 510 步 → Damped Cosine 降到 8e-5
梯度累积   从 1 线性爬到 128(前 5% 步)
掩码率     固定 15%(动态 curriculum 的代码在,默认关)
```

梯度累积爬升等价于 batch size warmup。它与 LR warmup 独立计算,训练日志因此
同时打印 `accum` 和 `lr` 两列。

**`finetune_mt.py` / `finetune_csc.py`** —— MT 是三个任务共享一个骨干:
CWS(CRF)、POS(逐位置分类)、NER(CRF)。两处非显然:

- POS loss 需加权 α=2,否则只占 1.6% 的梯度
- 骨干 lr 2e-5、头 lr 5e-4。骨干已训好,大 lr 会破坏已有表示;头是随机初始化的,
  需要较大 lr

CSC 是双头(纠错 + 检测),纠错头必须与词嵌入绑权重,否则 F1 差 0.05。
见 [`docs/WHY.md`](../docs/WHY.md#csc-的纠错头必须与词嵌入绑权重)。

## 两条边界

**只依赖 torch。** CRF、优化器、LR 调度、整词掩码、safetensors 读写、memmap
数据集全部自己实现,替掉的是 `torchcrf` / `optimi` / `transformers` /
`safetensors` / `datasets`。

**不处理文本。** 没有字符串处理,不 import 任何 tokenizer。分词、字→id、
标签构造全在 [`prepare/`](../prepare/),这里只读预编码好的 id 张量,
因此 PieceTokenizer 不是 `src/` 的依赖。

## 怎么跑

这是个 package,模块间用相对 import,要从仓库根目录用 `-m` 跑:

```bash
python -m src.pretrain     --train_data ... --output_dir ...
python -m src.finetune_mt  --ckpt_dir ... --train_data ... --dev_data ...
python -m src.finetune_csc --ckpt_dir ... --train_data ... --test_data ...
```

直接 `python src/pretrain.py` 会因为相对 import 失败。完整参数由
[`prepare/Makefile`](../prepare/Makefile) 拼好,平时用 `make -C prepare pretrain`。

用 package 而非平铺目录,是因为 `data` / `model` / `optim` 这几个模块名过于
通用,放在 `sys.path` 上会与其他模块互相遮蔽。
