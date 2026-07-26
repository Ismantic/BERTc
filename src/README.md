# src/ —— 模型与训练

**只依赖 torch 的两千行。** 这一层是整个项目的教学重点:BERT 的每个部件都在
这里,没有一个是 import 来的。

```
model.py       344   Transformer 编码器 + MLM 头
pretrain.py    337   预训练循环
finetune_mt.py 264   分词 / 词性 / 实体联合微调
data.py        250   memmap 语料 + 变长数据集
evaluate.py    218   F1 / 准确率,以及 CSC 的 pycorrector 口径
crf.py         192   线性链 CRF
finetune_csc.py 194  拼写纠错微调
optim.py       152   StableAdamW
masking.py     109   整词掩码
checkpoint.py   85   safetensors 读写
```

## 建议的阅读顺序

不建议从 `pretrain.py` 开始 —— 它把所有部件串起来,先读会被细节淹没。
从小的、自洽的模块入手:

### 1. `checkpoint.py`(85 行)—— 先建立信心

safetensors 的格式就三段:8 字节小端 uint64 表示头长度、一段 JSON 头、
剩下全是裸张量数据。85 行读完你会发现「模型文件」没有任何神秘之处。

从这里开始还有个好处:它证明了「不装库也能做」这件事是真的,而不是口号。

### 2. `optim.py`(152 行)—— 优化器不是黑盒

StableAdamW 在 AdamW 上加了一件事:**按每个张量的梯度 RMS 裁剪有效学习率**。

```python
rms = grad.pow(2).div_(exp_avg_sq.maximum(eps_sq)).mean().sqrt()
eff_lr = lr / max(1.0, rms)
```

梯度相对于二阶动量估计偏大时,就按比例压学习率。这两行是 bf16 下训练更稳的
全部原因。

顺带能看清 **bias correction 是怎么折进 β 的** —— 不单独做修正,而是把它吸收
进 β 的计算,数值等价但少一处除法。这种「论文里一句话带过、实现时想半天」的
东西,读代码比读论文快。

### 3. `model.py`(344 行)—— 主体

按 forward 的顺序读:嵌入 → N 层 → MLM 头。

值得停下来看的三处:

- **ScaledSinusoidal 位置编码**在嵌入层就加完了,attention 里没有任何位置相关
  的代码。对比 RoPE 需要在每层做旋转
- **GeGLU** 用两个投影矩阵 + 门控,所以中间层是 2752 而不是 4096(2/3 宽度补偿)
- **MLM 头就一行**:`logits = h @ embed.weight.T`。原版 BERT 这里有
  Dense + LayerNorm + GeLU + bias,微调时全丢掉

`flex_attention` 那部分可以先跳过,它解决的是「一个块里装了多篇文章」的工程
问题,不影响理解 Transformer 本身。

### 4. `masking.py`(109 行)—— MLM 的真正难点

「随机遮 15%」一句话说完,实现起来的麻烦是**整词掩码**:遮「沁县」要两个字
一起遮,而语料是打包成定长块的,一个块里横跨多篇文章、几百个词。

做法是给每个 token 一个 word id(同一个词共享),按 word id 分组遮。

⚠️ 这里有个静默失败:word id 要是没生成对,整词掩码退化成逐字掩码,
**训练照跑、loss 照降,不会报任何错**。见 [`docs/WHY.md`](../docs/WHY.md#整词掩码退化成逐字掩码)。

### 5. `crf.py`(192 行)—— 唯一需要动笔的一段

分词和实体识别用 BIO 标注,相邻标签有硬约束(`I-` 不能跟在 `O` 后面)。
CRF 把这个约束建进模型。

两个算法:

- **前向算法**算配分函数(所有可能路径的分数和)。关键技巧是在**对数空间**里
  做,用 `logsumexp` 代替求和,否则几百步的连乘立刻下溢
- **Viterbi** 解码最优路径,和前向算法结构一样,只是 `logsumexp` 换成 `max`
  并记住选择

如果只想读一段代码就理解「结构化预测」是什么意思,读这个文件。

### 6. `data.py`(250 行)—— 80 GB 语料怎么读

预训练语料是 80 GB 的 int32 数组,内存装不下。用
`torch.from_file(shared=True)` 做 memmap,操作系统按页惰性加载。

⚠️ `shared=False` 会尝试拷贝一份私有副本,直接 `Cannot allocate memory` ——
而报错信息指向内存不足,容易让人去调 batch size,方向就错了。

微调数据是变长的,用**扁平数组 + offsets** 存,不做 padding 也不存 list of
tensors。格式定义在文件头的模块文档里。

### 7. `pretrain.py`(337 行)—— 串起来

现在可以读主循环了。里面有三个调度在同时跑,而且**必须互相对齐**:

```
学习率      warmup 510 步 → Damped Cosine 降到 8e-5
梯度累积    从 1 线性爬到 128(前 5% 步)
掩码率      固定 15%(动态 curriculum 的代码在,默认关)
```

梯度累积爬升等价于 batch size warmup。它和 LR warmup 各算各的,两者在前几百步
的组合效果需要留意 —— 这是训练日志里 `accum` 和 `lr` 两列都要打出来的原因。

### 8. `finetune_mt.py` / `finetune_csc.py` —— 骨干怎么用

MT 是三个任务共享一个骨干:分词(CRF)、词性(逐位置分类)、实体(CRF)。
两处非显然的地方:

- **词性 loss 要加权 α=2**,否则只占 1.6% 的梯度,学不动
- **骨干 lr 2e-5、头 lr 5e-4**。骨干已经训好了,大 lr 会冲坏;头是随机初始化的,
  小 lr 学不动

CSC 是双头(纠错 + 检测),**纠错头必须与词嵌入绑权重** —— 否则 F1 差 0.05。
原因见 [`docs/WHY.md`](../docs/WHY.md#csc-的纠错头必须与词嵌入绑权重)。

## 两条边界

### 只依赖 torch

CRF、优化器、LR 调度、整词掩码、safetensors 读写、memmap 数据集,全部自己实现。
替掉的是 `torchcrf` / `optimi` / `transformers` / `safetensors` / `datasets`。

这不是造轮子的癖好 —— **装个 transformers 能让这两千行缩到三百行,也就没什么
可读的了**。这一层的价值就在于它没被抽象掉。

### 不碰文本

`src/` 里没有一个字符串处理函数,也不 import 任何 tokenizer。分词、字→id、
标签构造全在 [`prepare/`](../prepare/),`src/` 只读预编码好的 id 张量。

好处有两层:PieceTokenizer 不是 `src/` 的依赖(纯 torch 这条才成立);
以及边界一清晰,「模型看到的张量长什么样」就成了可以单独讲清楚的一件事。

## 怎么跑

这是个 package,模块间用相对 import,必须从仓库根目录用 `-m` 跑:

```bash
python -m src.pretrain     --train_data ... --output_dir ...
python -m src.finetune_mt  --ckpt_dir ... --train_data ... --dev_data ...
python -m src.finetune_csc --ckpt_dir ... --train_data ... --test_data ...
```

直接 `python src/pretrain.py` 会因为相对 import 失败。完整的参数由
[`prepare/Makefile`](../prepare/Makefile) 拼好,平时用 `make -C prepare pretrain`。

改成 package 是因为 `data` / `model` / `optim` 这几个名字太通用,平铺在
`sys.path` 上会跟别的模块互相遮蔽 —— 重构时撞过一次。
