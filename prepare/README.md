# prepare/ —— 文本怎么变成张量

第二层要解决的问题:**模型吃的是 int32 张量,你手上的是几十 GB 纯文本。
中间这一步怎么做。**

这一层是**唯一**碰文本的地方 —— 分词、字→id、标签构造全在这里。
[`src/`](../src/) 从头到尾只认预编码好的 id。

```bash
make -C prepare help       # 全部命令
make -C prepare status     # 每一步产物在不在
```

## 预训练语料:三个平行文件

一句 `tokenizer(text)` 只能给你 id。而预训练还需要另外两样东西,
所以产出是**三个逐 token 对齐的文件**:

| 文件 | 类型 | 回答的问题 |
|---|---|---|
| `v4.pt` | int32 | 这个位置是哪个字 |
| `v4.pt.wid` | int32 | **它属于哪个词** —— 遮「沁县」要两个字一起遮 |
| `v4.pt.seg` | uint8 | **它属于哪篇文章** —— 一个 512 的块里常装了好几篇 |

`.wid` 是整词掩码的依据,`.seg` 是 attention 跨文档隔离的依据。
两者都是「打包成定长块」这个决定带来的后果 —— 不打包就没这两个问题,
但也就浪费掉大量 padding。

### 词边界怎么算出来的

用 Wapic(CRF 分词器)切词,拿 `word_starts` 标出每个词的首字,同一个词的
字共享一个 word id。

绕不过去的地方是:**不能逐字编码**。字模式下 tokenizer 只对中文一字一 token,
英文单词整体成一个 piece、空格成 `▁` —— 逐字编会把英文拆碎。

所以 `pretokenize.py` 的做法是**整串编码,再按字符游标把 piece 对回原文位置**,
这样才能同时拿到正确的 id 和正确的词边界。MT / CSC 用的是纯中文短句,
才可以逐字编。

⚠️ 这一步做错了不会报错。整词掩码会静默退化成逐字掩码,训练照跑、loss 照降。
验证方法(看词长分布)见 [`docs/PRETRAIN.md`](../docs/PRETRAIN.md#验证产出)。

## 下游数据集:扁平数组 + offsets

`build_mt.py` / `build_csc.py` 产出 `torch.save` 的 dict。变长序列不存
list of tensors,也不 padding,而是**一个扁平数组 + 一个 offsets 数组**:

```
flat     [句1的所有id][句2的所有id][句3的所有id]...
offsets  [0, len1, len1+len2, ...]
```

省内存,而且 memmap 友好。字段定义见 [`src/data.py`](../src/data.py) 的模块文档。

**不在这一步截断。** `max_chars`(MT 254)和 `max_len`(CSC 128)是训练超参,
在 `src/data.py` 里截 —— 想换长度不用重跑预编码。

## 两个 C++ 依赖

```bash
make -C prepare deps
```

- **PieceTokenizer** —— 字级分词器,同时提供词表(`BERTc-Tokenizer.pt`)。
  本仓库不留副本,靠 `piece_tokenizer.__file__` 反查定位
- **Wapic** —— CRF 分词器,只用来标预训练的词边界。微调不需要它

它们是 C++ 写的,没法用纯 torch 替代,也不该替代 —— 解决的是中文分词问题,
不是深度学习问题。

⚠️ 重建这两个依赖**之前**要先 `python test/capture_baseline.py` 抓基线。
顺序反了就失去意义 —— 基线是用来发现「重建把行为改了」的,事后抓的基线
记录的已经是改过之后的行为。

编码行为一旦变了,词表就和已发布模型对不上,而**代码不会报错**。
`make deps` 装完会自动比对,输出 `校验通过` 才算成功。

## 文件

```
install_deps.sh   clone + 编译 PieceTokenizer / Wapic,装完自动校验
tokenizer.py      PieceTokenizer 适配:字→id 缓存、id→字 反查表
labels.py         CWS / POS / NER 标签表,PD-1998 → LTP 的词性与实体映射
pack.py           扁平数组 + offsets 的打包格式
build_mt.py       PD-1998 jsonl → mt_{train,dev}.pt
build_csc.py      CSC 句对 → csc_{train,test}.pt
pretokenize.py    预训练语料 → .pt / .wid / .seg
Makefile          全流程入口,也是所有超参的实际来源
```

产物落在 `prepare/datasets/`(下游)和 `prepare/corpus/`(预训练),都不进 git。
`corpus` 和 `datasets` 是 Make 的**文件目标** —— 产物在就不重跑,毕竟切块要
4.6 小时。

## Makefile 是超参的来源

`prepare/Makefile` 不只是快捷方式,**已发布模型的完整命令行就在里面**,
每个非默认参数旁边都有注释说明为什么是这个值。想知道 `--alpha_pos 2.0`
从哪来的,看那里。

两个规格用同一份配方,`SIZE` 切换(`large` = 315M,`mid` = 165M,默认 mid)。
预训练配方**完全相同**,只差层数和 batch 切分 —— 有效 batch 都是 4096。
CSC 微调配方不同,因为 [315M 用 5 epoch 严重欠训](../docs/WHY.md#315m-用-5-epoch-严重欠训)。
