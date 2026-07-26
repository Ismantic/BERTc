# prepare/

词表、预编码、语料切块、调训练。这一层是唯一碰文本的地方 ——
[`src/`](../src/) 只认预编码好的 id。

```bash
make -C prepare help       # 全部命令
make -C prepare status     # 每一步产物在不在
```

## 预训练语料

产出三个逐 token 对齐的文件:

| 文件 | 类型 | 用途 |
|---|---|---|
| `v4.pt` | int32 | 模型输入 |
| `v4.pt.wid` | int32 | word id,整词掩码用 —— 遮「沁县」要两个字一起遮 |
| `v4.pt.seg` | uint8 | 文档 id,跨文档隔离用 —— 一个 512 块里常装了好几篇 |

后两个都是「打包成定长块」带来的:不打包就没这两个问题,但会浪费大量 padding。

词边界用 Wapic 切词、取 `word_starts` 得到,同一个词的字共享一个 word id。

**语料不能逐字编码。** 字模式下 tokenizer 只对中文一字一 token,英文单词整体
成一个 piece、空格成 `▁`,逐字编会把英文拆碎。`pretokenize.py` 整串编码后按
字符游标把 piece 对回原文位置,才能同时拿到正确的 id 和词边界。MT / CSC 是
纯中文短句,可以逐字编。

这一步做错不会报错 —— 整词掩码会静默退化成逐字掩码。验证方法(看词长分布)见
[`docs/PRETRAIN.md`](../docs/PRETRAIN.md#验证产出)。

## 下游数据集

`build_mt.py` / `build_csc.py` 产出 `torch.save` 的 dict。变长序列用扁平数组 +
offsets,不 padding 也不存 list of tensors:

```
flat     [句1的所有id][句2的所有id][句3的所有id]...
offsets  [0, len1, len1+len2, ...]
```

字段定义见 [`src/data.py`](../src/data.py) 的模块文档。

**不在这一步截断。** `max_chars`(MT 254)和 `max_len`(CSC 128)是训练超参,
在 `src/data.py` 里截,换长度不用重跑预编码。

## C++ 依赖

```bash
make -C prepare deps
```

- **PieceTokenizer** —— 字级分词器,同时提供词表(`BERTc-Tokenizer.pt`)。
  本仓库不留副本,靠 `piece_tokenizer.__file__` 反查定位
- **Wapic** —— CRF 分词器,用来标预训练的词边界;仓库里还自带 PD-1998 标注
  语料(`data/PeopleDaily1998.zip`),微调也要用。只做微调的话不必编译,
  `bash prepare/install_deps.sh wapic-data` 只 clone

编码行为一旦变了,词表就和已发布模型对不上,而代码不会报错。`make deps`
装完会自动比对,输出 `校验通过` 才算成功。

重建这两个依赖**之前**先跑 `python test/capture_baseline.py` 抓基线 ——
顺序反了,记录的就已经是改过之后的行为。

改了数据流之后跑 `python test/test_provenance.py`:它逐条核对每个中间产物
都有代码负责生成,不允许出现只能靠手工放上去的文件。

## 文件

```
install_deps.sh   clone + 编译 PieceTokenizer / Wapic,装完自动校验
tokenizer.py      PieceTokenizer 适配:字→id 缓存、id→字 反查表
labels.py         CWS / POS / NER 标签表,PD-1998 → LTP 的词性与实体映射
pack.py           扁平数组 + offsets 的打包格式
build_mt.py       PD-1998 jsonl → mt_{train,dev}.pt
build_csc.py      CSC 句对 → csc_{train,test}.pt
pretokenize.py    预训练语料 → .pt / .wid / .seg
Makefile          全流程入口
```

产物落在 `prepare/datasets/`(下游)和 `prepare/corpus/`(预训练),都不进 git。
两者都是 Make 的**文件目标** —— 产物在就不重跑,切块要 4.6 小时。

## Makefile

已发布模型的完整命令行就在里面,每个非默认参数旁边有注释说明取值理由。
想知道 `--alpha_pos 2.0` 从哪来,看那里。

两个规格用 `SIZE` 切换(`large` = 315M,`mid` = 165M,默认 mid)。预训练配方
完全相同,只差层数和 batch 切分,有效 batch 都是 4096。CSC 微调配方不同,
因为 [315M 用 5 epoch 严重欠训](../docs/WHY.md#315m-用-5-epoch-严重欠训)。
