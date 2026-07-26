# CLAUDE.md

给 Claude Code 的项目说明。这些约定优先于默认行为。

## 这是什么

BERTc:字级中文 Modern BERT,从零预训练,纯 PyTorch。已发布
`Ismantic/BERTc-{165M,315M}` 及其 MT / CSC 微调版共六个 HF 仓库。

**这是一个教学项目,不是一个刷榜项目。** 价值在于两千行能读完的代码把「从零
训练中文 BERT」讲清楚了,不在于那 0.0035 的领先。这条决定了很多取舍:

- **可读性优先于抽象**。`src/` 只依赖 torch —— 装个 transformers 能让两千行
  缩到三百行,也就没什么可读的了。要加抽象层先想清楚读者会不会因此看不见
  原来能看见的东西。
- **SOTA 数字是证据,不是目标**。它证明的是这份代码里没藏着「其实少了一步」
  的问题。所以指标必须诚实(见「不要为了让数字好看去改评测代码」),
  但也不必为了再涨 0.001 去加复杂度。
- **踩过的坑要写下来**,尤其是**改错了不报错**的那类。集中在 `docs/WHY.md`,
  开篇就是「会静默出错的地方」。否定结果也写(见那篇里两个花掉整轮预训练的
  实验)—— 论文不发失败尝试,教材最该有。
- **文档不重复**。层 README 讲「这一层解决什么问题」,`make help` 讲怎么跑,
  `docs/WHY.md` 讲为什么。同一件事只写在一处。

**核心约束:整个仓库只依赖 Hugging Face 和 GitHub。** 语料、标注数据、词表、
C++ 依赖全部可从公网重建,不允许指向本机既有目录。加新数据源时必须登记公开
出处;找不到出处的宁可不用,也不要引入本地文件。这条也是教学定位的一部分 ——
读者跑不通的教程没有价值。

## 四层结构

```
data/       下载 + 加工。source.py 是数据源注册表,唯一的真相来源
src/        模型 + 训练。**只依赖 torch**
prepare/    编排:词表、预编码、语料切块、调训练
save/       导出 HF 发布包 + 上传;save/sota/ 存 SOTA checkpoint
```

外加 `deps/`(clone 的 C++ 依赖,gitignore)、`docs/`、`test/`。

两条分层约束,改代码时不要破坏:

- **`src/` 只依赖 torch**。CRF、StableAdamW、LR 调度都是自己实现的
  (替掉 torchcrf / optimi / transformers),memmap 用 `torch.from_file`。
  加依赖前先想能不能放 `prepare/`。
- **`src/` 不碰文本**。分词、字→id、标签构造全在 `prepare/`,`src/` 只读
  预编码好的 id。所以 PieceTokenizer 不是 `src/` 的依赖。

`src/` 是 package,模块间用相对 import。训练脚本要用 `-m` 从仓库根目录跑:

```bash
python -m src.pretrain / src.finetune_mt / src.finetune_csc
```

## 环境

- venv:`/home/tfbao/.venv/bin/python`(Python 3.14,torch 2.11+cu13)。
  **用 `uv pip install`,这个 venv 里没有 pip。**
- GPU:单张 RTX 4090(24GB,bf16)。**没有多卡代码路径。**
- 两个 C++ 依赖用 `bash prepare/install_deps.sh` 装,会 clone 到 `deps/`:
  - **PieceTokenizer** 提供字级分词器**和词表**。词表是
    `deps/PieceTokenizer/save/BERTc-Tokenizer.pt`,通过 `piece_tokenizer.__file__`
    反查定位,本仓库不留副本。加载必须传 `dict="no"`(字模式)。
  - **Wapic** 是 CRF 分词器,预训练做整词掩码时用。模型从 HF
    `Ismantic/wapic-cws` 下到 `deps/Wapic/data/model/wapic-cws.wac`。
    用 `segment`(不是 `segment_raw`,后者把空白也当 token);
    `word_starts` 直接给词首字符偏移,做 WWM 最省事。
    它还自带 **PD-1998 标注语料**(`deps/Wapic/data/PeopleDaily1998.zip`,
    与上游 chenhui-bupt/PeopleDaily1998 逐字节相同),所以 MT 微调也依赖这个
    clone —— 但不用编译,`install_deps.sh wapic-data` 只 clone。

## 不能改错的地方

- **`src/model.py` 的 state_dict key 不能动**。改模块名或嵌套层级会让已发布的
  六个 HF 模型权重全部失配,而模型照样能随机初始化跑起来、不报错。
- **CSC 的纠错头必须与词嵌入绑权重**(`cor_head.weight = bert.embed.weight`)。
  预训练的 MLM 头就是 `h @ embed.weightᵀ`,换独立 Linear 会废掉这层对齐,
  F1 差 0.05。
- **CSC 评测的参照必须用原文**,不能从 id 还原。字级 tokenizer 多对一
  (`撘/檡/暸` 都是 id 233),SIGHAN-15 的 707 条里有 16 条还原不回去,
  拿还原文本当参照 F1 会虚高 0.006。`prepare/build_csc.py` 把原文写进测试集
  文件,`src/evaluate.py` 用它。
- **CSC 的 `det_labels` 按字比对,不能按 id**。理由同上,按 id 会漏掉错字。
- **CSC 的 `correct()` 只用纠错头,不用检测头**。检测头是训练时的辅助信号,
  这跟训出 0.8346 的口径一致,别"顺手"改成用 det 过滤。
- **预训练语料不能逐字编码**。字模式下 tokenizer 只对中文一字一 token,
  英文单词整体成一个 piece、空格成 `▁`。`prepare/pretokenize.py` 整串编码后
  按字符游标把 piece 对回原文才能拿到词边界。
- **`torch.from_file` 必须 `shared=True`**。`shared=False` 要私有副本,
  70GB 语料直接 `Cannot allocate memory`。
- **MT 的指标口径是 dev 前 2000 句**(`--dev_limit 2000`),不是全量 21,143 句。
  全量上是 0.9790 / 0.9787 / 0.9597 / 1.4646。
- **`flex_attention` 必须 `torch.compile` 包**,`model.py` 在模块导入时已经做了。
  裸调会走未融合的核,慢约 3 倍。
- **不要为了让数字好看去改评测代码**。SIGHAN-15 官方 707 条是 CSC 的权威基准。

## 测试

```bash
python test/test_reproduce_sota.py   # 拿真 checkpoint 复现 MT 1.4712 / CSC 0.8346
python test/test_save.py             # 发布目录能否独立跑
python test/test_tokenizer.py        # C++ 依赖重建后行为有没有变
python test/test_crf.py              # vs torchcrf(软依赖,没装就跳过)
python test/test_optim.py            # vs optimi + transformers
```

`test_reproduce_sota.py` 是**长期回归防线** —— 它不依赖任何旧代码,只依赖
`save/sota/` 的 checkpoint 和记录的数字。改动 `src/` 或 `prepare/` 之后跑它,
数字不对就是改坏了。

重建 PieceTokenizer / Wapic **之前**要先 `python test/capture_baseline.py`
抓基线,顺序反了就失去意义 —— 基线是用来发现"重建把行为改了"的。

## 提交

commit message 不要带 `Co-Authored-By: Claude ...`(或任何 AI 署名),
这条覆盖 Claude Code 的默认约定。

**删目录之前先确认里面有没有 gitignored 的数据。** 2026-07-25 删旧目录时
连带删掉了 `csc/data/`(CSC 原始数据)和 158GB 预处理语料 —— 已发布权重和
SOTA checkpoint 因为提前 `git mv` 走了才幸免。教训是:`git ls-files` 看不到
的东西才是危险的,先 `du -sh` 每个子目录。
