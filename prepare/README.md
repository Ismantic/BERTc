# prepare/ — 编排层

把 `data/` 的原料变成 `src/` 能吃的张量,再把训练调起来。这一层是**唯一**需要
tokenizer 和分词器的地方 —— `src/` 只认预编码好的 id,从头到尾不碰文本。

## 全流程

```bash
bash prepare/install_deps.sh                 # 重编译 PieceTokenizer / Wapic(仅在它们更新后)
bash prepare/run_v4_large.sh data            # 下载 + 加工 + 预编码下游数据集
bash prepare/run_v4_large.sh pretokenize     # 语料 → 定长 chunk
bash prepare/run_v4_large.sh pretrain        # 预训练(3-5 天,单卡 4090)
bash prepare/run_v4_large.sh finetune        # MT + CSC 微调
```

## 文件

```
install_deps.sh   clone + 编译安装 PieceTokenizer / Wapic,装完自动跑行为校验
fetch_backbone.py HF 发布包 → 微调脚本认的 model.pt
tokenizer.py      PieceTokenizer 适配:字→id 缓存、id→字 反查表
labels.py         CWS / POS / NER 标签表,PD-1998 → LTP 的词性与实体映射
pack.py           预编码数据的打包格式(扁平数组 + offsets)
build_mt.py       PD-1998 jsonl → mt_{train,dev}.pt
build_csc.py      CSC 句对 → csc_{train,test}.pt
pretokenize.py    预训练语料 → .pt / .wid / .seg
run_v4_large.sh   v4-Large 全流程入口
```

产物落在 `prepare/datasets/`(下游)和 `prepare/corpus/`(预训练),都不进 git。

## 三个不能改错的地方

**`det_labels` 必须按字比对,不能按 id。** 两个不同的字可能都落到 UNK,按 id
比会漏掉那处错误。这不是理论风险:实测 `all_pairs.pkl` 前 2000 条里就有 1 处
(实测 82 万对里确实存在)。

**CSC 的 `id_to_char` 反查表只覆盖编码时见过的字。** `src/evaluate.py` 靠它把
预测还原成句子跟标准答案比。表放大会让本该"保留原字"的未知 id 变成真解码,
口径就跟 pycorrector 对不上了。

**预训练语料不能逐字编码。** 字模式下 tokenizer 只对中文一字一 token,英文
单词整体成一个 piece、空格成 `▁`。所以 `pretokenize.py` 整串编码后再按字符
游标把 piece 对回原文,才能拿到词边界。MT / CSC 是纯中文短句,才可以逐字编。

## 数据集格式

`build_*.py` 产出 `torch.save` 的 dict,变长序列用扁平数组 + offsets 表示。
字段定义见 `src/data.py` 的模块文档。**不在这一步截断** —— `max_chars`(MT 254)
和 `max_len`(CSC 128)是训练超参,在 `src/data.py` 里截,换长度不用重跑。

## 校验

```bash
python test/test_reproduce_sota.py   # 拿真 checkpoint 复现 MT 1.4712 / CSC 0.8346
python test/test_tokenizer.py        # C++ 依赖重建后行为有没有变
```

重建 PieceTokenizer / Wapic **之前**要先 `python test/capture_baseline.py` 抓基线,
顺序反了就失去意义。
