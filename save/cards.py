"""生成 HF 模型卡(README.md)。"""

_HEADER = """---
license: apache-2.0
language:
- zh
{extra_lang}tags:
{tags}
pipeline_tag: {pipeline}
library_name: pytorch
---
"""

_TOKENIZER_SECTION = """
## 依赖

只需要 **PyTorch** 和 **PieceTokenizer**,没有别的。

| | |
|---|---|
| 模型定义 | 目录内的 `model.py`(纯 torch,不 import transformers) |
| 权重读取 | 目录内的 `checkpoint.py`(85 行 safetensors 读取,不需要 safetensors 库) |
| 分词器 | PieceTokenizer,提供字级切分和词表 |

```bash
pip install torch
pip install git+https://github.com/Ismantic/PieceTokenizer
```

目录是自包含的:进到目录里直接 `python example_*.py` 就能跑,不依赖目录外的
任何文件。

## Tokenizer

字级 SentencePiece,`BERTc-Tokenizer.pt`,词表 12536(pad=12531,mask=12535)。**必须用 `dict="no"`
加载**(字模式,不挂分词词典)——挂了词典编码结果会跟训练时不一致,而且不报错。

```bash
pip install git+https://github.com/Ismantic/PieceTokenizer
```

## 文件

| 文件 | 说明 |
|---|---|
{files}

## 许可

Apache-2.0。训练语料各自的许可见对应数据集卡。
"""


def backbone_card(name: str, spec: dict) -> str:
    header = _HEADER.format(
        extra_lang="- en\n",
        tags="\n".join(f"- {t}" for t in
                       ["bert", "fill-mask", "chinese", "modernbert",
                        "masked-language-modeling"]),
        pipeline="fill-mask",
    )
    files = "\n".join([
        "| `model.safetensors` | 权重 |",
        "| `config.json` | 架构配置 |",
        "| `model.py` | 模型定义(纯 PyTorch,无 transformers 依赖)|",
        "| `BERTc-Tokenizer.pt` | 词表(与 PieceTokenizer 仓库里那份相同)|",
        "| `tokenizer.py` | 字级 tokenizer 封装 |",
        "| `example_load.py` | 加载 + 掩码预测示例 |",
    ])
    return f"""{header}
# {name}

字级中文 Modern BERT,**从零预训练**。纯 PyTorch 实现,不依赖 transformers。

## 架构

- 参数量 {spec['params']}
- {spec['arch']}
- ScaledSinusoidal 位置编码、GeGLU 前馈、LayerNorm 无 bias、输入输出词嵌入绑定
- 全部 Linear 无 bias;Megatron 式初始化(残差支路 ×1/√2L)
- 预训练用 flex_attention 做跨文档隔离,微调走 SDPA

## 下游表现

| 任务 | 结果 |
|---|---|
| PD-1998 分词 / 词性 / 实体(联合微调)| {spec['mt']} |
| SIGHAN-15 拼写纠错 | {spec['csc']} |

{spec['notes']}

## 用法

```python
import json, torch
from safetensors.torch import load_file
from model import ModernBertConfig, ModernBertForMLM
from tokenizer import PieceCharTokenizer

cfg = ModernBertConfig.from_dict(json.load(open("config.json")))
model = ModernBertForMLM(cfg)
model.load_state_dict(load_file("model.safetensors"), strict=True)
model.eval()

tok = PieceCharTokenizer(".")
ids = torch.tensor([tok.encode("北京是中国的首都")])
ids[0, 2] = tok.mask_token_id
print(tok.id_to_char(int(model(ids)["logits"][0, 2].argmax())))
```
{_TOKENIZER_SECTION.format(files=files)}"""


def finetune_card(name: str, spec: dict) -> str:
    is_mt = spec["task"] == "mt"
    header = _HEADER.format(
        extra_lang="",
        tags="\n".join(f"- {t}" for t in (
            ["bert", "chinese", "token-classification", "word-segmentation",
             "pos-tagging", "ner"] if is_mt else
            ["bert", "chinese", "text2text-generation", "spelling-correction"])),
        pipeline="token-classification" if is_mt else "fill-mask",
    )
    metrics = "\n".join(f"| {k} | {v} |" for k, v in spec["metrics"].items())

    if is_mt:
        title = "中文分词 + 词性标注 + 命名实体识别"
        desc = ("三任务联合微调。分词和实体走 CRF(序列约束强),词性走 softmax。"
                "词性只在词首字上有监督。")
        usage = """```python
from mt_model import BERTcForMT

model = BERTcForMT.from_pretrained(".")
print(model.predict("中国科学院计算技术研究所在北京"))
# {'words': ['中国科学院计算技术研究所', '在', '北京'],
#  'pos': ['ni', 'p', 'ns'],
#  'ner': [{'type': 'Ni', 'start': 0, 'end': 12}, {'type': 'Ns', 'start': 13, 'end': 15}]}
```"""
        files = "\n".join([
            "| `model.safetensors` | 骨干 + 三个任务头 |",
            "| `mt_model.py` | 推理入口 `BERTcForMT` |",
            "| `crf.py` | 线性链 CRF |",
            "| `model.py` | 骨干定义 |",
            "| `tokenizer.py` | 字级 tokenizer |",
            "| `example_decode.py` | 示例 |",
        ])
        extra = """
## 评测口径

指标在 **dev 集前 2000 句**上测 —— 与训练时选 best.pt 的口径一致
(训练脚本 `--dev_limit 2000`)。全量 21,143 句上是
分词 0.9790 / 词性 0.9787 / 实体 0.9597 / joint 1.4646。

joint score = 分词 F1 + 0.3 × 词性准确率 + 0.2 × 实体 F1。

## 标签体系

- 分词:BIES
- 词性:LTP base1 的 27 个标签(PD-1998 的 43 个映射过来)
- 实体:BIES × {人名 Nh / 地名 Ns / 机构 Ni},PD 的 MISC 丢弃

词性代号可读性差(`ns` 地名 / `ni` 机构名 / `nh` 人名 / `nd` 方位词 /
`nl` 处所词 / `wp` 标点),BERTc 仓库的 `save/cws.py` 有个把它们翻成中文的
交互式脚本。
"""
    else:
        title = "中文拼写纠错"
        desc = ("双头:cor 逐位置预测正确的字(权重与词嵌入绑定),"
                "det 判断该位置有没有错(focal loss)。**只做等长替换**,"
                "不处理多字少字。")
        usage = """```python
from csc_model import BERTcForCSC

model = BERTcForCSC.from_pretrained(".")
print(model.correct("他平时喜欢锻练身体"))   # 他平时喜欢锻炼身体
print(model.correct(["句子一", "句子二"]))   # 也接列表
```"""
        files = "\n".join([
            "| `model.safetensors` | 骨干 + 双头 |",
            "| `csc_model.py` | 推理入口 `BERTcForCSC` |",
            "| `model.py` | 骨干定义 |",
            "| `tokenizer.py` | 字级 tokenizer |",
            "| `example_correct.py` | 示例 |",
        ])
        extra = """
## 评测口径

指标在 **SIGHAN-15 官方 707 条**上测(`shibing624/pycorrector` 里
`pycorrector/data/sighan2015_test.tsv` 那一版)。注意 CTCDataset 里还有个
1100 条的 `sighan15_test.jsonl`,不是同一个东西。

判定是**整句**级:整句完全一致才算对,改对一半不给分。

## 阈值

`correct(..., threshold=0.7)`:纠错置信度低于阈值就保留原字。调低提召回、
调高提精确率。0.7 是与 MacBERT4CSC 对齐的默认值,报告的指标都基于它。

## 一个反直觉的行为

`correct()` **只用纠错头,不用检测头**。检测头是训练时的辅助信号,推理不参与。

所以会出现"模型知道这里有错、但选不出正确的字"的情况。比如「我今天很稿兴」,
`稿` 位置的检测分是 0.98,但纠错头的 top-1 仍是 `稿` 本身(0.22),
`高` 只排第 4(0.11)—— 这种时候**调低阈值没有任何用**,阈值只能否决改动,
不能凭空造出改动。
"""

    return f"""{header}
# {name}

{title}。基于 [{spec['base']}](https://huggingface.co/Ismantic/{spec['base']}) 微调。

{desc}

## 指标

| 指标 | 值 |
|---|---|
{metrics}

## 训练

- 配方:{spec['recipe']}
- 数据:{spec['data']}

## 用法

{usage}
{extra}{_TOKENIZER_SECTION.format(files=files)}"""
