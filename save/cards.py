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
## Tokenizer

字级 SentencePiece,词表 12536(pad=12531,mask=12535)。**必须用 `dict="no"`
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
        "| `piece.model` / `mask_token_id.txt` | tokenizer |",
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

cfg = ModernBertConfig(**json.load(open("config.json")))
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
## 标签体系

- 分词:BIES
- 词性:LTP base1 的 27 个标签(PD-1998 的 43 个映射过来)
- 实体:BIES × {人名 Nh / 地名 Ns / 机构 Ni},PD 的 MISC 丢弃
"""
    else:
        title = "中文拼写纠错"
        desc = ("双头:cor 逐位置预测正确的字(权重与词嵌入绑定),"
                "det 判断该位置有没有错(focal loss)。**只做等长替换**,"
                "不处理多字少字。")
        usage = """```python
from csc_model import BERTcForCSC

model = BERTcForCSC.from_pretrained(".")
print(model.correct("我今天很稿兴"))     # 我今天很高兴
```"""
        files = "\n".join([
            "| `model.safetensors` | 骨干 + 双头 |",
            "| `csc_model.py` | 推理入口 `BERTcForCSC` |",
            "| `model.py` | 骨干定义 |",
            "| `tokenizer.py` | 字级 tokenizer |",
            "| `example_correct.py` | 示例 |",
        ])
        extra = """
## 阈值

`correct(..., threshold=0.7)`:纠错置信度低于阈值就保留原字。调低提召回、
调高提精确率。0.7 是与 MacBERT4CSC 对齐的默认值,报告的指标都基于它。
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
