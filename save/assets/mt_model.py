"""BERTc-MT 推理:中文分词 + 词性标注 + 命名实体识别。

这份代码随模型一起发到 HF,只依赖同目录的 model.py / crf.py / tokenizer.py
和 torch,不 import 仓库里的东西。

    from mt_model import BERTcForMT
    from tokenizer import PieceCharTokenizer

    model = BERTcForMT.from_pretrained(".")
    print(model.predict("中国科学院计算技术研究所在北京"))
    # {'words': ['中国科学院计算技术研究所', '在', '北京'],
    #  'pos': ['ni', 'p', 'ns'],
    #  'ner': [{'type': 'Ni', 'start': 0, 'end': 12}, ...]}
"""
import json
from pathlib import Path

import torch
import torch.nn as nn
from checkpoint import load_safetensors

from crf import CRF
from model import ModernBertConfig, ModernBertModel
from tokenizer import PieceCharTokenizer

CWS_TAGS = ["B", "I", "E", "S"]

# LTP base1 的 27 个词性
POS_TAGS = [
    "a", "b", "c", "d", "e", "h", "i", "j", "k", "m",
    "n", "nd", "nh", "ni", "nl", "ns", "nt", "nz", "o", "p",
    "q", "r", "u", "v", "wp", "x", "z",
]

# BIES × {人名 Nh, 地名 Ns, 机构 Ni} + O
NER_TAGS = [
    "O",
    "B-Nh", "I-Nh", "E-Nh", "S-Nh",
    "B-Ns", "I-Ns", "E-Ns", "S-Ns",
    "B-Ni", "I-Ni", "E-Ni", "S-Ni",
]


def bies_to_words(chars, tag_ids):
    """BIES 标签 → 词列表。对不合法序列容错:B/S 起新词,I/E 并入当前词。"""
    words, cur = [], ""
    for c, t in zip(chars, tag_ids):
        tag = CWS_TAGS[int(t)] if not isinstance(t, str) else t
        if tag in ("B", "S"):
            if cur:
                words.append(cur)
            cur = c
        else:
            cur += c
    if cur:
        words.append(cur)
    return words


def bies_to_entities(tag_ids):
    """BIES-类型 标签 → [{type, start, end}]。B 后面没等到 E 的残片丢弃。"""
    out, i, n = [], 0, len(tag_ids)
    while i < n:
        t = NER_TAGS[int(tag_ids[i])]
        if t.startswith("S-"):
            out.append({"type": t[2:], "start": i, "end": i + 1})
            i += 1
        elif t.startswith("B-"):
            et, j = t[2:], i + 1
            while j < n:
                tj = NER_TAGS[int(tag_ids[j])]
                if tj == f"I-{et}":
                    j += 1
                elif tj == f"E-{et}":
                    j += 1
                    out.append({"type": et, "start": i, "end": j})
                    break
                else:
                    break
            i = max(j, i + 1)
        else:
            i += 1
    return out


class BERTcForMT(nn.Module):
    """预训练骨干 + 三个任务头。CWS 和 NER 走 CRF,POS 走 softmax。"""

    def __init__(self, config, num_cws=4, num_pos=27, num_ner=13, dropout=0.1):
        super().__init__()
        self.config = config
        self.bert = ModernBertModel(config)
        h = config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.cws_classifier = nn.Linear(h, num_cws)
        self.cws_crf = CRF(num_cws, batch_first=True)
        self.pos_classifier = nn.Linear(h, num_pos)
        self.ner_classifier = nn.Linear(h, num_ner)
        self.ner_crf = CRF(num_ner, batch_first=True)

    @classmethod
    def from_pretrained(cls, model_dir=".", map_location="cpu"):
        model_dir = Path(model_dir)
        cfg = ModernBertConfig.from_dict(
            json.loads((model_dir / "config.json").read_text()))
        model = cls(cfg)
        state = load_safetensors(model_dir / "model.safetensors",
                                 device=str(map_location))
        missing, unexpected = model.load_state_dict(state, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"权重不匹配:缺 {missing},多 {unexpected}")
        model.eval()
        return model

    def forward(self, input_ids, attention_mask):
        h = self.dropout(self.bert(input_ids, attention_mask=attention_mask))
        return (self.cws_classifier(h), self.pos_classifier(h),
                self.ner_classifier(h))

    @torch.no_grad()
    def predict(self, text, tokenizer=None, max_len=254, device=None):
        """一句或一批句子 → [{text, words, pos, ner}]。"""
        if tokenizer is None:
            tokenizer = PieceCharTokenizer(Path(__file__).resolve().parent)
        single = isinstance(text, str)
        texts = [text] if single else list(text)

        device = device or next(self.parameters()).device
        self.eval()
        input_ids, attn, lengths = tokenizer.batch(texts, max_len, device)

        cws_emi, pos_logits, ner_emi = self(input_ids, attn)
        mask = attn.bool()
        cws_preds = self.cws_crf.decode(cws_emi.float(), mask=mask)
        ner_preds = self.ner_crf.decode(ner_emi.float(), mask=mask)
        pos_preds = pos_logits.argmax(-1).cpu().tolist()

        out = []
        for i, s in enumerate(texts):
            chars = list(s[:lengths[i]])
            n = len(chars)
            words = bies_to_words(chars, cws_preds[i][:n])
            pos, offset = [], 0
            for w in words:
                pid = pos_preds[i][offset] if offset < n else None
                pos.append(POS_TAGS[int(pid)] if pid is not None else "x")
                offset += len(w)
            out.append({"text": s, "words": words, "pos": pos,
                        "ner": bies_to_entities(ner_preds[i][:n])})
        return out[0] if single else out
