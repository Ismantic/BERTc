#!/usr/bin/env python3
"""Prepare Hugging Face release folder for BERTc-315M-MT."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_DIR = ROOT / "pretrain" / "modern_bertc" / "tokenizer"
MODEL_CODE = ROOT / "pretrain" / "modern_bertc" / "model.py"


README_TEMPLATE = """---
license: apache-2.0
language:
- zh
tags:
- chinese
- cws
- pos
- ner
- multi-task
- bert
library_name: pytorch
---

# {name}

{name} is a Chinese multi-task tagging model fine-tuned from
`{base_model}`. It predicts:

- CWS: Chinese word segmentation
- POS: PD-1998 POS tags mapped to the LTP tag set
- NER: Nh / Ns / Ni entity spans in BIES format

## Metrics

PD-1998 dev joint score:

- Joint score: **{joint_score}**
- CWS micro F1: **{cws_f1}**
- POS per-word acc: **{pos_acc}**
- NER micro F1: **{ner_f1}**

Training recipe:

- backbone: `{base_model}`
- epochs: {epochs}
- batch size: {batch_size}
- learning rate: `bert_lr={bert_lr}`, `head_lr={head_lr}`
- warmup ratio: {warmup_ratio}
- FGM: enabled, `eps={fgm_eps}`
- loss weights: `alpha_pos={alpha_pos}`, `beta_ner={beta_ner}`

## Files

- `model.safetensors`: MT state dict for the backbone and task heads.
- `config.json`: BERTc backbone architecture.
- `model.py`: Modern BERTc implementation.
- `mt_model.py`: MT wrapper, tokenizer adapter, and decode helpers.
- `piece.model`: tokenizer model; load with `piece_tokenizer` using `dict="no"`.
- `mask_token_id.txt`: mask token id used by the backbone tokenizer.
- `mt_config.json`: task metadata and source checkpoint.

## Usage

```python
from mt_model import BERTcForMT, PieceCharTokenizer

tok = PieceCharTokenizer(".")
model = BERTcForMT.from_pretrained(".")
out = model.predict("中华人民共和国万岁", tok)
print(out["words"])
print(out["pos"])
print(out["ner"])
```
"""


MT_MODEL = '''import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from torchcrf import CRF

from model import ModernBertConfig, ModernBertModel


CWS_ID2TAG = {0: "B", 1: "I", 2: "E", 3: "S"}
LTP_POS_TAGS = [
    "a", "b", "c", "d", "e", "h", "i", "j", "k", "m",
    "n", "nd", "nh", "ni", "nl", "ns", "nt", "nz", "o", "p",
    "q", "r", "u", "v", "wp", "x", "z",
]
LTP_ID2POS = {i: t for i, t in enumerate(LTP_POS_TAGS)}
LTP_NER_TAGS = [
    "O",
    "B-Nh", "I-Nh", "E-Nh", "S-Nh",
    "B-Ns", "I-Ns", "E-Ns", "S-Ns",
    "B-Ni", "I-Ni", "E-Ni", "S-Ni",
]
LTP_ID2NER = {i: t for i, t in enumerate(LTP_NER_TAGS)}


def bies_to_words(chars, tag_ids):
    words = []
    buf = []
    for ch, tid in zip(chars, tag_ids):
        tag = CWS_ID2TAG.get(int(tid), "S")
        if tag == "S":
            if buf:
                words.append("".join(buf))
                buf = []
            words.append(ch)
        elif tag == "B":
            if buf:
                words.append("".join(buf))
            buf = [ch]
        elif tag == "I":
            buf.append(ch)
        elif tag == "E":
            buf.append(ch)
            words.append("".join(buf))
            buf = []
        else:
            words.append(ch)
    if buf:
        words.append("".join(buf))
    return words


def bies_tags_to_spans(tag_ids):
    spans = []
    i = 0
    n = len(tag_ids)
    while i < n:
        tag = LTP_ID2NER.get(int(tag_ids[i]), "O")
        if tag.startswith("S-"):
            spans.append({"type": tag[2:], "start": i, "end": i + 1})
            i += 1
            continue
        if tag.startswith("B-"):
            ent = tag[2:]
            j = i + 1
            while j < n:
                nxt = LTP_ID2NER.get(int(tag_ids[j]), "O")
                if nxt == f"I-{ent}":
                    j += 1
                    continue
                if nxt == f"E-{ent}":
                    spans.append({"type": ent, "start": i, "end": j + 1})
                j += 1
                break
            i = j
            continue
        i += 1
    return spans


class PieceCharTokenizer:
    def __init__(self, model_dir):
        import piece_tokenizer as pt

        model_dir = Path(model_dir)
        self._tok = pt.Tokenizer()
        self._tok.load(str(model_dir / "piece.model"), dict="no")
        self.pad_token_id = self._tok.piece_to_id("<pad>")
        self.unk_token_id = 0
        mask_path = model_dir / "mask_token_id.txt"
        self.mask_token_id = int(mask_path.read_text().strip()) if mask_path.exists() else self._tok.vocab_size()
        self.vocab_size = self._tok.vocab_size() + 1
        self._cache = {}

    def char_to_id(self, char):
        if char in self._cache:
            return self._cache[char]
        ids = self._tok.encode_as_ids(char)
        tid = ids[0] if ids else self.unk_token_id
        self._cache[char] = tid
        return tid


class BERTcForMT(nn.Module):
    def __init__(self, config, num_pos=27, num_cws=4, num_ner=13, dropout=0.1):
        super().__init__()
        self.config = config
        self.bert = ModernBertModel(config)
        hidden = config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.cws_classifier = nn.Linear(hidden, num_cws)
        self.cws_crf = CRF(num_cws, batch_first=True)
        self.pos_classifier = nn.Linear(hidden, num_pos)
        self.ner_classifier = nn.Linear(hidden, num_ner)
        self.ner_crf = CRF(num_ner, batch_first=True)

    @classmethod
    def from_pretrained(cls, model_dir, map_location="cpu"):
        model_dir = Path(model_dir)
        cfg = ModernBertConfig(**json.loads((model_dir / "config.json").read_text()))
        model = cls(cfg)
        state = load_file(str(model_dir / "model.safetensors"), device=str(map_location))
        missing, unexpected = model.load_state_dict(state, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"Bad state dict: missing={missing}, unexpected={unexpected}")
        model.eval()
        return model

    def forward(self, input_ids, attention_mask, cws_labels=None, pos_labels=None, ner_labels=None):
        hs = self.bert(input_ids, attention_mask=attention_mask)
        hs = self.dropout(hs)
        cws_emi = self.cws_classifier(hs)
        pos_logits = self.pos_classifier(hs)
        ner_emi = self.ner_classifier(hs)
        mask = attention_mask.bool()
        losses = {}
        if cws_labels is not None:
            losses["cws"] = -self.cws_crf(cws_emi, cws_labels, mask=mask, reduction="mean")
        if pos_labels is not None:
            losses["pos"] = F.cross_entropy(
                pos_logits.view(-1, pos_logits.size(-1)),
                pos_labels.view(-1),
                ignore_index=-100,
            )
        if ner_labels is not None:
            losses["ner"] = -self.ner_crf(ner_emi, ner_labels, mask=mask, reduction="mean")
        return losses, (cws_emi, pos_logits, ner_emi)

    @torch.no_grad()
    def decode_cws(self, input_ids, attention_mask):
        hs = self.bert(input_ids, attention_mask=attention_mask)
        emi = self.cws_classifier(self.dropout(hs)).float()
        return self.cws_crf.decode(emi, mask=attention_mask.bool())

    @torch.no_grad()
    def decode_ner(self, input_ids, attention_mask):
        hs = self.bert(input_ids, attention_mask=attention_mask)
        emi = self.ner_classifier(self.dropout(hs)).float()
        return self.ner_crf.decode(emi, mask=attention_mask.bool())

    @torch.no_grad()
    def predict_pos(self, input_ids, attention_mask):
        hs = self.bert(input_ids, attention_mask=attention_mask)
        logits = self.pos_classifier(self.dropout(hs))
        return logits.argmax(-1)

    @torch.no_grad()
    def predict(self, text, tokenizer=None, max_len=254, device=None):
        if tokenizer is None:
            tokenizer = PieceCharTokenizer(Path(__file__).resolve().parent)
        if isinstance(text, str):
            single = True
            texts = [text]
        else:
            single = False
            texts = list(text)
        device = device or next(self.parameters()).device
        self.eval()
        lengths = [min(len(t), max_len) for t in texts]
        max_l = max(lengths) if lengths else 0
        input_ids = torch.full((len(texts), max_l), tokenizer.pad_token_id, dtype=torch.long, device=device)
        attn = torch.zeros((len(texts), max_l), dtype=torch.long, device=device)
        for i, s in enumerate(texts):
            ids = [tokenizer.char_to_id(c) for c in s[:lengths[i]]]
            if ids:
                input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
                attn[i, :len(ids)] = 1
        cws_preds = self.decode_cws(input_ids, attn)
        pos_pred = self.predict_pos(input_ids, attn).cpu().tolist()
        ner_preds = self.decode_ner(input_ids, attn)

        out = []
        for i, s in enumerate(texts):
            chars = list(s[:lengths[i]])
            cws_ids = cws_preds[i][:len(chars)]
            pos_ids = pos_pred[i][:len(chars)]
            ner_ids = ner_preds[i][:len(chars)]
            words = bies_to_words(chars, cws_ids)
            pos = []
            offset = 0
            for w in words:
                if offset < len(pos_ids):
                    pos.append(LTP_ID2POS.get(int(pos_ids[offset]), "x"))
                else:
                    pos.append("x")
                offset += len(w)
            out.append({
                "text": s,
                "words": words,
                "pos": pos,
                "ner": bies_tags_to_spans(ner_ids),
            })
        return out[0] if single else out
'''


EXAMPLE = """from mt_model import BERTcForMT, PieceCharTokenizer

tok = PieceCharTokenizer(".")
model = BERTcForMT.from_pretrained(".")
result = model.predict("中华人民共和国万岁", tok)
print(result["words"])
print(result["pos"])
print(result["ner"])
"""


def copy_required(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="BERTc-315M-MT")
    parser.add_argument("--out", default=None)
    parser.add_argument("--base-model", default="Ismantic/BERTc-315M")
    parser.add_argument("--backbone-dir", default=str(ROOT / "pretrain" / "modern_bertc" / "output_v4_large" / "checkpoint-8500"))
    parser.add_argument("--ckpt", default=str(ROOT / "finetune" / "sota" / "sota_mt_v4large_fgm_5ep_best.pt"))
    parser.add_argument("--joint-score", type=float, default=1.4712)
    parser.add_argument("--cws-f1", type=float, default=0.9840)
    parser.add_argument("--pos-acc", type=float, default=0.9800)
    parser.add_argument("--ner-f1", type=float, default=0.9660)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bert-lr", default="2e-5")
    parser.add_argument("--head-lr", default="5e-4")
    parser.add_argument("--warmup-ratio", default="0.1")
    parser.add_argument("--fgm-eps", default="1.0")
    parser.add_argument("--alpha-pos", default="2.0")
    parser.add_argument("--beta-ner", default="0.5")
    args = parser.parse_args()

    out = Path(args.out) if args.out else ROOT / "hf_release" / args.name
    out.mkdir(parents=True, exist_ok=True)
    backbone_dir = Path(args.backbone_dir)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise RuntimeError("Expected a checkpoint dict")
    save_file(ckpt, out / "model.safetensors")

    copy_required(backbone_dir / "config.json", out / "config.json")
    copy_required(backbone_dir / "mask_token_id.txt", out / "mask_token_id.txt")
    copy_required(TOKENIZER_DIR / "piece.model", out / "piece.model")
    copy_required(MODEL_CODE, out / "model.py")

    mt_config = {
        "base_model": args.base_model,
        "task": "Chinese Word Segmentation + POS + NER",
        "metric": {
            "joint_score": args.joint_score,
            "cws_micro_f1": args.cws_f1,
            "pos_per_word_acc": args.pos_acc,
            "ner_micro_f1": args.ner_f1,
        },
        "training_recipe": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "bert_lr": args.bert_lr,
            "head_lr": args.head_lr,
            "warmup_ratio": args.warmup_ratio,
            "fgm": True,
            "fgm_eps": args.fgm_eps,
            "alpha_pos": args.alpha_pos,
            "beta_ner": args.beta_ner,
        },
        "source_checkpoint": str(Path(args.ckpt).resolve().relative_to(ROOT)),
    }
    (out / "mt_config.json").write_text(
        json.dumps(mt_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out / "README.md").write_text(
        README_TEMPLATE.format(
            name=args.name,
            base_model=args.base_model,
            joint_score=args.joint_score,
            cws_f1=args.cws_f1,
            pos_acc=args.pos_acc,
            ner_f1=args.ner_f1,
            epochs=args.epochs,
            batch_size=args.batch_size,
            bert_lr=args.bert_lr,
            head_lr=args.head_lr,
            warmup_ratio=args.warmup_ratio,
            fgm_eps=args.fgm_eps,
            alpha_pos=args.alpha_pos,
            beta_ner=args.beta_ner,
        ),
        encoding="utf-8",
    )
    (out / "mt_model.py").write_text(MT_MODEL, encoding="utf-8")
    (out / "example_decode.py").write_text(EXAMPLE, encoding="utf-8")
    print(f"prepared {out}")


if __name__ == "__main__":
    main()
