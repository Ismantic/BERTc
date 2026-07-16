#!/usr/bin/env python3
"""Prepare Hugging Face release folder for BERTc-315M-CSC."""
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
- spelling-correction
- csc
- bert
- text-correction
library_name: pytorch
---

# {name}

{name} is a Chinese spelling correction model fine-tuned from
`{base_model}`. It uses a Modern BERTc encoder with two heads:

- correction head: tied to the input embedding matrix
- detection head: binary error detection

## Metrics

SIGHAN-15 sentence-level evaluation using the pycorrector-style 707-sample protocol:

- Sentence F1: **{sentence_f1:.4f}**
- Accuracy: **{accuracy}**
- Precision: **{precision}**
- Recall: **{recall}**
- TP/FP/FN/TN: {tp} / {fp} / {fn} / {tn}

Training recipe:

- backbone: `{base_model}`
- epochs: {epochs}
- batch size: {batch_size}
- learning rate: {learning_rate}
- warmup ratio: {warmup_ratio}
- detection loss weight: {det_weight}
- inference threshold: {threshold}
- max length: {max_len}

## Files

- `model.safetensors`: CSC state dict. `cor_head.weight` is intentionally omitted and tied to `bert.embed.weight` by `csc_model.py`.
- `config.json`: BERTc backbone architecture.
- `csc_config.json`: task and metric metadata.
- `model.py`: Modern BERTc implementation.
- `csc_model.py`: CSC wrapper and batch correction helper.
- `piece.model`: tokenizer model; load with `piece_tokenizer` using `cn_dict="no"`.

## Usage

```python
from csc_model import BERTcForCSC, PieceCharTokenizer

tok = PieceCharTokenizer(".")
model = BERTcForCSC.from_pretrained(".")
texts = ["少先队员因该为老人让坐。"]
print(model.correct(texts, tok, threshold=0.7))
```

This is not a generative model. It performs same-length character replacement for
Chinese spelling correction.
"""


CSC_MODEL = '''import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

from model import ModernBertConfig, ModernBertModel


class PieceCharTokenizer:
    def __init__(self, model_dir):
        import piece_tokenizer as pt
        model_dir = Path(model_dir)
        self._tok = pt.Tokenizer()
        self._tok.load(str(model_dir / "piece.model"), cn_dict="no")
        mask_path = model_dir / "mask_token_id.txt"
        self.mask_token_id = int(mask_path.read_text().strip()) if mask_path.exists() else self._tok.vocab_size()
        self.vocab_size = self._tok.vocab_size() + 1
        self.pad_token_id = self._tok.piece_to_id("<pad>")
        self.unk_token_id = 0
        self.cache = {}
        self.inv_cache = {}

    def char_to_id(self, char):
        if char in self.cache:
            return self.cache[char]
        ids = self._tok.encode_as_ids(char)
        tid = ids[0] if ids else self.unk_token_id
        self.cache[char] = tid
        self.inv_cache.setdefault(tid, char)
        return tid


class BERTcForCSC(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = ModernBertModel(config)
        self.cor_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.cor_head.weight = self.bert.embed.weight
        self.det_head = nn.Linear(config.hidden_size, 1)

    @classmethod
    def from_pretrained(cls, model_dir, map_location="cpu"):
        model_dir = Path(model_dir)
        cfg = ModernBertConfig(**json.loads((model_dir / "config.json").read_text()))
        model = cls(cfg)
        state = load_file(str(model_dir / "model.safetensors"), device=str(map_location))
        missing, unexpected = model.load_state_dict(state, strict=False)
        allowed_missing = {"cor_head.weight"}
        if set(missing) != allowed_missing or unexpected:
            raise RuntimeError(f"Bad state dict: missing={missing}, unexpected={unexpected}")
        model.cor_head.weight = model.bert.embed.weight
        model.eval()
        return model

    def forward(self, input_ids, attention_mask=None):
        h = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cor_logits = self.cor_head(h)
        det_logits = self.det_head(h).squeeze(-1)
        return cor_logits, det_logits

    @torch.no_grad()
    def correct(self, texts, tokenizer, threshold=0.7, max_len=128, device=None):
        if isinstance(texts, str):
            single = True
            texts = [texts]
        else:
            single = False
        device = device or next(self.parameters()).device
        self.eval()
        lengths = [min(len(t), max_len) for t in texts]
        max_l = max(lengths) if lengths else 0
        input_ids = torch.full((len(texts), max_l), tokenizer.pad_token_id, dtype=torch.long, device=device)
        attn = torch.zeros((len(texts), max_l), dtype=torch.long, device=device)
        for i, text in enumerate(texts):
            ids = [tokenizer.char_to_id(c) for c in text[:lengths[i]]]
            if ids:
                input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
                attn[i, :len(ids)] = 1
        cor_logits, _ = self(input_ids, attn)
        probs = F.softmax(cor_logits, dim=-1)
        top_probs, top_ids = probs.max(dim=-1)
        out = []
        for i, text in enumerate(texts):
            chars = list(text[:lengths[i]])
            pred = []
            for j, orig in enumerate(chars):
                tid = int(top_ids[i, j].item())
                prob = float(top_probs[i, j].item())
                pred.append(tokenizer.inv_cache.get(tid, orig) if prob >= threshold else orig)
            if len(text) > lengths[i]:
                pred.extend(list(text[lengths[i]:]))
            out.append("".join(pred))
        return out[0] if single else out
'''


EXAMPLE = """from csc_model import BERTcForCSC, PieceCharTokenizer

tok = PieceCharTokenizer(".")
model = BERTcForCSC.from_pretrained(".")
print(model.correct(["少先队员因该为老人让坐。"], tok, threshold=0.7))
"""


def copy_required(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="BERTc-315M-CSC")
    parser.add_argument("--out", default=None)
    parser.add_argument("--base-model", default="Ismantic/BERTc-315M")
    parser.add_argument("--backbone-dir", default=str(ROOT / "pretrain" / "modern_bertc" / "output_v4_large" / "checkpoint-8500"))
    parser.add_argument("--ckpt", default=str(ROOT / "finetune" / "sota" / "sota_csc_v4large_v8_best.pt"))
    parser.add_argument("--sentence-f1", type=float, default=0.8346)
    parser.add_argument("--accuracy", default="0.8430")
    parser.add_argument("--precision", default="0.9396")
    parser.add_argument("--recall", default="0.7507")
    parser.add_argument("--tp", default="280")
    parser.add_argument("--fp", default="18")
    parser.add_argument("--fn", default="93")
    parser.add_argument("--tn", default="316")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", default="3e-5")
    parser.add_argument("--warmup-ratio", default="0.1")
    parser.add_argument("--det-weight", default="0.3")
    parser.add_argument("--threshold", default="0.7")
    parser.add_argument("--max-len", type=int, default=128)
    args = parser.parse_args()

    out = Path(args.out) if args.out else ROOT / "hf_release" / args.name
    out.mkdir(parents=True, exist_ok=True)
    backbone_dir = Path(args.backbone_dir)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = dict(ckpt["model"])
    state.pop("cor_head.weight", None)
    save_file(state, out / "model.safetensors")

    copy_required(backbone_dir / "config.json", out / "config.json")
    copy_required(backbone_dir / "mask_token_id.txt", out / "mask_token_id.txt")
    copy_required(TOKENIZER_DIR / "piece.model", out / "piece.model")
    copy_required(MODEL_CODE, out / "model.py")

    csc_config = {
        "base_model": args.base_model,
        "task": "Chinese Spelling Correction",
        "threshold": float(args.threshold),
        "max_len": args.max_len,
        "epoch": ckpt["epoch"],
        "metrics": ckpt["metrics"],
        "args": ckpt["args"],
        "source_checkpoint": str(Path(args.ckpt).resolve().relative_to(ROOT)),
    }
    (out / "csc_config.json").write_text(
        json.dumps(csc_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out / "README.md").write_text(
        README_TEMPLATE.format(
            name=args.name,
            base_model=args.base_model,
            sentence_f1=args.sentence_f1,
            accuracy=args.accuracy,
            precision=args.precision,
            recall=args.recall,
            tp=args.tp,
            fp=args.fp,
            fn=args.fn,
            tn=args.tn,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            det_weight=args.det_weight,
            threshold=args.threshold,
            max_len=args.max_len,
        ),
        encoding="utf-8",
    )
    (out / "csc_model.py").write_text(CSC_MODEL, encoding="utf-8")
    (out / "example_correct.py").write_text(EXAMPLE, encoding="utf-8")
    print(f"prepared {out}")


if __name__ == "__main__":
    main()
