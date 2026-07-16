#!/usr/bin/env python3
"""Prepare local Hugging Face release folders for Modern BERTc backbones."""
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

MODELS = {
    "BERTc-165M": {
        "source": ROOT / "pretrain" / "modern_bertc" / "output_v4_mid" / "checkpoint-8500",
        "params": "165M",
        "layers": "12L / 1024H / 2752I / 16 heads",
        "mt": "score 1.4689 (CWS 0.9836 / POS 0.9753 / NER 0.9632)",
        "csc": "SIGHAN-15 sentence F1 0.8308",
        "notes": "First Modern BERTc backbone to reach broad MT/CSC SOTA at the 165M scale.",
    },
    "BERTc-315M": {
        "source": ROOT / "pretrain" / "modern_bertc" / "output_v4_large" / "checkpoint-8500",
        "params": "315M",
        "layers": "24L / 1024H / 2752I / 16 heads",
        "mt": "score 1.4712 (CWS 0.9840 / POS 0.9800 / NER 0.9660)",
        "csc": "SIGHAN-15 sentence F1 0.8346",
        "notes": "Current strongest Modern BERTc backbone, trained on the full 17.65B-token corpus.",
    },
}


README_TEMPLATE = """---
license: apache-2.0
language:
- zh
- en
tags:
- bert
- fill-mask
- chinese
- modernbert
- masked-language-modeling
pipeline_tag: fill-mask
library_name: pytorch
---

# {name}

{name} is a char-level Chinese Modern BERTc masked language model trained from scratch.
It uses a custom ModernBERT-style PyTorch architecture from the BERTc repository, with
ScaledSinusoidal positional embeddings, GeGLU MLPs, no linear biases, tied input/output
embeddings, and a SentencePiece-based char/BPE tokenizer.

## Model Details

- Parameters: {params}
- Architecture: {layers}
- Vocabulary size: 12,536
- Max position length: 1,024
- Pretraining data: 17.65B-token BERTc mixed corpus
- License: Apache-2.0

## Reported Downstream Results

These are internal BERTc evaluations using fine-tuned heads:

- PD-1998 CWS/POS/NER multi-task: {mt}
- SIGHAN-15 Chinese spelling correction: {csc}

{notes}

## Files

- `model.safetensors`: `ModernBertForMLM` state dict.
- `config.json`: architecture configuration.
- `model.py`: model implementation used by the original training code.
- `piece.model`: tokenizer model; load with `piece_tokenizer` using `cn_dict="no"`.
- `mask_token_id.txt`: mask token id.

## Loading

```python
import json
import torch
from safetensors.torch import load_file
from model import ModernBertConfig, ModernBertForMLM

with open("config.json") as f:
    cfg = ModernBertConfig(**json.load(f))

model = ModernBertForMLM(cfg)
state = load_file("model.safetensors")
model.load_state_dict(state, strict=True)
model.eval()
```

Tokenization in the original code uses the sibling `piece_tokenizer` package:

```python
import piece_tokenizer as pt

tok = pt.Tokenizer()
tok.load("piece.model", cn_dict="no")
ids = tok.encode_as_ids("中文测试")
```

## Intended Use

Use this model as a Chinese encoder/MLM backbone for fine-tuning tasks such as CWS,
POS, NER, and Chinese spelling correction. This release is not an instruction model
and is not intended for text generation.
"""


EXAMPLE_TEMPLATE = """import json
import torch
from safetensors.torch import load_file
from model import ModernBertConfig, ModernBertForMLM


def load_bertc(model_dir="."):
    with open(f"{model_dir}/config.json") as f:
        cfg = ModernBertConfig(**json.load(f))
    model = ModernBertForMLM(cfg)
    state = load_file(f"{model_dir}/model.safetensors")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


if __name__ == "__main__":
    model = load_bertc(".")
    ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    with torch.no_grad():
        out = model(ids)
    print(out["logits"].shape)
"""


def copy_required(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    shutil.copy2(src, dst)


def export_one(name: str, spec: dict[str, str], out_root: Path) -> None:
    src = Path(spec["source"])
    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(src / "model.pt", map_location="cpu", weights_only=False)
    state = ckpt.get("ema") or ckpt["model"]
    save_file(state, out / "model.safetensors")

    copy_required(src / "config.json", out / "config.json")
    copy_required(src / "mask_token_id.txt", out / "mask_token_id.txt")
    copy_required(TOKENIZER_DIR / "piece.model", out / "piece.model")
    copy_required(MODEL_CODE, out / "model.py")

    (out / "README.md").write_text(
        README_TEMPLATE.format(name=name, **spec),
        encoding="utf-8",
    )
    (out / "example_load.py").write_text(EXAMPLE_TEMPLATE, encoding="utf-8")
    (out / "release_metadata.json").write_text(
        json.dumps(
            {
                "name": name,
                "source_checkpoint": str(src.relative_to(ROOT)),
                "step": ckpt.get("step"),
                "metrics": {"mt": spec["mt"], "csc": spec["csc"]},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "hf_release"))
    parser.add_argument("--model", choices=sorted(MODELS), action="append")
    args = parser.parse_args()

    out_root = Path(args.out)
    selected = args.model or sorted(MODELS)
    for name in selected:
        export_one(name, MODELS[name], out_root)
        print(f"prepared {out_root / name}")


if __name__ == "__main__":
    main()
