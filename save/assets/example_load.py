"""加载 BERTc 骨干做掩码语言建模。"""
import torch

from model import ModernBertConfig, ModernBertForMLM   # noqa: F401
from tokenizer import PieceCharTokenizer

import json
from safetensors.torch import load_file


def load_bertc(model_dir="."):
    with open(f"{model_dir}/config.json") as f:
        cfg = ModernBertConfig(**json.load(f))
    model = ModernBertForMLM(cfg)
    model.load_state_dict(load_file(f"{model_dir}/model.safetensors"), strict=True)
    model.eval()
    return model


if __name__ == "__main__":
    tok = PieceCharTokenizer(".")
    model = load_bertc(".")

    text = "北京是中国的首都"
    ids = torch.tensor([tok.encode(text)], dtype=torch.long)
    ids[0, 2] = tok.mask_token_id                     # 把"是"盖住
    with torch.no_grad():
        logits = model(ids)["logits"]
    pred = int(logits[0, 2].argmax())
    print(f"{text} → 第 3 个字预测为 {tok.id_to_char(pred)!r}")
