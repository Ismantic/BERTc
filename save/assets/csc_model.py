"""BERTc-CSC 推理:中文拼写纠错。

这份代码随模型一起发到 HF,只依赖同目录的 model.py / tokenizer.py
和 torch。

    from csc_model import BERTcForCSC
    model = BERTcForCSC.from_pretrained(".")
    print(model.correct("我今天很稿兴"))   # 我今天很高兴
"""
import json
from pathlib import Path

import torch
import torch.nn as nn
from checkpoint import load_safetensors

from model import ModernBertConfig, ModernBertModel
from tokenizer import PieceCharTokenizer


class BERTcForCSC(nn.Module):
    """骨干 + 双头:cor 逐位置预测正确的字,det 判断该位置有没有错。

    cor_head 的权重与词嵌入绑定 —— 预训练的 MLM 头就是 logits = h @ embed.weightᵀ,
    换成独立的 Linear 会废掉这层对齐,纠错效果掉一大截。
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = ModernBertModel(config)
        self.cor_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.cor_head.weight = self.bert.embed.weight
        self.det_head = nn.Linear(config.hidden_size, 1)

    @classmethod
    def from_pretrained(cls, model_dir=".", map_location="cpu"):
        model_dir = Path(model_dir)
        cfg = ModernBertConfig.from_dict(
            json.loads((model_dir / "config.json").read_text()))
        model = cls(cfg)
        state = load_safetensors(model_dir / "model.safetensors",
                                 device=str(map_location))
        missing, unexpected = model.load_state_dict(state, strict=False)
        # cor_head.weight 跟 embed 共享,safetensors 不重复存,所以它一定"缺"
        if set(missing) != {"cor_head.weight"} or unexpected:
            raise RuntimeError(f"权重不匹配:缺 {missing},多 {unexpected}")
        model.cor_head.weight = model.bert.embed.weight
        model.eval()
        return model

    def forward(self, input_ids, attention_mask=None):
        h = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.cor_head(h), self.det_head(h).squeeze(-1)

    @torch.no_grad()
    def correct(self, texts, tokenizer=None, threshold=0.7, max_len=128,
                device=None):
        """纠错。返回修正后的句子(输入是单句就返回单句)。

        threshold 是纠错置信度下限:softmax 概率低于它就保留原字。
        低置信度的"纠正"绝大多数是误伤,这个阈值是精确率的主要来源。
        """
        if tokenizer is None:
            tokenizer = PieceCharTokenizer(Path(__file__).resolve().parent)
        single = isinstance(texts, str)
        texts = [texts] if single else list(texts)

        device = device or next(self.parameters()).device
        self.eval()
        input_ids, attn, lengths = tokenizer.batch(texts, max_len, device)

        cor_logits, _ = self(input_ids, attn)
        probs = torch.softmax(cor_logits.float(), dim=-1)
        top_prob, top_id = probs.max(dim=-1)

        out = []
        for i, text in enumerate(texts):
            n = lengths[i]
            chars = list(text[:n])
            for j in range(n):
                if float(top_prob[i, j]) < threshold:
                    continue
                c = tokenizer.id_to_char(int(top_id[i, j]))
                if len(c) == 1:            # 只接受单字替换,保证不改变长度
                    chars[j] = c
            out.append("".join(chars) + text[n:])   # 超长部分原样接回
        return out[0] if single else out
