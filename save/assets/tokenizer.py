"""BERTc 发布包自带的字级 tokenizer。

这份代码会**随模型一起发到 HF**,所以只能依赖 piece_tokenizer 本身,
不能 import 仓库里的任何东西。

词表文件 BERTc-Tokenizer.pt 与 PieceTokenizer 仓库 save/ 下的那份逐字节相同 ——
同名是为了让来源一目了然。

装 tokenizer:
    pip install git+https://github.com/Ismantic/PieceTokenizer
"""
from pathlib import Path

import piece_tokenizer as _pt


class PieceCharTokenizer:
    """字级 tokenizer。

    必须用 dict="no" 加载(字模式,不挂分词词典)—— 挂了词典编码结果会跟
    训练时不一致,而且不会报错。
    """

    MODEL_NAME = "BERTc-Tokenizer.pt"

    def __init__(self, model_dir="."):
        model_dir = Path(model_dir)
        self._tok = _pt.Tokenizer()
        self._tok.load(str(model_dir / self.MODEL_NAME), dict="no")

        self.pad_token_id = self._tok.piece_to_id("<pad>")
        self.unk_token_id = 0
        # [MASK] 追加在 piece 词表之后,id 就等于词表大小 —— 不需要单独存一个文件
        self.mask_token_id = self._tok.vocab_size()
        self.vocab_size = self._tok.vocab_size() + 1
        self._cache = {}

    def char_to_id(self, char: str) -> int:
        tid = self._cache.get(char)
        if tid is None:
            ids = self._tok.encode_as_ids(char)
            tid = ids[0] if ids else self.unk_token_id
            self._cache[char] = tid
        return tid

    def id_to_char(self, tid: int) -> str:
        piece = self._tok.id_to_piece(int(tid))
        return piece.replace("▁", "")

    def encode(self, text: str) -> list:
        return [self.char_to_id(c) for c in text]

    def batch(self, texts, max_len, device=None):
        """一批文本 → (input_ids, attention_mask, 每条的有效长度)。"""
        import torch

        lengths = [min(len(t), max_len) for t in texts]
        width = max(lengths) if lengths else 0
        input_ids = torch.full((len(texts), width), self.pad_token_id,
                               dtype=torch.long, device=device)
        attn = torch.zeros((len(texts), width), dtype=torch.long, device=device)
        for i, t in enumerate(texts):
            ids = self.encode(t[:lengths[i]])
            if ids:
                input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long,
                                                       device=device)
                attn[i, :len(ids)] = 1
        return input_ids, attn, lengths
