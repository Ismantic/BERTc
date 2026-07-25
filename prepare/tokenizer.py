"""字级 PieceTokenizer 的适配层。

只有 prepare/ 用得到 —— src/ 从头到尾读预编码好的 id,不碰文本,
所以 PieceTokenizer 不是 src/ 的依赖。

词表约定(来自 pretrain/modern_bertc/tokenizer/):
  piece.model         SentencePiece 模型,vocab_size() = 12535
  mask_token_id.txt   [MASK] 的 id = 12535,追加在 piece 词表之后
  → BERT 侧的 vocab_size = 12535 + 1 = 12536

**必须用 dict="no" 加载**(字模式,不挂分词词典)。挂了词典 SentencePiece
的 SplitTextCn 行为就跟训练时不一致,编码结果整体偏掉,而且不会报错。
(参数在 2026-07 的 PieceTokenizer 里从 cn_dict 改名成了 dict。)
"""
import shutil
from pathlib import Path

import piece_tokenizer as _pt


class PieceTokenizer:
    """字级 tokenizer,带 字→id 缓存。

    缓存不只是为了快:CSC 的评测要把预测的 id 还原成字,而反查表的范围
    必须跟编码时见过的字一致 —— 用全词表反查会让本该"保留原字"的未知 id
    变成真解码,口径就跟 pycorrector 对不上了。见 id_to_char()。
    """

    def __init__(self, model_dir):
        self.dir = Path(model_dir)
        piece_path = self.dir / "piece.model"
        if not piece_path.exists():
            raise FileNotFoundError(f"{self.dir} 下没有 piece.model")

        self._tok = _pt.Tokenizer()
        self._tok.load(str(piece_path), dict="no")

        piece_vocab = self._tok.vocab_size()
        mask_file = self.dir / "mask_token_id.txt"
        self.mask_token_id = (int(mask_file.read_text().strip())
                              if mask_file.exists() else piece_vocab)
        self.vocab_size = piece_vocab + 1

        self.pad_token_id = self._tok.piece_to_id("<pad>")
        if self.pad_token_id <= 0:
            self.pad_token_id = 16259          # sp_char_v1 的老默认值
        self.unk_token_id = 0
        self._cache: dict[str, int] = {}

    def __repr__(self) -> str:
        return (f"PieceTokenizer(vocab={self.vocab_size}, pad={self.pad_token_id}, "
                f"unk={self.unk_token_id}, mask={self.mask_token_id})")

    def char_to_id(self, c: str) -> int:
        """单字 → id。编码不出东西时落到 unk。"""
        tid = self._cache.get(c)
        if tid is None:
            ids = self._tok.encode_as_ids(c)
            tid = ids[0] if ids else self.unk_token_id
            self._cache[c] = tid
        return tid

    def encode(self, text: str) -> list[int]:
        """整串 → id 序列。字模式下逐字走缓存,比整串 encode 快得多。"""
        return [self.char_to_id(c) for c in text]

    def encode_raw(self, text: str) -> list[int]:
        """不走缓存,直接交给 SentencePiece。用于非字模式的场合。"""
        return self._tok.encode_as_ids(text) if text else []

    def id_to_char(self) -> dict[int, str]:
        """当前缓存的反查表(id → 字)。

        只覆盖**编码过程中真正见过的字**。CSC 评测靠它把预测还原成句子,
        范围放大会改变口径 —— 见类文档。同一个 id 对应多个字时保留先见到的那个。
        """
        out: dict[int, str] = {}
        for c, i in self._cache.items():
            out.setdefault(i, c)
        return out

    def warm_cache(self, texts) -> int:
        """预热缓存。返回见过的不同字数。"""
        for t in texts:
            for c in t:
                if c not in self._cache:
                    self.char_to_id(c)
        return len(self._cache)

    def copy_assets(self, output_dir) -> None:
        """把 tokenizer 文件拷到 ckpt 旁,方便推理时直接加载。"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in ("piece.model", "mask_token_id.txt", "config.json"):
            src = self.dir / name
            if src.exists():
                shutil.copy2(src, output_dir / name)


DEFAULT_TOKENIZER_DIR = (Path(__file__).resolve().parents[1]
                         / "pretrain" / "modern_bertc" / "tokenizer")


def load_tokenizer(model_dir=None) -> PieceTokenizer:
    return PieceTokenizer(model_dir or DEFAULT_TOKENIZER_DIR)
