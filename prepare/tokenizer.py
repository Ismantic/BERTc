"""字级 PieceTokenizer 的适配层。

只有 prepare/ 用得到 —— src/ 从头到尾读预编码好的 id,不碰文本,
所以 PieceTokenizer 不是 src/ 的依赖。

词表不放在本仓库,而是取自 PieceTokenizer 仓库的 save/BERTc-Tokenizer.pt ——
只有一个来源,不会出现两边各存一份、慢慢漂移。路径通过已安装的 piece_tokenizer 模块反查,
所以 PieceTokenizer clone 在哪都能找到;也可以用 BERTC_PIECE_MODEL 覆盖。

  vocab_size() = 12535 是 piece 词表大小
  [MASK] 追加在其后,id = 12535
  → BERT 侧 vocab_size = 12536

**必须用 dict="no" 加载**(字模式,不挂分词词典)。挂了词典 SentencePiece
的 SplitTextCn 行为就跟训练时不一致,编码结果整体偏掉,而且不会报错。
"""
import os
import shutil
from pathlib import Path

import piece_tokenizer as _pt

PIECE_MODEL_NAME = "BERTc-Tokenizer.pt"


def default_piece_model() -> Path:
    """定位词表文件。

    piece_tokenizer 是 editable 安装,__file__ 就在 PieceTokenizer 仓库根目录,
    所以能顺着找到同仓库的 save/BERTc-Tokenizer.pt。
    """
    env = os.environ.get("BERTC_PIECE_MODEL")
    if env:
        return Path(env)
    return Path(_pt.__file__).resolve().parent / "save" / PIECE_MODEL_NAME


class PieceTokenizer:
    """字级 tokenizer,带 字→id 缓存。

    缓存不只是为了快:CSC 的评测要把预测的 id 还原成字,而反查表的范围
    必须跟编码时见过的字一致 —— 用全词表反查会让本该"保留原字"的未知 id
    变成真解码,口径就跟 pycorrector 对不上了。见 id_to_char()。
    """

    def __init__(self, piece_model=None):
        self.path = Path(piece_model) if piece_model else default_piece_model()
        if not self.path.exists():
            raise FileNotFoundError(
                f"找不到词表 {self.path}。\n"
                f"跑 bash prepare/install_deps.sh piece 安装 PieceTokenizer,"
                f"或用 BERTC_PIECE_MODEL 指定路径。")

        self._tok = _pt.Tokenizer()
        self._tok.load(str(self.path), dict="no")

        piece_vocab = self._tok.vocab_size()
        # [MASK] 追加在 piece 词表之后,所以它的 id 就等于 piece 词表大小
        self.mask_token_id = piece_vocab
        self.vocab_size = piece_vocab + 1

        self.pad_token_id = self._tok.piece_to_id("<pad>")
        if self.pad_token_id <= 0:
            self.pad_token_id = 16259          # sp_char_v1 的老默认值
        self.unk_token_id = 0
        self._cache: dict[str, int] = {}

    def __repr__(self) -> str:
        return (f"PieceTokenizer({self.path.name}: vocab={self.vocab_size}, "
                f"pad={self.pad_token_id}, unk={self.unk_token_id}, "
                f"mask={self.mask_token_id})")

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

    def copy_to(self, output_dir, name: str = PIECE_MODEL_NAME) -> Path:
        """把词表拷到发布目录。沿用 PieceTokenizer 仓库里的文件名,来源一目了然。"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        dst = output_dir / name
        shutil.copy2(self.path, dst)
        return dst


def load_tokenizer(piece_model=None) -> PieceTokenizer:
    return PieceTokenizer(piece_model)
