"""BertTokenizer-compatible adapter over piece_tokenizer.

让 NLP_BERT_CRF 的 train.py / data.py 能用 Summer/BERT 训出的 char-level
PieceTokenizer 模型,不动现有代码 API。

需求:
  - vocab_size / pad_token_id / unk_token_id 属性
  - encode(text, add_special_tokens=False) → list[int]
  - save_pretrained(output_dir):把 piece.model 等拷过去
"""
import os
import shutil


class PieceTokenizerAdapter:
    def __init__(self, model_dir):
        import piece_tokenizer as pt
        self._dir = model_dir
        piece_path = os.path.join(model_dir, "piece.model")
        if not os.path.exists(piece_path):
            raise FileNotFoundError(f"piece.model missing in {model_dir}")
        self._tok = pt.Tokenizer()
        # 训推一致:推理也用 cn_dict='no' 让 SP 端 SplitTextCn 跟训练对齐
        self._tok.load(piece_path, cn_dict="no")

        # 解析 vocab metadata。piece tokenizer 的 specials 来自训练时设置:
        #   sp_char_v1: <unk>=0, <s>=1, </s>=2, <pad>=16259, <user>=16260,
        #               <assistant>=16261, <system>=16262
        # BERT vocab 在 piece 基础上 +1 给 [MASK](mask_token_id 由 mask_token_id.txt 给定)
        piece_vocab = self._tok.vocab_size()
        mask_file = os.path.join(model_dir, "mask_token_id.txt")
        if os.path.exists(mask_file):
            self.mask_token_id = int(open(mask_file).read().strip())
            self.vocab_size = piece_vocab + 1  # BERT vocab = piece + 1 mask
        else:
            self.mask_token_id = piece_vocab
            self.vocab_size = piece_vocab + 1

        self.pad_token_id = self._tok.piece_to_id("<pad>")
        if self.pad_token_id <= 0:
            self.pad_token_id = 16259  # fallback to sp_char_v1 default
        self.unk_token_id = 0
        print(f"[PieceAdapter] vocab={self.vocab_size}, pad={self.pad_token_id}, "
              f"unk={self.unk_token_id}, mask={self.mask_token_id}")

    def encode(self, text, add_special_tokens=False):
        """单 char 或 短串 → token ids。NLP_BERT_CRF data.py 只传单个汉字。"""
        return self._tok.encode_as_ids(text)

    def save_pretrained(self, output_dir):
        """拷 piece tokenizer 文件到 ckpt 旁,方便 inference 重新加载。"""
        os.makedirs(output_dir, exist_ok=True)
        for f in ["piece.model", "mask_token_id.txt", "config.json"]:
            src = os.path.join(self._dir, f)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(output_dir, f))
