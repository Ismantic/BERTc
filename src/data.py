"""下游任务的 Dataset / Collator。只依赖 torch。

src/ 不碰文本 —— 分词、字→id、标签构造全部在 prepare/ 完成,这里只读预编码好的
id 序列并 pad 成 batch。所以 tokenizer 不是 src/ 的依赖。

预编码文件用 torch.save 存一个 dict,变长序列用「扁平数组 + offsets」表示
(N 条样本就存 N+1 个 offset,比存 N 个小 tensor 省内存也快得多):

  MT(CWS + POS + NER 联合):
    format      "bertc-mt-v1"
    offsets     int64 (N+1,)
    input_ids   int32 (T,)     T = offsets[-1]
    cws_tags    int32 (T,)
    pos_tags    int32 (T,)     -100 = 该位置无 POS 监督
    ner_tags    int32 (T,)
    cws_vocab   list[str]      如 ["B","I","E","S"]
    pos_vocab   list[str]
    ner_vocab   list[str]      如 ["O","B-Nh",...,"S-Ni"]
    pad_token_id int

  CSC:
    format      "bertc-csc-v1"
    offsets     int64 (N+1,)
    input_ids   int32 (T,)     错句
    cor_labels  int32 (T,)     正句
    det_labels  uint8 (T,)     该位置是否有错
    pad_token_id int

两点约定,改了会静默影响结果:

  - **不在预编码阶段截断**。max_len 是训练超参(CSC 用 128,MT 用 254),
    留在这里截,换个长度不用重跑 prepare。
  - **det_labels 必须预先算好**,不能在这里用 input_ids != cor_labels 现推。
    检测标签是按**字**比对的;两个不同的字可能都落到 UNK,按 id 比会漏掉那处错误。
"""
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset


def _load(path, expect_format: str) -> dict:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    got = blob.get("format")
    if got != expect_format:
        raise ValueError(f"{path} 的 format 是 {got!r},期望 {expect_format!r} —— "
                         f"是不是传错了数据集文件?")
    return blob


class _PackedDataset(Dataset):
    """扁平数组 + offsets 的公共部分。"""

    def __init__(self, blob: dict, fields: tuple[str, ...]):
        self.offsets = blob["offsets"]
        self.fields = fields
        self.arrays = {f: blob[f] for f in fields}
        self.pad_token_id = blob["pad_token_id"]

    def __len__(self) -> int:
        return self.offsets.numel() - 1

    def _slice(self, idx: int) -> dict:
        lo, hi = int(self.offsets[idx]), int(self.offsets[idx + 1])
        return {f: arr[lo:hi] for f, arr in self.arrays.items()}


# ---------------------------------------------------------------- MT

class MTDataset(_PackedDataset):
    """CWS + POS + NER 三任务共用一条字序列。

    max_chars 截断后会修 CWS 标签:如果最后一个字被切成 B(词首)或 I(词中),
    改成 S / E,否则留下一个语法上不可能的 BIES 序列,CRF 会被这些噪声拉偏。
    """

    FIELDS = ("input_ids", "cws_tags", "pos_tags", "ner_tags")

    def __init__(self, path, max_chars: int = 254):
        blob = _load(path, "bertc-mt-v1")
        super().__init__(blob, self.FIELDS)
        self.max_chars = max_chars
        self.cws_vocab = blob["cws_vocab"]
        self.pos_vocab = blob["pos_vocab"]
        self.ner_vocab = blob["ner_vocab"]
        self._cws_id = {t: i for i, t in enumerate(self.cws_vocab)}

    @property
    def num_cws_tags(self) -> int:
        return len(self.cws_vocab)

    @property
    def num_pos_tags(self) -> int:
        return len(self.pos_vocab)

    @property
    def num_ner_tags(self) -> int:
        return len(self.ner_vocab)

    def __getitem__(self, idx: int) -> dict:
        item = self._slice(idx)
        if item["input_ids"].numel() > self.max_chars:
            item = {k: v[:self.max_chars].clone() for k, v in item.items()}
            last = int(item["cws_tags"][-1])
            if last == self._cws_id["B"]:
                item["cws_tags"][-1] = self._cws_id["S"]
            elif last == self._cws_id["I"]:
                item["cws_tags"][-1] = self._cws_id["E"]
        return item


class MTCollator:
    """pad 成 (B, L)。pos 的 pad 值是 -100(cross entropy 忽略),
    cws / ner 走 CRF,pad 位置由 attention_mask 屏蔽,填 0 即可。"""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict]) -> dict:
        lengths = [item["input_ids"].numel() for item in batch]
        B, L = len(batch), max(lengths)
        out = {
            "input_ids": torch.full((B, L), self.pad_token_id, dtype=torch.long),
            "attention_mask": torch.zeros((B, L), dtype=torch.long),
            "cws_labels": torch.zeros((B, L), dtype=torch.long),
            "pos_labels": torch.full((B, L), -100, dtype=torch.long),
            "ner_labels": torch.zeros((B, L), dtype=torch.long),
        }
        for i, (item, n) in enumerate(zip(batch, lengths)):
            out["input_ids"][i, :n] = item["input_ids"].long()
            out["attention_mask"][i, :n] = 1
            out["cws_labels"][i, :n] = item["cws_tags"].long()
            out["pos_labels"][i, :n] = item["pos_tags"].long()
            out["ner_labels"][i, :n] = item["ner_tags"].long()
        return out


# ---------------------------------------------------------------- CSC

class CSCDataset(_PackedDataset):
    FIELDS = ("input_ids", "cor_labels", "det_labels")

    def __init__(self, path, max_len: int = 128):
        blob = _load(path, "bertc-csc-v1")
        super().__init__(blob, self.FIELDS)
        self.max_len = max_len

    def __getitem__(self, idx: int) -> dict:
        item = self._slice(idx)
        if item["input_ids"].numel() > self.max_len:
            item = {k: v[:self.max_len] for k, v in item.items()}
        return item


class CSCCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict]) -> dict:
        lengths = [item["input_ids"].numel() for item in batch]
        B, L = len(batch), max(lengths)
        out = {
            "input_ids": torch.full((B, L), self.pad_token_id, dtype=torch.long),
            "attention_mask": torch.zeros((B, L), dtype=torch.long),
            "cor_labels": torch.full((B, L), -100, dtype=torch.long),
            "det_labels": torch.zeros((B, L), dtype=torch.float),
        }
        for i, (item, n) in enumerate(zip(batch, lengths)):
            out["input_ids"][i, :n] = item["input_ids"].long()
            out["attention_mask"][i, :n] = 1
            out["cor_labels"][i, :n] = item["cor_labels"].long()
            out["det_labels"][i, :n] = item["det_labels"].float()
        return out


# ---------------------------------------------------------------- 预训练

class PackedMLMDataset(Dataset):
    """预训练语料:定长 chunk 的 memmap。

    prepare/ 产出三个平行文件,形状相同、逐 token 对齐:
      <name>.pt       int32/uint16  token id
      <name>.pt.wid   同 dtype      词 id,WWM 用(同一个词的字共享 wid)
      <name>.pt.seg   uint8         文档 id,跨文档 attention 隔离用

    每个文件旁边有 <file>.meta(json,记 dtype 和 shape)。用 torch.from_file
    做内存映射 —— 17.65B token 的语料展开是 70GB,不可能全读进内存。
    """

    def __init__(self, path, word_ids_path=None, seg_ids_path=None,
                 max_chunks: Optional[int] = None):
        self.data = _memmap(path)
        if max_chunks is not None and self.data.shape[0] > max_chunks:
            self.data = self.data[:max_chunks]
        self.n_chunks, self.seq_len = self.data.shape

        self.word_ids = self.seg_ids = None
        if word_ids_path is not None:
            self.word_ids = _memmap(word_ids_path)[:self.n_chunks]
        if seg_ids_path is not None:
            self.seg_ids = _memmap(seg_ids_path)[:self.n_chunks]

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, idx: int):
        out = [self.data[idx].long()]
        if self.word_ids is not None:
            out.append(self.word_ids[idx].long())
        if self.seg_ids is not None:
            out.append(self.seg_ids[idx].long())
        return out[0] if len(out) == 1 else tuple(out)


_DTYPES = {
    "uint8": torch.uint8, "int8": torch.int8,
    "uint16": torch.uint16, "int16": torch.int16,
    "uint32": torch.uint32, "int32": torch.int32,
    "int64": torch.int64,
}


def _memmap(path) -> torch.Tensor:
    """按 <path>.meta 里的 dtype/shape 内存映射一个平铺的二进制文件。

    用 torch.from_file 而不是 numpy.memmap —— src/ 只依赖 torch。
    没有 .meta 时退回 torch.load(旧格式,整份读进内存)。
    """
    import json

    path = Path(path)
    meta_path = Path(str(path) + ".meta")
    if not meta_path.exists():
        return torch.load(path, map_location="cpu", weights_only=True)

    meta = json.loads(meta_path.read_text())
    dtype = _DTYPES.get(meta["dtype"])
    if dtype is None:
        raise ValueError(f"{meta_path} 里未知的 dtype: {meta['dtype']}")
    shape = tuple(meta["shape"])
    numel = 1
    for s in shape:
        numel *= s
    # shared=True 必须的:它走 MAP_SHARED,按页惰性加载;shared=False 要一份私有
    # 副本,17.65B token 的语料展开是 70GB,直接 "Cannot allocate memory"。
    # 代价是这块映射可写、写入会落回文件 —— 所以下面一律只读,
    # __getitem__ 里 .long() 会复制出新张量,不会碰到原映射。
    t = torch.from_file(str(path), shared=True, size=numel, dtype=dtype)
    return t.view(*shape)
