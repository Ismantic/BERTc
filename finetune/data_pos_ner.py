"""POS / NER dataset (LTP-aligned label set)。

POS: PD-1998 43 tag → LTP 27 tag (alphabetic fixed vocab, see ltp_label_align)
NER: BIES + 3 type (Nh/Ns/Ni) = 13 tag, 弃 MISC + parser-bug 类
"""
import json
from torch.utils.data import Dataset

from ltp_label_align import (
    LTP_POS2ID, LTP_ID2POS, LTP_POS_TAGS,
    LTP_NER2ID, LTP_ID2NER, LTP_NER_TAGS,
    map_pd_pos, entity_to_bies, PD2LTP_NER,
)


def build_pos_vocab(jsonl_path=None):
    """返回 LTP fixed 27-tag vocab(忽略 jsonl 参数,仅 sanity)。"""
    return dict(LTP_POS2ID)


NER_TAGS = LTP_NER2ID
ID2NER = LTP_ID2NER


class POSDataset(Dataset):
    """LTP-aligned POS: 每 word PD tag → LTP tag → 复制到每个 char。"""
    def __init__(self, jsonl_path, pos2id=None, max_chars=254):
        if pos2id is None: pos2id = LTP_POS2ID
        self.items = []
        with open(jsonl_path, encoding="utf8") as f:
            for line in f:
                obj = json.loads(line)
                words = obj.get("words", [])
                pos_seq = obj.get("pos", [])
                if not words or len(words) != len(pos_seq): continue
                chars, tags = [], []
                for w, p in zip(words, pos_seq):
                    ltp_p = map_pd_pos(p)
                    pid = pos2id.get(ltp_p, pos2id['x'])
                    for c in w:
                        chars.append(c)
                        tags.append(pid)
                if not chars: continue
                if len(chars) > max_chars:
                    chars = chars[:max_chars]
                    tags = tags[:max_chars]
                self.items.append({"chars": chars, "tags": tags})

    def __len__(self): return len(self.items)
    def __getitem__(self, idx): return self.items[idx]


class NERDataset(Dataset):
    """LTP-aligned BIES NER: 弃 MISC / 异常类。"""
    def __init__(self, jsonl_path, max_chars=254):
        self.items = []
        with open(jsonl_path, encoding="utf8") as f:
            for line in f:
                obj = json.loads(line)
                text = obj.get("text", "")
                if not text: continue
                chars = list(text)
                tags = [LTP_NER2ID["O"]] * len(chars)
                for ent in obj.get("entities", []):
                    t = ent.get("type", "MISC")
                    if t not in PD2LTP_NER: continue
                    s, e = ent["start"], ent["end"]
                    bies = entity_to_bies(s, e, t, len(chars))
                    if not bies: continue
                    for pos, tid in bies:
                        tags[pos] = tid
                if len(chars) > max_chars:
                    chars = chars[:max_chars]
                    tags = tags[:max_chars]
                if not chars: continue
                self.items.append({"chars": chars, "tags": tags})

    def __len__(self): return len(self.items)
    def __getitem__(self, idx): return self.items[idx]


def ner_tags_to_spans(tags):
    """BIES seq → entity spans [{type, start, end}]"""
    from ltp_label_align import bies_tags_to_spans
    raw = []
    for t in tags:
        if hasattr(t, 'item'): raw.append(t.item())
        else: raw.append(int(t))
    return [{"type": ty, "start": s, "end": e} for ty, s, e in bies_tags_to_spans(raw)]
