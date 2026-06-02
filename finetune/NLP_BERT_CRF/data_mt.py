"""Multi-task dataset: 同时返回 char-level CWS + POS + NER tags。

输入 3 个 jsonl(cws/pos/ner.pd98),按 line idx 对齐(同 sentence)。

输出: items[i] = {chars, cws_tags, pos_tags, ner_tags}
"""
import json
import torch
from torch.utils.data import Dataset

from data import TAG2ID as CWS_TAG2ID  # B/I/E/S → 0/1/2/3
from data_pos_ner import NER_TAGS
from ltp_label_align import LTP_POS2ID, map_pd_pos, entity_to_bies, PD2LTP_NER


def words_to_cws_bies(words):
    chars, tags = [], []
    for w in words:
        if not w: continue
        if len(w) == 1:
            chars.append(w); tags.append(CWS_TAG2ID["S"])
        else:
            chars.append(w[0]); tags.append(CWS_TAG2ID["B"])
            for c in w[1:-1]:
                chars.append(c); tags.append(CWS_TAG2ID["I"])
            chars.append(w[-1]); tags.append(CWS_TAG2ID["E"])
    return chars, tags


class MTDataset(Dataset):
    def __init__(self, cws_path, pos_path, ner_path, pos2id, max_chars=254):
        self.items = []
        # 读 3 个文件,同 line idx 对齐
        cws_items, pos_items, ner_items = [], [], []
        with open(cws_path, encoding="utf8") as f:
            for line in f: cws_items.append(json.loads(line))
        with open(pos_path, encoding="utf8") as f:
            for line in f: pos_items.append(json.loads(line))
        with open(ner_path, encoding="utf8") as f:
            for line in f: ner_items.append(json.loads(line))
        n = min(len(cws_items), len(pos_items), len(ner_items))
        for i in range(n):
            cws_obj = cws_items[i]
            pos_obj = pos_items[i]
            ner_obj = ner_items[i]
            words = cws_obj.get("gold", [])
            if not words: continue
            chars, cws_tags = words_to_cws_bies(words)
            # POS: PD tag → LTP tag → per-char
            pos_words = pos_obj.get("words", [])
            pos_seq = pos_obj.get("pos", [])
            pos_tags = [-100] * len(chars)
            if pos_words == words and len(pos_words) == len(pos_seq):
                idx_c = 0
                for w, p in zip(pos_words, pos_seq):
                    ltp_p = map_pd_pos(p)
                    pid = pos2id.get(ltp_p, pos2id.get('x', 0))
                    for _ in w:
                        if idx_c < len(pos_tags):
                            pos_tags[idx_c] = pid
                        idx_c += 1
            # NER: LTP-aligned BIES,弃 MISC/I/L
            # 兼容旧 type (PER/LOC/ORG) 和新 type (Nh/Ns/Ni)
            ner_tags = [NER_TAGS["O"]] * len(chars)
            for ent in ner_obj.get("entities", []):
                t = ent.get("type", "")
                if t in PD2LTP_NER: t = PD2LTP_NER[t]
                if t not in ("Nh", "Ns", "Ni"): continue
                s, e = ent["start"], min(ent["end"], len(chars))
                length = e - s
                if s >= len(chars) or length <= 0: continue
                if length == 1:
                    ner_tags[s] = NER_TAGS[f"S-{t}"]
                else:
                    ner_tags[s] = NER_TAGS[f"B-{t}"]
                    for j in range(s + 1, e - 1):
                        ner_tags[j] = NER_TAGS[f"I-{t}"]
                    ner_tags[e - 1] = NER_TAGS[f"E-{t}"]
            if len(chars) > max_chars:
                chars = chars[:max_chars]
                cws_tags = cws_tags[:max_chars]
                pos_tags = pos_tags[:max_chars]
                ner_tags = ner_tags[:max_chars]
                # 修正 cws 切到非 S/E 收尾
                if cws_tags[-1] == CWS_TAG2ID["B"]: cws_tags[-1] = CWS_TAG2ID["S"]
                elif cws_tags[-1] == CWS_TAG2ID["I"]: cws_tags[-1] = CWS_TAG2ID["E"]
            self.items.append({
                "chars": chars,
                "cws_tags": cws_tags,
                "pos_tags": pos_tags,
                "ner_tags": ner_tags,
            })

    def __len__(self): return len(self.items)
    def __getitem__(self, idx): return self.items[idx]


class DistillMTDataset(Dataset):
    """LTP-distilled 单 jsonl(text/words/pos/entities)→ MT items。
    POS 直接用 LTP tag(不走 PD→LTP 映射),NER type 已是 Nh/Ns/Ni。
    """
    def __init__(self, jsonl_path, pos2id=None, max_chars=254):
        if pos2id is None: pos2id = LTP_POS2ID
        self.items = []
        with open(jsonl_path, encoding="utf8") as f:
            for line in f:
                obj = json.loads(line)
                words = obj.get("words", [])
                pos = obj.get("pos", [])
                if not words or len(pos) != len(words): continue
                chars, cws_tags = words_to_cws_bies(words)
                pos_tags = [-100] * len(chars)
                idx_c = 0
                for w, p in zip(words, pos):
                    pid = pos2id.get(p, pos2id.get('x', 0))
                    for _ in w:
                        if idx_c < len(pos_tags): pos_tags[idx_c] = pid
                        idx_c += 1
                ner_tags = [NER_TAGS["O"]] * len(chars)
                for ent in obj.get("entities", []):
                    t = ent.get("type", "")
                    if t not in ("Nh", "Ns", "Ni"): continue
                    s, e = ent["start"], min(ent["end"], len(chars))
                    length = e - s
                    if s >= len(chars) or length <= 0: continue
                    if length == 1:
                        ner_tags[s] = NER_TAGS[f"S-{t}"]
                    else:
                        ner_tags[s] = NER_TAGS[f"B-{t}"]
                        for j in range(s + 1, e - 1):
                            ner_tags[j] = NER_TAGS[f"I-{t}"]
                        ner_tags[e - 1] = NER_TAGS[f"E-{t}"]
                if len(chars) > max_chars:
                    chars = chars[:max_chars]
                    cws_tags = cws_tags[:max_chars]
                    pos_tags = pos_tags[:max_chars]
                    ner_tags = ner_tags[:max_chars]
                    if cws_tags[-1] == CWS_TAG2ID["B"]: cws_tags[-1] = CWS_TAG2ID["S"]
                    elif cws_tags[-1] == CWS_TAG2ID["I"]: cws_tags[-1] = CWS_TAG2ID["E"]
                if not chars: continue
                self.items.append({
                    "chars": chars, "cws_tags": cws_tags,
                    "pos_tags": pos_tags, "ner_tags": ner_tags,
                })

    def __len__(self): return len(self.items)
    def __getitem__(self, idx): return self.items[idx]


class ConcatMTDataset(Dataset):
    """简单拼接两个 MT dataset(PD strong + LTP distill)。"""
    def __init__(self, *datasets):
        self.items = []
        for d in datasets:
            self.items.extend(d.items)
    def __len__(self): return len(self.items)
    def __getitem__(self, idx): return self.items[idx]


class MTCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id
        self.unk_id = tokenizer.unk_token_id
        self._cache = {}

    def _char_to_id(self, c):
        if c in self._cache: return self._cache[c]
        ids = self.tokenizer.encode(c, add_special_tokens=False)
        tid = ids[0] if ids else self.unk_id
        self._cache[c] = tid
        return tid

    def __call__(self, batch):
        seqs = [[self._char_to_id(c) for c in item["chars"]] for item in batch]
        B = len(batch)
        max_l = max(len(s) for s in seqs)
        input_ids = torch.full((B, max_l), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((B, max_l), dtype=torch.long)
        cws_labels = torch.zeros((B, max_l), dtype=torch.long)
        pos_labels = torch.full((B, max_l), -100, dtype=torch.long)
        ner_labels = torch.zeros((B, max_l), dtype=torch.long)
        for i, item in enumerate(batch):
            n = len(seqs[i])
            input_ids[i, :n] = torch.tensor(seqs[i], dtype=torch.long)
            attention_mask[i, :n] = 1
            cws_labels[i, :n] = torch.tensor(item["cws_tags"], dtype=torch.long)
            pos_labels[i, :n] = torch.tensor(item["pos_tags"], dtype=torch.long)
            ner_labels[i, :n] = torch.tensor(item["ner_tags"], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "cws_labels": cws_labels,
            "pos_labels": pos_labels,
            "ner_labels": ner_labels,
        }
