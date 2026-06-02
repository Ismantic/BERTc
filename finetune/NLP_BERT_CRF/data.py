"""PD-1998 jsonl → char + BIES tags Dataset for BertCRF.

cws.jsonl schema: {"gold": ["word1", "word2", ...], ...}
"""
import json
import torch
from torch.utils.data import Dataset

TAG2ID = {"B": 0, "I": 1, "E": 2, "S": 3}
ID2TAG = {v: k for k, v in TAG2ID.items()}


def words_to_bies(words):
    """['北京', '是', '首都'] → chars ['北','京','是','首','都'], tags [B,E,S,B,E]"""
    chars, tags = [], []
    for w in words:
        if not w:
            continue
        if len(w) == 1:
            chars.append(w)
            tags.append(TAG2ID["S"])
        else:
            chars.append(w[0])
            tags.append(TAG2ID["B"])
            for c in w[1:-1]:
                chars.append(c)
                tags.append(TAG2ID["I"])
            chars.append(w[-1])
            tags.append(TAG2ID["E"])
    return chars, tags


def bies_to_words(chars, tags):
    """[chars], [tag_id...] → word list, robust to malformed sequences."""
    words = []
    cur = ""
    for c, t in zip(chars, tags):
        tag = ID2TAG[t] if isinstance(t, int) else t
        if tag in ("B", "S"):
            if cur:
                words.append(cur)
            cur = c
        else:  # I or E
            cur += c
    if cur:
        words.append(cur)
    return words


class CWSDataset(Dataset):
    def __init__(self, jsonl_path, max_chars=254):
        self.items = []
        with open(jsonl_path, encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                words = obj.get("gold") or obj.get("cut", "").split()
                if not words:
                    continue
                chars, tags = words_to_bies(words)
                if len(chars) < 1:
                    continue
                if len(chars) > max_chars:
                    chars = chars[:max_chars]
                    tags = tags[:max_chars]
                    # If the truncation left the last char as B or I, downgrade
                    # to keep a valid sequence (set last to E if I, S if B).
                    if tags[-1] == TAG2ID["B"]:
                        tags[-1] = TAG2ID["S"]
                    elif tags[-1] == TAG2ID["I"]:
                        tags[-1] = TAG2ID["E"]
                self.items.append({"chars": chars, "tags": tags})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class Collator:
    """Char-by-char encoding with MacBERT tokenizer (no [CLS]/[SEP] — CRF sees pure char seq)."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id
        self.unk_id = tokenizer.unk_token_id
        self._cache = {}

    def _char_to_id(self, c):
        if c in self._cache:
            return self._cache[c]
        ids = self.tokenizer.encode(c, add_special_tokens=False)
        tid = ids[0] if ids else self.unk_id
        self._cache[c] = tid
        return tid

    def __call__(self, batch):
        seqs = [[self._char_to_id(c) for c in item["chars"]] for item in batch]
        tags = [item["tags"] for item in batch]
        max_l = max(len(s) for s in seqs)
        B = len(batch)
        input_ids = torch.full((B, max_l), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((B, max_l), dtype=torch.long)
        labels = torch.zeros((B, max_l), dtype=torch.long)
        for i, (s, t) in enumerate(zip(seqs, tags)):
            n = len(s)
            input_ids[i, :n] = torch.tensor(s, dtype=torch.long)
            attention_mask[i, :n] = 1
            labels[i, :n] = torch.tensor(t, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
