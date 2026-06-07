"""Modern BERTc backbone → CWS fine-tune 评估(inline_eval 用)。

Modern BERTc 保存格式跟 HF BertModel 不兼容(我们用自定义 model.pt + config.json),
所以不能用 NLP_BERT_CRF/train.py。本脚本自己实现:
  1. 加载 Modern BERTc ckpt
  2. 加一个 Linear(H, 4) BIES 头
  3. 在 PD-1998 strong-labeled 上跑 1 ep(~50K 样本,batch 64,~10min)
  4. dev 2K 子集报 cws_F1
  5. 写 inline_track.tsv + stdout 输出

调用:
  python eval_modern_cws.py --ckpt path/to/checkpoint-20000 [--track_tsv path]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

# 加载 Modern BERTc 模型
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import ModernBertConfig, ModernBertModel

# 复用 NLP_BERT_CRF 的 piece tokenizer adapter
sys.path.insert(0, "/home/tfbao/Shiyu/BERTc/finetune/NLP_BERT_CRF")
from piece_tokenizer_adapter import PieceTokenizerAdapter


# ============ Model ============

class ModernBertCWS(nn.Module):
    def __init__(self, ckpt_dir, num_tags=4):
        super().__init__()
        # 加载 config
        with open(os.path.join(ckpt_dir, "config.json")) as f:
            cfg_dict = json.load(f)
        cfg = ModernBertConfig(**{k: v for k, v in cfg_dict.items()
                                   if k in ModernBertConfig.__dataclass_fields__})
        self.bert = ModernBertModel(cfg)
        # 加载 weight
        ckpt = torch.load(os.path.join(ckpt_dir, "model.pt"),
                          map_location="cpu", weights_only=False)
        # 优先用 EMA shadow(更稳),回退 raw model
        sd = ckpt.get("ema") or ckpt["model"]
        # 只取 bert.* 部分(可能含 head_*,跳过)
        bert_sd = {k[len("bert."):]: v for k, v in sd.items() if k.startswith("bert.")}
        self.bert.load_state_dict(bert_sd, strict=True)
        H = cfg.hidden_size
        self.cws_head = nn.Linear(H, num_tags)
        self.cfg = cfg

    def forward(self, input_ids, attention_mask=None):
        h = self.bert(input_ids, attention_mask=attention_mask)
        return self.cws_head(h)


# ============ Dataset(BIES tagging)============

# B=0, I=1, E=2, S=3
TAG_B, TAG_I, TAG_E, TAG_S = 0, 1, 2, 3


def words_to_tags(words):
    tags = []
    for w in words:
        if len(w) == 1:
            tags.append(TAG_S)
        else:
            tags.append(TAG_B)
            for _ in range(len(w) - 2):
                tags.append(TAG_I)
            tags.append(TAG_E)
    return tags


class PD98CWSDataset(Dataset):
    def __init__(self, jsonl_path, char_to_id, max_len=128, max_samples=None):
        self.samples = []
        with open(jsonl_path) as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                obj = json.loads(line.strip())
                # NLP_BERT_CRF jsonl 格式:{"gold": [...words...], ...}
                words = obj.get("gold") or obj.get("words") or obj.get("tokens")
                if not words:
                    continue
                if not words:
                    continue
                chars = list("".join(words))
                tags = words_to_tags(words)
                if len(chars) != len(tags):
                    continue
                if len(chars) > max_len:
                    chars = chars[:max_len]
                    tags = tags[:max_len]
                self.samples.append((chars, tags))
        self.char_to_id = char_to_id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        chars, tags = self.samples[idx]
        ids = [self.char_to_id(c) for c in chars]
        return ids, tags


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        max_l = max(len(ids) for ids, _ in batch)
        B = len(batch)
        input_ids = torch.full((B, max_l), self.pad_id, dtype=torch.long)
        attn = torch.zeros((B, max_l), dtype=torch.long)
        labels = torch.full((B, max_l), -100, dtype=torch.long)
        for i, (ids, tags) in enumerate(batch):
            n = len(ids)
            input_ids[i, :n] = torch.tensor(ids, dtype=torch.long)
            attn[i, :n] = 1
            labels[i, :n] = torch.tensor(tags, dtype=torch.long)
        return input_ids, attn, labels


class CharToId:
    def __init__(self, tok):
        self.tok = tok
        self.cache = {}

    def __call__(self, c):
        if c in self.cache:
            return self.cache[c]
        ids = self.tok.encode(c, add_special_tokens=False)
        tid = ids[0] if ids else self.tok.unk_token_id
        self.cache[c] = tid
        return tid


# ============ Eval ============

@torch.no_grad()
def eval_cws_f1(model, loader, device):
    model.eval()
    TP = FP = FN = 0
    for input_ids, attn, labels in loader:
        input_ids = input_ids.to(device)
        attn = attn.to(device)
        logits = model(input_ids, attn)
        preds = logits.argmax(-1).cpu().numpy()
        labels_np = labels.numpy()
        for p_seq, l_seq in zip(preds, labels_np):
            # 提取 gold 词边界 + pred 词边界
            gold_words = _tags_to_words(l_seq)
            pred_words = _tags_to_words(p_seq, l_seq)  # 同长度
            gold_set = set(gold_words)
            pred_set = set(pred_words)
            TP += len(gold_set & pred_set)
            FP += len(pred_set - gold_set)
            FN += len(gold_set - pred_set)
    P = TP / max(1, TP + FP)
    R = TP / max(1, TP + FN)
    F1 = 2 * P * R / max(1e-9, P + R)
    return F1, P, R


def _tags_to_words(tags, ref_tags=None):
    """根据 BIES 序列输出 (start, end) 词边界集合。-100 跳过。"""
    if ref_tags is None:
        ref_tags = tags
    words = []
    start = None
    for i, (t, r) in enumerate(zip(tags, ref_tags)):
        if r == -100:
            if start is not None:
                start = None
            continue
        t = int(t)
        if t == TAG_B:
            if start is not None:
                words.append((start, i - 1))
            start = i
        elif t == TAG_S:
            if start is not None:
                words.append((start, i - 1))
                start = None
            words.append((i, i))
        elif t == TAG_I:
            if start is None:
                start = i
        elif t == TAG_E:
            if start is None:
                start = i
            words.append((start, i))
            start = None
    if start is not None:
        words.append((start, len(tags) - 1))
    return words


# ============ Main ============

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint-NNNNN dir")
    ap.add_argument("--train_jsonl",
                    default="/home/tfbao/Shiyu/BERTc/finetune/NLP_BERT_CRF/data/cws.jsonl")
    ap.add_argument("--dev_jsonl",
                    default="/home/tfbao/Shiyu/BERTc/finetune/NLP_BERT_CRF/data/cws_dev.jsonl")
    ap.add_argument("--track_tsv", default="/home/tfbao/Shiyu/BERTc/pretrain/inline_track.tsv")
    ap.add_argument("--max_train", type=int, default=50000)
    ap.add_argument("--max_dev", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max_len", type=int, default=128)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # tokenizer:用 Modern BERTc backbone 同一个 piece tokenizer
    piece_path = "/home/tfbao/Shiyu/BERTc/pretrain/modern_bertc/tokenizer"
    tok = PieceTokenizerAdapter(piece_path)
    char_to_id = CharToId(tok)
    pad_id = tok.pad_token_id

    # data
    print(f"[modern_cws_eval] loading data...", flush=True)
    train_ds = PD98CWSDataset(args.train_jsonl, char_to_id,
                               max_len=args.max_len, max_samples=args.max_train)
    dev_ds = PD98CWSDataset(args.dev_jsonl, char_to_id,
                             max_len=args.max_len, max_samples=args.max_dev)
    print(f"  train: {len(train_ds)} | dev: {len(dev_ds)}", flush=True)
    collator = Collator(pad_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               collate_fn=collator, num_workers=0)
    dev_loader = DataLoader(dev_ds, batch_size=64, shuffle=False,
                             collate_fn=collator, num_workers=0)

    # model
    print(f"[modern_cws_eval] loading backbone from {args.ckpt}...", flush=True)
    model = ModernBertCWS(args.ckpt, num_tags=4).to(device)

    # optim
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * len(train_loader)
    sched = get_linear_schedule_with_warmup(optim,
                                             num_warmup_steps=int(total_steps * 0.1),
                                             num_training_steps=total_steps)

    # train(per-epoch dev eval,best 保留终态)
    t0 = time.time()
    print(f"[modern_cws_eval] training {total_steps} steps...", flush=True)
    best_f1, best_p, best_r = 0.0, 0.0, 0.0
    for ep in range(args.epochs):
        model.train()
        for step, (input_ids, attn, labels) in enumerate(train_loader):
            input_ids = input_ids.to(device, non_blocking=True)
            attn = attn.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(input_ids, attn)
            loss = F.cross_entropy(logits.view(-1, 4), labels.view(-1), ignore_index=-100)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            if step % 100 == 0:
                print(f"  ep{ep+1} step {step}/{len(train_loader)} loss {loss.item():.4f}",
                      flush=True)
        # per-epoch dev eval
        f1, p, r = eval_cws_f1(model, dev_loader, device)
        if f1 > best_f1:
            best_f1, best_p, best_r = f1, p, r
            tag = " ★ new best"
        else:
            tag = ""
        print(f"  ep{ep+1} dev: cws_F1={f1:.4f} P={p:.4f} R={r:.4f}{tag}", flush=True)

    # final = best across epochs(更稳健 vs 一定取最后一个)
    f1, p, r = best_f1, best_p, best_r
    elapsed = time.time() - t0
    step_label = os.path.basename(args.ckpt.rstrip("/")).replace("checkpoint-", "")
    print(f"[modern_cws_eval] step={step_label} cws_F1={f1:.4f} P={p:.4f} R={r:.4f}  "
          f"({elapsed:.0f}s)  [best across {args.epochs} ep]", flush=True)

    # append to track tsv
    track = Path(args.track_tsv)
    if not track.exists() or track.stat().st_size == 0:
        track.write_text("timestamp\tstep\tckpt\tcws_F1\tP\tR\n")
    with track.open("a") as f:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{ts}\t{step_label}\t{args.ckpt}\t{f1:.4f}\t{p:.4f}\t{r:.4f}\n")
    print(f"[modern_cws_eval] appended → {args.track_tsv}", flush=True)


if __name__ == "__main__":
    main()
