"""Modern BERTc backbone → CSC fine-tune 评估(inline_eval 用)。

跟 eval_modern_cws.py 同理,但跑 CSC(纠错)而非 CWS(分词):
  1. 加载 Modern BERTc ckpt
  2. 加 cor_head(Linear→vocab)+ det_head(Linear→1),仿 MacBERT4CSC
  3. 在 SIGHAN+Wang271K 50K 子集上跑 1 ep(~10min)
  4. SIGHAN-15 官方 707 test 报 sentence F1
  5. 写 inline_track_csc.tsv

调用:
  python eval_modern_csc.py --ckpt path/to/checkpoint-20000
"""
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import ModernBertConfig, ModernBertModel

sys.path.insert(0, "/home/tfbao/Shiyu/BERTc/finetune/NLP_BERT_CRF")
from piece_tokenizer_adapter import PieceTokenizerAdapter


# ============ Model ============

class ModernBertCSC(nn.Module):
    def __init__(self, ckpt_dir, vocab_size):
        super().__init__()
        with open(os.path.join(ckpt_dir, "config.json")) as f:
            cfg_dict = json.load(f)
        cfg = ModernBertConfig(**{k: v for k, v in cfg_dict.items()
                                   if k in ModernBertConfig.__dataclass_fields__})
        self.bert = ModernBertModel(cfg)
        ckpt = torch.load(os.path.join(ckpt_dir, "model.pt"),
                          map_location="cpu", weights_only=False)
        # 优先用 EMA shadow,回退 raw model
        sd = ckpt.get("ema") or ckpt["model"]
        bert_sd = {k[len("bert."):]: v for k, v in sd.items() if k.startswith("bert.")}
        self.bert.load_state_dict(bert_sd, strict=True)
        H = cfg.hidden_size
        self.vocab_size = vocab_size
        self.cor_head = nn.Linear(H, vocab_size)
        self.det_head = nn.Linear(H, 1)
        self.cfg = cfg

    def forward(self, input_ids, attention_mask=None):
        h = self.bert(input_ids, attention_mask=attention_mask)
        return self.cor_head(h), self.det_head(h).squeeze(-1)


# ============ Dataset ============

class CSCDataset(Dataset):
    def __init__(self, pairs, char_to_id, max_len=128, max_samples=None):
        if max_samples:
            pairs = pairs[:max_samples]
        self.pairs = pairs
        self.char_to_id = char_to_id
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        L = min(len(src), len(tgt), self.max_len)
        src_ids = [self.char_to_id(c) for c in src[:L]]
        cor_ids = [self.char_to_id(c) for c in tgt[:L]]
        det_lbl = [1.0 if src[i] != tgt[i] else 0.0 for i in range(L)]
        return src_ids, cor_ids, det_lbl


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        max_l = max(len(s) for s, _, _ in batch)
        B = len(batch)
        input_ids = torch.full((B, max_l), self.pad_id, dtype=torch.long)
        cor_labels = torch.full((B, max_l), -100, dtype=torch.long)
        det_labels = torch.zeros((B, max_l), dtype=torch.float)
        attn = torch.zeros((B, max_l), dtype=torch.long)
        for i, (s, c, d) in enumerate(batch):
            n = len(s)
            input_ids[i, :n] = torch.tensor(s, dtype=torch.long)
            cor_labels[i, :n] = torch.tensor(c, dtype=torch.long)
            det_labels[i, :n] = torch.tensor(d, dtype=torch.float)
            attn[i, :n] = 1
        return input_ids, attn, cor_labels, det_labels


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


def focal_bce_loss(logits, labels, gamma=2.0, valid_mask=None):
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * labels + (1 - p) * (1 - labels)
    fw = (1 - pt) ** gamma
    loss = fw * bce
    if valid_mask is not None:
        loss = loss * valid_mask
        return loss.sum() / valid_mask.sum().clamp(min=1.0)
    return loss.mean()


# ============ Inference + Eval ============

@torch.no_grad()
def correct_batch(model, char_to_id, texts, device, max_len=128, threshold=0.7):
    model.eval()
    pad_id = char_to_id.tok.pad_token_id
    B = len(texts)
    encoded, lengths = [], []
    for t in texts:
        L = min(len(t), max_len)
        ids = [char_to_id(c) for c in t[:L]]
        encoded.append(ids); lengths.append(L)
    max_l = max(lengths)
    input_ids = torch.full((B, max_l), pad_id, dtype=torch.long)
    attn = torch.zeros((B, max_l), dtype=torch.long)
    for i, ids in enumerate(encoded):
        input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        attn[i, :len(ids)] = 1
    input_ids = input_ids.to(device); attn = attn.to(device)
    cor_logits, _ = model(input_ids, attn)
    probs = F.softmax(cor_logits, dim=-1)
    top_probs, top_ids = probs.max(dim=-1)
    # 反向 id->char
    id_to_char = {}
    for c, tid in char_to_id.cache.items():
        id_to_char.setdefault(tid, c)
    preds = []
    for i, text in enumerate(texts):
        L = lengths[i]
        orig = list(text[:L])
        pred = []
        for j in range(L):
            tid = top_ids[i, j].item()
            p = top_probs[i, j].item()
            if p < threshold or tid not in id_to_char:
                pred.append(orig[j])
            else:
                pred.append(id_to_char[tid])
        if len(text) > L:
            pred.extend(list(text[L:]))
        preds.append("".join(pred))
    return preds


def eval_sighan(model, char_to_id, test_tsv, device, max_len=128, threshold=0.7):
    samples = []
    with open(test_tsv) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = line.split("\t")
            if len(parts) != 2: continue
            samples.append((parts[0], parts[1]))
    srcs = [s for s, _ in samples]
    tgts = [t for _, t in samples]
    preds = []
    for i in range(0, len(srcs), 32):
        preds.extend(correct_batch(model, char_to_id, srcs[i:i+32], device,
                                    max_len=max_len, threshold=threshold))
    TP = FP = FN = TN = 0
    for src, tgt, pred in zip(srcs, tgts, preds):
        if src == tgt:
            if tgt == pred: TN += 1
            else: FP += 1
        else:
            if tgt == pred: TP += 1
            else: FN += 1
    n = len(samples)
    acc = (TP + TN) / n
    P = TP / max(1, TP + FP)
    R = TP / max(1, TP + FN)
    F1 = 2 * P * R / max(1e-9, P + R)
    return F1, P, R, acc, TP, FP, FN, TN


# ============ Main ============

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train_pkl",
                    default="/home/tfbao/Shiyu/BERTc/csc/data/sighan_wang271k_pairs.pkl")
    ap.add_argument("--test_tsv",
                    default="/home/tfbao/Shiyu/BERTc/csc/data/test/sighan2015_test_official.tsv")
    ap.add_argument("--track_tsv",
                    default="/home/tfbao/Shiyu/BERTc/pretrain/inline_track_csc.tsv")
    ap.add_argument("--max_train", type=int, default=50000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--threshold", type=float, default=0.7)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = PieceTokenizerAdapter("/home/tfbao/Shiyu/BERTc/pretrain/modern_bertc/tokenizer")
    char_to_id = CharToId(tok)
    pad_id = tok.pad_token_id
    vocab_size = tok.vocab_size

    # data
    print(f"[modern_csc_eval] loading {args.train_pkl}...", flush=True)
    with open(args.train_pkl, "rb") as f:
        pairs = pickle.load(f)
    print(f"  total {len(pairs)} pairs, using {args.max_train}", flush=True)
    # warm cache 全字符,inference 需要
    seen = set()
    for s, t in pairs[:args.max_train]:
        for c in s + t:
            if c not in seen: seen.add(c); char_to_id(c)
    with open(args.test_tsv) as f:
        for line in f:
            for c in line.strip():
                if c not in seen: seen.add(c); char_to_id(c)
    print(f"  cache: {len(char_to_id.cache)} chars", flush=True)

    train_ds = CSCDataset(pairs, char_to_id, max_len=args.max_len,
                           max_samples=args.max_train)
    collator = Collator(pad_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               collate_fn=collator, num_workers=0)
    total_steps = args.epochs * len(train_loader)

    # model
    print(f"[modern_csc_eval] loading backbone from {args.ckpt}...", flush=True)
    model = ModernBertCSC(args.ckpt, vocab_size).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(optim,
                                             int(total_steps * 0.1), total_steps)

    # train
    t0 = time.time()
    print(f"[modern_csc_eval] training {total_steps} steps...", flush=True)
    model.train()
    for ep in range(args.epochs):
        for step, (ids, attn, cor, det) in enumerate(train_loader):
            ids = ids.to(device, non_blocking=True)
            attn = attn.to(device, non_blocking=True)
            cor = cor.to(device, non_blocking=True)
            det = det.to(device, non_blocking=True)
            cor_logits, det_logits = model(ids, attn)
            cor_loss = F.cross_entropy(cor_logits.view(-1, vocab_size),
                                       cor.view(-1), ignore_index=-100)
            det_loss = focal_bce_loss(det_logits, det,
                                       valid_mask=attn.float())
            loss = 0.3 * det_loss + cor_loss
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            if step % 100 == 0:
                print(f"  step {step}/{len(train_loader)} cor {cor_loss.item():.3f} "
                      f"det {det_loss.item():.4f}", flush=True)

    # eval
    F1, P, R, acc, TP, FP, FN, TN = eval_sighan(
        model, char_to_id, args.test_tsv, device,
        max_len=args.max_len, threshold=args.threshold)
    elapsed = time.time() - t0
    step_label = os.path.basename(args.ckpt.rstrip("/")).replace("checkpoint-", "")
    print(f"[modern_csc_eval] step={step_label} SIGHAN15 F1={F1:.4f} "
          f"P={P:.4f} R={R:.4f} acc={acc:.4f} TP={TP} ({elapsed:.0f}s)", flush=True)

    track = Path(args.track_tsv)
    if not track.exists() or track.stat().st_size == 0:
        track.write_text("timestamp\tstep\tckpt\tcsc_F1\tP\tR\tacc\n")
    with track.open("a") as f:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{ts}\t{step_label}\t{args.ckpt}\t{F1:.4f}\t{P:.4f}\t{R:.4f}\t{acc:.4f}\n")


if __name__ == "__main__":
    main()
