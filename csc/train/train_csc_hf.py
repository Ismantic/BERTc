"""CSC 训练脚本(HuggingFace BertTokenizer 版),用于 RoBERTa-wwm-ext / MacBERT 等基线对比。

跟 train_csc.py 架构、loss、threshold 完全一致,只换 tokenizer:
  - 用 transformers.BertTokenizer(vocab.txt + WordPiece)
  - 加 [CLS] / [SEP] 包裹(跟原版 MacBERT4CSC 一致)
  - 中文部分每字 1 token(WordPiece 字级)

用法:
  python train_csc_hf.py \\
      --backbone_path /home/tfbao/Shiyu/Summer/BERT/NLP_BERT_CRF/roberta-wwm-ext \\
      --output_dir /home/tfbao/Shiyu/BERTc/csc/output_roberta_csc_v1 \\
      --epochs 10 --batch_size 32 --lr 5e-5
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
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup


# ============ Model(跟 train_csc.py BERTcForCSC 同构,只改名)============

class BertForCSC(nn.Module):
    def __init__(self, backbone_path, vocab_size):
        super().__init__()
        self.bert = BertModel.from_pretrained(backbone_path)
        H = self.bert.config.hidden_size
        self.vocab_size = vocab_size
        self.cor_head = nn.Linear(H, vocab_size)
        self.det_head = nn.Linear(H, 1)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        h = self.bert(input_ids=input_ids,
                      attention_mask=attention_mask,
                      token_type_ids=token_type_ids).last_hidden_state
        return self.cor_head(h), self.det_head(h).squeeze(-1)


def focal_bce_loss(logits, labels, gamma=2.0, valid_mask=None):
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * labels + (1 - p) * (1 - labels)
    focal_weight = (1.0 - pt) ** gamma
    loss = focal_weight * bce
    if valid_mask is not None:
        loss = loss * valid_mask
        return loss.sum() / valid_mask.sum().clamp(min=1.0)
    return loss.mean()


# ============ Dataset(带 CLS / SEP)============

class CSCDatasetHF(Dataset):
    def __init__(self, pairs, tokenizer, max_len=128):
        self.pairs = pairs
        self.tok = tokenizer
        self.max_len = max_len  # 包含 CLS + body + SEP

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        # 截断 body 到 max_len - 2(留给 CLS + SEP)
        body_len = min(len(src), len(tgt), self.max_len - 2)
        src_body = src[:body_len]
        tgt_body = tgt[:body_len]

        # 单字编码(中文)。WordPiece 对中文一字一 token,但若遇到 ASCII 等可能不对齐 → 退化为 unk
        src_ids = [self.tok.cls_token_id]
        cor_ids = [-100]              # CLS 不参与 loss
        det_lbl = [0.0]                # CLS detection label = 0
        for sc, tc in zip(src_body, tgt_body):
            sid = self.tok.convert_tokens_to_ids(sc)
            tid = self.tok.convert_tokens_to_ids(tc)
            if sid == self.tok.unk_token_id:
                # 未知字符跳过(罕见),但还是占一个位置以维持对齐
                pass
            src_ids.append(sid)
            cor_ids.append(tid)
            det_lbl.append(1.0 if sc != tc else 0.0)
        src_ids.append(self.tok.sep_token_id)
        cor_ids.append(-100)
        det_lbl.append(0.0)
        return {
            "input_ids": src_ids,
            "cor_labels": cor_ids,
            "det_labels": det_lbl,
            "length": len(src_ids),
        }


class CSCCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        max_l = max(item["length"] for item in batch)
        B = len(batch)
        input_ids = torch.full((B, max_l), self.pad_id, dtype=torch.long)
        cor_labels = torch.full((B, max_l), -100, dtype=torch.long)
        det_labels = torch.zeros((B, max_l), dtype=torch.float)
        attn = torch.zeros((B, max_l), dtype=torch.long)
        for i, item in enumerate(batch):
            n = item["length"]
            input_ids[i, :n] = torch.tensor(item["input_ids"], dtype=torch.long)
            cor_labels[i, :n] = torch.tensor(item["cor_labels"], dtype=torch.long)
            det_labels[i, :n] = torch.tensor(item["det_labels"], dtype=torch.float)
            attn[i, :n] = 1
        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "token_type_ids": torch.zeros_like(input_ids),
            "cor_labels": cor_labels,
            "det_labels": det_labels,
        }


# ============ Inference + Eval ============

@torch.no_grad()
def correct_batch(model, tokenizer, texts, device, max_len=128, threshold=0.7):
    model.eval()
    pad_id = tokenizer.pad_token_id
    B = len(texts)
    encoded = []
    for t in texts:
        body_len = min(len(t), max_len - 2)
        body = t[:body_len]
        ids = [tokenizer.cls_token_id]
        for c in body:
            ids.append(tokenizer.convert_tokens_to_ids(c))
        ids.append(tokenizer.sep_token_id)
        encoded.append((t, body_len, ids))
    max_l = max(len(x[2]) for x in encoded)
    input_ids = torch.full((B, max_l), pad_id, dtype=torch.long)
    attn = torch.zeros((B, max_l), dtype=torch.long)
    for i, (_, _, ids) in enumerate(encoded):
        input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        attn[i, :len(ids)] = 1
    input_ids = input_ids.to(device)
    attn = attn.to(device)
    cor_logits, _ = model(input_ids, attn,
                          token_type_ids=torch.zeros_like(input_ids))
    probs = F.softmax(cor_logits, dim=-1)
    top_probs, top_ids = probs.max(dim=-1)

    preds = []
    for i, (text, body_len, _) in enumerate(encoded):
        orig = list(text[:body_len])
        pred = []
        for j in range(body_len):
            tid = top_ids[i, 1 + j].item()       # +1 跳过 CLS
            p = top_probs[i, 1 + j].item()
            if p < threshold:
                pred.append(orig[j])
                continue
            t = tokenizer.convert_ids_to_tokens(tid)
            t = t.replace("##", "") if t else ""
            if not t or len(t) != 1 or t.startswith("["):
                pred.append(orig[j])
            else:
                pred.append(t)
        if len(text) > body_len:
            pred.extend(list(text[body_len:]))
        preds.append("".join(pred))
    return preds


def eval_sighan(model, tokenizer, test_tsv, device, max_len=128, threshold=0.7, batch_size=32):
    samples = []
    with open(test_tsv, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = line.split("\t")
            if len(parts) != 2: continue
            samples.append((parts[0], parts[1]))
    srcs = [s for s, _ in samples]
    tgts = [t for _, t in samples]
    preds = []
    for i in range(0, len(srcs), batch_size):
        preds.extend(correct_batch(model, tokenizer, srcs[i:i+batch_size],
                                   device, max_len=max_len, threshold=threshold))
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
    prec = TP / max(1, TP + FP)
    rec = TP / max(1, TP + FN)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return {"acc": acc, "precision": prec, "recall": rec, "f1": f1,
            "TP": TP, "FP": FP, "FN": FN, "TN": TN, "n": n}


# ============ Train ============

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained(args.backbone_path)
    vocab_size = tokenizer.vocab_size
    print(f"Tokenizer: vocab={vocab_size}, cls={tokenizer.cls_token_id}, "
          f"sep={tokenizer.sep_token_id}, pad={tokenizer.pad_token_id}", flush=True)

    print(f"Loading pairs from {args.train_pkl}...", flush=True)
    with open(args.train_pkl, "rb") as f:
        pairs = pickle.load(f)
    print(f"  {len(pairs)} pairs", flush=True)

    train_ds = CSCDatasetHF(pairs, tokenizer, max_len=args.max_len)
    collator = CSCCollator(pad_id=tokenizer.pad_token_id)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collator, num_workers=2, pin_memory=True)
    total_steps = args.epochs * len(loader)
    print(f"steps/epoch: {len(loader)}, total: {total_steps}", flush=True)

    model = BertForCSC(args.backbone_path, vocab_size).to(device)

    no_decay = ["bias", "LayerNorm.weight"]
    params = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    best_f1 = 0.0
    global_step = 0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        sum_cor = sum_det = 0.0
        sum_n = 0
        for batch in loader:
            global_step += 1
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attn = batch["attention_mask"].to(device, non_blocking=True)
            tti = batch["token_type_ids"].to(device, non_blocking=True)
            cor_labels = batch["cor_labels"].to(device, non_blocking=True)
            det_labels = batch["det_labels"].to(device, non_blocking=True)

            cor_logits, det_logits = model(input_ids, attn, tti)
            cor_loss = F.cross_entropy(cor_logits.view(-1, vocab_size),
                                       cor_labels.view(-1), ignore_index=-100)
            det_loss = focal_bce_loss(det_logits, det_labels,
                                      gamma=2.0, valid_mask=attn.float())
            loss = args.det_weight * det_loss + cor_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            sum_cor += cor_loss.item()
            sum_det += det_loss.item()
            sum_n += 1
            if global_step % args.log_every == 0:
                elapsed = time.time() - t0
                rate = global_step / elapsed
                eta = (total_steps - global_step) / rate / 60
                print(f"  ep{ep} step {global_step}/{total_steps}  "
                      f"cor={cor_loss.item():.3f} det={det_loss.item():.4f}  "
                      f"lr={scheduler.get_last_lr()[0]:.2e}  "
                      f"{rate:.1f}/s ETA {eta:.1f}m", flush=True)

        print(f"\nEval after epoch {ep}...", flush=True)
        metrics = eval_sighan(model, tokenizer, args.test_tsv, device,
                              max_len=args.max_len, threshold=args.threshold,
                              batch_size=32)
        avg_cor = sum_cor / sum_n
        avg_det = sum_det / sum_n
        print(f"=== Epoch {ep}/{args.epochs}  "
              f"avg_cor={avg_cor:.4f} avg_det={avg_det:.4f}  "
              f"SIGHAN-15: acc={metrics['acc']:.4f} P={metrics['precision']:.4f} "
              f"R={metrics['recall']:.4f} F1={metrics['f1']:.4f} "
              f"(TP={metrics['TP']} FP={metrics['FP']} FN={metrics['FN']} TN={metrics['TN']}) ===",
              flush=True)

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save({"model": model.state_dict(), "epoch": ep,
                        "metrics": metrics, "args": vars(args)},
                       out / "best.pt")
            print(f"    ↑ saved best.pt (F1={metrics['f1']:.4f})", flush=True)
        torch.save({"model": model.state_dict(), "epoch": ep,
                    "metrics": metrics, "args": vars(args)},
                   out / "final.pt")

    total_min = (time.time() - t0) / 60
    print(f"\nDone. Best SIGHAN-15 F1: {best_f1:.4f}", flush=True)
    print(f"Total: {total_min:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone_path", required=True)
    ap.add_argument("--train_pkl", default="/home/tfbao/Shiyu/BERTc/csc/data/sighan_wang271k_pairs.pkl")
    ap.add_argument("--test_tsv",
                    default="/home/tfbao/Shiyu/BERTc/csc/data/test/sighan2015_test_official.tsv")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--det_weight", type=float, default=0.3)
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--log_every", type=int, default=400)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
