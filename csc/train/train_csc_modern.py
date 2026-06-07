"""BERTc-CSC v1 训练脚本

架构(MacBERT4CSC 风格):
  - Backbone: BertModel(BERTc v7 165M, char-level)
  - Correction head: Linear(H, vocab)  # MLM 头,预测每个位置的正确字
  - Detection head: Linear(H, 1)       # 二分类:该位置是否出错(focal loss)
  - Loss = det_weight * focal(det) + CE(cor)

数据:
  - 826K (src, tgt) 同长 char-to-char 对(/home/tfbao/Shiyu/BERTc/csc/data/all_pairs.pkl)
  - 无 [CLS]/[SEP](跟 MT 训练对齐,纯 char seq)

评估:
  - SIGHAN-15 官方 707 测试(/home/tfbao/Shiyu/BERTc/csc/data/test/sighan2015_test_official.tsv)
  - pycorrector 口径 sentence-level P/R/F1
  - threshold=0.7 过滤低置信度纠错

用法:
  python train_csc.py \\
      --backbone_path /home/tfbao/Shiyu/BERTc/finetune/backbones/bert_train_v7_mid \\
      --output_dir /home/tfbao/Shiyu/BERTc/csc/output_v7_csc_v1 \\
      --epochs 5 --batch_size 64 --lr 5e-5
"""
import argparse
import json
import math
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

# Modern BERTc 适配
import json as _json
sys.path.insert(0, "/home/tfbao/Shiyu/BERTc/pretrain/modern_bertc")
from model import ModernBertConfig, ModernBertModel

sys.path.insert(0, "/home/tfbao/Shiyu/BERTc/finetune/NLP_BERT_CRF")
from piece_tokenizer_adapter import PieceTokenizerAdapter


# ============ Model ============

class BERTcForCSC(nn.Module):
    """Modern BERTc backbone + 2 heads(cor + det),跟原 train_csc.py 接口一致。"""
    def __init__(self, ckpt_dir, vocab_size):
        super().__init__()
        # 1) config
        with open(os.path.join(ckpt_dir, "config.json")) as f:
            cfg_dict = _json.load(f)
        cfg = ModernBertConfig(**{k: v for k, v in cfg_dict.items()
                                   if k in ModernBertConfig.__dataclass_fields__})
        self.bert = ModernBertModel(cfg)
        H = cfg.hidden_size
        # 2) weights(state["model"] 是 ModernBertForMLM,取 bert.* 前缀)
        ckpt = torch.load(os.path.join(ckpt_dir, "model.pt"),
                          map_location="cpu", weights_only=False)
        sd = ckpt.get("ema") or ckpt["model"]
        bert_sd = {k[len("bert."):]: v for k, v in sd.items() if k.startswith("bert.")}
        self.bert.load_state_dict(bert_sd, strict=True)
        # 3) heads
        self.vocab_size = vocab_size
        # cor_head **tied 到 embed**(关键):预训 MLM head 是简化版 logits = h @ embed.weight.T,
        # 预训完 h 已经跟 embed 对齐,fresh Linear 会废掉这个对齐,导致 CSC 学不动。
        self.cor_head = nn.Linear(H, vocab_size, bias=False)
        self.cor_head.weight = self.bert.embed.weight   # weight tying
        self.det_head = nn.Linear(H, 1)

    def forward(self, input_ids, attention_mask=None):
        # ModernBertModel: 直接返回 last_hidden_state(无 .last_hidden_state attr)
        h = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cor_logits = self.cor_head(h)                # [B, L, V]
        det_logits = self.det_head(h).squeeze(-1)    # [B, L]
        return cor_logits, det_logits


def focal_bce_loss(logits, labels, gamma=2.0, valid_mask=None):
    """二分类 focal loss(忽略 padding)。"""
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * labels + (1 - p) * (1 - labels)
    focal_weight = (1.0 - pt) ** gamma
    loss = focal_weight * bce
    if valid_mask is not None:
        loss = loss * valid_mask
        return loss.sum() / valid_mask.sum().clamp(min=1.0)
    return loss.mean()


# ============ Dataset ============

class CSCDataset(Dataset):
    def __init__(self, pairs, char_to_id, max_len=128, pad_id=0):
        self.pairs = pairs
        self.char_to_id = char_to_id
        self.max_len = max_len
        self.pad_id = pad_id

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        L = min(len(src), len(tgt), self.max_len)
        src_ids = [self.char_to_id(c) for c in src[:L]]
        cor_ids = [self.char_to_id(c) for c in tgt[:L]]
        det_lbl = [1.0 if src[i] != tgt[i] else 0.0 for i in range(L)]
        return {
            "input_ids": src_ids,
            "cor_labels": cor_ids,
            "det_labels": det_lbl,
            "length": L,
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
        attention_mask = torch.zeros((B, max_l), dtype=torch.long)
        for i, item in enumerate(batch):
            n = item["length"]
            input_ids[i, :n] = torch.tensor(item["input_ids"], dtype=torch.long)
            cor_labels[i, :n] = torch.tensor(item["cor_labels"], dtype=torch.long)
            det_labels[i, :n] = torch.tensor(item["det_labels"], dtype=torch.float)
            attention_mask[i, :n] = 1
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "cor_labels": cor_labels,
            "det_labels": det_labels,
        }


# ============ Char-to-id with cache ============

class CharToId:
    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.cache = {}
        self.unk_id = tokenizer.unk_token_id

    def __call__(self, c):
        if c in self.cache:
            return self.cache[c]
        ids = self.tok.encode(c, add_special_tokens=False)
        tid = ids[0] if ids else self.unk_id
        self.cache[c] = tid
        return tid


# ============ Inference + Eval ============

@torch.no_grad()
def correct_batch(model, char_to_id, texts, device, max_len=128, threshold=0.7):
    """跟 pycorrector MacBertCorrector 对齐:
      - argmax cor_logits → 候选字
      - softmax cor 取 top1 prob,< threshold 保留原字
      - 长度不一致保留原字"""
    model.eval()
    pad_id = char_to_id.tok.pad_token_id
    B = len(texts)
    # 编码
    encoded = []
    lengths = []
    for t in texts:
        L = min(len(t), max_len)
        ids = [char_to_id(c) for c in t[:L]]
        encoded.append(ids)
        lengths.append(L)
    max_l = max(lengths)
    input_ids = torch.full((B, max_l), pad_id, dtype=torch.long)
    attn = torch.zeros((B, max_l), dtype=torch.long)
    for i, ids in enumerate(encoded):
        input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        attn[i, :len(ids)] = 1
    input_ids = input_ids.to(device)
    attn = attn.to(device)
    cor_logits, _ = model(input_ids, attn)
    probs = F.softmax(cor_logits, dim=-1)
    top_probs, top_ids = probs.max(dim=-1)   # [B, L]

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
            if p < threshold:
                pred.append(orig[j])
            elif tid in id_to_char:
                pred.append(id_to_char[tid])
            else:
                pred.append(orig[j])  # 未知 id 保留原字
        # 补回截掉的尾部
        if len(text) > L:
            pred.extend(list(text[L:]))
        preds.append("".join(pred))
    return preds


def eval_sighan(model, char_to_id, test_tsv, device, max_len=128, threshold=0.7, batch_size=32):
    """pycorrector 口径 sentence-level eval。"""
    samples = []
    with open(test_tsv, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            samples.append((parts[0], parts[1]))
    srcs = [s for s, _ in samples]
    tgts = [t for _, t in samples]
    preds = []
    for i in range(0, len(srcs), batch_size):
        chunk = srcs[i:i+batch_size]
        preds.extend(correct_batch(model, char_to_id, chunk, device,
                                   max_len=max_len, threshold=threshold))
    TP = FP = FN = TN = 0
    for src, tgt, pred in zip(srcs, tgts, preds):
        if src == tgt:
            if tgt == pred:
                TN += 1
            else:
                FP += 1
        else:
            if tgt == pred:
                TP += 1
            else:
                FN += 1
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

    # tokenizer: Modern BERTc ckpt dir 没有 piece.model,用共享 tokenizer dir
    tok_dir = args.tokenizer_dir
    if not os.path.exists(os.path.join(tok_dir, "piece.model")):
        raise SystemExit(f"piece.model 不在 {tok_dir}")
    tokenizer = PieceTokenizerAdapter(tok_dir)
    char_to_id = CharToId(tokenizer)
    pad_id = tokenizer.pad_token_id
    vocab_size = tokenizer.vocab_size

    # 数据
    print(f"Loading pairs from {args.train_pkl}...", flush=True)
    with open(args.train_pkl, "rb") as f:
        pairs = pickle.load(f)
    print(f"  {len(pairs)} pairs", flush=True)

    # 预热 char_to_id cache(全量,避免训练时反复 SP encode + 给 inference 用)
    print("Warming char->id cache (full corpus)...", flush=True)
    t_warm = time.time()
    seen = set()
    for src, tgt in pairs:
        for c in src:
            if c not in seen:
                seen.add(c); char_to_id(c)
        for c in tgt:
            if c not in seen:
                seen.add(c); char_to_id(c)
    # 加上测试集 + sighan 字符以防漏
    with open(args.test_tsv, "r", encoding="utf-8") as f:
        for line in f:
            for c in line.strip():
                if c not in seen:
                    seen.add(c); char_to_id(c)
    print(f"  cache warmed: {len(char_to_id.cache)} unique chars in {time.time()-t_warm:.1f}s",
          flush=True)

    train_ds = CSCDataset(pairs, char_to_id, max_len=args.max_len, pad_id=pad_id)
    collator = CSCCollator(pad_id=pad_id)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collator, num_workers=0, pin_memory=True)
    total_steps = args.epochs * len(loader)
    print(f"steps/epoch: {len(loader)}, total: {total_steps}", flush=True)

    # 模型
    model = BERTcForCSC(args.backbone_path, vocab_size).to(device)

    # 优化器
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

    # 输出目录
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
            cor_labels = batch["cor_labels"].to(device, non_blocking=True)
            det_labels = batch["det_labels"].to(device, non_blocking=True)

            cor_logits, det_logits = model(input_ids, attn)
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

        # epoch end eval
        print(f"\nEval after epoch {ep}...", flush=True)
        metrics = eval_sighan(model, char_to_id, args.test_tsv, device,
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

        # save
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save({"model": model.state_dict(),
                        "epoch": ep,
                        "metrics": metrics,
                        "args": vars(args)},
                       out / "best.pt")
            print(f"    ↑ saved best.pt (F1={metrics['f1']:.4f})", flush=True)
        torch.save({"model": model.state_dict(),
                    "epoch": ep,
                    "metrics": metrics,
                    "args": vars(args)},
                   out / "final.pt")

    total_min = (time.time() - t0) / 60
    print(f"\nDone. Best SIGHAN-15 F1: {best_f1:.4f}", flush=True)
    print(f"Total: {total_min:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone_path", required=True,
                    help="Modern BERTc ckpt dir, e.g. .../output_v4_mid/checkpoint-8500")
    ap.add_argument("--tokenizer_dir",
                    default="/home/tfbao/Shiyu/BERTc/pretrain/modern_bertc/tokenizer",
                    help="piece.model 所在目录")
    ap.add_argument("--train_pkl", default="/home/tfbao/Shiyu/BERTc/csc/data/all_pairs.pkl")
    ap.add_argument("--test_tsv",
                    default="/home/tfbao/Shiyu/BERTc/csc/data/test/sighan2015_test_official.tsv")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--det_weight", type=float, default=0.3,
                    help="detection loss 权重(MacBERT4CSC 默认 0.3)")
    ap.add_argument("--threshold", type=float, default=0.7,
                    help="纠错置信度阈值,< threshold 保留原字(MacBERT4CSC 默认 0.7)")
    ap.add_argument("--log_every", type=int, default=100)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
