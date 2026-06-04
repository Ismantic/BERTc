"""Modern BERTc MT fine-tune(CWS + POS + NER joint)。

跟 NLP_BERT_CRF/train_mt.py 完全同 recipe(SOTA 配置:v6.5+FGM 5ep,score 1.4636),
仅 backbone 改用 ModernBertModel 加载 v3/v4 ckpt(.pt + ema)。

用法(SOTA 配置):
  python train_modern_mt.py \
      --ckpt_dir /home/tfbao/Shiyu/BERTc/pretrain/modern_bertc/output_v3/checkpoint-7000 \
      --output_dir /tmp/v3_mt_out \
      --alpha_pos 2.0 --beta_ner 0.5 \
      --fgm --fgm_eps 1.0 \
      --epochs 5 --batch_size 64 \
      --bert_lr 2e-5 --head_lr 5e-4 --warmup_ratio 0.1
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
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchcrf import CRF
from transformers import get_linear_schedule_with_warmup

# NLP_BERT_CRF utilities(MTDataset, MTCollator, evaluate, etc.)
sys.path.insert(0, "/home/tfbao/Shiyu/BERTc/finetune/NLP_BERT_CRF")
from data_mt import MTDataset, MTCollator, DistillMTDataset, ConcatMTDataset
from data_pos_ner import build_pos_vocab
from piece_tokenizer_adapter import PieceTokenizerAdapter
from train_mt import evaluate as _evaluate_mt  # reuse upstream evaluate()

# ModernBertModel(v3/v4 backbone)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import ModernBertConfig, ModernBertModel


# ============ Model ============

class ModernBertMT(nn.Module):
    """ModernBertModel + 3 heads(CWS CRF / POS Linear / NER CRF)。仿 NLP_BERT_CRF.model_mt.BertMT。"""
    def __init__(self, ckpt_dir: str, num_cws: int = 4, num_pos: int = 27, num_ner: int = 13,
                 dropout: float = 0.1):
        super().__init__()
        with open(os.path.join(ckpt_dir, "config.json")) as f:
            cfg_dict = json.load(f)
        cfg = ModernBertConfig(**{k: v for k, v in cfg_dict.items()
                                   if k in ModernBertConfig.__dataclass_fields__})
        self.bert = ModernBertModel(cfg)
        ckpt = torch.load(os.path.join(ckpt_dir, "model.pt"),
                          map_location="cpu", weights_only=False)
        # 优先 EMA shadow(更稳),回退 raw
        sd = ckpt.get("ema") or ckpt["model"]
        bert_sd = {k[len("bert."):]: v for k, v in sd.items() if k.startswith("bert.")}
        self.bert.load_state_dict(bert_sd, strict=True)

        hidden = cfg.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.cws_classifier = nn.Linear(hidden, num_cws)
        self.cws_crf = CRF(num_cws, batch_first=True)
        self.pos_classifier = nn.Linear(hidden, num_pos)
        self.ner_classifier = nn.Linear(hidden, num_ner)
        self.ner_crf = CRF(num_ner, batch_first=True)
        self.cfg = cfg

    def forward(self, input_ids, attention_mask,
                cws_labels=None, pos_labels=None, ner_labels=None):
        # fine-tune 走 SDPA + attention_mask(seg_ids=None)
        hs = self.bert(input_ids, attention_mask=attention_mask)
        hs = self.dropout(hs)
        cws_emi = self.cws_classifier(hs)
        pos_logits = self.pos_classifier(hs)
        ner_emi = self.ner_classifier(hs)
        mask = attention_mask.bool()
        losses = {}
        if cws_labels is not None:
            losses["cws"] = -self.cws_crf(cws_emi, cws_labels, mask=mask, reduction="mean")
        if pos_labels is not None:
            losses["pos"] = F.cross_entropy(
                pos_logits.view(-1, pos_logits.size(-1)),
                pos_labels.view(-1), ignore_index=-100)
        if ner_labels is not None:
            losses["ner"] = -self.ner_crf(ner_emi, ner_labels, mask=mask, reduction="mean")
        return losses, (cws_emi, pos_logits, ner_emi)

    @torch.no_grad()
    def decode_cws(self, input_ids, attention_mask):
        hs = self.bert(input_ids, attention_mask=attention_mask)
        emi = self.cws_classifier(self.dropout(hs)).float()
        return self.cws_crf.decode(emi, mask=attention_mask.bool())

    @torch.no_grad()
    def decode_ner(self, input_ids, attention_mask):
        hs = self.bert(input_ids, attention_mask=attention_mask)
        emi = self.ner_classifier(self.dropout(hs)).float()
        return self.ner_crf.decode(emi, mask=attention_mask.bool())

    @torch.no_grad()
    def predict_pos(self, input_ids, attention_mask):
        hs = self.bert(input_ids, attention_mask=attention_mask)
        logits = self.pos_classifier(self.dropout(hs))
        return logits.argmax(-1)


# ============ FGM ============

class FGM:
    """对 embedding 加 norm-bounded perturbation。
    注意:跟 train_mt.py 不同,我们 embedding 命名是 `bert.embed.weight`(而非 word_embeddings)。
    """
    def __init__(self, model, eps=1.0):
        self.model = model
        self.eps = eps
        self.backup = {}

    def attack(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad and ("bert.embed.weight" in n or "word_embeddings" in n):
                self.backup[n] = p.data.clone()
                if p.grad is None:
                    continue
                norm = torch.norm(p.grad)
                if norm and not torch.isnan(norm):
                    p.data.add_(self.eps * p.grad / norm)

    def restore(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.backup:
                p.data = self.backup[n]
        self.backup = {}


# ============ Main ============

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True,
                    help="Modern BERTc ckpt 目录(含 model.pt + config.json)")
    ap.add_argument("--tokenizer_dir",
                    default="/home/tfbao/Shiyu/BERTc/pretrain/modern_bertc/tokenizer",
                    help="piece.model + mask_token_id.txt 所在目录")
    ap.add_argument("--cws_train", default="/home/tfbao/Shiyu/BERTc/finetune/NLP_BERT_CRF/data/cws.pd98.jsonl")
    ap.add_argument("--cws_dev",   default="/home/tfbao/Shiyu/BERTc/finetune/NLP_BERT_CRF/data/cws_dev.pd98.jsonl")
    ap.add_argument("--pos_train", default="/home/tfbao/Shiyu/BERTc/finetune/NLP_BERT_CRF/data/pos.pd98.jsonl")
    ap.add_argument("--pos_dev",   default="/home/tfbao/Shiyu/BERTc/finetune/NLP_BERT_CRF/data/pos_dev.pd98.jsonl")
    ap.add_argument("--ner_train", default="/home/tfbao/Shiyu/BERTc/finetune/NLP_BERT_CRF/data/ner.pd98.jsonl")
    ap.add_argument("--ner_dev",   default="/home/tfbao/Shiyu/BERTc/finetune/NLP_BERT_CRF/data/ner_dev.pd98.jsonl")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_chars", type=int, default=254)
    ap.add_argument("--bert_lr", type=float, default=2e-5)
    ap.add_argument("--head_lr", type=float, default=5e-4)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--alpha_pos", type=float, default=2.0,
                    help="POS loss 权重(SOTA: 2.0)")
    ap.add_argument("--beta_ner", type=float, default=0.5,
                    help="NER loss 权重(SOTA: 0.5)")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--eval_dev_limit", type=int, default=2000)
    ap.add_argument("--fgm", action="store_true")
    ap.add_argument("--fgm_eps", type=float, default=1.0)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    # POS vocab
    pos2id = build_pos_vocab(args.pos_train)
    with open(out_dir / "pos_vocab.json", "w") as f:
        json.dump(pos2id, f, ensure_ascii=False, indent=2)
    print(f"POS vocab: {len(pos2id)} tags")

    # Tokenizer(我们的 piece tokenizer)
    tokenizer = PieceTokenizerAdapter(args.tokenizer_dir)

    # Datasets
    print("Loading train(PD strong)...")
    train_ds = MTDataset(args.cws_train, args.pos_train, args.ner_train,
                         pos2id, max_chars=args.max_chars)
    print(f"  PD: {len(train_ds)} samples")
    print("Loading dev...")
    full_dev_ds = MTDataset(args.cws_dev, args.pos_dev, args.ner_dev,
                            pos2id, max_chars=args.max_chars)
    print(f"  {len(full_dev_ds)} samples")

    class DevSubset:
        def __init__(self, items): self.items = items
        def __len__(self): return len(self.items)
        def __getitem__(self, i): return self.items[i]
    dev_subset = DevSubset(full_dev_ds.items[:args.eval_dev_limit])

    collator = MTCollator(tokenizer)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collator, num_workers=0, pin_memory=True)

    # Model
    model = ModernBertMT(args.ckpt_dir, num_pos=len(pos2id)).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ModernBertMT: {n_params/1e6:.1f}M params  "
          f"H={model.cfg.hidden_size} L={model.cfg.num_hidden_layers}")

    # Optimizer(bert_lr + head_lr 分组)
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight")
    bert_params = list(model.bert.named_parameters())
    head_params = (list(model.cws_classifier.named_parameters()) +
                   list(model.cws_crf.named_parameters()) +
                   list(model.pos_classifier.named_parameters()) +
                   list(model.ner_classifier.named_parameters()) +
                   list(model.ner_crf.named_parameters()))
    grouped = [
        {"params": [p for n, p in bert_params if not any(nd in n for nd in no_decay)],
         "lr": args.bert_lr, "weight_decay": args.weight_decay},
        {"params": [p for n, p in bert_params if any(nd in n for nd in no_decay)],
         "lr": args.bert_lr, "weight_decay": 0.0},
        {"params": [p for _, p in head_params],
         "lr": args.head_lr, "weight_decay": 0.0},
    ]
    optim = AdamW(grouped)
    total_steps = len(loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optim, int(total_steps * args.warmup_ratio), total_steps)
    print(f"Total steps: {total_steps}\n")

    fgm = FGM(model, eps=args.fgm_eps) if args.fgm else None
    if args.fgm:
        print(f"FGM enabled, eps={args.fgm_eps}")

    model.train()
    global_step = 0
    best_score = 0.0
    t_start = time.time()
    for epoch in range(args.epochs):
        ep_losses = {"cws": 0, "pos": 0, "ner": 0}
        ep_n = 0
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                losses, _ = model(batch["input_ids"], batch["attention_mask"],
                                  cws_labels=batch["cws_labels"],
                                  pos_labels=batch["pos_labels"],
                                  ner_labels=batch["ner_labels"])
            loss = losses["cws"] + args.alpha_pos * losses["pos"] + args.beta_ner * losses["ner"]
            loss.backward()
            if fgm is not None:
                fgm.attack()
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    losses_adv, _ = model(batch["input_ids"], batch["attention_mask"],
                                          cws_labels=batch["cws_labels"],
                                          pos_labels=batch["pos_labels"],
                                          ner_labels=batch["ner_labels"])
                (losses_adv["cws"] + args.alpha_pos * losses_adv["pos"]
                 + args.beta_ner * losses_adv["ner"]).backward()
                fgm.restore()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()
            scheduler.step()
            optim.zero_grad()
            for k in ep_losses:
                ep_losses[k] += losses[k].item()
            ep_n += 1
            global_step += 1
            if global_step % args.log_every == 0:
                el = time.time() - t_start
                sps = global_step / el
                eta = (total_steps - global_step) / sps / 60
                print(f"  ep{epoch+1} step {global_step}/{total_steps}  "
                      f"cws={losses['cws'].item():.3f} pos={losses['pos'].item():.3f} "
                      f"ner={losses['ner'].item():.3f}  "
                      f"lr={optim.param_groups[0]['lr']:.2e}  "
                      f"{sps:.1f}/s ETA {eta:.1f}m", flush=True)

        cws_f1, pos_acc, ner_f1 = _evaluate_mt(model, dev_subset, collator, device)
        avg = {k: v / max(1, ep_n) for k, v in ep_losses.items()}
        print(f"\n=== Epoch {epoch+1}/{args.epochs}  "
              f"avg_cws_loss={avg['cws']:.3f} pos_loss={avg['pos']:.3f} ner_loss={avg['ner']:.3f}  "
              f"dev: cws_F1={cws_f1:.4f}  pos_acc={pos_acc:.4f}  ner_F1={ner_f1:.4f} ===",
              flush=True)
        # SOTA scoring: cws_f1 + 0.3*pos_acc + 0.2*ner_f1
        score = cws_f1 + 0.3 * pos_acc + 0.2 * ner_f1
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), out_dir / "best.pt")
            print(f"    ↑ saved best.pt (score={score:.4f})", flush=True)
        print(flush=True)

    torch.save(model.state_dict(), out_dir / "final.pt")
    print(f"\nDone. Best score: {best_score:.4f}")
    print(f"Total: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
