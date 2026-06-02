"""MacBERT-large + CRF training for CWS on PD-1998."""
import os
import sys
import time
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import CWSDataset, Collator, bies_to_words  # noqa: E402
from model import BertCRF  # noqa: E402


def boundary_f1(pred_words, gold_words):
    def spans(ws):
        out, pos = set(), 0
        for w in ws:
            out.add((pos, pos + len(w)))
            pos += len(w)
        return out
    P, G = spans(pred_words), spans(gold_words)
    if not P or not G:
        return 0.0
    tp = len(P & G)
    if tp == 0:
        return 0.0
    p = tp / len(P)
    r = tp / len(G)
    return 2 * p * r / (p + r)


@torch.no_grad()
def evaluate(model, dataset, collator, device, batch_size=64):
    """Iterate dataset in order, decode with CRF, compute boundary F1."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collator, num_workers=0)
    f1s = []
    item_idx = 0
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            preds = model.decode(batch["input_ids"], batch["attention_mask"])
        for pred_tags in preds:
            item = dataset.items[item_idx]
            chars = item["chars"]
            n = min(len(chars), len(pred_tags))
            pred_words = bies_to_words(chars[:n], pred_tags[:n])
            gold_words = bies_to_words(chars[:n], item["tags"][:n])
            f1s.append(boundary_f1(pred_words, gold_words))
            item_idx += 1
    model.train()
    return sum(f1s) / max(1, len(f1s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="./macbert-large")
    ap.add_argument("--train_jsonl", default="./data/cws.jsonl")
    ap.add_argument("--dev_jsonl", default="./data/cws_dev.jsonl")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_chars", type=int, default=254)
    ap.add_argument("--bert_lr", type=float, default=2e-5)
    ap.add_argument("--crf_lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--eval_dev_limit", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--init_state_dict", default=None,
                    help="启动时 load 整个 BertCRF state_dict(用于 stage 2 接 stage 1)")
    ap.add_argument("--fgm", action="store_true",
                    help="启用 FGM adversarial training(对 word_embeddings 加扰动)")
    ap.add_argument("--fgm_eps", type=float, default=1.0,
                    help="FGM 扰动大小,通常 0.5-1.0")
    ap.add_argument("--ema", action="store_true",
                    help="EMA shadow weights, eval 用 EMA copy")
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--rdrop", action="store_true",
                    help="R-Drop: 两次 forward + KL 一致性 loss")
    ap.add_argument("--rdrop_alpha", type=float, default=1.0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if os.path.exists(os.path.join(args.model_path, "piece.model")):
        from piece_tokenizer_adapter import PieceTokenizerAdapter
        tokenizer = PieceTokenizerAdapter(args.model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    print(f"Tokenizer: vocab={tokenizer.vocab_size} pad={tokenizer.pad_token_id} "
          f"unk={tokenizer.unk_token_id}")

    print(f"\nLoading train: {args.train_jsonl}")
    t0 = time.time()
    train_ds = CWSDataset(args.train_jsonl, max_chars=args.max_chars)
    print(f"  {len(train_ds)} samples  ({time.time()-t0:.1f}s)")

    print(f"Loading dev: {args.dev_jsonl}")
    t0 = time.time()
    full_dev_ds = CWSDataset(args.dev_jsonl, max_chars=args.max_chars)
    print(f"  {len(full_dev_ds)} samples  ({time.time()-t0:.1f}s)")

    class DevSubset:
        def __init__(self, items):
            self.items = items
        def __len__(self):
            return len(self.items)
        def __getitem__(self, idx):
            return self.items[idx]
    dev_subset = DevSubset(full_dev_ds.items[:args.eval_dev_limit])
    print(f"  eval subset for per-epoch monitor: {len(dev_subset)}")

    collator = Collator(tokenizer)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collator, num_workers=0, pin_memory=True,
                              drop_last=False)

    device = torch.device("cuda")
    model = BertCRF(args.model_path, num_tags=4).to(device)
    if args.init_state_dict:
        sd = torch.load(args.init_state_dict, map_location=device, weights_only=True)
        model.load_state_dict(sd)
        print(f"Initialized BertCRF from {args.init_state_dict}")

    # Param groups: BERT (decoupled lr from CRF head)
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    bert_params = list(model.bert.named_parameters())
    head_params = list(model.classifier.named_parameters()) + list(model.crf.named_parameters())
    grouped = [
        {"params": [p for n, p in bert_params if not any(nd in n for nd in no_decay)],
         "lr": args.bert_lr, "weight_decay": args.weight_decay},
        {"params": [p for n, p in bert_params if any(nd in n for nd in no_decay)],
         "lr": args.bert_lr, "weight_decay": 0.0},
        {"params": [p for _, p in head_params],
         "lr": args.crf_lr, "weight_decay": 0.0},
    ]
    optimizer = AdamW(grouped)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"\nTotal steps: {total_steps}  warmup: {warmup_steps}\n")

    # FGM helper
    class FGM:
        def __init__(self, model, eps=1.0, emb_name="word_embeddings"):
            self.model = model
            self.eps = eps
            self.emb_name = emb_name
            self.backup = {}
        def attack(self):
            for n, p in self.model.named_parameters():
                if p.requires_grad and self.emb_name in n:
                    self.backup[n] = p.data.clone()
                    if p.grad is None: continue
                    norm = torch.norm(p.grad)
                    if norm and not torch.isnan(norm):
                        p.data.add_(self.eps * p.grad / norm)
        def restore(self):
            for n, p in self.model.named_parameters():
                if p.requires_grad and self.emb_name in n and n in self.backup:
                    p.data = self.backup[n]
            self.backup = {}
    fgm = FGM(model, eps=args.fgm_eps) if args.fgm else None
    if args.fgm:
        print(f"FGM enabled, eps={args.fgm_eps}")

    # EMA shadow
    class EMA:
        def __init__(self, model, decay):
            self.decay = decay
            self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        def update(self, model):
            for n, p in model.named_parameters():
                if p.requires_grad and n in self.shadow:
                    self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1-self.decay)
        def apply(self, model):
            self.backup = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
            for n, p in model.named_parameters():
                if p.requires_grad and n in self.shadow:
                    p.data.copy_(self.shadow[n])
        def restore(self, model):
            for n, p in model.named_parameters():
                if p.requires_grad and n in self.backup:
                    p.data.copy_(self.backup[n])
            self.backup = {}
    ema = EMA(model, args.ema_decay) if args.ema else None
    if args.ema: print(f"EMA enabled, decay={args.ema_decay}")
    if args.rdrop: print(f"R-Drop enabled, alpha={args.rdrop_alpha}")

    def compute_loss(input_ids, mask, labels):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            l, _ = model(input_ids, mask, labels)
        if args.rdrop:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                l2, _ = model(input_ids, mask, labels)
            # 两次 forward 不同 dropout → CRF NLL 平均 + 一致性,简化:平均两个 loss
            l = 0.5*(l + l2) + args.rdrop_alpha * (l - l2).pow(2).mean()
        return l

    model.train()
    global_step = 0
    best_f1 = 0.0
    t_start = time.time()

    for epoch in range(args.epochs):
        ep_loss, ep_n = 0.0, 0
        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            loss = compute_loss(batch["input_ids"], batch["attention_mask"], batch["labels"])
            loss.backward()
            # FGM adversarial step: 在 embedding 上加扰动,二次 forward + backward
            if fgm is not None:
                fgm.attack()
                loss_adv = compute_loss(batch["input_ids"], batch["attention_mask"], batch["labels"])
                loss_adv.backward()
                fgm.restore()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if ema is not None:
                ema.update(model)
            ep_loss += loss.item()
            ep_n += 1
            global_step += 1
            if global_step % args.log_every == 0:
                el = time.time() - t_start
                sps = global_step / el
                eta_min = (total_steps - global_step) / sps / 60
                print(f"  ep{epoch+1} step {global_step}/{total_steps}  "
                      f"loss {loss.item():.3f}  "
                      f"lr_bert {optimizer.param_groups[0]['lr']:.2e}  "
                      f"{sps:.1f} step/s  ETA {eta_min:.1f}m", flush=True)

        # eval raw + EMA shadow
        dev_f1 = evaluate(model, dev_subset, collator, device)
        ema_f1 = None
        if ema is not None:
            ema.apply(model)
            ema_f1 = evaluate(model, dev_subset, collator, device)
            ema.restore(model)
        avg_loss = ep_loss / max(1, ep_n)
        ema_str = f" ema={ema_f1:.4f}" if ema_f1 is not None else ""
        print(f"\n=== Epoch {epoch+1}/{args.epochs}  avg_loss {avg_loss:.4f}  "
              f"dev_F1(subset {len(dev_subset)}) raw={dev_f1:.4f}{ema_str} ===", flush=True)
        # prefer EMA for best
        score = ema_f1 if ema_f1 is not None else dev_f1
        if score > best_f1:
            best_f1 = score
            if ema is not None:
                ema.apply(model)
                torch.save(model.state_dict(), out_dir / "best.pt")
                ema.restore(model)
            else:
                torch.save(model.state_dict(), out_dir / "best.pt")
            print(f"    ↑ saved best.pt (score={score:.4f})", flush=True)
        print(flush=True)

    torch.save(model.state_dict(), out_dir / "final.pt")
    tokenizer.save_pretrained(out_dir)
    print(f"\nDone. Best dev_F1(subset): {best_f1:.4f}")
    print(f"Total: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
