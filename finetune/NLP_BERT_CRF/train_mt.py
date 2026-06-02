"""Multi-task training: CWS + POS + NER joint。

joint loss = cws_loss + alpha * pos_loss + beta * ner_loss
eval per task,save best (best_cws_f1 + 0.5*pos_acc + 0.3*ner_f1 weighted)。
"""
import argparse, os, sys, time, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import bies_to_words
from data_pos_ner import build_pos_vocab, ID2NER, NER_TAGS
from data_mt import MTDataset, MTCollator, DistillMTDataset, ConcatMTDataset
from model_mt import BertMT


def boundary_f1(pred_words, gold_words):
    def spans(ws):
        out, pos = set(), 0
        for w in ws:
            out.add((pos, pos + len(w))); pos += len(w)
        return out
    P, G = spans(pred_words), spans(gold_words)
    if not P or not G: return 0.0
    tp = len(P & G)
    if tp == 0: return 0.0
    p = tp / len(P); r = tp / len(G)
    return 2*p*r/(p+r)


def ner_tags_to_spans(tag_ids):
    from ltp_label_align import bies_tags_to_spans
    return [(t, s, e) for t, s, e in bies_tags_to_spans([int(x) for x in tag_ids])]


def ner_f1(pred_spans, gold_spans):
    P = set(pred_spans); G = set(gold_spans)
    if not P or not G: return 0.0
    tp = len(P & G)
    if tp == 0: return 0.0
    p = tp/len(P); r = tp/len(G)
    return 2*p*r/(p+r)


@torch.no_grad()
def evaluate(model, dataset, collator, device, batch_size=64):
    """对齐 LTP/eval_ltp_mt.py:micro CWS F1 + per-word POS acc + micro NER F1。"""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collator, num_workers=0)
    cws_tp, cws_p, cws_g = 0, 0, 0
    ner_tp, ner_p, ner_g = 0, 0, 0
    pos_word_correct, pos_word_total = 0, 0
    pos_char_correct, pos_char_total = 0, 0
    idx = 0
    for batch in loader:
        batch_d = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            cws_preds = model.decode_cws(batch_d["input_ids"], batch_d["attention_mask"])
            ner_preds = model.decode_ner(batch_d["input_ids"], batch_d["attention_mask"])
            pos_pred = model.predict_pos(batch_d["input_ids"], batch_d["attention_mask"]).cpu().numpy()
        pos_labels = batch_d["pos_labels"].cpu().numpy()
        valid = pos_labels != -100
        pos_char_correct += int(((pos_pred == pos_labels) & valid).sum())
        pos_char_total += int(valid.sum())
        for i in range(len(cws_preds)):
            item = dataset.items[idx]
            chars = item["chars"]
            n = min(len(chars), len(cws_preds[i]), len(ner_preds[i]))
            gold_words = bies_to_words(chars[:n], item["cws_tags"][:n])
            pred_words = bies_to_words(chars[:n], cws_preds[i][:n])
            # CWS micro
            def _spans(ws):
                out, p = set(), 0
                for w in ws: out.add((p, p+len(w))); p += len(w)
                return out
            P, G = _spans(pred_words), _spans(gold_words)
            cws_tp += len(P & G); cws_p += len(P); cws_g += len(G)
            # POS per-word(仅当 cws pred==gold)
            if pred_words == gold_words:
                # 重建 gold word POS:item['pos_tags'] per-char,取每个 word 第 1 char
                cpos = 0
                pp_pred = pos_pred[i]
                for w in gold_words:
                    gp = int(item['pos_tags'][cpos]) if cpos < len(item['pos_tags']) else -100
                    pr = int(pp_pred[cpos]) if cpos < len(pp_pred) else -1
                    if gp >= 0:
                        pos_word_total += 1
                        if pr == gp: pos_word_correct += 1
                    cpos += len(w)
            # NER micro
            gold_ner = set(ner_tags_to_spans(item["ner_tags"][:n]))
            pred_ner = set(ner_tags_to_spans(ner_preds[i][:n]))
            ner_tp += len(gold_ner & pred_ner)
            ner_p += len(pred_ner)
            ner_g += len(gold_ner)
            idx += 1
    model.train()
    cws_pp, cws_rr = cws_tp/max(1,cws_p), cws_tp/max(1,cws_g)
    cws_f1 = 2*cws_pp*cws_rr/max(1e-9, cws_pp+cws_rr)
    ner_pp, ner_rr = ner_tp/max(1,ner_p), ner_tp/max(1,ner_g)
    ner_f1 = 2*ner_pp*ner_rr/max(1e-9, ner_pp+ner_rr)
    return (cws_f1, pos_word_correct/max(1,pos_word_total), ner_f1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--cws_train", default="./data/cws.pd98.jsonl")
    ap.add_argument("--cws_dev",   default="./data/cws_dev.pd98.jsonl")
    ap.add_argument("--pos_train", default="./data/pos.pd98.jsonl")
    ap.add_argument("--pos_dev",   default="./data/pos_dev.pd98.jsonl")
    ap.add_argument("--ner_train", default="./data/ner.pd98.jsonl")
    ap.add_argument("--ner_dev",   default="./data/ner_dev.pd98.jsonl")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_chars", type=int, default=254)
    ap.add_argument("--bert_lr", type=float, default=2e-5)
    ap.add_argument("--head_lr", type=float, default=5e-4)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--alpha_pos", type=float, default=0.5)
    ap.add_argument("--beta_ner", type=float, default=0.5)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--eval_dev_limit", type=int, default=2000)
    ap.add_argument("--fgm", action="store_true")
    ap.add_argument("--fgm_eps", type=float, default=1.0)
    ap.add_argument("--distill_jsonl", default=None,
                    help="LTP-distill MT jsonl(单文件 text/words/pos/entities)")
    ap.add_argument("--distill_limit", type=int, default=0,
                    help="限 distill items 数(0=全用)")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    # Build POS vocab from train
    pos2id = build_pos_vocab(args.pos_train)
    with open(out_dir / "pos_vocab.json", "w") as f:
        json.dump(pos2id, f, ensure_ascii=False, indent=2)
    print(f"POS vocab: {len(pos2id)} tags")

    # Tokenizer
    if os.path.exists(os.path.join(args.model_path, "piece.model")):
        from piece_tokenizer_adapter import PieceTokenizerAdapter
        tokenizer = PieceTokenizerAdapter(args.model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)

    # Datasets
    print("Loading train(PD strong)...")
    train_ds = MTDataset(args.cws_train, args.pos_train, args.ner_train,
                         pos2id, max_chars=args.max_chars)
    print(f"  PD: {len(train_ds)} samples")
    if args.distill_jsonl:
        print(f"Loading distill: {args.distill_jsonl}")
        d_ds = DistillMTDataset(args.distill_jsonl, pos2id, max_chars=args.max_chars)
        if args.distill_limit and args.distill_limit < len(d_ds):
            d_ds.items = d_ds.items[:args.distill_limit]
        print(f"  distill: {len(d_ds)} samples")
        train_ds = ConcatMTDataset(train_ds, d_ds)
        print(f"  combined: {len(train_ds)} samples")
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
    model = BertMT(args.model_path, num_pos=len(pos2id)).to(device)
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    bert_params = list(model.bert.named_parameters())
    head_params = (list(model.cws_classifier.named_parameters()) +
                   list(model.cws_crf.named_parameters()) +
                   list(model.pos_classifier.named_parameters()) +
                   list(model.ner_classifier.named_parameters()) +
                   list(model.ner_crf.named_parameters()))
    grouped = [
        {"params": [p for n,p in bert_params if not any(nd in n for nd in no_decay)],
         "lr": args.bert_lr, "weight_decay": args.weight_decay},
        {"params": [p for n,p in bert_params if any(nd in n for nd in no_decay)],
         "lr": args.bert_lr, "weight_decay": 0.0},
        {"params": [p for _,p in head_params],
         "lr": args.head_lr, "weight_decay": 0.0},
    ]
    optim = AdamW(grouped)
    total_steps = len(loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optim, int(total_steps*args.warmup_ratio), total_steps)
    print(f"Total steps: {total_steps}\n")

    # FGM
    class FGM:
        def __init__(self, model, eps=1.0):
            self.model = model; self.eps = eps; self.backup = {}
        def attack(self):
            for n, p in self.model.named_parameters():
                if p.requires_grad and "word_embeddings" in n:
                    self.backup[n] = p.data.clone()
                    if p.grad is None: continue
                    norm = torch.norm(p.grad)
                    if norm and not torch.isnan(norm):
                        p.data.add_(self.eps * p.grad / norm)
        def restore(self):
            for n, p in self.model.named_parameters():
                if p.requires_grad and "word_embeddings" in n and n in self.backup:
                    p.data = self.backup[n]
            self.backup = {}
    fgm = FGM(model, eps=args.fgm_eps) if args.fgm else None
    if args.fgm: print(f"FGM enabled, eps={args.fgm_eps}")

    model.train()
    global_step = 0
    best_score = 0.0
    t_start = time.time()
    for epoch in range(args.epochs):
        ep_losses = {"cws":0,"pos":0,"ner":0}
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
                (losses_adv["cws"] + args.alpha_pos*losses_adv["pos"] + args.beta_ner*losses_adv["ner"]).backward()
                fgm.restore()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()
            scheduler.step()
            optim.zero_grad()
            for k in ep_losses: ep_losses[k] += losses[k].item()
            ep_n += 1
            global_step += 1
            if global_step % args.log_every == 0:
                el = time.time() - t_start
                sps = global_step / el
                eta = (total_steps - global_step)/sps/60
                print(f"  ep{epoch+1} step {global_step}/{total_steps}  "
                      f"cws={losses['cws'].item():.3f} pos={losses['pos'].item():.3f} "
                      f"ner={losses['ner'].item():.3f}  lr={optim.param_groups[0]['lr']:.2e}  "
                      f"{sps:.1f}/s ETA {eta:.1f}m", flush=True)

        cws_f1, pos_acc, ner_f1 = evaluate(model, dev_subset, collator, device)
        avg = {k: v/max(1,ep_n) for k,v in ep_losses.items()}
        print(f"\n=== Epoch {epoch+1}/{args.epochs}  "
              f"avg_cws_loss={avg['cws']:.3f} pos_loss={avg['pos']:.3f} ner_loss={avg['ner']:.3f}  "
              f"dev: cws_F1={cws_f1:.4f}  pos_acc={pos_acc:.4f}  ner_F1={ner_f1:.4f} ===", flush=True)
        # weighted score: cws primary, pos/ner secondary
        score = cws_f1 + 0.3*pos_acc + 0.2*ner_f1
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
