"""Dump v4 CWS-v1 (BertCRF) 对 PD-06 dev 全 20973 sample 的预测到 jsonl。

每条:
  {
    "idx": int,
    "text": "...",          # 原文
    "gold": [...],          # PD-06 标注
    "pred": [...],          # 模型预测
    "match": bool,          # 是否完全一致
    "fp": [(s, e, word)],   # pred 多/错的 span
    "fn": [(s, e, word)]    # gold 有但 pred 没的 span
  }

用法:
  python dump_all_predictions.py --ckpt output_char_mid_v4_crf/best.pt \
      --model_path ../bert_train_v4_mid --device cpu \
      --output all_preds.jsonl
"""
import argparse, json, os, sys, time
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import CWSDataset, Collator, bies_to_words
from model import BertCRF
from piece_tokenizer_adapter import PieceTokenizerAdapter


def spans(words):
    out, pos = [], 0
    for w in words:
        out.append((pos, pos + len(w), w))
        pos += len(w)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--dev_jsonl", default="./data/cws_dev.jsonl")
    ap.add_argument("--output", default="all_preds.jsonl")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"Loading tokenizer + model from {args.model_path}")
    if os.path.exists(os.path.join(args.model_path, "piece.model")):
        tokenizer = PieceTokenizerAdapter(args.model_path)
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    model = BertCRF(args.model_path, num_tags=4).to(device)
    sd = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(sd)
    model.eval()
    print(f"  ckpt: {args.ckpt}")

    dev = CWSDataset(args.dev_jsonl)
    if args.limit:
        dev.items = dev.items[:args.limit]
    print(f"Dev: {len(dev)} samples")

    collator = Collator(tokenizer)
    loader = DataLoader(dev, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collator, num_workers=0)

    print(f"Predicting → {args.output}")
    t0 = time.time()
    idx = 0
    with open(args.output, "w", encoding="utf8") as fout, torch.no_grad():
        for bi, batch in enumerate(loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            if args.device == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    preds = model.decode(batch["input_ids"], batch["attention_mask"])
            else:
                preds = model.decode(batch["input_ids"], batch["attention_mask"])
            for pred_tags in preds:
                item = dev.items[idx]
                chars = item["chars"]
                n = min(len(chars), len(pred_tags))
                pred_words = bies_to_words(chars[:n], pred_tags[:n])
                gold_words = bies_to_words(chars[:n], item["tags"][:n])
                text = "".join(chars[:n])
                gset = {(s, e) for s, e, _ in spans(gold_words)}
                pset = {(s, e) for s, e, _ in spans(pred_words)}
                fp = [(s, e, w) for s, e, w in spans(pred_words) if (s, e) not in gset]
                fn = [(s, e, w) for s, e, w in spans(gold_words) if (s, e) not in pset]
                match = (gold_words == pred_words)
                rec = {
                    "idx": idx,
                    "text": text,
                    "gold": gold_words,
                    "pred": pred_words,
                    "match": match,
                    "fp": fp,
                    "fn": fn,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                idx += 1
            if bi % 50 == 0:
                el = time.time() - t0
                rate = idx / max(1, el)
                eta = (len(dev) - idx) / rate / 60
                print(f"  {idx:,}/{len(dev):,} ({100*idx/len(dev):.1f}%) "
                      f"{el:.0f}s ({rate:.0f}/s) ETA {eta:.0f}m", flush=True)

    print(f"\nDone in {time.time()-t0:.0f}s → {args.output}")
    print(f"\n常用 jq 查询例子:")
    print(f"  全部不一致: jq 'select(.match==false)' {args.output} | less")
    print(f"  搜含某字:    jq 'select(.text|contains(\"漏\"))' {args.output}")
    print(f"  按 FP 多排:  jq -c 'select(.fp|length > 2)' {args.output} | head")


if __name__ == "__main__":
    main()
