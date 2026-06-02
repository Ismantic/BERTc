"""跑多个 baseline 在 cws_dev.pd98 上 inference,找出 N-way disagree gold 的 sample。

输出:ambiguous_dev_cases.jsonl,含每个 case 的:
  - text
  - gold (PD-1998)
  - 各 baseline 的 pred
  - disagree spans(各 baseline 都不同意 gold 的 span)

这些是"高概率 gold 自身有问题"的 case,送 LLM judge。
"""
import argparse, json, os, sys, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import CWSDataset, Collator, bies_to_words
from model import BertCRF


def get_tokenizer(model_path):
    if os.path.exists(os.path.join(model_path, "piece.model")):
        from piece_tokenizer_adapter import PieceTokenizerAdapter
        return PieceTokenizerAdapter(model_path)
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_path, use_fast=False)


def run_inference(model_path, ckpt_path, dev_ds, device, batch_size=16):
    tokenizer = get_tokenizer(model_path)
    model = BertCRF(model_path, num_tags=4).to(device)
    sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(sd)
    model.eval()
    collator = Collator(tokenizer)
    loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collator, num_workers=0)
    preds = []
    t0 = time.time()
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            if device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    tag_seqs = model.decode(batch["input_ids"], batch["attention_mask"])
            else:
                tag_seqs = model.decode(batch["input_ids"], batch["attention_mask"])
            for tags in tag_seqs:
                preds.append(tags)
            if bi % 50 == 0:
                el = time.time() - t0
                rate = len(preds) / max(1, el)
                eta = (len(dev_ds) - len(preds)) / rate / 60
                print(f"  {len(preds):,}/{len(dev_ds):,} {el:.0f}s ({rate:.0f}/s) ETA {eta:.1f}m",
                      flush=True)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev_jsonl", default="./data/cws_dev.pd98.jsonl")
    ap.add_argument("--output", default="./data/ambiguous_dev_cases.jsonl")
    ap.add_argument("--device", default="cpu", help="cuda or cpu(默认 cpu 不抢 GPU)")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--min_disagree", type=int, default=1,
                    help="多少 baseline disagree gold span 才记 ambiguous(1=any,3=unanimous)")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"Loading dev: {args.dev_jsonl}")
    dev_ds = CWSDataset(args.dev_jsonl)
    print(f"  {len(dev_ds)} samples")

    # 3 baseline 配置
    HERE = os.path.dirname(os.path.abspath(__file__))
    SUM = os.path.dirname(HERE)
    baselines = [
        ("v4_pd98",    f"{SUM}/bert_train_v4_mid",      f"{HERE}/output_v4_pd98_crf/best.pt"),
        ("roberta",    f"{HERE}/roberta-wwm-ext",       f"{HERE}/output_roberta_pd98_crf/best.pt"),
        ("macbert",    f"{HERE}/macbert-large",         f"{HERE}/output_macbert_pd98_crf/best.pt"),
    ]

    all_preds = {}  # name → list of pred tags
    for name, mp, ckpt in baselines:
        if not os.path.exists(ckpt):
            print(f"⚠️  skip {name}: ckpt {ckpt} 不存在")
            continue
        print(f"\n=== Inference {name} on {args.device} ===")
        all_preds[name] = run_inference(mp, ckpt, dev_ds, device, args.batch_size)

    # 算 disagree gold 的 span
    print(f"\n=== 找 N-way disagree case ===")
    ambig_cases = []
    n_total = 0
    for idx in range(len(dev_ds.items)):
        item = dev_ds.items[idx]
        chars = item["chars"]
        gold_tags = item["tags"]
        gold_words = bies_to_words(chars, gold_tags)
        gold_spans = set()
        pos = 0
        for w in gold_words:
            gold_spans.add((pos, pos+len(w)))
            pos += len(w)
        # 各 baseline pred
        pred_words_per_model = {}
        pred_spans_per_model = {}
        for name in all_preds:
            tags = all_preds[name][idx]
            n = min(len(chars), len(tags))
            pw = bies_to_words(chars[:n], tags[:n])
            pred_words_per_model[name] = pw
            ps = set()
            pp = 0
            for w in pw:
                ps.add((pp, pp+len(w)))
                pp += len(w)
            pred_spans_per_model[name] = ps
        # 找 spans: gold 有,但所有 baseline 都不在它们的 pred 里(N-way disagree gold)
        # 至少一个 gold span 满足这条件 → sample 进 ambig list
        disagree_spans = []
        for s in gold_spans:
            n_disagree = sum(1 for name in all_preds if s not in pred_spans_per_model[name])
            if n_disagree >= args.min_disagree:
                disagree_spans.append((s, n_disagree))
        if disagree_spans:
            text = "".join(chars)
            ambig_cases.append({
                "idx": idx,
                "text": text,
                "gold": gold_words,
                "preds": pred_words_per_model,
                "disagree_spans": [
                    {"start": s, "end": e, "gold_word": text[s:e], "n_disagree": n}
                    for (s, e), n in disagree_spans
                ],
            })
        n_total += 1

    print(f"  total {n_total}, ambiguous {len(ambig_cases)} ({100*len(ambig_cases)/max(1,n_total):.1f}%)")
    with open(args.output, "w", encoding="utf8") as f:
        for c in ambig_cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"  → {args.output}")


if __name__ == "__main__":
    main()
