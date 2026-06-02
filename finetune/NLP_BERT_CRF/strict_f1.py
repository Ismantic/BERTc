"""Strict boundary F1 vs Style-tolerant F1 — separate 'hard' errors from acceptable style choices.

Definitions (using IsCut 350K dict as "valid Chinese word" oracle):

HARD ERRORS (model definitely wrong):
  H1. INVENTED span     : pred has span ≥2 chars, NOT in dict, NOT in gold
                           e.g. "应哈萨克斯坦" — not a word, model fabricated it
  H2. SPLIT real word   : gold has span ≥2 chars IN dict, pred breaks the chars apart
                           AND the pred's coverage of those chars is NOT all single-char only
                           (i.e., pred mixed g's chars with adjacent context = boundary cross)
                           e.g. "著名/作家" → "著 / 名作家" — pred merged "名" with adjacent

STYLE ERRORS (acceptable alternative):
  S1. Finer granularity : gold = ≥2-char word, pred = same chars as single chars only
                           e.g. "踢球" → "踢 / 球"  (1 word vs 2 words)
  S2. Coarser merge     : pred = ≥2-char span IN dict, gold has it split
                           e.g. "黎/以" → "黎以"  (PD splits, MSR merges)
  S3. Boundary on single chars (both sides have single-char words at the disputed position)
"""
import argparse
import os
import sys
import json
import torch
from collections import defaultdict
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import CWSDataset, Collator, bies_to_words  # noqa: E402
from model import BertCRF  # noqa: E402


def load_dict(path, min_freq=0):
    """Load IsCut-style dict: 'word\tfreq' or just 'word' per line."""
    words = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            w = parts[0]
            if len(parts) >= 2:
                try:
                    if int(parts[1]) < min_freq:
                        continue
                except ValueError:
                    pass
            words.add(w)
    return words


def words_to_spans(words):
    out, pos = [], 0
    for w in words:
        out.append((pos, pos + len(w), w))
        pos += len(w)
    return out


def classify_errors(pred_words, gold_words, dict_set):
    """Return (hard_pred_fp, hard_gold_fn, style_pred_fp, style_gold_fn)."""
    pred_spans = words_to_spans(pred_words)
    gold_spans = words_to_spans(gold_words)
    pred_set = {(s, e): w for s, e, w in pred_spans}
    gold_set = {(s, e): w for s, e, w in gold_spans}
    pred_keys = set(pred_set.keys())
    gold_keys = set(gold_set.keys())

    fp_keys = pred_keys - gold_keys
    fn_keys = gold_keys - pred_keys

    # Build char-position → which gold/pred span it belongs to
    def boundary_set(spans):
        # set of (s, e) tuples
        return {(s, e) for s, e, _ in spans}

    pred_chars_to_span = {}
    for s, e, w in pred_spans:
        for i in range(s, e):
            pred_chars_to_span[i] = (s, e, w)
    gold_chars_to_span = {}
    for s, e, w in gold_spans:
        for i in range(s, e):
            gold_chars_to_span[i] = (s, e, w)

    hard_pred = []
    style_pred = []
    for s, e in fp_keys:
        w = pred_set[(s, e)]
        if len(w) == 1:
            # single-char pred not matching gold = gold has a multi-char word there
            # → that's a style "finer" cut (gold = multi-char, pred = singles)
            style_pred.append(("S1_finer_pred", s, e, w))
            continue
        # pred span has ≥2 chars
        if w in dict_set:
            # pred created a valid Chinese word (just different grouping than gold)
            # → style
            style_pred.append(("S2_valid_merge", s, e, w))
        else:
            # pred created an invalid multi-char span — INVENTED
            hard_pred.append(("H1_invented", s, e, w))

    hard_gold = []
    style_gold = []
    for s, e in fn_keys:
        w = gold_set[(s, e)]
        if len(w) == 1:
            # gold single char missed — pred merged it with neighbors
            # → check if neighbor merge is valid
            # find pred span covering position s
            ps = pred_chars_to_span.get(s)
            if ps and ps[2] in dict_set:
                style_gold.append(("S2_valid_merge", s, e, w))
            elif ps and len(ps[2]) == 1:
                # both sides single char, shouldn't happen since gold span IS pred span then
                style_gold.append(("S3_uncategorized", s, e, w))
            else:
                # pred merged into non-dict word — counted on pred side as H1
                style_gold.append(("S_pred_merged_invalid", s, e, w))
            continue
        # gold span ≥2 chars
        if w in dict_set:
            # gold has a valid word; did pred split it into all single chars?
            chars_covered_by_singles = all(
                pred_chars_to_span.get(i) is not None
                and pred_chars_to_span[i][1] - pred_chars_to_span[i][0] == 1
                for i in range(s, e)
            )
            if chars_covered_by_singles:
                style_gold.append(("S1_finer_pred", s, e, w))
            else:
                # pred boundary crosses the word (merged part of it with neighbor)
                hard_gold.append(("H2_split_real_word", s, e, w))
        else:
            # gold span ≥2 chars but NOT in dict (rare/proper name/PD-specific)
            # can't reliably tell hard vs style — call it style (give benefit of doubt)
            style_gold.append(("S_unknown_word", s, e, w))

    return hard_pred, hard_gold, style_pred, style_gold, pred_keys, gold_keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="output_macbert_large_crf/final.pt")
    ap.add_argument("--model_path", default="./macbert-large")
    ap.add_argument("--name", default="MacBERT-large", help="label for this run")
    ap.add_argument("--dev_jsonl", default="./data/cws_dev.jsonl")
    ap.add_argument("--dict", default="/home/tfbao/Shiyu/IsCut/dict.txt")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--show_examples", type=int, default=15)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    print(f"=== {args.name} ===")

    print(f"Loading dict {args.dict} ...")
    dict_set = load_dict(args.dict)
    print(f"  {len(dict_set)} dict words")

    print(f"Loading dev {args.dev_jsonl} ...")
    dev = CWSDataset(args.dev_jsonl)
    if args.limit:
        dev.items = dev.items[:args.limit]
    print(f"  {len(dev)} dev samples\n")

    print(f"Loading model {args.ckpt} ...")
    if os.path.exists(os.path.join(args.model_path, "piece.model")):
        from piece_tokenizer_adapter import PieceTokenizerAdapter
        tokenizer = PieceTokenizerAdapter(args.model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    device = torch.device(args.device)
    model = BertCRF(args.model_path, num_tags=4).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    collator = Collator(tokenizer)

    # Predict + classify
    print("Predicting + classifying errors ...")
    total_tp, total_fp, total_fn = 0, 0, 0
    hard_fp_count, hard_fn_count = 0, 0
    style_fp_count, style_fn_count = 0, 0
    error_cat_counts = defaultdict(int)
    hard_examples = []
    style_examples = []
    from torch.utils.data import DataLoader
    loader = DataLoader(dev, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collator, num_workers=0)
    idx = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            if args.device == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    preds = model.decode(batch["input_ids"], batch["attention_mask"])
            else:
                preds = model.decode(batch["input_ids"], batch["attention_mask"])
            for pred_tags in preds:
                item = dev.items[idx]
                idx += 1
                n = min(len(item["chars"]), len(pred_tags))
                pred_words = bies_to_words(item["chars"][:n], pred_tags[:n])
                gold_words = bies_to_words(item["chars"][:n], item["tags"][:n])
                text = "".join(item["chars"][:n])
                hp, hg, sp, sg, pk, gk = classify_errors(pred_words, gold_words, dict_set)
                tp = len(pk & gk)
                total_tp += tp
                total_fp += len(pk - gk)
                total_fn += len(gk - pk)
                hard_fp_count += len(hp)
                hard_fn_count += len(hg)
                style_fp_count += len(sp)
                style_fn_count += len(sg)
                for cat, *_ in hp + hg:
                    error_cat_counts[cat] += 1
                for cat, *_ in sp + sg:
                    error_cat_counts[cat] += 1
                # Collect examples
                if (hp or hg) and len(hard_examples) < args.show_examples:
                    hard_examples.append({"text": text, "gold": gold_words, "pred": pred_words,
                                          "hard_pred": hp, "hard_gold": hg})
                if sp or sg:
                    if not (hp or hg) and len(style_examples) < args.show_examples:
                        style_examples.append({"text": text, "gold": gold_words, "pred": pred_words,
                                                "style_pred": sp, "style_gold": sg})

    print("\n" + "=" * 70)
    print(f"GLOBAL counts over {len(dev)} dev samples")
    print("=" * 70)
    print(f"  TP (correct spans)        : {total_tp:7d}")
    print(f"  FP (pred span not in gold): {total_fp:7d}")
    print(f"  FN (gold span not in pred): {total_fn:7d}")

    def f1_from(tp, fp, fn):
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        if p + r == 0: return 0, 0, 0
        return 2 * p * r / (p + r), p, r

    overall_f1, p, r = f1_from(total_tp, total_fp, total_fn)
    print(f"  Overall F1 = {overall_f1:.4f}  (P={p:.4f}  R={r:.4f})")
    print()
    print(f"  Hard errors  : FP={hard_fp_count}  FN={hard_fn_count}  (total {hard_fp_count + hard_fn_count})")
    print(f"  Style errors : FP={style_fp_count}  FN={style_fn_count}  (total {style_fp_count + style_fn_count})")
    print()

    # Strict F1: only hard errors count
    strict_f1, sp_, sr_ = f1_from(total_tp, hard_fp_count, hard_fn_count)
    print(f"  ⚙  Strict (hard-only) F1 = {strict_f1:.4f}  (P={sp_:.4f}  R={sr_:.4f})")
    print(f"     → upper-bound on 'truly correct' rate, treating style as match")
    print()

    print("Error category breakdown:")
    for cat, n in sorted(error_cat_counts.items(), key=lambda x: -x[1]):
        kind = "HARD" if cat.startswith("H") else "STYLE"
        print(f"  {kind:5s}  {cat:30s}: {n:6d}")
    print()

    print("=" * 70)
    print(f"HARD ERROR examples (top {len(hard_examples)})")
    print("=" * 70)
    for ex in hard_examples:
        print(f"\n  text  : {ex['text']}")
        print(f"  gold  : {' / '.join(ex['gold'])}")
        print(f"  pred  : {' / '.join(ex['pred'])}")
        for cat, s, e, w in ex["hard_pred"]:
            print(f"     ❌ HARD pred-side  [{cat}]  '{w}'  (pos {s}-{e})")
        for cat, s, e, w in ex["hard_gold"]:
            print(f"     ❌ HARD gold-side  [{cat}]  '{w}'  (pos {s}-{e})")

    print()
    print("=" * 70)
    print(f"STYLE-only error examples (top {min(8, len(style_examples))})")
    print("=" * 70)
    for ex in style_examples[:8]:
        print(f"\n  text  : {ex['text']}")
        print(f"  gold  : {' / '.join(ex['gold'])}")
        print(f"  pred  : {' / '.join(ex['pred'])}")
        for cat, s, e, w in ex["style_pred"]:
            print(f"     ⊙ STYLE pred  [{cat}]  '{w}'")
        for cat, s, e, w in ex["style_gold"]:
            print(f"     ⊙ STYLE gold  [{cat}]  '{w}'")


if __name__ == "__main__":
    main()
