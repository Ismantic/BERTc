"""Case-by-case comparison: MacBERT-CRF vs IsCut on PD-06 dev samples.

Per case:
  - source text
  - gold (PD-1998 北大 style)
  - MacBERT-CRF prediction + F1
  - IsCut prediction + F1
  - mark disagreements
"""
import argparse
import json
import os
import sys
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import CWSDataset, Collator, bies_to_words  # noqa: E402
from model import BertCRF  # noqa: E402

import iscut


def spans(words):
    out, pos = set(), 0
    for w in words:
        out.add((pos, pos + len(w)))
        pos += len(w)
    return out


def f1(pred, gold):
    P, G = spans(pred), spans(gold)
    if not P or not G:
        return 0.0
    tp = len(P & G)
    if tp == 0:
        return 0.0
    p, r = tp / len(P), tp / len(G)
    return 2 * p * r / (p + r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="output_macbert_large_crf/best.pt")
    ap.add_argument("--model_path", default="./macbert-large")
    ap.add_argument("--dev_jsonl", default="./data/cws_dev.jsonl")
    ap.add_argument("--iscut_dict", default="/home/tfbao/Shiyu/IsCut/dict.txt")
    ap.add_argument("--n_cases", type=int, default=100)
    ap.add_argument("--show_n", type=int, default=20, help="how many full cases to print")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    # IsCut
    print(f"Loading IsCut dict: {args.iscut_dict}")
    cutter = iscut.Cutter(args.iscut_dict)

    # MacBERT-CRF
    print(f"Loading MacBERT-CRF: {args.ckpt}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    device = torch.device("cuda")
    model = BertCRF(args.model_path, num_tags=4).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    collator = Collator(tokenizer)

    # Dev
    print(f"Loading dev: {args.dev_jsonl}")
    full_dev = CWSDataset(args.dev_jsonl)
    # Random sample by seed
    import random
    rng = random.Random(args.seed)
    indices = rng.sample(range(len(full_dev.items)), args.n_cases)
    cases = [full_dev.items[i] for i in indices]
    print(f"  {len(cases)} sampled cases\n")

    # MacBERT-CRF predictions in one batch (per case)
    macbert_preds = []
    with torch.no_grad():
        # batch them for speed
        batch_size = 32
        for start in range(0, len(cases), batch_size):
            chunk = cases[start:start + batch_size]
            batch = collator(chunk)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                preds = model.decode(batch["input_ids"], batch["attention_mask"])
            for case, tag_seq in zip(chunk, preds):
                chars = case["chars"]
                n = min(len(chars), len(tag_seq))
                pred_words = bies_to_words(chars[:n], tag_seq[:n])
                macbert_preds.append(pred_words)

    # IsCut predictions
    iscut_preds = []
    for case in cases:
        text = "".join(case["chars"])
        iscut_preds.append(cutter.cut(text))

    # Per-case F1
    macbert_f1s = []
    iscut_f1s = []
    rows = []
    for i, case in enumerate(cases):
        gold = bies_to_words(case["chars"], case["tags"])
        m_f1 = f1(macbert_preds[i], gold)
        i_f1 = f1(iscut_preds[i], gold)
        macbert_f1s.append(m_f1)
        iscut_f1s.append(i_f1)
        rows.append((i, gold, macbert_preds[i], iscut_preds[i], m_f1, i_f1))

    # Sort by diff: biggest gap (favoring each side)
    rows.sort(key=lambda r: r[4] - r[5])  # ascending: IsCut wins first, MacBERT wins last

    # Aggregate
    n = len(cases)
    print("=" * 80)
    print(f"Aggregate over {n} cases")
    print("=" * 80)
    print(f"  MacBERT-CRF avg F1 : {sum(macbert_f1s)/n:.4f}")
    print(f"  IsCut       avg F1 : {sum(iscut_f1s)/n:.4f}")
    print(f"  MacBERT >  IsCut : {sum(1 for m,i in zip(macbert_f1s,iscut_f1s) if m > i):4d} cases")
    print(f"  IsCut   >  MacBERT: {sum(1 for m,i in zip(macbert_f1s,iscut_f1s) if i > m):4d} cases")
    print(f"  TIE              : {sum(1 for m,i in zip(macbert_f1s,iscut_f1s) if m == i):4d} cases")
    print(f"  Both perfect (1.0): {sum(1 for m,i in zip(macbert_f1s,iscut_f1s) if m==1 and i==1):4d} cases")
    print(f"  Both 0 F1        : {sum(1 for m,i in zip(macbert_f1s,iscut_f1s) if m==0 and i==0):4d} cases")
    print()

    # Show extreme cases
    half = args.show_n // 2
    print("=" * 80)
    print(f"TOP {half} cases where IsCut > MacBERT (sorted by gap)")
    print("=" * 80)
    for i, gold, m_pred, i_pred, m_f1, i_f1 in rows[:half]:
        if i_f1 - m_f1 < 0.01:
            break
        print(f"\n#{i}  gap=+{i_f1-m_f1:.3f}  (IsCut better)")
        print(f"   text : {''.join(gold)}")
        print(f"   gold : {' / '.join(gold)}")
        print(f"   mac  : {' / '.join(m_pred)}    F1={m_f1:.3f}")
        print(f"   iscut: {' / '.join(i_pred)}    F1={i_f1:.3f}")

    print()
    print("=" * 80)
    print(f"TOP {half} cases where MacBERT > IsCut (sorted by gap)")
    print("=" * 80)
    for i, gold, m_pred, i_pred, m_f1, i_f1 in rows[-half:][::-1]:
        if m_f1 - i_f1 < 0.01:
            break
        print(f"\n#{i}  gap=+{m_f1-i_f1:.3f}  (MacBERT better)")
        print(f"   text : {''.join(gold)}")
        print(f"   gold : {' / '.join(gold)}")
        print(f"   mac  : {' / '.join(m_pred)}    F1={m_f1:.3f}")
        print(f"   iscut: {' / '.join(i_pred)}    F1={i_f1:.3f}")


if __name__ == "__main__":
    main()
