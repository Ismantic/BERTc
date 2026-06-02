"""Case-by-case 3-way comparison: MacBERT-CRF vs Wapic-CRF vs IsCut-Unigram on PD-06."""
import argparse
import json
import os
import sys
import subprocess
import tempfile
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


def run_wapic(texts, wapic_bin, model):
    """Pipe lines to wapic. In non-TTY mode wapic writes cut to stdout (raw, no prefix),
    and the '>>> ' prompt to stderr — so parse stdout lines directly."""
    inp = "\n".join(texts) + "\n"
    proc = subprocess.run(
        [wapic_bin, "-m", model],
        input=inp.encode("utf-8"),
        capture_output=True,
        timeout=300,
    )
    out_lines = proc.stdout.decode("utf-8", errors="replace").splitlines()
    preds = [line.strip().split() for line in out_lines]
    # Pad or trim to match texts
    while len(preds) < len(texts):
        preds.append([])
    preds = preds[:len(texts)]
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="output_macbert_large_crf/best.pt")
    ap.add_argument("--model_path", default="./macbert-large")
    ap.add_argument("--dev_jsonl", default="./data/cws_dev.jsonl")
    ap.add_argument("--iscut_dict", default="/home/tfbao/Shiyu/IsCut/dict.txt")
    ap.add_argument("--wapic_bin", default="/home/tfbao/Shiyu/Wapic/build/wapic")
    ap.add_argument("--wapic_model", default="/home/tfbao/Shiyu/Wapic/data/cut.wac")
    ap.add_argument("--n_cases", type=int, default=200)
    ap.add_argument("--show_n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    # IsCut
    print(f"[1/3] Loading IsCut dict: {args.iscut_dict}")
    cutter = iscut.Cutter(args.iscut_dict)

    # MacBERT-CRF
    print(f"[2/3] Loading MacBERT-CRF: {args.ckpt}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    device = torch.device("cuda")
    model = BertCRF(args.model_path, num_tags=4).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    collator = Collator(tokenizer)

    print(f"[3/3] Wapic binary: {args.wapic_bin}")

    # Dev
    print(f"\nLoading dev: {args.dev_jsonl}")
    full_dev = CWSDataset(args.dev_jsonl)
    import random
    rng = random.Random(args.seed)
    indices = rng.sample(range(len(full_dev.items)), args.n_cases)
    cases = [full_dev.items[i] for i in indices]
    texts = ["".join(c["chars"]) for c in cases]
    print(f"  {len(cases)} sampled cases\n")

    # MacBERT-CRF preds (batched)
    print("Running MacBERT-CRF ...")
    macbert_preds = []
    with torch.no_grad():
        for start in range(0, len(cases), 32):
            chunk = cases[start:start + 32]
            batch = collator(chunk)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                preds = model.decode(batch["input_ids"], batch["attention_mask"])
            for case, tag_seq in zip(chunk, preds):
                chars = case["chars"]
                n = min(len(chars), len(tag_seq))
                macbert_preds.append(bies_to_words(chars[:n], tag_seq[:n]))

    # Wapic preds (subprocess)
    print("Running Wapic ...")
    wapic_preds = run_wapic(texts, args.wapic_bin, args.wapic_model)
    if len(wapic_preds) != len(texts):
        print(f"  WARN: wapic returned {len(wapic_preds)} lines, expected {len(texts)}")
        wapic_preds = (wapic_preds + [[] for _ in range(len(texts))])[:len(texts)]

    # IsCut preds
    print("Running IsCut ...")
    iscut_preds = [cutter.cut(t) for t in texts]

    # Per-case F1s
    m_f1s, w_f1s, i_f1s = [], [], []
    rows = []
    for k, case in enumerate(cases):
        gold = bies_to_words(case["chars"], case["tags"])
        # Wapic / IsCut may not reproduce same char sequence (English splitting etc) — F1 is span-based so we just compare directly
        m = f1(macbert_preds[k], gold)
        w = f1(wapic_preds[k], gold)
        i = f1(iscut_preds[k], gold)
        m_f1s.append(m); w_f1s.append(w); i_f1s.append(i)
        rows.append((k, gold, macbert_preds[k], wapic_preds[k], iscut_preds[k], m, w, i))

    n = len(cases)
    print()
    print("=" * 88)
    print(f"Aggregate over {n} random PD-06 cases")
    print("=" * 88)
    print(f"  MacBERT-CRF (330M, our train)  : avg F1 = {sum(m_f1s)/n:.4f}")
    print(f"  Wapic-CRF   (hand features, PD): avg F1 = {sum(w_f1s)/n:.4f}")
    print(f"  IsCut-Unigram (35w wiki dict)  : avg F1 = {sum(i_f1s)/n:.4f}")
    print()
    # Pairwise wins (strict)
    def cmp(a, b):
        return sum(1 for x, y in zip(a, b) if x > y)
    print(f"  Wins (MacBERT vs Wapic)  : {cmp(m_f1s, w_f1s):3d} / {cmp(w_f1s, m_f1s):3d}  (tie {n - cmp(m_f1s, w_f1s) - cmp(w_f1s, m_f1s)})")
    print(f"  Wins (MacBERT vs IsCut)  : {cmp(m_f1s, i_f1s):3d} / {cmp(i_f1s, m_f1s):3d}  (tie {n - cmp(m_f1s, i_f1s) - cmp(i_f1s, m_f1s)})")
    print(f"  Wins (Wapic   vs IsCut)  : {cmp(w_f1s, i_f1s):3d} / {cmp(i_f1s, w_f1s):3d}  (tie {n - cmp(w_f1s, i_f1s) - cmp(i_f1s, w_f1s)})")
    print(f"  All 3 perfect (1.0)       : {sum(1 for m,w,i in zip(m_f1s,w_f1s,i_f1s) if m==1 and w==1 and i==1):3d}")
    print(f"  Only MacBERT perfect      : {sum(1 for m,w,i in zip(m_f1s,w_f1s,i_f1s) if m==1 and w<1 and i<1):3d}")
    print(f"  Only Wapic   perfect      : {sum(1 for m,w,i in zip(m_f1s,w_f1s,i_f1s) if w==1 and m<1 and i<1):3d}")
    print(f"  Only IsCut   perfect      : {sum(1 for m,w,i in zip(m_f1s,w_f1s,i_f1s) if i==1 and m<1 and w<1):3d}")

    # Show worst MacBERT cases (model error analysis)
    rows_by_macbert = sorted(rows, key=lambda r: r[5])
    print()
    print("=" * 88)
    print(f"BOTTOM {args.show_n} cases where MacBERT struggles (sorted by MacBERT F1 ascending)")
    print("=" * 88)
    for k, gold, m_p, w_p, i_p, m, w, i in rows_by_macbert[:args.show_n]:
        print(f"\n#{k}")
        print(f"   text  : {''.join(gold)}")
        print(f"   gold  : {' / '.join(gold)}")
        print(f"   mac   : {' / '.join(m_p)}   F1={m:.3f}")
        print(f"   wapic : {' / '.join(w_p)}   F1={w:.3f}")
        print(f"   iscut : {' / '.join(i_p)}   F1={i:.3f}")

    # Show cases where IsCut wins big (style differences PD vs wiki)
    print()
    print("=" * 88)
    print(f"TOP cases where IsCut beats MacBERT (PD vs wiki style)")
    print("=" * 88)
    rows_by_gap = sorted(rows, key=lambda r: r[7] - r[5], reverse=True)
    shown = 0
    for k, gold, m_p, w_p, i_p, m, w, i in rows_by_gap:
        if i - m < 0.05:
            break
        print(f"\n#{k}  IsCut +{i-m:.3f} over MacBERT")
        print(f"   text  : {''.join(gold)}")
        print(f"   gold  : {' / '.join(gold)}")
        print(f"   mac   : {' / '.join(m_p)}   F1={m:.3f}")
        print(f"   wapic : {' / '.join(w_p)}   F1={w:.3f}")
        print(f"   iscut : {' / '.join(i_p)}   F1={i:.3f}")
        shown += 1
        if shown >= args.show_n // 2:
            break


if __name__ == "__main__":
    main()
