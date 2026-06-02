"""Dump diverse case examples from 3-way comparison, organized by category."""
import argparse
import os
import sys
import random
import subprocess
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import CWSDataset, Collator, bies_to_words  # noqa: E402
from model import BertCRF  # noqa: E402
from compare_all import run_wapic, spans, f1  # noqa: E402

import iscut


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="output_macbert_large_crf/final.pt")
    ap.add_argument("--model_path", default="./macbert-large")
    ap.add_argument("--dev_jsonl", default="./data/cws_dev.jsonl")
    ap.add_argument("--iscut_dict", default="/home/tfbao/Shiyu/IsCut/dict.txt")
    ap.add_argument("--wapic_bin", default="/home/tfbao/Shiyu/Wapic/build/wapic")
    ap.add_argument("--wapic_model", default="/home/tfbao/Shiyu/Wapic/data/cut.wac")
    ap.add_argument("--n_cases", type=int, default=500)
    ap.add_argument("--per_bucket", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    print("[1] Loading IsCut ...")
    cutter = iscut.Cutter(args.iscut_dict)
    print("[2] Loading MacBERT-CRF ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    device = torch.device("cuda")
    model = BertCRF(args.model_path, num_tags=4).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    collator = Collator(tokenizer)

    print(f"[3] Loading dev ...")
    full_dev = CWSDataset(args.dev_jsonl)
    indices = rng.sample(range(len(full_dev.items)), args.n_cases)
    cases = [full_dev.items[i] for i in indices]
    texts = ["".join(c["chars"]) for c in cases]
    print(f"   {len(cases)} cases\n")

    # MacBERT preds
    print("Running MacBERT-CRF ...")
    mac_preds = []
    with torch.no_grad():
        for s in range(0, len(cases), 32):
            chunk = cases[s:s+32]
            batch = collator(chunk)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                preds = model.decode(batch["input_ids"], batch["attention_mask"])
            for c, ts in zip(chunk, preds):
                n = min(len(c["chars"]), len(ts))
                mac_preds.append(bies_to_words(c["chars"][:n], ts[:n]))

    print("Running Wapic ...")
    wapic_preds = run_wapic(texts, args.wapic_bin, args.wapic_model)
    while len(wapic_preds) < len(texts):
        wapic_preds.append([])
    wapic_preds = wapic_preds[:len(texts)]

    print("Running IsCut ...\n")
    iscut_preds = [cutter.cut(t) for t in texts]

    # Compute
    rows = []
    for k, case in enumerate(cases):
        gold = bies_to_words(case["chars"], case["tags"])
        m = f1(mac_preds[k], gold)
        w = f1(wapic_preds[k], gold)
        i = f1(iscut_preds[k], gold)
        rows.append({
            "idx": k, "text": texts[k], "gold": gold,
            "mac": mac_preds[k], "wapic": wapic_preds[k], "iscut": iscut_preds[k],
            "m_f1": m, "w_f1": w, "i_f1": i,
            "len": len(case["chars"]),
        })

    def print_row(r, prefix=""):
        print(f"{prefix}#{r['idx']}  ({r['len']}字)")
        print(f"   text  : {r['text']}")
        print(f"   gold  : {' / '.join(r['gold'])}")
        print(f"   mac   : {' / '.join(r['mac'])}    F1={r['m_f1']:.3f}")
        print(f"   wapic : {' / '.join(r['wapic'])}    F1={r['w_f1']:.3f}")
        print(f"   iscut : {' / '.join(r['iscut'])}    F1={r['i_f1']:.3f}")
        print()

    def header(title):
        print()
        print("=" * 90)
        print(f"  {title}")
        print("=" * 90)
        print()

    # Bucket A: all 3 perfect
    perfect_all = [r for r in rows if r["m_f1"] == 1.0 and r["w_f1"] == 1.0 and r["i_f1"] == 1.0]
    header(f"A. 三方全对 ({len(perfect_all)} 例,展示前 {args.per_bucket})")
    for r in perfect_all[:args.per_bucket]:
        print_row(r)

    # Bucket B: only MacBERT perfect
    only_mac = [r for r in rows if r["m_f1"] == 1.0 and r["w_f1"] < 1.0 and r["i_f1"] < 1.0]
    header(f"B. 只有 MacBERT 全对 ({len(only_mac)} 例,展示前 {args.per_bucket})")
    only_mac.sort(key=lambda r: -r["len"])  # longest first
    for r in only_mac[:args.per_bucket]:
        print_row(r)

    # Bucket C: only Wapic perfect
    only_wap = [r for r in rows if r["w_f1"] == 1.0 and r["m_f1"] < 1.0 and r["i_f1"] < 1.0]
    header(f"C. 只有 Wapic 全对 ({len(only_wap)} 例,展示前 {args.per_bucket})")
    only_wap.sort(key=lambda r: r["m_f1"])
    for r in only_wap[:args.per_bucket]:
        print_row(r)

    # Bucket D: only IsCut perfect
    only_isc = [r for r in rows if r["i_f1"] == 1.0 and r["m_f1"] < 1.0 and r["w_f1"] < 1.0]
    header(f"D. 只有 IsCut 全对 ({len(only_isc)} 例,展示前 {args.per_bucket})")
    only_isc.sort(key=lambda r: r["m_f1"])
    for r in only_isc[:args.per_bucket]:
        print_row(r)

    # Bucket E: MacBERT 最差(error analysis)
    sorted_by_mac = sorted(rows, key=lambda r: r["m_f1"])
    header(f"E. MacBERT 最差的 {args.per_bucket} 例(error analysis)")
    for r in sorted_by_mac[:args.per_bucket]:
        print_row(r)

    # Bucket F: 长句对比(>=50 chars)
    long_rows = [r for r in rows if r["len"] >= 50]
    long_rows.sort(key=lambda r: -r["len"])
    header(f"F. 长句(≥50字)对比 ({len(long_rows)} 例,展示前 {args.per_bucket})")
    for r in long_rows[:args.per_bucket]:
        print_row(r)

    # Bucket G: 短句(<8 字)难句
    short_rows = [r for r in rows if r["len"] < 8 and r["m_f1"] < 1.0]
    header(f"G. 短句但 MacBERT 错的 ({len(short_rows)} 例,展示前 {args.per_bucket})")
    short_rows.sort(key=lambda r: r["m_f1"])
    for r in short_rows[:args.per_bucket]:
        print_row(r)

    # Final summary
    n = len(rows)
    print()
    print("=" * 90)
    print(f"  汇总({n} 例)")
    print("=" * 90)
    print(f"  MacBERT avg F1: {sum(r['m_f1'] for r in rows)/n:.4f}")
    print(f"  Wapic   avg F1: {sum(r['w_f1'] for r in rows)/n:.4f}")
    print(f"  IsCut   avg F1: {sum(r['i_f1'] for r in rows)/n:.4f}")
    print()
    print(f"  桶 A 三方全对          : {len(perfect_all):4d} ({100*len(perfect_all)/n:.1f}%)")
    print(f"  桶 B 只 MacBERT 全对   : {len(only_mac):4d} ({100*len(only_mac)/n:.1f}%)")
    print(f"  桶 C 只 Wapic 全对     : {len(only_wap):4d} ({100*len(only_wap)/n:.1f}%)")
    print(f"  桶 D 只 IsCut 全对     : {len(only_isc):4d} ({100*len(only_isc)/n:.1f}%)")
    mac_perfect = sum(1 for r in rows if r["m_f1"] == 1.0)
    print(f"  MacBERT 全对(任意)    : {mac_perfect:4d} ({100*mac_perfect/n:.1f}%)")


if __name__ == "__main__":
    main()
