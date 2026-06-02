"""4-way case comparison: MacBERT-large + RoBERTa-wwm-ext + Wapic + IsCut on PD-06."""
import argparse
import os
import sys
import random
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import CWSDataset, Collator, bies_to_words  # noqa: E402
from model import BertCRF  # noqa: E402
from compare_all import run_wapic, f1  # noqa: E402

import iscut


def bert_predict(ckpt, model_path, cases, batch=32):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    device = torch.device("cuda")
    m = BertCRF(model_path, num_tags=4).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device))
    m.eval()
    coll = Collator(tokenizer)
    preds = []
    with torch.no_grad():
        for s in range(0, len(cases), batch):
            chunk = cases[s:s+batch]
            b = coll(chunk)
            b = {k: v.to(device) for k, v in b.items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                ps = m.decode(b["input_ids"], b["attention_mask"])
            for c, ts in zip(chunk, ps):
                n = min(len(c["chars"]), len(ts))
                preds.append(bies_to_words(c["chars"][:n], ts[:n]))
    del m
    torch.cuda.empty_cache()
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mac_ckpt", default="output_macbert_large_crf/final.pt")
    ap.add_argument("--mac_path", default="./macbert-large")
    ap.add_argument("--rob_ckpt", default="output_roberta_wwm_ext_crf/best.pt")
    ap.add_argument("--rob_path", default="./roberta-wwm-ext")
    ap.add_argument("--dev_jsonl", default="./data/cws_dev.jsonl")
    ap.add_argument("--iscut_dict", default="/home/tfbao/Shiyu/IsCut/dict.txt")
    ap.add_argument("--wapic_bin", default="/home/tfbao/Shiyu/Wapic/build/wapic")
    ap.add_argument("--wapic_model", default="/home/tfbao/Shiyu/Wapic/data/cut.wac")
    ap.add_argument("--n_cases", type=int, default=500)
    ap.add_argument("--per_bucket", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    print(f"Loading dev: {args.dev_jsonl}")
    full_dev = CWSDataset(args.dev_jsonl)
    indices = rng.sample(range(len(full_dev.items)), args.n_cases)
    cases = [full_dev.items[i] for i in indices]
    texts = ["".join(c["chars"]) for c in cases]
    print(f"  {len(cases)} cases\n")

    print("Running MacBERT-large ...")
    mac_preds = bert_predict(args.mac_ckpt, args.mac_path, cases)
    print("Running RoBERTa-wwm-ext ...")
    rob_preds = bert_predict(args.rob_ckpt, args.rob_path, cases)
    print("Running Wapic ...")
    wapic_preds = run_wapic(texts, args.wapic_bin, args.wapic_model)
    while len(wapic_preds) < len(texts):
        wapic_preds.append([])
    wapic_preds = wapic_preds[:len(texts)]
    print("Running IsCut ...")
    cutter = iscut.Cutter(args.iscut_dict)
    iscut_preds = [cutter.cut(t) for t in texts]
    print()

    rows = []
    for k, case in enumerate(cases):
        gold = bies_to_words(case["chars"], case["tags"])
        rows.append({
            "idx": k, "text": texts[k], "gold": gold, "len": len(case["chars"]),
            "mac": mac_preds[k], "rob": rob_preds[k], "wap": wapic_preds[k], "isc": iscut_preds[k],
            "m_f1": f1(mac_preds[k], gold), "r_f1": f1(rob_preds[k], gold),
            "w_f1": f1(wapic_preds[k], gold), "i_f1": f1(iscut_preds[k], gold),
        })

    n = len(rows)
    print("=" * 95)
    print(f"AGGREGATE over {n} random PD-06 cases")
    print("=" * 95)
    print(f"  MacBERT-large + CRF (330M) : avg F1 = {sum(r['m_f1'] for r in rows)/n:.4f}")
    print(f"  RoBERTa-wwm-ext + CRF (110M): avg F1 = {sum(r['r_f1'] for r in rows)/n:.4f}")
    print(f"  Wapic-CRF (PD-trained)      : avg F1 = {sum(r['w_f1'] for r in rows)/n:.4f}")
    print(f"  IsCut-Unigram (wiki dict)   : avg F1 = {sum(r['i_f1'] for r in rows)/n:.4f}")

    # mac perfect, rob perfect, both perfect, neither
    mp = sum(1 for r in rows if r["m_f1"] == 1.0)
    rp = sum(1 for r in rows if r["r_f1"] == 1.0)
    both = sum(1 for r in rows if r["m_f1"] == 1.0 and r["r_f1"] == 1.0)
    print()
    print(f"  MacBERT 全对: {mp}  ({100*mp/n:.1f}%)")
    print(f"  RoBERTa 全对: {rp}  ({100*rp/n:.1f}%)")
    print(f"  俩都全对    : {both}  ({100*both/n:.1f}%)")

    # mac wins vs rob
    m_over_r = sum(1 for r in rows if r["m_f1"] > r["r_f1"])
    r_over_m = sum(1 for r in rows if r["r_f1"] > r["m_f1"])
    tied = n - m_over_r - r_over_m
    print()
    print(f"  Mac > Rob: {m_over_r}  |  Rob > Mac: {r_over_m}  |  tied: {tied}")

    # 4-way agreement check
    all4 = sum(1 for r in rows if r["m_f1"] == r["r_f1"] == r["w_f1"] == r["i_f1"] == 1.0)
    print(f"  All 4 perfect: {all4}")

    def print_row(r):
        print(f"  #{r['idx']}  ({r['len']}字)")
        print(f"     text  : {r['text']}")
        print(f"     gold  : {' / '.join(r['gold'])}")
        print(f"     mac   : {' / '.join(r['mac'])}    F1={r['m_f1']:.3f}")
        print(f"     rob   : {' / '.join(r['rob'])}    F1={r['r_f1']:.3f}")
        print(f"     wapic : {' / '.join(r['wap'])}    F1={r['w_f1']:.3f}")
        print(f"     iscut : {' / '.join(r['isc'])}    F1={r['i_f1']:.3f}")
        print()

    def header(title):
        print()
        print("=" * 95)
        print(f"  {title}")
        print("=" * 95)

    # A: Mac > Rob (gap >0.1)
    mac_wins = [r for r in rows if r["m_f1"] - r["r_f1"] > 0.10]
    mac_wins.sort(key=lambda r: -(r["m_f1"] - r["r_f1"]))
    header(f"A. MacBERT > RoBERTa (gap >0.10) — {len(mac_wins)} cases, top {args.per_bucket}")
    for r in mac_wins[:args.per_bucket]:
        print_row(r)

    # B: Rob > Mac (gap >0.05)
    rob_wins = [r for r in rows if r["r_f1"] - r["m_f1"] > 0.05]
    rob_wins.sort(key=lambda r: -(r["r_f1"] - r["m_f1"]))
    header(f"B. RoBERTa > MacBERT (gap >0.05) — {len(rob_wins)} cases, top {args.per_bucket}")
    for r in rob_wins[:args.per_bucket]:
        print_row(r)

    # C: both BERT-CRF win over both classical (mac=rob=1, wapic+iscut both<1)
    both_bert_strong = [r for r in rows
                        if r["m_f1"] == 1.0 and r["r_f1"] == 1.0
                        and r["w_f1"] < 1.0 and r["i_f1"] < 1.0]
    both_bert_strong.sort(key=lambda r: -(min(r["w_f1"], r["i_f1"])))
    header(f"C. 两 BERT-CRF 都全对,Wapic+IsCut 都没对 — {len(both_bert_strong)} cases, top {args.per_bucket}")
    for r in both_bert_strong[:args.per_bucket]:
        print_row(r)

    # D: cases where ALL fail (BERT can't either)
    all_fail = [r for r in rows if max(r["m_f1"], r["r_f1"], r["w_f1"], r["i_f1"]) < 0.85]
    all_fail.sort(key=lambda r: max(r["m_f1"], r["r_f1"]))
    header(f"D. 全员翻车 (max F1 < 0.85) — {len(all_fail)} cases, top {args.per_bucket}")
    for r in all_fail[:args.per_bucket]:
        print_row(r)


if __name__ == "__main__":
    main()
