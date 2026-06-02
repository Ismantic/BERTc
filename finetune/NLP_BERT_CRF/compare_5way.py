"""5-way: MacBERT-large + RoBERTa-wwm-ext + char-mid(self-trained) + Wapic + IsCut."""
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
    if os.path.exists(os.path.join(model_path, "piece.model")):
        from piece_tokenizer_adapter import PieceTokenizerAdapter
        tokenizer = PieceTokenizerAdapter(model_path)
    else:
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
    ap.add_argument("--cm_ckpt", default="output_char_mid_crf/best.pt")
    ap.add_argument("--cm_path", default="/home/tfbao/Shiyu/Summer/BERT/bert_train_mid")
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
    print("Running char-mid (self-trained) ...")
    cm_preds = bert_predict(args.cm_ckpt, args.cm_path, cases)
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
            "mac": mac_preds[k], "rob": rob_preds[k], "cm": cm_preds[k],
            "wap": wapic_preds[k], "isc": iscut_preds[k],
            "m_f1": f1(mac_preds[k], gold), "r_f1": f1(rob_preds[k], gold),
            "c_f1": f1(cm_preds[k], gold),
            "w_f1": f1(wapic_preds[k], gold), "i_f1": f1(iscut_preds[k], gold),
        })

    n = len(rows)
    print("=" * 95)
    print(f"AGGREGATE over {n} random PD-06 cases")
    print("=" * 95)
    avg = lambda k: sum(r[k] for r in rows) / n
    print(f"  MacBERT-large (330M, HFL ~262B tok)    : avg F1 = {avg('m_f1'):.4f}")
    print(f"  RoBERTa-wwm-ext (110M, HFL ~262B tok)  : avg F1 = {avg('r_f1'):.4f}")
    print(f"  char-mid (169M, self-trained 5B tok)   : avg F1 = {avg('c_f1'):.4f}")
    print(f"  Wapic-CRF (hand-feature, PD-trained)   : avg F1 = {avg('w_f1'):.4f}")
    print(f"  IsCut-Unigram (35w wiki dict)          : avg F1 = {avg('i_f1'):.4f}")

    perfects = {k: sum(1 for r in rows if r[k] == 1.0) for k in ["m_f1", "r_f1", "c_f1", "w_f1", "i_f1"]}
    print()
    print(f"  Mac    全对: {perfects['m_f1']}  ({100*perfects['m_f1']/n:.1f}%)")
    print(f"  Rob    全对: {perfects['r_f1']}  ({100*perfects['r_f1']/n:.1f}%)")
    print(f"  char-mid 全对: {perfects['c_f1']}  ({100*perfects['c_f1']/n:.1f}%)")

    # Compare char-mid vs the two HFL models
    cm_over_rob = sum(1 for r in rows if r["c_f1"] > r["r_f1"])
    rob_over_cm = sum(1 for r in rows if r["r_f1"] > r["c_f1"])
    cm_over_mac = sum(1 for r in rows if r["c_f1"] > r["m_f1"])
    mac_over_cm = sum(1 for r in rows if r["m_f1"] > r["c_f1"])
    print()
    print(f"  char-mid vs RoBERTa : {cm_over_rob} 胜 / {rob_over_cm} 负 / tied {n - cm_over_rob - rob_over_cm}")
    print(f"  char-mid vs MacBERT : {cm_over_mac} 胜 / {mac_over_cm} 负 / tied {n - cm_over_mac - mac_over_cm}")

    def print_row(r):
        print(f"  #{r['idx']}  ({r['len']}字)")
        print(f"     text  : {r['text']}")
        print(f"     gold  : {' / '.join(r['gold'])}")
        print(f"     mac   : {' / '.join(r['mac'])}    F1={r['m_f1']:.3f}")
        print(f"     rob   : {' / '.join(r['rob'])}    F1={r['r_f1']:.3f}")
        print(f"     cm    : {' / '.join(r['cm'])}    F1={r['c_f1']:.3f}")
        print(f"     wapic : {' / '.join(r['wap'])}    F1={r['w_f1']:.3f}")
        print(f"     iscut : {' / '.join(r['isc'])}    F1={r['i_f1']:.3f}")
        print()

    def header(title):
        print()
        print("=" * 95)
        print(f"  {title}")
        print("=" * 95)

    # A: HFL models 全对,char-mid 没对 → 我们自研 backbone 的盲区
    only_hfl = [r for r in rows if r["m_f1"] == 1.0 and r["r_f1"] == 1.0 and r["c_f1"] < 1.0]
    only_hfl.sort(key=lambda r: r["c_f1"])
    header(f"A. Mac + Rob 都全对,但 char-mid 错了 ({len(only_hfl)} 例,top {args.per_bucket}) — 自研 backbone 短板")
    for r in only_hfl[:args.per_bucket]:
        print_row(r)

    # B: char-mid 全对,有 HFL 一个错 → 自研模型偶有亮点
    cm_only = [r for r in rows if r["c_f1"] == 1.0 and (r["m_f1"] < 1.0 or r["r_f1"] < 1.0)]
    cm_only.sort(key=lambda r: min(r["m_f1"], r["r_f1"]))
    header(f"B. char-mid 全对,Mac 或 Rob 错 ({len(cm_only)} 例,top {args.per_bucket}) — 自研模型不输大模型场景")
    for r in cm_only[:args.per_bucket]:
        print_row(r)

    # C: 三个 BERT 都对,经典方法没对 → BERT 系优势区
    all_bert = [r for r in rows
                if r["m_f1"] == 1.0 and r["r_f1"] == 1.0 and r["c_f1"] == 1.0
                and r["w_f1"] < 1.0 and r["i_f1"] < 1.0]
    header(f"C. 三个 BERT 都全对,Wapic+IsCut 错 ({len(all_bert)} 例,top {args.per_bucket}) — 神经独霸")
    for r in all_bert[:args.per_bucket]:
        print_row(r)

    # D: 全员翻车
    all_fail = [r for r in rows if max(r["m_f1"], r["r_f1"], r["c_f1"], r["w_f1"], r["i_f1"]) < 0.85]
    all_fail.sort(key=lambda r: max(r["m_f1"], r["r_f1"], r["c_f1"]))
    header(f"D. 全员翻车 (max F1 < 0.85) — {len(all_fail)} cases, top {args.per_bucket}")
    for r in all_fail[:args.per_bucket]:
        print_row(r)


if __name__ == "__main__":
    main()
