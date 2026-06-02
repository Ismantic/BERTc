"""Dump LTP vs PD-06 gold mismatch cases,分类:
  - 合并(LTP 把 PD-06 多词合一)
  - 拆分(LTP 把 PD-06 一词拆多个)
  - 边界混乱(span 重叠错)

让人肉眼看是真错还是分词标准差异。

用法:
  python dump_ltp_cases.py --n_cases 200
"""
import argparse, json, random
from ltp import LTP


def spans(words):
    """word list → set of (start, end) byte spans"""
    out, pos = set(), 0
    for w in words:
        out.add((pos, pos + len(w)))
        pos += len(w)
    return out


def classify_mismatch(gold, pred):
    """Classify each mismatched span pair as merge / split / boundary."""
    G = spans(gold)
    P = spans(pred)
    only_g = G - P
    only_p = P - G

    # Build chars
    sent = "".join(gold)
    # Find merge: 一个 pred span 覆盖多个 gold spans(连续)
    merges, splits, others = [], [], []
    # 简单实现:对每个 only_p span,看它是否完全覆盖多个连续 gold spans
    only_p_sorted = sorted(only_p)
    only_g_sorted = sorted(only_g)
    matched_g = set()
    for ps, pe in only_p_sorted:
        contained = [(gs, ge) for gs, ge in only_g_sorted if ps <= gs and ge <= pe]
        if len(contained) >= 2 and contained[0][0] == ps and contained[-1][1] == pe:
            ptext = sent[ps:pe]
            gtexts = [sent[gs:ge] for gs, ge in contained]
            merges.append((ptext, gtexts))
            for c in contained: matched_g.add(c)
    # Split: 一个 gold span 被多个 pred spans 覆盖
    matched_p = set()
    for gs, ge in only_g_sorted:
        if (gs, ge) in matched_g: continue
        contained = [(ps, pe) for ps, pe in only_p_sorted if gs <= ps and pe <= ge]
        if len(contained) >= 2 and contained[0][0] == gs and contained[-1][1] == ge:
            gtext = sent[gs:ge]
            ptexts = [sent[ps:pe] for ps, pe in contained]
            splits.append((gtext, ptexts))
            for c in contained: matched_p.add(c)
    # 其它 = 真边界错
    for ps, pe in only_p_sorted:
        if (ps, pe) in matched_p: continue
        # 找最近的 gold span
        others.append(("P", sent[ps:pe], ps, pe))
    for gs, ge in only_g_sorted:
        if (gs, ge) in matched_g: continue
        others.append(("G", sent[gs:ge], gs, ge))
    return merges, splits, others


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="LTP/small")
    ap.add_argument("--dev_jsonl", default="./data/cws_dev.jsonl")
    ap.add_argument("--n_cases", type=int, default=200, help="sample 多少 dev case")
    ap.add_argument("--max_show", type=int, default=15, help="每类显示多少")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"Loading LTP {args.model}...")
    ltp = LTP(args.model)

    items = []
    with open(args.dev_jsonl, encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            obj = json.loads(line)
            words = obj.get("gold") or obj.get("cut", "").split()
            if words: items.append(words)
    print(f"Dev total: {len(items):,}, sample {args.n_cases}")

    rng = random.Random(args.seed)
    idx = rng.sample(range(len(items)), min(args.n_cases, len(items)))
    cases = [items[i] for i in idx]
    sents = ["".join(w) for w in cases]
    preds = ltp.pipeline(sents, tasks=["cws"]).cws

    all_merges, all_splits, all_others = [], [], []
    n_diff = 0
    for sent, gold, pred in zip(sents, cases, preds):
        if list(gold) == list(pred):
            continue
        n_diff += 1
        m, s, o = classify_mismatch(gold, pred)
        for x in m: all_merges.append((sent, gold, pred, x))
        for x in s: all_splits.append((sent, gold, pred, x))
        for x in o: all_others.append((sent, gold, pred, x))

    print(f"\n=== Stats ===")
    print(f"  total cases:       {len(cases)}")
    print(f"  diff cases:        {n_diff} ({100*n_diff/len(cases):.1f}%)")
    print(f"  merge mismatches:  {len(all_merges)}  (LTP 粗粒度,合并 gold 多词)")
    print(f"  split mismatches:  {len(all_splits)}  (LTP 细粒度,拆 gold 一词)")
    print(f"  other mismatches:  {len(all_others)} (真边界错 / 不对齐)")

    def show(title, cases, n):
        print(f"\n=== {title} (前 {min(n, len(cases))} 个) ===")
        for sent, gold, pred, x in cases[:n]:
            if isinstance(x, tuple) and len(x) == 2:  # merge or split
                a, b = x
                if isinstance(b, list) and isinstance(a, str):
                    print(f"  PD-06: {' '.join(b)}")
                    print(f"  LTP:   {a}")
                else:
                    print(f"  {a} | {b}")
            else:
                print(f"  {x}")
            print(f"  full sent: {sent[:60]}{'...' if len(sent)>60 else ''}")
            print(f"  gold: {' '.join(gold)[:80]}")
            print(f"  pred: {' '.join(pred)[:80]}")
            print()

    show("MERGE (LTP 比 PD-06 粗)", all_merges, args.max_show)
    show("SPLIT (LTP 比 PD-06 细)", all_splits, args.max_show)
    show("OTHER (真边界错)", all_others, args.max_show)


if __name__ == "__main__":
    main()
