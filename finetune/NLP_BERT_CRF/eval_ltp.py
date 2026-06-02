"""Eval LTP CWS on full PD-06 dev → boundary F1。

LTP 4.x 用 transformer 模型(HuggingFace 自动下),装 `uv pip install ltp`。
不用 PD-06 训练集,只评测 PD-06 dev set 跟其它 baseline 对比。

用法:
  python eval_ltp.py
  python eval_ltp.py --model LTP/base    # 大点的 model
  python eval_ltp.py --batch_size 64
"""
import argparse, json, time
from ltp import LTP


def boundary_f1(pred_words, gold_words):
    def spans(ws):
        out, pos = set(), 0
        for w in ws:
            out.add((pos, pos + len(w)))
            pos += len(w)
        return out
    P, G = spans(pred_words), spans(gold_words)
    if not P or not G:
        return 0.0
    tp = len(P & G)
    if tp == 0:
        return 0.0
    p = tp / len(P)
    r = tp / len(G)
    return 2 * p * r / (p + r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="LTP/small",
                    help="LTP/small (default) / LTP/base / LTP/base1 / LTP/base2 等")
    ap.add_argument("--dev_jsonl", default="./data/cws_dev.jsonl")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    print(f"Loading LTP model: {args.model}")
    ltp = LTP(args.model)
    if args.device == "cuda":
        try:
            ltp.to("cuda")
            print("  on cuda")
        except Exception as e:
            print(f"  cuda 失败 ({e}),用 cpu")
            args.device = "cpu"

    # Load dev set
    items = []
    with open(args.dev_jsonl, encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            obj = json.loads(line)
            words = obj.get("gold") or obj.get("cut", "").split()
            if not words: continue
            items.append(words)
    if args.limit:
        items = items[:args.limit]
    print(f"Dev: {len(items):,} samples")

    # Reconstruct sentences from gold word list
    sentences = ["".join(w) for w in items]

    # Batch inference
    f1s = []
    t0 = time.time()
    for batch_start in range(0, len(sentences), args.batch_size):
        batch_sents = sentences[batch_start:batch_start + args.batch_size]
        batch_gold = items[batch_start:batch_start + args.batch_size]
        try:
            out = ltp.pipeline(batch_sents, tasks=["cws"])
            preds = out.cws
        except Exception as e:
            # Fallback to one-by-one if batch fails(LTP 对超长 / 特殊 char 偶尔崩)
            preds = []
            for s in batch_sents:
                try:
                    o = ltp.pipeline([s], tasks=["cws"])
                    preds.append(o.cws[0])
                except Exception:
                    preds.append([s])  # 整句当一个词 → F1 极低
        for pred_words, gold_words in zip(preds, batch_gold):
            f1s.append(boundary_f1(pred_words, gold_words))
        n_done = batch_start + len(batch_sents)
        if (batch_start // args.batch_size) % 20 == 0:
            el = time.time() - t0
            cur_f1 = sum(f1s) / max(1, len(f1s))
            print(f"  {n_done:,}/{len(items):,} ({100*n_done/len(items):.1f}%) "
                  f"running F1 = {cur_f1:.4f}  {el:.0f}s", flush=True)

    final_f1 = sum(f1s) / max(1, len(f1s))
    print(f"\n=== LTP {args.model} on PD-06 dev ===")
    print(f"  samples:  {len(f1s):,}")
    print(f"  F1:       {final_f1:.4f}")
    print(f"  elapsed:  {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
