"""LTP/base1 在 PD-1998 dev 上 zero-shot 三任务评测,与我们 MT 对齐。

数据:
  cws_dev.pd98.jsonl (21143)  — gold words
  pos_dev.pd98.jsonl (21143)  — gold PD POS  → map LTP
  ner_dev.pd98.jsonl (21143)  — gold PER/LOC/ORG (BIES → Nh/Ns/Ni)

Metrics(全部 char-level):
  CWS boundary F1
  POS per-char accuracy (跳过 -100)
  NER span F1 (type+span 都匹配)
"""
import argparse, json, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ltp_label_align import map_pd_pos, PD2LTP_NER, LTP_POS2ID


def boundary_f1(pred_words, gold_words):
    def spans(ws):
        out, pos = set(), 0
        for w in ws:
            out.add((pos, pos + len(w))); pos += len(w)
        return out
    P, G = spans(pred_words), spans(gold_words)
    if not P or not G: return 0.0
    tp = len(P & G)
    if tp == 0: return 0.0
    p, r = tp/len(P), tp/len(G)
    return 2*p*r/(p+r)


def ner_span_f1(pred_spans, gold_spans):
    P, G = set(pred_spans), set(gold_spans)
    if not P or not G: return None
    tp = len(P & G)
    if tp == 0: return 0.0
    p, r = tp/len(P), tp/len(G)
    return 2*p*r/(p+r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cws", default="./data/cws_dev.pd98.jsonl")
    ap.add_argument("--pos", default="./data/pos_dev.pd98.jsonl")
    ap.add_argument("--ner", default="./data/ner_dev.pd98.jsonl")
    ap.add_argument("--model", default="LTP/base1")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from ltp import LTP
    print(f"Load LTP {args.model}")
    ltp = LTP(args.model)
    if args.device == "cuda": ltp.to("cuda")

    # 同行对齐 3 个 jsonl
    cws_items, pos_items, ner_items = [], [], []
    with open(args.cws, encoding="utf8") as f:
        for line in f: cws_items.append(json.loads(line))
    with open(args.pos, encoding="utf8") as f:
        for line in f: pos_items.append(json.loads(line))
    with open(args.ner, encoding="utf8") as f:
        for line in f: ner_items.append(json.loads(line))
    n = min(len(cws_items), len(pos_items), len(ner_items))
    if args.limit: n = min(n, args.limit)
    print(f"Eval {n} samples")

    # micro CWS counters
    cws_tp, cws_p_total, cws_g_total = 0, 0, 0
    # POS per-word
    pos_word_correct, pos_word_total = 0, 0
    pos_correct, pos_total = 0, 0  # per-char(对照)
    # micro NER
    ner_tp, ner_p_total, ner_g_total = 0, 0, 0
    t0 = time.time()
    for i0 in range(0, n, args.batch_size):
        batch_idx = list(range(i0, min(i0 + args.batch_size, n)))
        texts = [cws_items[i].get("text") or "".join(cws_items[i].get("gold", []))
                 for i in batch_idx]
        try:
            out = ltp.pipeline(texts, tasks=["cws","pos","ner"])
            cws_b, pos_b, ner_b = out.cws, out.pos, out.ner
        except Exception as e:
            print(f"  batch {i0} error: {e}, skip"); continue
        for k, idx in enumerate(batch_idx):
            text = texts[k]
            pred_words = cws_b[k]
            pred_pos = pos_b[k]
            pred_ner_w = ner_b[k]
            if "".join(pred_words) != text: continue

            # CWS micro F1: 累计 char-span 集合
            gold_words = cws_items[idx].get("gold", [])
            if gold_words:
                def _spans(ws):
                    out, p = set(), 0
                    for w in ws: out.add((p, p+len(w))); p += len(w)
                    return out
                P, G = _spans(pred_words), _spans(gold_words)
                cws_tp += len(P & G)
                cws_p_total += len(P)
                cws_g_total += len(G)

            # POS per-word(对齐 LTP):仅当 cws pred == gold words 时算 word-level acc
            gold_words_pos = pos_items[idx].get("words", [])
            gold_pos_seq = pos_items[idx].get("pos", [])
            if gold_words_pos == pred_words and len(gold_pos_seq) == len(gold_words_pos) \
               and len(pred_pos) == len(pred_words):
                for gp, pp in zip(gold_pos_seq, pred_pos):
                    pos_word_total += 1
                    if map_pd_pos(gp) == pp: pos_word_correct += 1
            # 同时保留 per-char acc(参考)— 不论分词是否对齐都算
            if gold_words_pos == gold_words and len(gold_pos_seq) == len(gold_words):
                gold_pos_chars = []
                for w, p in zip(gold_words_pos, gold_pos_seq):
                    for _ in w: gold_pos_chars.append(map_pd_pos(p))
                pred_pos_chars = []
                for w, p in zip(pred_words, pred_pos):
                    for _ in w: pred_pos_chars.append(p)
                m = min(len(gold_pos_chars), len(pred_pos_chars))
                for c in range(m):
                    pos_total += 1
                    if gold_pos_chars[c] == pred_pos_chars[c]:
                        pos_correct += 1

            # NER span F1: char-level
            gold_spans = set()
            for e in ner_items[idx].get("entities", []):
                t = e.get("type")
                # 兼容旧 (PER/LOC/ORG) + 新 (Nh/Ns/Ni)
                if t in PD2LTP_NER: t = PD2LTP_NER[t]
                if t not in ("Nh", "Ns", "Ni"): continue
                gold_spans.add((t, e["start"], e["end"]))
            # pred: ner_b[k] → (type, text, word_start, word_end)
            word_off = []
            pc = 0
            for w in pred_words:
                word_off.append(pc); pc += len(w)
            word_off.append(pc)
            pred_spans = set()
            for e in pred_ner_w:
                et, _, ws, we = e[0], e[1], e[2], e[3]
                if ws >= len(pred_words) or we >= len(pred_words): continue
                pred_spans.add((et, word_off[ws], word_off[we+1]))
            # NER micro
            ner_tp += len(pred_spans & gold_spans)
            ner_p_total += len(pred_spans)
            ner_g_total += len(gold_spans)

        if (i0 // args.batch_size) % 30 == 0:
            el = time.time()-t0
            cws_p = cws_tp/max(1,cws_p_total); cws_r = cws_tp/max(1,cws_g_total)
            cws_f1 = 2*cws_p*cws_r/max(1e-9, cws_p+cws_r)
            ner_p = ner_tp/max(1,ner_p_total); ner_r = ner_tp/max(1,ner_g_total)
            ner_f1 = 2*ner_p*ner_r/max(1e-9, ner_p+ner_r)
            print(f"  {i0+len(batch_idx):,}/{n:,} {el:.0f}s "
                  f"cws_F1={cws_f1:.4f} "
                  f"pos_word_acc={pos_word_correct/max(1,pos_word_total):.4f} "
                  f"ner_F1={ner_f1:.4f}", flush=True)

    cws_p = cws_tp/max(1,cws_p_total); cws_r = cws_tp/max(1,cws_g_total)
    cws_f1 = 2*cws_p*cws_r/max(1e-9, cws_p+cws_r)
    ner_p = ner_tp/max(1,ner_p_total); ner_r = ner_tp/max(1,ner_g_total)
    ner_f1 = 2*ner_p*ner_r/max(1e-9, ner_p+ner_r)
    print(f"\n=== LTP/base1 zero-shot on PD-1998 dev (n={n}) ===")
    print(f"  CWS micro F1:     {cws_f1:.4f}  (P={cws_p:.4f} R={cws_r:.4f})")
    print(f"  POS per-word acc: {pos_word_correct/max(1,pos_word_total):.4f}  (only when CWS pred==gold,n={pos_word_total:,})")
    print(f"  POS per-char acc: {pos_correct/max(1,pos_total):.4f}  (对照,n={pos_total:,})")
    print(f"  NER micro F1:     {ner_f1:.4f}  (P={ner_p:.4f} R={ner_r:.4f},n_gold={ner_g_total:,})")
    print(f"  total time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
