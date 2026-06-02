"""从 ambiguous_dev_cases.sorted.jsonl 中抽 N 个 max_disagree=3 span(随机),
准备成 LLM judge 的 query 格式。

每个 query 含:
  - text(原文)
  - target_span(disagree gold span)
  - context_before / context_after
  - gold_word(PD-1998 切法)
  - baseline_word(3 个 baseline 都切的同样 span 的词)
"""
import argparse, json, random


def get_word_at_char(words, char_offset):
    """根据 char offset 找它所在的词,返回 (word, word_start_char, word_end_char)。"""
    pos = 0
    for w in words:
        if pos <= char_offset < pos + len(w):
            return w, pos, pos + len(w)
        pos += len(w)
    return None, -1, -1


def find_alt_word(text, pred_words, gold_start, gold_end):
    """找 baseline pred 中覆盖 gold [start,end) 的所有 words,返回它们的切法字串。"""
    pos = 0
    covered = []
    for w in pred_words:
        w_start, w_end = pos, pos + len(w)
        if not (w_end <= gold_start or w_start >= gold_end):
            covered.append(w)
        pos += len(w)
    if not covered:
        return None
    # 用 " / " 显示 baseline 的多 word 切法
    return " / ".join(covered)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="./data/ambiguous_dev_cases.sorted.jsonl")
    ap.add_argument("--output", default="./data/judge_query_100.jsonl")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--context_chars", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # 加载所有 max=3 case
    cases = []
    with open(args.input, encoding="utf8") as f:
        for line in f:
            c = json.loads(line)
            if c["max_disagree"] == 3:
                cases.append(c)
    print(f"max_disagree=3 sample 总数: {len(cases):,}")

    # 抽 N 个,然后每个 sample 取 1 个 max=3 span
    rng = random.Random(args.seed)
    rng.shuffle(cases)
    queries = []
    for c in cases:
        if len(queries) >= args.n: break
        # 选这个 sample 里的一个 max=3 span(取第一个或随机)
        m3_spans = [s for s in c["disagree_spans"] if s["n_disagree"] == 3]
        if not m3_spans: continue
        sp = rng.choice(m3_spans)
        gold_start, gold_end = sp["start"], sp["end"]
        text = c["text"]
        gold_word = sp["gold_word"]
        # 找 3 个 baseline 在这位置的切法
        baseline_alts = []
        for name, pred in c["preds"].items():
            alt = find_alt_word(text, pred, gold_start, gold_end)
            if alt:
                baseline_alts.append((name, alt))
        # context
        ctx_before = text[max(0, gold_start - args.context_chars):gold_start]
        ctx_after = text[gold_end:gold_end + args.context_chars]
        queries.append({
            "query_id": f"{c['idx']}_{gold_start}_{gold_end}",
            "text_sample_idx": c["idx"],
            "gold_span": (gold_start, gold_end),
            "gold_word": gold_word,
            "context_before": ctx_before,
            "context_after": ctx_after,
            "baseline_alts": baseline_alts,
            "full_text": text[:200] + ("..." if len(text) > 200 else ""),
        })

    with open(args.output, "w", encoding="utf8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"\nSampled {len(queries)} queries → {args.output}")


if __name__ == "__main__":
    main()
