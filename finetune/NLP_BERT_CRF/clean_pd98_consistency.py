"""保守清洗 cws.pd98 / cws_dev.pd98 的标注 inconsistency。

策略:
  1. 从 train 统计每个 2-char phrase 的 MERGE vs SPLIT 分布
  2. minority < 15% 的 phrase 算 annotation noise,做 majority vote
  3. 同 phrase 字典应用到 train + dev,保证一致

只处理 2-char phrase(拆/合明确,无歧义拆点)。

输出:
  cws.pd98.clean.jsonl + cws_dev.pd98.clean.jsonl
"""
import argparse, json, os
from collections import defaultdict, Counter
from pathlib import Path


def collect_phrase_stats(path, max_phrase_len=2):
    """统计每个 2-char phrase 的 MERGE/SPLIT 分布。"""
    stats = defaultdict(Counter)
    with open(path, encoding="utf8") as f:
        for line in f:
            gold = json.loads(line)["gold"]
            # MERGE:整 word 是 phrase
            for w in gold:
                if len(w) == max_phrase_len:
                    stats[w]["MERGE"] += 1
            # SPLIT:相邻 word 合 = phrase
            for i in range(len(gold) - 1):
                if len(gold[i]) + len(gold[i+1]) == max_phrase_len:
                    merged = gold[i] + gold[i+1]
                    stats[merged]["SPLIT"] += 1
    return stats


def build_force_dict(stats, min_count=50, minority_threshold=0.15):
    """对每个 phrase,如果 minority < 15% 就 force majority。返回:
        force_merge: set of phrases that should always be MERGED (single word)
        force_split: set of phrases that should always be SPLIT (2 single-char words)
    """
    force_merge, force_split = set(), set()
    for p, cnt in stats.items():
        m, s = cnt["MERGE"], cnt["SPLIT"]
        total = m + s
        if total < min_count: continue
        minority = min(m, s) / total
        if minority >= minority_threshold: continue  # 分歧大,不动
        if m > s: force_merge.add(p)
        else: force_split.add(p)
    return force_merge, force_split


def clean_gold(gold, force_merge, force_split):
    """对 gold 应用 force_merge / force_split。返回新 gold list。"""
    # Step 1: force_merge — 找连续 2 word 合并 = force_merge phrase 的位置,合并
    out = []
    i = 0
    while i < len(gold):
        if i + 1 < len(gold):
            merged = gold[i] + gold[i+1]
            if merged in force_merge and len(merged) == 2:
                out.append(merged)
                i += 2
                continue
        out.append(gold[i])
        i += 1
    # Step 2: force_split — 找单 word = force_split phrase,拆成 2 single-char
    new_out = []
    for w in out:
        if w in force_split and len(w) == 2:
            new_out.extend(list(w))  # 'A B' → ['A', 'B']
        else:
            new_out.append(w)
    return new_out


def apply_clean(in_path, out_path, force_merge, force_split):
    n_total, n_changed = 0, 0
    with open(in_path, encoding="utf8") as fin, open(out_path, "w", encoding="utf8") as fout:
        for line in fin:
            obj = json.loads(line)
            old_gold = obj["gold"]
            new_gold = clean_gold(old_gold, force_merge, force_split)
            n_total += 1
            if new_gold != old_gold:
                n_changed += 1
                obj["gold"] = new_gold
                # 同步 messages.assistant.content
                for m in obj.get("messages", []):
                    if m.get("role") == "assistant":
                        m["content"] = " ".join(new_gold)
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return n_total, n_changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_in", default="data/cws.pd98.jsonl")
    ap.add_argument("--dev_in",   default="data/cws_dev.pd98.jsonl")
    ap.add_argument("--train_out", default="data/cws.pd98.clean.jsonl")
    ap.add_argument("--dev_out",   default="data/cws_dev.pd98.clean.jsonl")
    ap.add_argument("--minority_threshold", type=float, default=0.15)
    ap.add_argument("--min_count", type=int, default=50)
    args = ap.parse_args()

    print(f"Collecting phrase stats from {args.train_in}...")
    stats = collect_phrase_stats(args.train_in, max_phrase_len=2)
    print(f"  {len(stats):,} unique 2-char phrases")

    force_merge, force_split = build_force_dict(
        stats, min_count=args.min_count,
        minority_threshold=args.minority_threshold)
    print(f"\n  force MERGE phrases: {len(force_merge):,}")
    print(f"  force SPLIT phrases: {len(force_split):,}")
    print(f"  not touched (high disagreement ≥{100*args.minority_threshold:.0f}%): "
          f"{sum(1 for p,c in stats.items() if c['MERGE']+c['SPLIT']>=args.min_count) - len(force_merge) - len(force_split):,}")

    print("\nApplying to train + dev:")
    n_train, n_train_changed = apply_clean(args.train_in, args.train_out, force_merge, force_split)
    n_dev, n_dev_changed = apply_clean(args.dev_in, args.dev_out, force_merge, force_split)
    print(f"  train: {n_train_changed:,}/{n_train:,} samples changed ({100*n_train_changed/n_train:.1f}%) → {args.train_out}")
    print(f"  dev:   {n_dev_changed:,}/{n_dev:,} samples changed ({100*n_dev_changed/n_dev:.1f}%) → {args.dev_out}")


if __name__ == "__main__":
    main()
