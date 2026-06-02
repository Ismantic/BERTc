"""按 disagree count 排序 ambiguous_dev_cases.jsonl,输出分层统计 + 排序文件。

n_disagree:
  3 = unanimous(3 baseline 全 disagree)→ gold 大概率错
  2 = majority(2/3 baseline disagree)→ gold 50/50
  1 = minority(仅 1 baseline disagree)→ baseline 错概率高

每 sample 取它含的最大 n_disagree(sample 内有多 span)做 sort key。
"""
import argparse, json
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="./data/ambiguous_dev_cases.jsonl")
    ap.add_argument("--output", default="./data/ambiguous_dev_cases.sorted.jsonl")
    args = ap.parse_args()

    cases = []
    with open(args.input, encoding="utf8") as f:
        for line in f:
            obj = json.loads(line)
            obj["max_disagree"] = max((s["n_disagree"] for s in obj["disagree_spans"]), default=0)
            obj["sum_disagree"] = sum(s["n_disagree"] for s in obj["disagree_spans"])
            obj["n_disagree_spans"] = len(obj["disagree_spans"])
            cases.append(obj)

    # 排序:max_disagree 优先(降),其次 sum_disagree(降),最后 idx(升,稳定)
    cases.sort(key=lambda c: (-c["max_disagree"], -c["sum_disagree"], c["idx"]))

    # 分层统计
    by_max = Counter(c["max_disagree"] for c in cases)
    by_sum = Counter(c["sum_disagree"] for c in cases)
    print(f"=== 分层统计 (total {len(cases)} ambiguous sample) ===\n")
    print(f"按 max_disagree(sample 内最严重的 disagree level):")
    for k in sorted(by_max.keys(), reverse=True):
        print(f"  max_disagree={k}:  {by_max[k]:>6,}  ({100*by_max[k]/len(cases):.1f}%)")
    print(f"\n按 sum_disagree(sample 内累计 disagree 次数):")
    for k in sorted(by_sum.keys(), reverse=True)[:10]:
        print(f"  sum={k}:  {by_sum[k]:>6,}")

    with open(args.output, "w", encoding="utf8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"\nSorted → {args.output}")


if __name__ == "__main__":
    main()
