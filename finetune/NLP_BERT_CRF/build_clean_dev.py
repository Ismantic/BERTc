"""根据 verdict_all.jsonl 过滤 cws_dev.pd98.jsonl:
  - 排除 WRONG sample(LLM 确认 gold 错)
  - 排除 AMBIGUOUS sample(50/50,加噪)
  - 保留 CORRECT + 不在 ambig 列表的(剩 max_disagree<3 的 ambig 也保留 + 所有 baseline 完全同意 gold 的)

输出 cws_dev.pd98.cleanjudge.jsonl(LLM-judged clean dev)"""
import json
from pathlib import Path

# 加载 verdict
verdicts = {}
with open("data/judge_verdict_all.jsonl") as f:
    for line in f:
        v = json.loads(line)
        # query_id 格式: "<text_sample_idx>_<start>_<end>"
        sample_idx = int(v["query_id"].split("_")[0])
        verdicts[sample_idx] = v["verdict"]

excluded = {idx for idx, v in verdicts.items() if v in ("WRONG", "AMBIGUOUS")}
print(f"Verdict 总数: {len(verdicts)}")
print(f"  CORRECT keep: {sum(1 for v in verdicts.values() if v=='CORRECT')}")
print(f"  WRONG exclude: {sum(1 for v in verdicts.values() if v=='WRONG')}")
print(f"  AMBIGUOUS exclude: {sum(1 for v in verdicts.values() if v=='AMBIGUOUS')}")
print(f"  → 排除 {len(excluded)} sample")

# 过滤 dev
n_in, n_out = 0, 0
with open("data/cws_dev.pd98.jsonl") as fin, \
     open("data/cws_dev.pd98.cleanjudge.jsonl", "w") as fout:
    for i, line in enumerate(fin):
        n_in += 1
        if i in excluded:
            continue
        fout.write(line)
        n_out += 1

print(f"\nDev: {n_in} → {n_out} ({100*n_out/n_in:.1f}% kept)")
print(f"→ data/cws_dev.pd98.cleanjudge.jsonl")
