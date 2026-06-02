"""全面 sanity check cws.jsonl / cws_dev.jsonl 找数据预处理问题。

检测项:
  1. [xxx 残留 NER 起始标记
  2. xxx]nt/ns/nz/nr 残留 NER 结束标记
  3. /n /v 之类 POS 标签残留
  4. text(raw, in messages content)跟 gold join 一致性
  5. 超长 token / 异常 token
  6. 重复 sample
  7. 空 sample / 标签全空
  8. 非常见字符
"""
import json
import sys
import re
from collections import Counter, defaultdict


def extract_text(obj):
    """从 LLM-style messages 提取 user content 中的原文。"""
    msgs = obj.get("messages", [])
    for m in msgs:
        if m.get("role") == "user":
            c = m.get("content", "")
            # 形如 "...原文: <text>" 取后半
            if "原文:" in c:
                return c.split("原文:", 1)[1].strip()
            return c
    return ""


def check(path, name):
    print(f"\n{'='*70}\n=== {name}: {path}\n{'='*70}")
    total = 0
    issues = defaultdict(int)
    examples = defaultdict(list)
    text_lens = []
    gold_lens = []
    seen_texts = set()
    dupes = 0

    bracket_start = Counter()    # [xxx
    bracket_end = Counter()      # xxx]
    pos_tag = Counter()          # xxx/n
    chars_per_token_dist = Counter()
    weird_tokens = []

    with open(path, encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            obj = json.loads(line)
            gold = obj.get("gold", [])
            text = extract_text(obj)
            total += 1
            gold_joined = "".join(gold)
            text_lens.append(len(text))
            gold_lens.append(len(gold_joined))

            # 1. text vs gold 一致性
            if text and gold_joined and text != gold_joined:
                issues["text_neq_gold_join"] += 1
                if len(examples["text_neq_gold_join"]) < 3:
                    examples["text_neq_gold_join"].append(
                        f"text='{text[:80]}' | gold_join='{gold_joined[:80]}'")

            # 2. 重复
            if text in seen_texts:
                dupes += 1
            seen_texts.add(text)

            # 3. token-level 检查
            for tok in gold:
                if not tok: continue
                if tok.startswith("["):
                    bracket_start[tok[:6] if len(tok) > 6 else tok] += 1
                if re.search(r"\][a-z]+$", tok):
                    bracket_end[tok[-6:]] += 1
                if re.search(r"/[a-z]+$", tok) and not tok.startswith("/"):
                    pos_tag[tok[-4:]] += 1
                chars_per_token_dist[len(tok)] += 1
                # weird:含空白 / 非常用字符
                if " " in tok or "\t" in tok or "\n" in tok:
                    if len(weird_tokens) < 5:
                        weird_tokens.append(("whitespace", tok))
                if len(tok) > 15:
                    if len(weird_tokens) < 10:
                        weird_tokens.append(("超长", tok))

            # 4. 空 gold
            if not gold or all(not t for t in gold):
                issues["empty_gold"] += 1

    print(f"\n[1] 总 sample 数: {total:,}")
    print(f"[2] 重复 text: {dupes:,} ({100*dupes/total:.1f}%)")
    print(f"[3] text != gold.join: {issues['text_neq_gold_join']:,}")
    for ex in examples['text_neq_gold_join'][:3]:
        print(f"     例: {ex}")
    print(f"[4] empty gold: {issues['empty_gold']:,}")
    print(f"[5] text len 分布:    min={min(text_lens) if text_lens else 0}  "
          f"max={max(text_lens) if text_lens else 0}  "
          f"mean={sum(text_lens)/max(1,len(text_lens)):.0f}")
    print(f"[6] gold join len:    min={min(gold_lens) if gold_lens else 0}  "
          f"max={max(gold_lens) if gold_lens else 0}  "
          f"mean={sum(gold_lens)/max(1,len(gold_lens)):.0f}")
    print(f"\n[7] NER 起始 [xxx artifact: {sum(bracket_start.values()):,} 次")
    for k, v in bracket_start.most_common(8):
        print(f"     {k}: {v}")
    print(f"\n[8] NER 结束 ]xx artifact: {sum(bracket_end.values()):,} 次")
    for k, v in bracket_end.most_common(8):
        print(f"     {k}: {v}")
    print(f"\n[9] POS tag /xx artifact: {sum(pos_tag.values()):,} 次")
    for k, v in pos_tag.most_common(8):
        print(f"     {k}: {v}")
    print(f"\n[10] token 长度分布(char):")
    for l in sorted(chars_per_token_dist.keys())[:8]:
        print(f"     {l} char: {chars_per_token_dist[l]:,}")
    if len(chars_per_token_dist) > 8:
        print(f"     ... 最大 {max(chars_per_token_dist):,} char")
    print(f"\n[11] 异常 token 例:")
    for cat, tok in weird_tokens[:8]:
        print(f"     [{cat}] '{tok[:30]}'")


if __name__ == "__main__":
    for path, name in [
        ("/home/tfbao/Shiyu/Summer/BERT/NLP_BERT_CRF/data/cws.jsonl", "train"),
        ("/home/tfbao/Shiyu/Summer/BERT/NLP_BERT_CRF/data/cws_dev.jsonl", "dev"),
    ]:
        check(path, name)
