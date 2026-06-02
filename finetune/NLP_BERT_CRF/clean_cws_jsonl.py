"""清理 cws.jsonl / cws_dev.jsonl 的 PD-06 预处理 artifacts:
  A. [xxx 起始 NER 标记 → 去掉 [(同时更新 text 里的 [)
  B. xxx/yy POS 标签残留 → 去掉 /yy 后缀
  C. 重复 text → 保留首次出现

输出:同名加 .clean.jsonl 后缀。"""
import json, re, sys
from pathlib import Path


def clean_token(tok):
    """去 [ 前缀 + 去 /xx POS 后缀。"""
    # 去 [ 前缀(只去开头那个)
    while tok.startswith("["):
        tok = tok[1:]
    # 去 /xx 后缀(/字母 1-3 个)
    m = re.search(r"/[a-z]{1,3}$", tok)
    if m:
        tok = tok[:m.start()]
    return tok


def clean_text(text):
    """text 里的 [(NER 起始)去掉。注意保留可能的方括号字面值(罕见)。"""
    # PD raw 里 [ 总在地名/机构名前(汉字开头)→ 安全去掉 [
    return text.replace("[", "")


def clean_jsonl(in_path, out_path):
    seen_texts = set()
    n_total, n_kept, n_dup, n_bracket_removed, n_pos_removed = 0, 0, 0, 0, 0
    with open(in_path, encoding="utf8") as fin, open(out_path, "w", encoding="utf8") as fout:
        for line in fin:
            line = line.strip()
            if not line: continue
            n_total += 1
            obj = json.loads(line)
            gold = obj.get("gold", [])
            new_gold = []
            for tok in gold:
                cleaned = clean_token(tok)
                if cleaned != tok:
                    if "[" in tok and "[" not in cleaned:
                        n_bracket_removed += 1
                    if re.search(r"/[a-z]+$", tok):
                        n_pos_removed += 1
                if cleaned:
                    new_gold.append(cleaned)
            # 同步 messages 里的 text
            for m in obj.get("messages", []):
                if m.get("role") == "user":
                    m["content"] = clean_text(m["content"])
                elif m.get("role") == "assistant":
                    # assistant content 是分词结果(用空格连接),也清
                    parts = m["content"].split()
                    m["content"] = " ".join(clean_token(p) for p in parts if clean_token(p))
            obj["gold"] = new_gold
            # text 去重
            user_text = ""
            for m in obj.get("messages", []):
                if m.get("role") == "user":
                    c = m.get("content", "")
                    if "原文:" in c:
                        user_text = c.split("原文:", 1)[1].strip()
                    else:
                        user_text = c
                    break
            if user_text in seen_texts:
                n_dup += 1
                continue
            seen_texts.add(user_text)
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n_kept += 1
    print(f"  {in_path} → {out_path}")
    print(f"  total {n_total:,} → kept {n_kept:,}  (去重 {n_dup:,})")
    print(f"  [xxx → xxx:  {n_bracket_removed:,}")
    print(f"  xxx/yy → xxx: {n_pos_removed:,}")
    return n_total, n_kept


if __name__ == "__main__":
    base = "/home/tfbao/Shiyu/Summer/BERT/NLP_BERT_CRF/data"
    for f in ["cws.jsonl", "cws_dev.jsonl"]:
        in_path = f"{base}/{f}"
        out_path = f"{base}/{f[:-6]}.clean.jsonl"
        print(f"\n=== clean {f} ===")
        clean_jsonl(in_path, out_path)
