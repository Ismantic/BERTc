"""从 PFR PD-1998 原始标注语料解析,生成 CWS / POS / NER 三种 jsonl。

PFR 格式:
  19980101-01-001-001/m  迈向/v  充满/v  希望/n  的/u  新/a  世纪/n  ——/w
  ...通过/p  [中央/n  人民/n  广播/vn  电台/n]nt  、/w  [中国/ns  ...]nt  ...

每行:
  - 头一个 token 是 ID(`yyyymmdd-NN-NNN-NNN/m`)→ skip
  - 后面 `word/POS` 序列,用 2 空格分隔
  - NER 嵌套:`[word/x word/y]NER_TAG`,内部 word 都是独立 CWS token

输出 split:
  train: 199801-199805
  dev:   199806

源:chenhui-bupt/PeopleDaily1998 的 199801.zip —— 目录名叫 199801,里面其实是
199801.txt ~ 199806.txt 六个月。先 `python data/download.py pd1998` 下载+解压。

产出与 v4-Large MT SOTA 实际训练输入一致:train 102,739 行 / dev 21,143 行。

"""
import argparse, json, os, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import source


ID_RE = re.compile(r"^\d{8}-\d+-\d+-\d+/m$")


def parse_token(tok):
    """解析一个 'word/POS' 或 '[word/POS' 或 'word/POS]NER' 返回 (is_bracket_open, word, pos, ner_close_tag)。"""
    is_open = False
    if tok.startswith("["):
        is_open = True
        tok = tok[1:]
    # 找 NER 关闭 ]xx
    ner_close = None
    m = re.search(r"\]([a-zA-Z]+)$", tok)
    if m:
        ner_close = m.group(1)
        tok = tok[:m.start()]
    # 最后一个 / 之前是 word,之后是 POS
    if "/" not in tok:
        return is_open, tok, "", ner_close  # 异常,无 POS
    word, _, pos = tok.rpartition("/")
    return is_open, word, pos, ner_close


def parse_line(line):
    """返回 (words, pos_tags, ner_spans) 或 None。"""
    parts = line.strip().split()
    if not parts: return None
    if ID_RE.match(parts[0]):
        parts = parts[1:]
    if not parts: return None
    words, pos_tags, ner_spans = [], [], []
    ner_open_idx = -1
    for tok in parts:
        is_open, word, pos, ner_close = parse_token(tok)
        if not word:
            continue
        if is_open:
            ner_open_idx = len(words)
        words.append(word)
        pos_tags.append(pos)
        if ner_close:
            start = ner_open_idx if ner_open_idx >= 0 else len(words) - 1
            ner_spans.append((start, len(words), ner_close))
            ner_open_idx = -1
    return words, pos_tags, ner_spans


def make_cws_jsonl(records, out_path, source_label):
    """写 CWS jsonl,跟旧 cws.jsonl 兼容。"""
    with open(out_path, "w", encoding="utf8") as f:
        n = 0
        for words, _, _ in records:
            if not words: continue
            text = "".join(words)
            obj = {
                "task": "cws",
                "lang": "zh",
                "messages": [
                    {"role": "user",
                     "content": f"任务: 中文分词\n请将下面这句中文分词,词与词之间用单个空格隔开,原文不增不减:\n原文: {text}"},
                    {"role": "assistant", "content": " ".join(words)},
                ],
                "gold": words,
                "source": source_label,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1
    return n


def make_pos_jsonl(records, out_path, source_label):
    """写 POS jsonl: per-word POS tag。"""
    with open(out_path, "w", encoding="utf8") as f:
        n = 0
        for words, pos_tags, _ in records:
            if not words or len(words) != len(pos_tags): continue
            text = "".join(words)
            obj = {
                "task": "pos",
                "lang": "zh",
                "text": text,
                "words": words,
                "pos": pos_tags,
                "source": source_label,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1
    return n


def make_ner_jsonl(records, out_path, source_label):
    """写 NER jsonl: entity list,LTP-aligned 类型 (Nh/Ns/Ni)。

    PFR 标注分两源:
      a) bracket `[X/n Y/n]nt` 或 `[X/ns Y/n]ns` — 多 word 复合实体
      b) 单 word POS = nr / ns / nt — 没 bracket 的裸专名
    我们都要收,但要避免重叠(bracket 包含的 word 不再单独算)。

    类型映射(PD → LTP):
      nt → Ni  (机构)
      ns → Ns  (地名)
      nr → Nh  (人名,相邻 nr 合并 = 姓+名)
      nz → 弃  (LTP 没 nz 实体类别;LTP/base1 自己把"小米/小红书"之类归 Ni/Nh)
    """
    NER_MAP = {"nt": "Ni", "ns": "Ns", "nr": "Nh"}  # 严格
    with open(out_path, "w", encoding="utf8") as f:
        n = 0
        for words, pos_tags, ner_spans in records:
            if not words: continue
            text = "".join(words)
            offsets = []
            pos = 0
            for w in words:
                offsets.append(pos)
                pos += len(w)
            in_bracket = [False] * len(words)  # word 是否已在 bracket 实体内
            entities = []
            # 1) bracket 复合实体
            for start_w, end_w, tag in ner_spans:
                if start_w >= len(words): continue
                end_w = min(end_w, len(words))
                if tag not in NER_MAP: continue  # 弃 unknown(如 i / l / nz / vd 等)
                char_start = offsets[start_w]
                char_end = (offsets[end_w] if end_w < len(offsets) else pos)
                entities.append({
                    "type": NER_MAP[tag],
                    "start": char_start,
                    "end": char_end,
                    "text": text[char_start:char_end],
                })
                for k in range(start_w, end_w):
                    in_bracket[k] = True
            # 2) 单 word fallback(裸 nr/ns/nt POS,不在 bracket 内)
            i = 0
            while i < len(words):
                if in_bracket[i]:
                    i += 1; continue
                p = pos_tags[i]
                if p not in NER_MAP:
                    i += 1; continue
                # 合并连续同 POS 的 word(主要 nr 姓+名;ns/nt 偶尔也有)
                j = i + 1
                while j < len(words) and not in_bracket[j] and pos_tags[j] == p:
                    j += 1
                cs = offsets[i]
                ce = (offsets[j] if j < len(offsets) else pos)
                entities.append({
                    "type": NER_MAP[p],
                    "start": cs,
                    "end": ce,
                    "text": text[cs:ce],
                })
                i = j
            # 按 start 排序便于下游
            entities.sort(key=lambda e: (e["start"], e["end"]))
            obj = {
                "task": "ner",
                "lang": "zh",
                "text": text,
                "entities": entities,
                "source": source_label,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir",
                    default=str(source.FINETUNE_SOURCES["pd1998"].dir() / "199801"),
                    help="解压后的月份 txt 所在目录")
    ap.add_argument("--out_dir",
                    default=str(source.PD98_DIR),
                    help="MT 训练脚本读的 jsonl 目录")
    ap.add_argument("--dev_month", default="199806",
                    help="哪个月作 dev,其余作 train")
    args = ap.parse_args()

    src_files = sorted(Path(args.src_dir).glob("19980?.txt"))
    print(f"Found {len(src_files)} month files in {args.src_dir}")

    train_records = []
    dev_records = []
    for fp in src_files:
        month = fp.stem  # '199801' etc
        is_dev = (month == args.dev_month)
        target = dev_records if is_dev else train_records
        before = len(target)
        with open(fp, "rb") as fb:
            raw = fb.read()
        # PFR 用 GBK 或 GB18030(常见)
        for enc in ["gb18030", "gbk", "utf-8"]:
            try:
                text = raw.decode(enc)
                break
            except Exception:
                continue
        for line in text.split("\n"):
            r = parse_line(line)
            if r:
                target.append(r)
        added = len(target) - before
        print(f"  {month}: +{added:,} lines ({'dev' if is_dev else 'train'})")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    # CWS
    n_train = make_cws_jsonl(train_records, out_dir / "cws.pd98.jsonl", "pd1998-train")
    n_dev = make_cws_jsonl(dev_records, out_dir / "cws_dev.pd98.jsonl", f"pd1998-{args.dev_month[-2:]}-dev")
    print(f"\n[CWS] train {n_train:,} → {out_dir}/cws.pd98.jsonl")
    print(f"[CWS] dev   {n_dev:,} → {out_dir}/cws_dev.pd98.jsonl")

    # POS
    n_train_pos = make_pos_jsonl(train_records, out_dir / "pos.pd98.jsonl", "pd1998-train")
    n_dev_pos = make_pos_jsonl(dev_records, out_dir / "pos_dev.pd98.jsonl", f"pd1998-{args.dev_month[-2:]}-dev")
    print(f"[POS] train {n_train_pos:,} → {out_dir}/pos.pd98.jsonl")
    print(f"[POS] dev   {n_dev_pos:,} → {out_dir}/pos_dev.pd98.jsonl")

    # NER
    n_train_ner = make_ner_jsonl(train_records, out_dir / "ner.pd98.jsonl", "pd1998-train")
    n_dev_ner = make_ner_jsonl(dev_records, out_dir / "ner_dev.pd98.jsonl", f"pd1998-{args.dev_month[-2:]}-dev")
    print(f"[NER] train {n_train_ner:,} → {out_dir}/ner.pd98.jsonl")
    print(f"[NER] dev   {n_dev_ner:,} → {out_dir}/ner_dev.pd98.jsonl")


if __name__ == "__main__":
    main()
