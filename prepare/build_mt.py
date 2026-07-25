"""PD-1998 的 cws/pos/ner jsonl → 预编码的 MT 训练集。

data/process_cws.py 产出三份 jsonl(按行对齐,同一句),这里把它们合成
每字一行的三套标签,再用 PieceTokenizer 编成 id,打包成 src/data.py 读的格式。

  cws  词 → BIES
  pos  PD 词性 → LTP 词性 → 摊到词的每个字上;词首字有标签,其余是 -100
  ner  实体区间 → BIES-类型;PD 的 PER/LOC/ORG → LTP 的 Nh/Ns/Ni,MISC 丢弃

**不截断**。max_chars 是训练时的超参(src/data.py 里截),换长度不用重跑这步。

用法:
    python -m prepare.build_mt                      # train + dev 都建
    python -m prepare.build_mt --split dev
"""
import argparse
import json
import sys
from pathlib import Path

from .labels import (CWS_TAGS, NER_TAGS, NER2ID, POS_TAGS, POS2ID,
                     entity_to_bies, map_pd_pos, words_to_cws_bies)
from .pack import pack, save
from .tokenizer import load_tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))
import source as corpus  # noqa: E402

DEFAULT_JSONL_DIR = corpus.PD98_DIR
DEFAULT_OUT_DIR = ROOT / "prepare" / "datasets"

FIELDS = ("input_ids", "cws_tags", "pos_tags", "ner_tags")


def read_jsonl(path: Path) -> list:
    with open(path, encoding="utf8") as f:
        return [json.loads(line) for line in f]


def build_items(cws_path: Path, pos_path: Path, ner_path: Path) -> list[dict]:
    """三份 jsonl 按行对齐 → [{chars, cws_tags, pos_tags, ner_tags}]。"""
    cws_rows = read_jsonl(cws_path)
    pos_rows = read_jsonl(pos_path)
    ner_rows = read_jsonl(ner_path)
    n = min(len(cws_rows), len(pos_rows), len(ner_rows))
    if not (len(cws_rows) == len(pos_rows) == len(ner_rows)):
        print(f"  ! 三份行数不一致 "
              f"({len(cws_rows)}/{len(pos_rows)}/{len(ner_rows)}),按 {n} 对齐")

    items = []
    for i in range(n):
        words = cws_rows[i].get("gold", [])
        if not words:
            continue
        chars, cws_tags = words_to_cws_bies(words)

        # POS:只有当 pos 那份的分词跟 cws 一致时才采信,否则整句无监督
        pos_tags = [-100] * len(chars)
        pos_words = pos_rows[i].get("words", [])
        pos_seq = pos_rows[i].get("pos", [])
        if pos_words == words and len(pos_words) == len(pos_seq):
            c = 0
            for w, p in zip(pos_words, pos_seq):
                pid = POS2ID.get(map_pd_pos(p), POS2ID["x"])
                for _ in w:
                    if c < len(pos_tags):
                        pos_tags[c] = pid
                    c += 1

        ner_tags = [NER2ID["O"]] * len(chars)
        for ent in ner_rows[i].get("entities", []):
            entity_to_bies(ner_tags, ent["start"], ent["end"], ent.get("type", ""))

        items.append({"chars": chars, "cws_tags": cws_tags,
                      "pos_tags": pos_tags, "ner_tags": ner_tags})
    return items


def build_split(split: str, jsonl_dir: Path, out_dir: Path, tok) -> Path:
    suffix = "" if split == "train" else "_dev"
    paths = {t: jsonl_dir / f"{t}{suffix}.pd98.jsonl" for t in ("cws", "pos", "ner")}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        sys.exit(f"缺少 {missing} —— 先跑 python data/process_cws.py")

    print(f"[{split}] 读 {jsonl_dir}")
    items = build_items(paths["cws"], paths["pos"], paths["ner"])
    print(f"  {len(items):,} 条")

    encoded = [{"input_ids": tok.encode("".join(it["chars"])),
                "cws_tags": it["cws_tags"], "pos_tags": it["pos_tags"],
                "ner_tags": it["ner_tags"]} for it in items]
    # 字级 tokenizer 下 1 字 = 1 id,不成立的话标签就对不齐了,这里挡一道
    bad = [i for i, (e, it) in enumerate(zip(encoded, items))
           if len(e["input_ids"]) != len(it["chars"])]
    if bad:
        sys.exit(f"  {len(bad)} 条样本编码后长度与字数不符(首个 idx={bad[0]}),"
                 f"tokenizer 不是字模式?")

    blob = pack(encoded, FIELDS, {
        "format": "bertc-mt-v1",
        "pad_token_id": tok.pad_token_id,
        "cws_vocab": CWS_TAGS,
        "pos_vocab": POS_TAGS,
        "ner_vocab": NER_TAGS,
    })
    return save(blob, out_dir / f"mt_{split}.pt")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jsonl_dir", type=Path, default=DEFAULT_JSONL_DIR)
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--tokenizer_dir", type=Path, default=None)
    ap.add_argument("--split", choices=("train", "dev"), default=None,
                    help="默认两个都建")
    args = ap.parse_args()

    tok = load_tokenizer(args.tokenizer_dir)
    print(tok)
    for split in ([args.split] if args.split else ["train", "dev"]):
        build_split(split, args.jsonl_dir, args.out_dir, tok)


if __name__ == "__main__":
    main()
