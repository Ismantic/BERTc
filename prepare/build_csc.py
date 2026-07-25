"""CSC 的 (错句, 正句) 对 → 预编码训练集与测试集。

训练对来自 data/process_csc.py 产出的 pkl,测试集是 SIGHAN-15 官方 707 条 tsv。

两个必须在这一步做好的事:

  det_labels  逐位置标"这里有没有错",按**字**比对而不是按 id ——
              两个不同的字可能都落到 UNK,按 id 比会漏掉那处错误。
  id_to_char  id → 字 的反查表,写进测试集文件。src/evaluate.py 靠它把预测
              还原成句子跟标准答案做字符串比对。表只覆盖**编码时见过的字**:
              范围放大会让本该"保留原字"的未知 id 变成真解码,口径就跟
              pycorrector 对不上了。

**不截断**。max_len 是训练超参(src/data.py 里截)。

用法:
    python -m prepare.build_csc
    python -m prepare.build_csc --train_pkl csc/data/all_pairs.rebuilt.pkl
"""
import argparse
import pickle
import sys
from pathlib import Path

from .pack import pack, save
from .tokenizer import load_tokenizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_PKL = ROOT / "csc" / "data" / "all_pairs.pkl"
DEFAULT_TEST_TSV = ROOT / "csc" / "data" / "test" / "sighan2015_test_official.tsv"
DEFAULT_OUT_DIR = ROOT / "prepare" / "datasets"

FIELDS = ("input_ids", "cor_labels", "det_labels")


def read_tsv(path: Path) -> list[tuple[str, str]]:
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) == 2:
            pairs.append((parts[0], parts[1]))
    return pairs


def encode_pairs(pairs, tok) -> list[dict]:
    items = []
    for src, tgt in pairs:
        n = min(len(src), len(tgt))
        if n == 0:
            continue
        items.append({
            "input_ids": tok.encode(src[:n]),
            "cor_labels": tok.encode(tgt[:n]),
            "det_labels": [1 if src[i] != tgt[i] else 0 for i in range(n)],
        })
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train_pkl", type=Path, default=DEFAULT_TRAIN_PKL)
    ap.add_argument("--test_tsv", type=Path, default=DEFAULT_TEST_TSV)
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--tokenizer_dir", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None, help="只取前 N 对(调试用)")
    args = ap.parse_args()

    for p in (args.train_pkl, args.test_tsv):
        if not p.exists():
            sys.exit(f"缺少 {p} —— 先跑 python data/process_csc.py")

    tok = load_tokenizer(args.tokenizer_dir)
    print(tok)

    with open(args.train_pkl, "rb") as f:
        train_pairs = pickle.load(f)
    if args.limit:
        train_pairs = train_pairs[:args.limit]
    test_pairs = read_tsv(args.test_tsv)
    print(f"训练 {len(train_pairs):,} 对 | 测试 {len(test_pairs):,} 条")

    # 先编训练集再编测试集,让反查表覆盖两边见过的所有字
    train_items = encode_pairs(train_pairs, tok)
    test_items = encode_pairs(test_pairs, tok)
    id_to_char = tok.id_to_char()
    print(f"字→id 缓存 {len(id_to_char):,} 个不同的字")

    common = {"format": "bertc-csc-v1", "pad_token_id": tok.pad_token_id,
              "vocab_size": tok.vocab_size, "id_to_char": id_to_char}
    save(pack(train_items, FIELDS, common), args.out_dir / "csc_train.pt")
    save(pack(test_items, FIELDS, common), args.out_dir / "csc_test.pt")


if __name__ == "__main__":
    main()
