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


def encode_pairs(pairs, tok, keep_full_src: bool = False) -> list[dict]:
    """句对 → id。

    keep_full_src=False(训练):按 min(len(src), len(tgt)) 对齐 —— 逐位置的
    纠错标签只在两边都有字的位置上有定义。
    keep_full_src=True(测试):按 src 的完整长度编码。SIGHAN-15 测试集里有
    增删类错误,src 和 tgt 不等长,按 min 截会把长的那句尾巴切掉,评测就跟
    官方口径对不上了。
    """
    items = []
    for src, tgt in pairs:
        n = len(src) if keep_full_src else min(len(src), len(tgt))
        if n == 0:
            continue
        cor = tok.encode(tgt[:n])
        src_ids = tok.encode(src[:n])
        if len(cor) < n:                       # tgt 比 src 短,补齐(评测不看这段)
            cor = cor + src_ids[len(cor):]
        items.append({
            "input_ids": src_ids,
            "cor_labels": cor,
            "det_labels": [1 if i < len(tgt) and src[i] != tgt[i] else 0
                           for i in range(n)],
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
    test_items = encode_pairs(test_pairs, tok, keep_full_src=True)
    id_to_char = tok.id_to_char()
    print(f"字→id 缓存 {len(id_to_char):,} 个不同的字")

    common = {"format": "bertc-csc-v1", "pad_token_id": tok.pad_token_id,
              "vocab_size": tok.vocab_size, "id_to_char": id_to_char}
    save(pack(train_items, FIELDS, common), args.out_dir / "csc_train.pt")

    # 测试集额外存原文。评测要跟标准答案做字符串比对,而 id→字 的往返是有损的
    # (不同的字可能撞到同一个 id,实测 SIGHAN-15 里有 錓→镒、ㄦ→㚖 这类),
    # 拿还原出来的文本当参照会让分数偏高。预测那一侧没有别的办法,只能走还原,
    # 这跟原实现一致。
    n_bad = sum(1 for s, _ in test_pairs
                if "".join(id_to_char.get(i, "") for i in tok.encode(s)) != s)
    print(f"测试集 {len(test_pairs)} 条中 {n_bad} 条 id→字 往返不还原,"
          f"所以额外存原文作参照")
    save(pack(test_items, FIELDS, {
        **common,
        "src_texts": [s for s, _ in test_pairs],
        "tgt_texts": [t for _, t in test_pairs],
    }), args.out_dir / "csc_test.pt")


if __name__ == "__main__":
    main()
