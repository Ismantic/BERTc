"""把 CSC 源合并成 (错句, 正句) 对,去重后写 pkl。

配方 sighan_wang271k —— 已发布的 CSC 模型(F1 0.8388)用的就是这份:

    wang271k/train.json                     Wang271K + SIGHAN,27.6 万对
    CTCDataset/sighan/sighan13_train.jsonl  SIGHAN-13 训练集
    CTCDataset/sighan/sighan14_train.jsonl  SIGHAN-14 训练集
    CTCDataset/sighan/sighan15_train.jsonl  SIGHAN-15 训练集

按上面的顺序读,先到先得(顺序即去重优先级),等长过滤后 249,975 对。

**等长过滤** `len(src) == len(tgt)`:BERTc-CSC 做的是狭义 CSC —— 同音 / 形似字
的等长替换。增删类的语法错误进来只会干扰。

格式:
  *.json    list of {original_text, correct_text}
  *.jsonl   {source, target, label}

用法:
    python data/process_csc.py             # 写 data/derived/csc/*.pkl
    python data/process_csc.py --verify    # 只统计,不写文件
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import source

# 已发布模型实际用的配方。顺序即去重优先级。
SIGHAN_WANG271K = [
    "wang271k/train.json",
    "CTCDataset/sighan/sighan13_train.jsonl",
    "CTCDataset/sighan/sighan14_train.jsonl",
    "CTCDataset/sighan/sighan15_train.jsonl",
]


def _clean(s: str) -> str:
    """去掉 CRLF 残留。这四个文件里有 4 对带尾随空白。"""
    return s.strip()


def load_json(path: Path):
    for it in json.load(open(path, encoding="utf8")):
        src, tgt = it.get("original_text"), it.get("correct_text")
        if src and tgt:
            yield _clean(src), _clean(tgt)


def load_jsonl(path: Path):
    for line in open(path, encoding="utf8", errors="ignore"):
        try:
            it = json.loads(line)
        except json.JSONDecodeError:
            continue
        src = it.get("source") or it.get("original_text") or it.get("text")
        tgt = it.get("target") or it.get("correct_text") or it.get("correct")
        if src and tgt:
            yield _clean(src), _clean(tgt)


def collect(raw_dir: Path, files: list[str],
            equal_length: bool = True) -> tuple[list, dict]:
    """按给定文件清单收集,返回 (去重后的 pair 列表, 每个文件的贡献数)。"""
    stats, seen, pairs = {}, set(), []
    for rel in files:
        path = raw_dir / rel
        if not path.exists():
            sys.exit(f"缺少 {path} —— 先跑 python data/download.py --csc")
        loader = load_json if path.suffix == ".json" else load_jsonl
        n_new = 0
        for pair in loader(path):
            if equal_length and len(pair[0]) != len(pair[1]):
                continue
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
                n_new += 1
        stats[rel] = n_new
    return pairs, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=source.DATA_ROOT / "csc")
    ap.add_argument("--output", type=Path, default=None,
                    help=f"默认 {source.CSC_PAIRS}")
    ap.add_argument("--verify", action="store_true", help="只统计,不写文件")
    ap.add_argument("--no-length-filter", action="store_true",
                    help="不做等长过滤(会混入增删类语法错误,与已发布口径不符)")
    args = ap.parse_args()

    if args.output is None:
        args.output = source.CSC_PAIRS
    if not args.raw_dir.exists():
        sys.exit(f"raw 目录不存在: {args.raw_dir} —— 先跑 python data/download.py --csc")

    print(f"配方 sighan_wang271k({len(SIGHAN_WANG271K)} 个文件)")
    pairs, stats = collect(args.raw_dir, SIGHAN_WANG271K,
                           equal_length=not args.no_length_filter)
    print(f"\n去重后 {len(pairs):,} 对")
    for name, n in stats.items():
        print(f"  {n:>8,}  {name}")

    if args.output.exists():
        with open(args.output, "rb") as f:
            old = set(pickle.load(f))
        new = set(pairs)
        print(f"\n对照已有 {args.output.name}: {len(old):,} 对")
        print(f"  重建 ∩ 原有 = {len(new & old):,} "
              f"(原有的 {len(new & old) / len(old):.2%})")
        print(f"  重建独有   = {len(new - old):,}")
        print(f"  原有独有   = {len(old - new):,}")

    if args.verify:
        print("\n--verify:不写文件")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(pairs, f)
    print(f"\n写出 {len(pairs):,} 对 → {args.output}")

    # SIGHAN-15 官方测试集原样拷一份到 derived/,评测直接读这里
    test_src = source.ALL_SOURCES["sighan15_test"].files()
    if test_src:
        source.SIGHAN_TEST.parent.mkdir(parents=True, exist_ok=True)
        source.SIGHAN_TEST.write_bytes(test_src[0].read_bytes())
        n = len(source.SIGHAN_TEST.read_text(encoding="utf8").splitlines())
        print(f"测试集 {n} 条 → {source.SIGHAN_TEST}")


if __name__ == "__main__":
    main()
