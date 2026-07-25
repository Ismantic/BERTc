"""把 csc/data/raw/ 下所有 CSC 源统一成 (错句, 正句) 对,去重后写 pkl。

背景:v4-Large CSC SOTA(SIGHAN-15 F1 0.8346)用的 csc/data/all_pairs.pkl
是当年临时拼的,**生成代码从未进过 git**。这份是补写的可复现版本。

配方是逐文件集合比对反推出来的,三条规则:

1. **扫 raw/ 下全部源**,包括两个容易漏的 `.jsonl.gz`
   (CTC2021/train_large_v2 贡献 10.2 万对,Wang271k/data 是全量 26.8 万)
2. **等长过滤** `len(src) == len(tgt)` —— 原 pkl 826,097 对里不等长的有 0 个。
   BERTc-CSC 做的是狭义 CSC(同音/形似字的等长替换),增删类的语法错误不要。
   佐证:CTC2021 等长部分 101,950 对 100% 在原 pkl,不等长部分 115,667 对 0% 在。
3. **整文件排除** EXCLUDE_PATTERNS 里那几个源(见下)

三条都加上后重建 826,205 对 vs 原 826,097 对,双向重合 99.96%
(重建独有 470 / 原有独有 362,是空白与全半角规范化的边角差异)。

各源格式:
  *.json          list of {original_text, correct_text}      (wang271k_csc)
  *.jsonl[.gz]    {source, target, label}                    (CTCDataset)
  *.sgml          <TEXT> + <MISTAKE><LOCATION|WRONG|CORRECTION>  (wang271k_raw)
  *.tsv / *.txt   src<TAB>tgt[<TAB>type]                     (其余)
坑:
  - CTCDataset 下两个 .jsonl.gz(CTC2021/train_large_v2、Wang271k/data)
    是主力源之一,漏掉会缺 10 万对
  - mcsc_* / shibing624 是 CRLF,不 strip 会让 tgt 带 \r
  - mcsc_set 用 {} 标出错误片段,要去掉
  - shibing624/*.tsv 有表头 source/target/type
  - cscd_ime/ ecspell/ 下是当年下载失败留下的 HTML 错误页,跳过

用法:
    python data/process_csc.py                  # → csc/data/all_pairs.pkl
    python data/process_csc.py --verify         # 跟现有 pkl 比对,不写文件
"""
import argparse
import gzip
import json
import pickle
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import source

# 明显是下载失败残留的目录(内容为 HTML 错误页 / "404: Not Found")
JUNK_MARKERS = ("404: Not Found", "Invalid username or password.", "<!DOCTYPE", "<html")

# 原 all_pairs.pkl 整文件排除掉的源。逐文件比对确认:这几个文件贡献的等长对
# 100%(HSK/lemon_v2/val_bak)或 99%(MuCGEC)都不在原 pkl 里。
EXCLUDE_PATTERNS = [
    "NLPCC2023/grammar/",   # HSK 10.4 万 + MuCGEC —— 语法纠错,非同音/形似字替换
    "lemon_v2/",            # 与 CTCDataset/lemon/ 同源的另一版本,原口径用前者
    "val_bak",              # 备份文件
]

TSV_PATTERNS = ["sighan/*.txt", "shibing624/*.tsv", "mcsc_full/*.txt",
                "mcsc_set/*.txt", "lemon_v2/*.txt", "ecspell/*.txt",
                "cscd_ime/*.txt", "cscd_ime/*.tsv"]


def _clean(s: str) -> str:
    """去掉 CRLF 残留和 MCSCSet 的 {} 错误标记。"""
    return s.strip().replace("{", "").replace("}", "")


def _open(path: Path):
    """透明处理 .gz。"""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf8", errors="ignore")
    return open(path, encoding="utf8", errors="ignore")


def _is_excluded(path: Path, raw_dir: Path) -> bool:
    rel = str(path.relative_to(raw_dir))
    return any(pat in rel for pat in EXCLUDE_PATTERNS)


def _is_junk(path: Path) -> bool:
    if path.suffix == ".gz":
        return False
    try:
        head = path.read_text(encoding="utf8", errors="ignore")[:200]
    except OSError:
        return True
    return any(m in head for m in JUNK_MARKERS)


def load_json(path: Path):
    for it in json.load(open(path, encoding="utf8")):
        src, tgt = it.get("original_text"), it.get("correct_text")
        if src and tgt:
            yield _clean(src), _clean(tgt)


def load_jsonl(path: Path):
    for line in _open(path):
        try:
            it = json.loads(line)
        except json.JSONDecodeError:
            continue
        src = it.get("source") or it.get("original_text") or it.get("text")
        tgt = it.get("target") or it.get("correct_text") or it.get("correct")
        if src and tgt:
            yield _clean(src), _clean(tgt)


def load_sgml(path: Path):
    """Wang271K 原始 SGML:一个 <SENTENCE> 里 <TEXT> 是错句,
    每个 <MISTAKE> 给出 1-based 位置 + 错字 + 正字,逐个替换得正句。"""
    text = path.read_text(encoding="utf8", errors="ignore")
    for block in re.findall(r"<SENTENCE>(.*?)</SENTENCE>", text, re.S):
        m = re.search(r"<TEXT>(.*?)</TEXT>", block, re.S)
        if not m:
            continue
        src = _clean(m.group(1))
        chars = list(src)
        for loc, wrong, corr in re.findall(
                r"<LOCATION>(\d+)</LOCATION>\s*<WRONG>(.*?)</WRONG>\s*"
                r"<CORRECTION>(.*?)</CORRECTION>", block, re.S):
            i = int(loc) - 1
            wrong, corr = wrong.strip(), corr.strip()
            if 0 <= i < len(chars) and chars[i] == wrong and len(corr) == 1:
                chars[i] = corr
        tgt = "".join(chars)
        if src and tgt:
            yield src, tgt


def load_tsv(path: Path):
    for i, line in enumerate(open(path, encoding="utf8", errors="ignore")):
        fields = line.rstrip("\r\n").split("\t")
        if len(fields) < 2:
            continue
        src, tgt = _clean(fields[0]), _clean(fields[1])
        if i == 0 and src == "source" and tgt == "target":   # 表头
            continue
        if src and tgt:
            yield src, tgt


def collect(raw_dir: Path, equal_length: bool = True,
            apply_excludes: bool = True) -> tuple[list, dict]:
    """扫全部源,返回 (去重后的 pair 列表, 每个文件的贡献数)。

    equal_length=True 只保留 len(src)==len(tgt) 的对 —— 狭义 CSC 的定义,
    也是原 all_pairs.pkl 的实际口径(其 826,097 对中不等长的为 0)。
    """
    stats = {}
    seen, pairs = set(), []

    def skip(path: Path) -> bool:
        return _is_junk(path) or (apply_excludes and _is_excluded(path, raw_dir))

    def add(path: Path, it):
        n_new = 0
        for pair in it:
            if equal_length and len(pair[0]) != len(pair[1]):
                continue
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
                n_new += 1
        stats[str(path.relative_to(raw_dir))] = n_new

    for path in sorted(raw_dir.glob("*/*.json")):
        if skip(path):
            continue
        add(path, load_json(path))
    for pat in ("**/*.jsonl", "**/*.jsonl.gz"):
        for path in sorted(raw_dir.glob(pat)):
            if skip(path):
                continue
            add(path, load_jsonl(path))
    for path in sorted(raw_dir.glob("**/*.sgml")):
        if skip(path):
            continue
        add(path, load_sgml(path))
    for pat in TSV_PATTERNS:
        for path in sorted(raw_dir.glob(pat)):
            if skip(path):
                continue
            add(path, load_tsv(path))
    return pairs, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=source.CSC_RAW_DIR)
    ap.add_argument("--output", type=Path,
                    default=source.REPO_ROOT / "csc" / "data" / "all_pairs.rebuilt.pkl",
                    help="默认写 .rebuilt.pkl,不覆盖 SOTA 实际用的 all_pairs.pkl")
    ap.add_argument("--verify", action="store_true",
                    help="跟现有 all_pairs.pkl 比对覆盖率,不写文件")
    ap.add_argument("--top", type=int, default=20, help="打印贡献最多的 N 个文件")
    ap.add_argument("--no-length-filter", action="store_true",
                    help="不做等长过滤(会混入增删类语法错误,与原 pkl 口径不符)")
    ap.add_argument("--no-excludes", action="store_true",
                    help="不排除 EXCLUDE_PATTERNS 里的源(会多出 12.6 万对语法/重复数据)")
    args = ap.parse_args()

    if not args.raw_dir.exists():
        sys.exit(f"raw 目录不存在: {args.raw_dir}")

    print(f"扫描 {args.raw_dir} ...")
    pairs, stats = collect(args.raw_dir,
                           equal_length=not args.no_length_filter,
                           apply_excludes=not args.no_excludes)
    print(f"\n去重后 {len(pairs):,} 对,来自 {len(stats)} 个文件")
    print(f"\n贡献最多的 {args.top} 个:")
    for name, n in sorted(stats.items(), key=lambda kv: -kv[1])[:args.top]:
        print(f"  {n:>8,}  {name}")

    ref = source.REPO_ROOT / "csc" / "data" / "all_pairs.pkl"
    if ref.exists():
        with open(ref, "rb") as f:
            old = set(pickle.load(f))
        new = set(pairs)
        print(f"\n对照 {ref.name}: {len(old):,} 对")
        print(f"  重建 ∩ 原有 = {len(new & old):,} "
              f"(原有的 {len(new & old) / len(old):.2%})")
        print(f"  重建独有   = {len(new - old):,}")
        print(f"  原有独有   = {len(old - new):,}")

    if args.verify:
        print("\n--verify:未写文件。")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(pairs, f)
    print(f"\n写入 {len(pairs):,} 对 -> {args.output}")


if __name__ == "__main__":
    main()
