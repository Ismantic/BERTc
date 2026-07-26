"""把下载的原始数据加工成 encode_corpus 能直接读的 documents.txt。

只有 PeopleDaily 和 CnnDailyMail 需要这一步 —— SkyPile / CCI3-HQ /
FineWeb-Edu / finewiki 本来就是 parquet / jsonl,encode_corpus.py 直接读。

documents.txt 格式:一行一篇,title 与 content 用空格连接,内部换行折叠成空格。
逻辑移植自 Shiyu/Data 的 src/process.py + src/documents.py(原本是两段:
raw → 统一 jsonl → documents.txt;这里合成一步,输出等价)。

用法:
    python data/process.py people_daily
    python data/process.py cnn_dailymail
    python data/process.py --all
"""
import argparse
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import source


def _write_doc(out, title: str, content: str) -> int:
    doc = title + "\n" + content if title else content
    doc = doc.replace("\n", " ").strip()
    if not doc:
        return 0
    out.write(doc + "\n")
    return 1


def process_people_daily(output: Path | None = None) -> Path:
    """人民日报 *.jsonl.gz(字段 title/text/date)→ PeopleDaily.documents.txt"""
    src = source.PRETRAIN_SOURCES["people_daily"]
    src_dir = src.dir()
    output = output or source.DERIVED_ROOT / source.DERIVED["people_daily_docs"]
    output.parent.mkdir(parents=True, exist_ok=True)

    gz_files = sorted(src_dir.glob("*.jsonl.gz"))
    if not gz_files:
        sys.exit(f"没找到 {src_dir}/*.jsonl.gz —— 先跑 download.py people_daily")

    count = 0
    with open(output, "w", encoding="utf-8") as out:
        for gz in gz_files:
            n = 0
            with gzip.open(gz, "rt", encoding="utf-8") as fin:
                for line in fin:
                    item = json.loads(line)
                    n += _write_doc(out,
                                    item.get("title", "").strip(),
                                    item.get("text", "").strip())
            count += n
            print(f"  {gz.name}: {n} 篇")
    print(f"共写入 {count} 篇 -> {output}")
    return output


def process_cnn_dailymail(output: Path | None = None) -> Path:
    """cnn_dailymail parquet(article/highlights)→ CnnDailyMail.documents.txt

    title 取 article 第一句(200 字符内的首个 ". "),其余为 content —— 跟
    Shiyu/Data 的切法保持一致,否则产出的 documents.txt 跟 v4-Large 实跑输入不一样。
    """
    import pandas as pd

    src = source.PRETRAIN_SOURCES["cnn_dailymail"]
    src_dir = src.dir()
    output = output or source.DERIVED_ROOT / source.DERIVED["cnn_dailymail_docs"]
    output.parent.mkdir(parents=True, exist_ok=True)

    pq_files = sorted(src_dir.rglob("*.parquet"))
    if not pq_files:
        sys.exit(f"没找到 {src_dir}/**/*.parquet —— 先跑 download.py cnn_dailymail")

    count = 0
    with open(output, "w", encoding="utf-8") as out:
        for pq in pq_files:
            df = pd.read_parquet(pq)
            n = 0
            for article in df.get("article", []):
                article = str(article or "").strip()
                if not article:
                    continue
                first_dot = article.find(". ")
                if 0 < first_dot < 200:
                    title = article[:first_dot + 1].strip()
                    content = article[first_dot + 2:].strip()
                else:
                    title, content = "", article
                n += _write_doc(out, title, content)
            count += n
            print(f"  {pq.parent.name}/{pq.name}: {n} 篇")
    print(f"共写入 {count} 篇 -> {output}")
    return output


COMMANDS = {
    "people_daily": process_people_daily,
    "cnn_dailymail": process_cnn_dailymail,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", choices=list(COMMANDS) + [[]],
                    help=f"可选: {', '.join(COMMANDS)}")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--output", type=Path, default=None,
                    help="覆盖输出路径(只在处理单个源时有意义)")
    args = ap.parse_args()

    names = list(COMMANDS) if args.all else args.names
    if not names:
        ap.error("给一个源名,或用 --all")

    for name in names:
        print(f"=== {name} ===")
        existing = source.derived_path(f"{name}_docs")
        if args.output is None and existing.exists():
            print(f"  已存在: {existing}({existing.stat().st_size / 1e9:.1f} GB)"
                  f",跳过。要重新生成请传 --output。")
            continue
        COMMANDS[name](args.output if len(names) == 1 else None)


if __name__ == "__main__":
    main()
