"""预训练语料 → 定长 chunk 的三个平行文件。

产出(src/data.py 的 PackedMLMDataset 读):
  <out>        int32 [N, seq_len]  token id
  <out>.wid    int32 [N, seq_len]  词 id,整词掩码用
  <out>.seg    uint8 [N, seq_len]  chunk 内的文档序号,跨文档 attention 隔离用
  各自带一个 .meta(json,记 dtype 和 shape)

中英按文档级加权轮询混合(默认 18:10),token 比例约 6:4 —— 英文文档平均比
中文长,所以文档数的比例和 token 的比例不是一回事。

三件容易出错的事:

  **不能逐字编码。** 字模式下 tokenizer 只对中文一字一 token,英文单词整体成
  一个 piece、空格成 ▁。所以这里整串 encode,再把 piece 按字符游标对回原文
  才能拿到词边界。(MT / CSC 是纯中文短句,才可以逐字编。)

  **词 id 要跨文档唯一。** 同一个 chunk 里可能塞了好几个文档,词 id 撞了的话
  WWM 会把两个文档的字当成同一个词一起掩掉。

  **文档序号在 chunk 内重新编号。** seg 是 uint8,存的是"本 chunk 内第几个
  文档",不是全局文档号。

用法:
    python -m prepare.encode_corpus --output prepare/corpus/train.pt \\
        --target_tokens 20_000_000_000 --num_workers 14
    python -m prepare.encode_corpus --output /tmp/smoke.pt \\
        --target_tokens 2_000_000 --num_workers 4      # 冒烟
"""
import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import piece_tokenizer as _pt
import wapic as _wapic

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))
import source as corpus                                          # noqa: E402

def default_wapic_model() -> Path:
    """Wapic 的分词模型。install_deps.sh 把仓库 clone 到 deps/,模型下到它的 data/model/。"""
    import os
    env = os.environ.get("BERTC_WAPIC_MODEL")
    if env:
        return Path(env)
    deps = Path(os.environ.get("BERTC_DEPS_DIR", str(ROOT / "deps")))
    return deps / "Wapic" / "data" / "model" / "wapic-cws.wac"


MIN_DOC_CHARS = 30
MAX_DOC_CHARS = 50_000
WS_RE = re.compile(r"\s+")

# 语料混合。v4-Large 的实跑配方,用量见 data/source.py 的注册表。
ZH_SOURCES = [
    ("finewiki_zh", "parquet"),
    ("people_daily", "documents"),
    ("skypile", "jsonl"),
    ("cci3", "jsonl"),
    ("fineweb_edu_zh", "parquet"),
]
EN_SOURCES = [
    ("cnn_dailymail", "documents"),
    ("finewiki_en", "parquet"),
]
TEXT_COLUMNS = ("text", "content", "raw_content", "document")


# ---------------------------------------------------------------- worker

_WAPIC = None
_PIECE = None


def init_worker(wapic_model: str, piece_model: str) -> None:
    """每个 worker 进程各自建一份分词器和 tokenizer。

    对象必须在子进程里构造 —— C++ 扩展对象 pickle 不过去,而且用的是 spawn
    上下文。模块本身在顶部 import 即可,spawn 的子进程会重新 import 本模块。
    """
    global _WAPIC, _PIECE
    _WAPIC = _wapic.Segmenter(wapic_model)
    _PIECE = _pt.Tokenizer()
    _PIECE.load(piece_model, dict="no")


def worker(text):
    """一篇文档 → (token id 列表, 词 id 列表),两者等长。切不动就返回 None。"""
    if not isinstance(text, str) or len(text) < MIN_DOC_CHARS:
        return None
    text = WS_RE.sub(" ", text).strip()
    if len(text) < MIN_DOC_CHARS:
        return None
    if len(text) > MAX_DOC_CHARS:
        text = text[:MAX_DOC_CHARS]

    try:
        ids = _PIECE.encode_as_ids(text)
        pieces = _PIECE.encode_as_pieces(text)
    except Exception:                                            # noqa: BLE001
        return None
    if not ids or len(ids) != len(pieces):
        return None

    # word_starts 直接给原文里每个词首的字符下标,末尾带一个 len(text) 哨兵。
    try:
        starts = _WAPIC.word_starts(text)
    except Exception:                                            # noqa: BLE001
        starts = list(range(len(text) + 1))                      # 退化成一字一词
    if not starts or starts[-1] != len(text):
        starts = list(starts) + [len(text)]

    char_wid = [0] * len(text)
    for w in range(len(starts) - 1):
        for i in range(starts[w], min(starts[w + 1], len(text))):
            char_wid[i] = w

    # piece 按字符游标对回原文:▁ 是 SentencePiece 的空格标记,吃掉原文里的空格
    out_ids, out_wids = [], []
    cursor = 0
    for piece, tid in zip(pieces, ids):
        if piece.startswith("▁"):
            while cursor < len(text) and text[cursor] == " ":
                cursor += 1
            real = piece.replace("▁", "")
        else:
            real = piece
        if not real:
            continue
        if cursor + len(real) > len(text):
            break
        out_ids.append(tid)
        out_wids.append(char_wid[cursor])
        cursor += len(real)
    return (out_ids, out_wids) if out_ids else None


# ---------------------------------------------------------------- 读语料

def iter_parquet(paths, columns=TEXT_COLUMNS):
    import pyarrow.parquet as pq
    for p in paths:
        pf = pq.ParquetFile(p)
        col = next((c for c in columns if c in pf.schema_arrow.names), None)
        if col is None:
            print(f"  ! {p} 里没有 {columns} 任何一列,跳过", flush=True)
            continue
        for batch in pf.iter_batches(batch_size=1024, columns=[col]):
            for t in batch.column(0).to_pylist():
                if t:
                    yield t


def iter_jsonl(paths, columns=TEXT_COLUMNS):
    for p in paths:
        with open(p, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for c in columns:
                    if obj.get(c):
                        yield obj[c]
                        break


def iter_documents(paths):
    """documents.txt:一行一篇。"""
    for p in paths:
        with open(p, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line


def source_files(name: str, kind: str) -> list:
    """从 data/source.py 的注册表解析出文件列表。"""
    if kind == "documents":
        key = {"people_daily": "people_daily_docs",
               "cnn_dailymail": "cnn_dailymail_docs"}[name]
        p = corpus.derived_path(key)
        return [p] if p.exists() else []
    return corpus.PRETRAIN_SOURCES[name].files()


def build_side(specs, label: str):
    """把一侧(中或英)的多个源拼成一个文档流,各源轮流出一篇。"""
    gens, desc = [], []
    for name, kind in specs:
        files = source_files(name, kind)
        if not files:
            print(f"  ! {name}: 没有文件,跳过(跑 python data/download.py {name})")
            continue
        reader = {"parquet": iter_parquet, "jsonl": iter_jsonl,
                  "documents": iter_documents}[kind]
        gens.append(reader(files))
        desc.append(f"{name}({len(files)})")
    print(f"  {label}: {', '.join(desc) if desc else '(空)'}")
    return gens


def round_robin(gens):
    """各源轮流出一篇,出完的踢掉。"""
    live = list(gens)
    while live:
        nxt = []
        for g in live:
            t = next(g, None)
            if t is not None:
                yield t
                nxt.append(g)
        live = nxt


def interleave(zh, en, zh_per_round: int, en_per_round: int):
    """中英按文档数加权轮询。"""
    zh_done = en_done = False
    while not (zh_done and en_done):
        for _ in range(zh_per_round):
            if zh_done:
                break
            t = next(zh, None)
            if t is None:
                zh_done = True
                break
            yield t
        for _ in range(en_per_round):
            if en_done:
                break
            t = next(en, None)
            if t is None:
                en_done = True
                break
            yield t


def batched(it, n):
    batch = []
    for x in it:
        batch.append(x)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


# ---------------------------------------------------------------- main

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", required=True)
    p.add_argument("--target_tokens", type=int, default=20_000_000_000)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--zh_per_round", type=int, default=18)
    p.add_argument("--en_per_round", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=14)
    p.add_argument("--batch_docs", type=int, default=2048)
    p.add_argument("--chunksize", type=int, default=32)
    p.add_argument("--wapic_model", default=None,
                   help="默认自动定位 deps/Wapic/data/model/wapic-cws.wac")
    p.add_argument("--piece_model", default=None,
                   help="默认自动定位 PieceTokenizer 仓库里的 BERTc-Tokenizer.pt")
    p.add_argument("--overalloc", type=float, default=1.10,
                   help="预分配倍数。写不满会截断,写满了会提前停")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.wapic_model is None:
        args.wapic_model = str(default_wapic_model())
    if args.piece_model is None:
        from .tokenizer import default_piece_model
        args.piece_model = str(default_piece_model())
    if not Path(args.piece_model).exists():
        sys.exit(f"找不到词表 {args.piece_model} —— "
                 f"跑 bash prepare/install_deps.sh piece")
    if not Path(args.wapic_model).exists():
        sys.exit(f"没有分词模型 {args.wapic_model} —— "
                 f"跑 bash prepare/install_deps.sh wapic")

    print(f"=== encode_corpus {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    print(f"目标 {args.target_tokens / 1e9:.1f}B token,"
          f"中英文档比 {args.zh_per_round}:{args.en_per_round}")
    print(f"输出 {out_path}")
    zh_gens = build_side(ZH_SOURCES, "中文")
    en_gens = build_side(EN_SOURCES, "英文")
    if not zh_gens and not en_gens:
        sys.exit("一个语料都没有,先跑 python data/download.py --pretrain")

    docs = interleave(round_robin(zh_gens), round_robin(en_gens),
                      args.zh_per_round, args.en_per_round)

    est = int(args.target_tokens / args.seq_len * args.overalloc)
    print(f"预分配 {est:,} chunk × {args.seq_len} "
          f"= {est * args.seq_len / 1e9:.1f}B 槽位")
    ids_mm = np.memmap(out_path, dtype=np.int32, mode="w+", shape=(est, args.seq_len))
    wid_mm = np.memmap(str(out_path) + ".wid", dtype=np.int32, mode="w+",
                       shape=(est, args.seq_len))
    seg_mm = np.memmap(str(out_path) + ".seg", dtype=np.uint8, mode="w+",
                       shape=(est, args.seq_len))

    buf_ids: list[int] = []
    buf_wid: list[int] = []
    buf_seg: list[int] = []
    written = 0
    total_tokens = 0
    wid_base = 0          # 词 id 的全局偏移,保证跨文档不撞
    doc_no = 0
    t0 = last_report = time.time()

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(args.num_workers, initializer=init_worker,
                    initargs=(args.wapic_model, args.piece_model))
    try:
        for batch in batched(docs, args.batch_docs):
            for result in pool.imap_unordered(worker, batch,
                                              chunksize=args.chunksize):
                if result is None:
                    continue
                ids, wids = result
                buf_ids.extend(ids)
                buf_wid.extend(w + wid_base for w in wids)
                wid_base += (max(wids) + 1) if wids else 1
                buf_seg.extend([doc_no] * len(ids))
                doc_no += 1

                while len(buf_ids) >= args.seq_len and written < est:
                    ids_mm[written] = buf_ids[:args.seq_len]
                    wid_mm[written] = buf_wid[:args.seq_len]
                    # 文档号在 chunk 内重编成 0,1,2...(uint8 上限 255,
                    # 实际一个 chunk 里不会超过几十篇)
                    local, remapped = {}, np.empty(args.seq_len, dtype=np.uint8)
                    for i, s in enumerate(buf_seg[:args.seq_len]):
                        if s not in local:
                            local[s] = min(len(local), 255)
                        remapped[i] = local[s]
                    seg_mm[written] = remapped
                    written += 1
                    total_tokens += args.seq_len
                    buf_ids = buf_ids[args.seq_len:]
                    buf_wid = buf_wid[args.seq_len:]
                    buf_seg = buf_seg[args.seq_len:]
                    if buf_wid:      # 词 id 归零,避免无限增长
                        off = buf_wid[0]
                        buf_wid = [w - off for w in buf_wid]
                        wid_base = max(buf_wid) + 1

                if written >= est:
                    break

            if written >= est or total_tokens >= args.target_tokens:
                break

            now = time.time()
            if now - last_report > 30:
                el = now - t0
                rate = total_tokens / el
                eta = (args.target_tokens - total_tokens) / max(1.0, rate) / 60
                print(f"  [{el:.0f}s] {written:,}/{est:,} chunk  "
                      f"{total_tokens / 1e9:.2f}B token  {rate / 1e6:.1f}M tok/s  "
                      f"ETA {eta:.0f} 分钟", flush=True)
                last_report = now
    finally:
        pool.close()
        pool.join()

    ids_mm.flush(); wid_mm.flush(); seg_mm.flush()
    del ids_mm, wid_mm, seg_mm

    if written < est:
        os.truncate(out_path, written * args.seq_len * 4)
        os.truncate(str(out_path) + ".wid", written * args.seq_len * 4)
        os.truncate(str(out_path) + ".seg", written * args.seq_len * 1)

    meta = {"shape": [written, args.seq_len], "seq_len": args.seq_len,
            "chunks_written": written, "tokens": written * args.seq_len,
            "dtype": "int32"}
    Path(str(out_path) + ".meta").write_text(json.dumps(meta, indent=2))
    Path(str(out_path) + ".wid.meta").write_text(json.dumps(meta, indent=2))
    Path(str(out_path) + ".seg.meta").write_text(
        json.dumps(dict(meta, dtype="uint8"), indent=2))

    print(f"\n=== 完成 {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    print(f"{written:,} chunk / {written * args.seq_len:,} token "
          f"({written * args.seq_len / 1e9:.2f}B),{doc_no:,} 篇文档")
    print(f"用时 {(time.time() - t0) / 60:.1f} 分钟")


if __name__ == "__main__":
    main()
