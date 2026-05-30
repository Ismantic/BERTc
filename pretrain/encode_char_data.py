"""中文语料 → [N, seq_len] int32 tensor,用 PieceTokenizer encode。

PieceTokenizer 训练时若用 --method sentencepiece --cn-dict no,词表里:
  - 中文按字独立 piece
  - 英文 BPE
  - 0 字符 → byte fallback(SP 内置 256 个 <0xXX>)

inference 不需要传 cn-dict(词表里没有中文 N-gram piece,BPE 也 merge 不出来)。

用法:
  python encode_char_data.py \
      --corpus /path/to/cn.txt \
      --tokenizer_model /path/to/piece.model \
      --output /path/to/train.pt \
      --seq_len 512 --total_tokens 5000000000
"""
import argparse, os, random, time
import numpy as np
import torch
import multiprocessing as mp


def iter_shuffled_lines(corpus_path, seed=42):
    """先 scan corpus 收集每行 byte offset → shuffle → seek-read 按 shuffled 顺序流式 yield。

    内存:offsets 列表 (64M lines × 8 byte = ~512 MB),不读 line 内容。
    适合 corpus 远大于内存的情况(避免读全 file 进 list)。
    """
    print("  [shuffle] scanning line offsets...", flush=True)
    t0 = time.time()
    offsets = []
    with open(corpus_path, "rb") as f:
        pos = 0
        while True:
            line = f.readline()
            if not line: break
            offsets.append(pos)
            pos += len(line)
    print(f"  [shuffle] {len(offsets):,} lines in {time.time()-t0:.0f}s, "
          f"shuffling (seed={seed})...", flush=True)
    random.Random(seed).shuffle(offsets)
    print(f"  [shuffle] done shuffling, streaming in random order", flush=True)

    with open(corpus_path, "rb") as f:
        for off in offsets:
            f.seek(off)
            line = f.readline()
            yield line.decode("utf-8", errors="replace")


def worker(args):
    """每个 worker 处理一段文档,返回 list[int] token ids。"""
    docs, tok_model = args
    import piece_tokenizer as pt
    tok = pt.Tokenizer()
    tok.load(tok_model)
    out = []
    for doc in docs:
        ids = tok.encode_as_ids(doc)
        if ids:
            out.extend(ids)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--tokenizer_model", required=True, help="piece.model")
    ap.add_argument("--output", required=True)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--total_tokens", type=int, required=True)
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--batch_docs", type=int, default=10000,
                    help="每批送给 worker 的 doc 数")
    ap.add_argument("--shuffle_docs", action="store_true",
                    help="按行随机抽样(避免只覆盖 corpus 前缀)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    target_chunks = max(1, args.total_tokens // args.seq_len)
    print(f"Target: {target_chunks:,} chunks × {args.seq_len} = "
          f"{target_chunks*args.seq_len:,} tokens")
    print(f"Tokenizer: {args.tokenizer_model}")
    print(f"Workers: {args.num_workers}")

    # 用 memmap 直接背靠 disk,避免 32G out array 全部驻 RAM(超 8B token 时会 OOM kill)
    tmp_path = args.output + ".tmp"
    out = np.memmap(tmp_path, dtype=np.int32, mode="w+",
                    shape=(target_chunks, args.seq_len))
    buf, n_filled = [], 0
    t0 = time.time()
    n_docs = 0

    # 流式读 + 批量送给 worker pool
    pool = mp.get_context("spawn").Pool(args.num_workers)
    batches_inflight = []  # list of (n_docs_in_batch, AsyncResult)

    def flush_one(r, ndocs):
        nonlocal n_filled, n_docs, buf
        ids = r.get()
        buf.extend(ids)
        n_docs += ndocs
        while len(buf) >= args.seq_len and n_filled < target_chunks:
            out[n_filled] = buf[:args.seq_len]
            del buf[:args.seq_len]
            n_filled += 1
        if n_docs % 200000 < ndocs:  # 大致每 200k docs log 一次
            print(f"  {n_docs:,} docs | {n_filled:,}/{target_chunks:,} chunks "
                  f"| {time.time()-t0:.0f}s", flush=True)
        return n_filled >= target_chunks

    def split_batches(docs):
        """把 docs 列表按 workers 数分成均等子列表。"""
        per = max(1, len(docs) // args.num_workers)
        return [docs[i:i+per] for i in range(0, len(docs), per)]

    done = False
    docs_buf = []
    # 选择 line iterator: shuffle = offset-shuffle 随机顺序;否则顺序流式
    if args.shuffle_docs:
        line_iter = iter_shuffled_lines(args.corpus, seed=args.seed)
    else:
        def _seq_iter():
            with open(args.corpus, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    yield line
        line_iter = _seq_iter()
    for line in line_iter:
        line = line.strip()
        if line:
            docs_buf.append(line)
        if len(docs_buf) >= args.batch_docs:
            sub_batches = split_batches(docs_buf)
            results = [pool.apply_async(worker, ((sb, args.tokenizer_model),))
                       for sb in sub_batches]
            for r, sb in zip(results, sub_batches):
                if flush_one(r, len(sb)):
                    done = True; break
            docs_buf = []
            if done: break
    # 收尾:最后一批
    if not done and docs_buf:
        sub_batches = split_batches(docs_buf)
        results = [pool.apply_async(worker, ((sb, args.tokenizer_model),))
                   for sb in sub_batches]
        for r, sb in zip(results, sub_batches):
            if flush_one(r, len(sb)):
                break

    pool.close(); pool.join()

    # flush memmap + truncate 到实际 n_filled + 写 .meta sidecar(用于 train load)
    out.flush()
    del out
    actual_bytes = n_filled * args.seq_len * 4  # int32 = 4 byte
    os.truncate(tmp_path, actual_bytes)
    os.rename(tmp_path, args.output)
    import json
    with open(args.output + ".meta", "w") as f:
        json.dump({"shape": [n_filled, args.seq_len], "dtype": "int32"}, f)
    print(f"\nDone in {time.time()-t0:.0f}s")
    print(f"  {n_docs:,} docs scanned")
    print(f"  saved {n_filled:,} × {args.seq_len} = {n_filled*args.seq_len:,} tokens → {args.output}")
    print(f"  file size: {os.path.getsize(args.output)/1e9:.2f}GB")


if __name__ == "__main__":
    main()
