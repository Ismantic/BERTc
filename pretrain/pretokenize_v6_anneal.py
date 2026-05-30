"""L3-mix parquet → Wapic 切词 → v6 piece char-id + word-id → train_v6_anneal.pt(+wid)

输出 2 个 memmap:
  train_v6_anneal.pt      [N, seq_len] int32  char token id (v6 piece tokenizer)
  train_v6_anneal.pt.wid  [N, seq_len] int32  word_id 同 word 内 char 共享,WWM mask 用
  + .pt.meta JSON (dtype/shape/seq_len)

WWM mask 逻辑(下游 train_bert_mlm.py 用):
  - 同 word_id 的 chars 当一组,15% 整组选中 → 整组 mask
  - 跨 chunk 的 word_id 不连续(chunk 间独立)

使用:
  python pretokenize_v6_anneal.py --target_tokens 2_000_000_000 --num_workers 16
"""
import argparse, glob, json, os, time, multiprocessing as mp
import numpy as np
import pyarrow.parquet as pq


def init_worker(model_path, piece_path):
    """worker 进程独立加载 wapic + piece tokenizer。"""
    import wapic
    import piece_tokenizer as pt
    global _WAPIC, _PIECE
    _WAPIC = wapic.Segmenter(model_path)
    _PIECE = pt.Tokenizer()
    _PIECE.load(piece_path)


def worker(text):
    """完美 WWM pretokenize:
      a) normalize 空白(多空格合一 + strip),跟 SentencePiece 内部 normalize 对齐
      b) 整段 piece encode(产生 BPE + ▁ 空格 marker,跟 v6 训练时一致)
      c) 构 char-level wid:
         - 空格(' ')自成 1 wid(被 piece 编为 ▁)
         - 全英文 segment 整体 1 wid
         - 含中文 segment 用 wapic CRF 切,每词独立 wid
      d) piece-by-piece 推算 char span,取 piece 起点 char 的 wid
    输出 token id sequence == 原 text 整段过 SentencePiece 的结果(含 ▁)。
    """
    if not isinstance(text, str) or len(text) < 30:
        return None
    import re as _re
    text = _re.sub(r'\s+', ' ', text).strip()
    if len(text) < 30: return None
    if len(text) > 50_000: text = text[:50_000]

    # a+b) 整段 piece encode
    try:
        ids = _PIECE.encode_as_ids(text)
        pieces = _PIECE.encode_as_pieces(text)
    except Exception:
        return None
    if not ids: return None

    # c) 构 char-level wid (len(text) 个 wid,1 char 1 wid)
    char_wids = [0] * len(text)
    word_id = 0
    i = 0
    while i < len(text):
        c = text[i]
        if c == ' ':
            char_wids[i] = word_id
            word_id += 1
            i += 1
            continue
        j = i
        while j < len(text) and text[j] != ' ':
            j += 1
        seg = text[i:j]
        has_chinese = any('一' <= ch <= '鿿' for ch in seg)
        if not has_chinese:
            for k in range(i, j):
                char_wids[k] = word_id
            word_id += 1
        else:
            try:
                wchars, wtags = _WAPIC.tag(seg)
            except Exception:
                for k in range(i, j):
                    char_wids[k] = word_id
                word_id += 1
                i = j
                continue
            # wapic 应返回 len(seg) 个 char + tag
            cur_w = word_id
            for k in range(min(len(wtags), j - i)):
                char_wids[i + k] = cur_w
                if wtags[k] in ('E', 'S'):
                    cur_w += 1
            word_id = cur_w
        i = j

    # d) piece-by-piece 取 wid
    char_ids_out = []
    char_wids_out = []
    text_pos = 0
    for piece, tid in zip(pieces, ids):
        # piece 占的 char 数:
        #   '▁' 单独 → 1 char(空格)
        #   其他 piece → len(piece) char(每 unicode char 1 项)
        if piece == '▁':
            span = 1
        else:
            span = len(piece)
        if text_pos < len(char_wids):
            wid = char_wids[text_pos]
        else:
            wid = word_id
        char_ids_out.append(tid)
        char_wids_out.append(wid)
        text_pos += span

    if not char_ids_out: return None
    return char_ids_out, char_wids_out


def iter_parquet_texts(parquet_paths):
    """按 parquet 顺序 yield text strings,自动找 text/content/document 字段。"""
    for p in parquet_paths:
        pf = pq.ParquetFile(p)
        # 推断字段名(每个 file 看 schema)
        cols = [f.name for f in pf.schema_arrow]
        text_field = next((c for c in ('content','text','document','data') if c in cols), None)
        if text_field is None:
            print(f"  WARN no text col in {p} (cols={cols})", flush=True)
            continue
        for batch in pf.iter_batches(batch_size=1024, columns=[text_field]):
            for row in batch.to_pylist():
                t = row.get(text_field)
                if t: yield t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l3_mix_root", default="/home/tfbao/Shiyu/data/L3-mix")
    ap.add_argument("--wapic_model", default="/home/tfbao/Shiyu/Wapic/data/wapic-20260529.wac")
    ap.add_argument("--piece_model", default="/home/tfbao/Shiyu/Summer/BERT/bert_train_v6_mid/piece.model")
    ap.add_argument("--output", default="/home/tfbao/Shiyu/data/train_v6_anneal.pt")
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--target_tokens", type=int, default=2_000_000_000)
    ap.add_argument("--num_workers", type=int, default=14)
    ap.add_argument("--batch_docs", type=int, default=2048)
    ap.add_argument("--chunksize", type=int, default=32)
    ap.add_argument("--limit_parquets", type=int, default=0,
                    help="如非 0,只用前 N parquet(debug)")
    args = ap.parse_args()

    # 列 4 个 config 所有 parquet,interleave 保多样性
    configs = [
        "data/ultrafineweb_zh_l3/qa",
        "data/ultrafineweb_zh_l3/multi_style",
        "data/ultrafineweb_en_l3/qa",
        "data/ultrafineweb_en_l3/multi_style",
    ]
    files_per_cfg = []
    for c in configs:
        ps = sorted(glob.glob(os.path.join(args.l3_mix_root, c, "*.parquet")))
        if args.limit_parquets: ps = ps[:args.limit_parquets]
        files_per_cfg.append(ps)
        print(f"  {c}: {len(ps)} parquets")
    # interleave
    parquets = []
    for i in range(max(len(x) for x in files_per_cfg)):
        for fs in files_per_cfg:
            if i < len(fs): parquets.append(fs[i])
    print(f"Total parquets to process: {len(parquets)}")

    # output memmap
    target_chunks = args.target_tokens // args.seq_len
    print(f"Target chunks: {target_chunks:,} × seq_len {args.seq_len} = {target_chunks*args.seq_len:,} tokens")
    out_ids = np.memmap(args.output, dtype=np.int32, mode='w+',
                        shape=(target_chunks, args.seq_len))
    # wid 用 int64:跨 doc 累计 offset 容易溢出 int32
    out_wids = np.memmap(args.output + ".wid", dtype=np.int64, mode='w+',
                         shape=(target_chunks, args.seq_len))

    # buffer
    buf_ids = []
    buf_wids = []
    wid_offset = 0  # 累计 word_id,跨 doc 不冲突
    chunks_written = 0

    pool = mp.Pool(args.num_workers,
                   initializer=init_worker,
                   initargs=(args.wapic_model, args.piece_model))

    t0 = time.time()
    docs_done = 0
    text_iter = iter_parquet_texts(parquets)

    def flush_buffer():
        """从 buf 填 chunks 写出。"""
        nonlocal chunks_written, buf_ids, buf_wids
        while len(buf_ids) >= args.seq_len and chunks_written < target_chunks:
            out_ids[chunks_written] = buf_ids[:args.seq_len]
            out_wids[chunks_written] = buf_wids[:args.seq_len]
            buf_ids = buf_ids[args.seq_len:]
            buf_wids = buf_wids[args.seq_len:]
            chunks_written += 1

    batch = []
    # 主进程改用 numpy array 累积,worker 直接返回 array — 减 Python list overhead
    while chunks_written < target_chunks:
        try:
            text = next(text_iter)
        except StopIteration:
            break
        batch.append(text)
        if len(batch) >= args.batch_docs:
            # 收集 worker 输出后,numpy concat 一次性
            cids_arrs = []
            cwids_arrs = []
            for r in pool.imap_unordered(worker, batch, chunksize=args.chunksize):
                if r is None: continue
                cids, cwids = r
                if not cwids: continue
                cids_arrs.append(np.asarray(cids, dtype=np.int32))
                # offset wids
                wid_arr = np.asarray(cwids, dtype=np.int64) + wid_offset
                cwids_arrs.append(wid_arr)
                wid_offset += int(max(cwids)) + 1
            if cids_arrs:
                # 把累计 buf 和新数据 concat 成一个连续 array
                cids_new = np.concatenate(cids_arrs)
                wids_new = np.concatenate(cwids_arrs)
                if buf_ids:
                    cids_new = np.concatenate([np.asarray(buf_ids, dtype=np.int32), cids_new])
                    wids_new = np.concatenate([np.asarray(buf_wids, dtype=np.int64), wids_new])
                    buf_ids = []; buf_wids = []
                # 切 chunks
                n_full = len(cids_new) // args.seq_len
                writable = min(n_full, target_chunks - chunks_written)
                if writable > 0:
                    end = writable * args.seq_len
                    out_ids[chunks_written:chunks_written+writable] = \
                        cids_new[:end].reshape(writable, args.seq_len)
                    out_wids[chunks_written:chunks_written+writable] = \
                        wids_new[:end].reshape(writable, args.seq_len)
                    chunks_written += writable
                # 留下尾巴下一轮拼
                tail_start = writable * args.seq_len
                if tail_start < len(cids_new):
                    buf_ids = cids_new[tail_start:].tolist()
                    buf_wids = wids_new[tail_start:].tolist()
            docs_done += len(batch)
            batch = []
            elapsed = time.time() - t0
            rate = chunks_written * args.seq_len / max(1, elapsed)
            eta = (target_chunks - chunks_written) * args.seq_len / max(1, rate) / 60
            print(f"  docs={docs_done:,} chunks={chunks_written:,}/{target_chunks:,} "
                  f"({100*chunks_written/target_chunks:.1f}%) "
                  f"{rate/1e6:.2f}M tok/s ETA {eta:.0f}m", flush=True)
    # 收尾
    if batch:
        results = pool.imap_unordered(worker, batch, chunksize=8)
        for r in results:
            if r is None: continue
            cids, cwids = r
            offset_wids = [w + wid_offset for w in cwids]
            if cwids: wid_offset += max(cwids) + 1
            buf_ids.extend(cids)
            buf_wids.extend(offset_wids)
        flush_buffer()
    pool.close(); pool.join()
    out_ids.flush(); out_wids.flush()

    # 写 meta(ids int32, wid int64)
    base_meta = {
        "shape": [target_chunks, args.seq_len],
        "seq_len": args.seq_len, "chunks_written": chunks_written,
        "tokens": chunks_written * args.seq_len,
    }
    with open(args.output + ".meta", "w") as f:
        json.dump({**base_meta, "dtype": "int32"}, f, indent=2)
    with open(args.output + ".wid.meta", "w") as f:
        json.dump({**base_meta, "dtype": "int64"}, f, indent=2)

    print(f"\nDone in {(time.time()-t0)/60:.1f}m")
    print(f"  {chunks_written:,} chunks × {args.seq_len} = {chunks_written*args.seq_len:,} tokens")
    print(f"  → {args.output}")
    print(f"  → {args.output}.wid")


if __name__ == "__main__":
    main()
