"""Modern BERTc v1 pretokenize — 15B token,中:英 = 6:4,Wapic WWM。

源(user 指定):
  cn 9B:
    - Wikipedia_cn  3 json files               ≈ 0.15B
    - PeopleDaily.documents.txt                ≈ 1.5B
    - SkyPile  00000-00013 全 14 parquet       ≈ 5.0B
    - CCI3-HQ  5 jsonl files                   ≈ 1.5B
    - Chinese-FineWeb-Edu 200 parquet           ≈ 0.9B  → 总 9B
  en 6B:
    - CnnDailyMail.documents.txt 全            ≈ 1.3B
    - Wikipedia_en 13 json files               ≈ 4.7B  → 总 6B

输出:
  data/train_modern.pt     [N, 512] int32  char token id (v6 piece tokenizer)
  data/train_modern.pt.wid [N, 512] int32  word_id (WWM)
  data/train_modern.pt.seg [N, 512] uint8  doc_id in chunk(cross-doc attention 隔离用)
                                            chunk 内按出现顺序 re-label 为 0,1,2,...
  + .meta JSON + .wid.meta JSON + .seg.meta JSON

WWM 规则同 v7_anneal:
  空格 → 1 word_id (▁)
  英文 segment → 1 word_id (整 BPE 组)
  含中文 → Wapic CRF 切词,每词独立 word_id
"""
import argparse, glob, json, os, time, multiprocessing as mp
import numpy as np
import pyarrow.parquet as pq


def init_worker(model_path, piece_path):
    import wapic
    import piece_tokenizer as pt
    global _WAPIC, _PIECE
    _WAPIC = wapic.Segmenter(model_path)
    _PIECE = pt.Tokenizer()
    _PIECE.load(piece_path)


def worker(text):
    """简洁版:piece 是 char-mode,每 piece = 1 char。Wapic 在 char-level 文本上切词决定 word_id。
       输出 (char_ids, word_ids)."""
    if not isinstance(text, str) or len(text) < 30:
        return None
    import re as _re
    text = _re.sub(r'\s+', ' ', text).strip()
    if len(text) < 30:
        return None
    if len(text) > 50_000:
        text = text[:50_000]

    try:
        ids = _PIECE.encode_as_ids(text)
        pieces = _PIECE.encode_as_pieces(text)
    except Exception:
        return None
    if not ids or len(ids) != len(pieces):
        return None

    # 用 wapic 切词:cut_smart(空白切段 → 英文整段不切 + 中文走 CRF)。
    # wapic 输出不含空白,需要重新对齐到原 text(含空白)。
    text_nows = "".join(c for c in text if not c.isspace())
    try:
        if hasattr(_WAPIC, 'cut_smart'):
            wapic_words = _WAPIC.cut_smart(text)
        else:
            wapic_words = _WAPIC.cut(text)
    except Exception:
        wapic_words = list(text_nows)
    rebuilt = "".join(wapic_words)
    if rebuilt != text_nows:
        # 失败 fallback:逐字符(每字一 word)
        wapic_words = list(text_nows)

    # Step 1: char_to_wid_nows 按 wapic_words 顺序对 text_nows 每个字符分配 word_id
    char_to_wid_nows = []
    wid = 0
    for w in wapic_words:
        for _ in w:
            char_to_wid_nows.append(wid)
        wid += 1
    # 此时 len(char_to_wid_nows) == len(text_nows)

    # Step 2: 映射回原 text(含空白)。空白复用下一个非空白字符的 word_id。
    char_to_wid = []
    nows_idx = 0
    last_wid = 0
    for c in text:
        if c.isspace():
            if nows_idx < len(char_to_wid_nows):
                char_to_wid.append(char_to_wid_nows[nows_idx])
            else:
                char_to_wid.append(last_wid)
        else:
            if nows_idx < len(char_to_wid_nows):
                cw = char_to_wid_nows[nows_idx]
                char_to_wid.append(cw)
                last_wid = cw
                nows_idx += 1
            else:
                char_to_wid.append(last_wid)
    # 现在 len(char_to_wid) == len(text)

    # piece 是 char-mode,每个 piece 对应原文中的 1-2 个字(▁ + 字)
    # 用 piece tokenizer 的 "encode + 反推到 char 偏移" — 简化:逐 piece scan
    char_ids_out = []
    char_wids_out = []
    cursor = 0  # 当前在 text 中的位置
    for piece, char_id in zip(pieces, ids):
        # 去掉 ▁(SentencePiece 空格标记)— 在 text 中对应一个空格 char
        if piece.startswith("▁"):
            # 跳过 text 中可能的前导空格
            while cursor < len(text) and text[cursor] == " ":
                cursor += 1
            real = piece.replace("▁", "")
        else:
            real = piece
        if not real:
            continue
        # 该 piece 对应 text[cursor : cursor + len(real)]
        if cursor + len(real) > len(text):
            break
        # 用 char_to_wid 的第一个字符的 word_id 当这个 piece 的 word_id
        w = char_to_wid[cursor] if cursor < len(char_to_wid) else wid
        char_ids_out.append(char_id)
        char_wids_out.append(w)
        cursor += len(real)
    return char_ids_out, char_wids_out


# ============ Source Iterators ============

def iter_parquet_text(parquets, text_col='text'):
    for p in parquets:
        pf = pq.ParquetFile(p)
        for batch in pf.iter_batches(batch_size=1024, columns=[text_col]):
            for t in batch.column(0).to_pylist():
                if t:
                    yield t


def iter_jsonl_text(json_paths, text_col='text'):
    """通用 jsonl:每行一个 json,取 text 字段."""
    for p in json_paths:
        with open(p, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = obj.get(text_col)
                if t:
                    yield t


def iter_txt_lines(txt_path):
    """每行一个 doc."""
    with open(txt_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def round_robin(*generators):
    """各 generator 轮流出 1 doc,直到全空."""
    done = [False] * len(generators)
    while not all(done):
        for i, g in enumerate(generators):
            if done[i]:
                continue
            t = next(g, None)
            if t is None:
                done[i] = True
            else:
                yield t


def iter_weighted(zh_gen, en_gen, zh_per_round=18, en_per_round=10):
    """中:英 = 18:10 doc-level weighted round-robin → token 比 ~ 6:4."""
    zh_done = en_done = False
    while not (zh_done and en_done):
        for _ in range(zh_per_round):
            if zh_done:
                break
            t = next(zh_gen, None)
            if t is None:
                zh_done = True
                break
            yield t
        for _ in range(en_per_round):
            if en_done:
                break
            t = next(en_gen, None)
            if t is None:
                en_done = True
                break
            yield t


# ============ Main ============

def main():
    ap = argparse.ArgumentParser()
    # cn sources
    ap.add_argument("--wiki_cn_root", default="/home/tfbao/a6000/Wikipedia_cn_json_files")
    ap.add_argument("--peopledaily_path",
                    default="/home/tfbao/Shiyu/Data/data/PeopleDaily.documents.txt")
    ap.add_argument("--skypile_root", default="/home/tfbao/a6000/SkyPile")
    ap.add_argument("--skypile_start", type=int, default=0)
    ap.add_argument("--skypile_end", type=int, default=13)
    ap.add_argument("--cci3_root", default="/home/tfbao/a6000/Summer-data/CCI3-HQ/data")
    ap.add_argument("--fineweb_root",
                    default="/home/tfbao/a6000/Summer-data/Chinese-FineWeb-Edu-V2.2/4_5")
    ap.add_argument("--fineweb_start", type=int, default=0)
    ap.add_argument("--fineweb_n", type=int, default=200)
    # en sources
    ap.add_argument("--cnndm_path",
                    default="/home/tfbao/Shiyu/Data/data/CnnDailyMail.documents.txt")
    ap.add_argument("--wiki_en_root", default="/home/tfbao/a6000/Wikipedia_en_json_files")
    ap.add_argument("--wiki_en_n", type=int, default=13)
    # tokenizer
    ap.add_argument("--wapic_model",
                    default="/home/tfbao/Shiyu/Wapic/data/wapic-20260602-h19_1-full.wac")
    ap.add_argument("--piece_model",
                    default="/home/tfbao/Shiyu/Summer/BERT/bert_train_v6_mid/piece.model")
    # output
    ap.add_argument("--output",
                    default="/home/tfbao/Shiyu/BERTc/pretrain/modern_bertc/data/train_modern.pt")
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--target_tokens", type=int, default=15_000_000_000)
    ap.add_argument("--num_workers", type=int, default=14)
    ap.add_argument("--batch_docs", type=int, default=2048)
    ap.add_argument("--chunksize", type=int, default=32)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print(f"=== Modern BERTc pretokenize start {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    print(f"target_tokens: {args.target_tokens / 1e9:.1f}B (zh:en = 6:4)")
    print(f"output: {args.output}")

    # 收集 cn 源文件
    wiki_cn_files = sorted(glob.glob(f"{args.wiki_cn_root}/*.json"))
    sky_files = sorted(glob.glob(f"{args.skypile_root}/*.parquet"))[
        args.skypile_start:args.skypile_end + 1
    ]
    cci3_files = sorted(glob.glob(f"{args.cci3_root}/*.jsonl"))
    all_fw = sorted(glob.glob(f"{args.fineweb_root}/*.parquet"))
    fw_files = all_fw[args.fineweb_start:args.fineweb_start + args.fineweb_n]
    pd_files = [args.peopledaily_path] if os.path.exists(args.peopledaily_path) else []

    # 英 源文件
    wiki_en_files = sorted(glob.glob(f"{args.wiki_en_root}/*.json"))[:args.wiki_en_n]
    cnndm_files = [args.cnndm_path] if os.path.exists(args.cnndm_path) else []

    print(f"\nCN sources:")
    print(f"  wiki_cn:   {len(wiki_cn_files)} files")
    print(f"  PeopleDaily: {len(pd_files)} files ({args.peopledaily_path})")
    print(f"  SkyPile:   {len(sky_files)} files ({args.skypile_start}-{args.skypile_end})")
    print(f"  CCI3-HQ:   {len(cci3_files)} files")
    print(f"  FineWeb:   {len(fw_files)} files ({args.fineweb_start}-{args.fineweb_start+args.fineweb_n-1})")
    print(f"EN sources:")
    print(f"  CnnDM:     {len(cnndm_files)} files")
    print(f"  Wiki_en:   {len(wiki_en_files)} files")

    # 构建 cn 多源 round-robin generator
    def make_zh_gen():
        return round_robin(
            iter_jsonl_text(wiki_cn_files),
            iter_txt_lines(args.peopledaily_path),
            iter_parquet_text(sky_files),
            iter_jsonl_text(cci3_files),
            iter_parquet_text(fw_files),
        )

    def make_en_gen():
        return round_robin(
            iter_txt_lines(args.cnndm_path),
            iter_jsonl_text(wiki_en_files),
        )

    doc_iter = iter_weighted(make_zh_gen(), make_en_gen())

    # memmap output
    # 估算总 chunks。如果 token=15B,seq=512 → ~29.3M chunks。预分配多 10% 安全
    est_chunks = int(args.target_tokens / args.seq_len * 1.10)
    print(f"\nPre-allocating {est_chunks:,} chunks × {args.seq_len} = "
          f"{est_chunks * args.seq_len / 1e9:.1f}B slots")
    out = np.memmap(args.output, dtype=np.int32, mode="w+",
                    shape=(est_chunks, args.seq_len))
    out_wid = np.memmap(args.output + ".wid", dtype=np.int32, mode="w+",
                        shape=(est_chunks, args.seq_len))
    out_seg = np.memmap(args.output + ".seg", dtype=np.uint8, mode="w+",
                        shape=(est_chunks, args.seq_len))

    # 池处理
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(args.num_workers, initializer=init_worker,
                    initargs=(args.wapic_model, args.piece_model))

    # streaming:积攒 doc batch → imap_unordered → 累 chunks
    buf_ids = []
    buf_wids = []
    buf_segs = []        # 每 token 的 全局 doc_id(切 chunk 时 re-label 为 chunk-local 0..N)
    written = 0
    total_tokens = 0
    word_id_global = 0   # 全局 word id 计数
    doc_seg_global = 0   # 全局 doc 计数(切 chunk 时 re-label)
    t0 = time.time()
    last_report = t0

    def batched(it, n):
        batch = []
        for item in it:
            batch.append(item)
            if len(batch) == n:
                yield batch
                batch = []
        if batch:
            yield batch

    try:
        for batch in batched(doc_iter, args.batch_docs):
            for result in pool.imap_unordered(worker, batch, chunksize=args.chunksize):
                if result is None:
                    continue
                ids, wids = result
                # 把 word_id 偏移到全局空间
                local_max = max(wids) if wids else 0
                wids_shifted = [w + word_id_global for w in wids]
                word_id_global += local_max + 1
                buf_ids.extend(ids)
                buf_wids.extend(wids_shifted)
                # 整个 doc 用同一个全局 seg id(切 chunk 时 re-label)
                buf_segs.extend([doc_seg_global] * len(ids))
                doc_seg_global += 1

                # 切 chunk
                while len(buf_ids) >= args.seq_len:
                    if written >= est_chunks:
                        break
                    out[written, :] = buf_ids[:args.seq_len]
                    out_wid[written, :] = buf_wids[:args.seq_len]
                    # seg: chunk 内按出现顺序 re-label 为 0..255 uint8
                    chunk_segs = buf_segs[:args.seq_len]
                    seg2local = {}
                    remapped = np.empty(args.seq_len, dtype=np.uint8)
                    for i, s in enumerate(chunk_segs):
                        if s not in seg2local:
                            n = len(seg2local)
                            seg2local[s] = min(n, 255)  # 防御 cap(实际 <50 docs/chunk)
                        remapped[i] = seg2local[s]
                    out_seg[written, :] = remapped
                    written += 1
                    buf_ids = buf_ids[args.seq_len:]
                    buf_wids = buf_wids[args.seq_len:]
                    buf_segs = buf_segs[args.seq_len:]
                    total_tokens += args.seq_len
                    # word_id chunk 间不连续:重置 buffer 的 word_id 偏移
                    if buf_wids:
                        # remap buf_wids 让它从 0 开始
                        offset = buf_wids[0]
                        buf_wids = [w - offset for w in buf_wids]
                        word_id_global = max(buf_wids) + 1

                if written >= est_chunks:
                    break

            if written >= est_chunks or total_tokens >= args.target_tokens:
                break

            now = time.time()
            if now - last_report > 30:
                elapsed = now - t0
                rate = total_tokens / elapsed / 1e6
                eta = (args.target_tokens - total_tokens) / max(1, total_tokens / elapsed) / 60
                print(f"  [{elapsed:.0f}s] written {written:,}/{est_chunks:,} chunks  "
                      f"{total_tokens / 1e9:.2f}B tokens  {rate:.1f}M tok/s  "
                      f"ETA {eta:.0f}min", flush=True)
                last_report = now
    finally:
        pool.close()
        pool.join()

    # truncate to actual size
    actual_chunks = written
    out.flush()
    out_wid.flush()
    out_seg.flush()
    del out
    del out_wid
    del out_seg

    # truncate file to exact size
    if actual_chunks < est_chunks:
        new_size_i32 = actual_chunks * args.seq_len * 4  # int32
        new_size_u8 = actual_chunks * args.seq_len * 1   # uint8
        os.truncate(args.output, new_size_i32)
        os.truncate(args.output + ".wid", new_size_i32)
        os.truncate(args.output + ".seg", new_size_u8)

    # write meta
    meta = {
        "shape": [actual_chunks, args.seq_len],
        "seq_len": args.seq_len,
        "chunks_written": actual_chunks,
        "tokens": actual_chunks * args.seq_len,
        "dtype": "int32",
    }
    seg_meta = dict(meta, dtype="uint8")
    with open(args.output + ".meta", "w") as f:
        json.dump(meta, f, indent=2)
    with open(args.output + ".wid.meta", "w") as f:
        json.dump(meta, f, indent=2)
    with open(args.output + ".seg.meta", "w") as f:
        json.dump(seg_meta, f, indent=2)

    print(f"\n=== DONE {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    print(f"chunks: {actual_chunks:,}  tokens: {actual_chunks * args.seq_len:,} "
          f"({actual_chunks * args.seq_len / 1e9:.2f}B)")
    print(f"time: {(time.time() - t0) / 60:.1f}min")


if __name__ == "__main__":
    main()
