"""src/data.py 与旧数据管线的对拍。

src/data.py 把「字→id + 标签构造」前移到了 prepare/,自己只读预编码 ids。
这里验证这次搬迁没有改变喂给模型的张量:

  1. MT —— 拿真实 pd98 jsonl,用旧的 MTDataset + MTCollator(带 tokenizer)
     产出 batch;再把同一批样本预编码成新格式,用新的 MTDataset + MTCollator
     产出 batch,逐张量比对。
  2. CSC —— 同样的思路,用真实 all_pairs.pkl 的一个切片。
  3. _memmap —— torch.from_file 取代 numpy.memmap,拿真实 data3 语料
     跟 numpy 读出来的值比对(numpy 只在测试里用,src/ 不依赖)。

    python tests/test_data.py
"""
import pickle
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import data as new_data                                          # noqa: E402

OLD_MT = ROOT / "finetune" / "NLP_BERT_CRF"
TOKENIZER = ROOT / "pretrain" / "modern_bertc" / "tokenizer"
PD98 = OLD_MT / "data"
CSC_PKL = ROOT / "csc" / "data" / "all_pairs.pkl"
CORPUS = ROOT / "pretrain" / "modern_bertc" / "data3" / "train_v3.pt"

N_SAMPLES = 512          # 对拍用的样本数,够覆盖长句截断和各种标签


def _pack(items, fields, extra):
    """items: list[dict[str, list[int]]] → 扁平数组 + offsets 的 blob。"""
    offsets = [0]
    cols = {f: [] for f in fields}
    for it in items:
        n = len(it[fields[0]])
        for f in fields:
            assert len(it[f]) == n, f"{f} 长度对不上"
            cols[f].extend(it[f])
        offsets.append(offsets[-1] + n)
    blob = {"offsets": torch.tensor(offsets, dtype=torch.int64)}
    for f in fields:
        dtype = torch.uint8 if f == "det_labels" else torch.int32
        blob[f] = torch.tensor(cols[f], dtype=dtype)
    blob.update(extra)
    return blob


def test_mt(tmp: Path) -> int:
    # 旧的 NLP_BERT_CRF/data.py 和新的 src/data.py 同名。data_mt.py 内部
    # `from data import TAG2ID`,所以必须让 OLD_MT 排在 src/ 前面,
    # 并且先把已经加载的 src/data 从 sys.modules 挪开。
    sys.path.insert(0, str(OLD_MT))
    shadowed = sys.modules.pop("data", None)
    try:
        from data_mt import MTDataset as OldMT, MTCollator as OldCollator
        from data_pos_ner import build_pos_vocab, NER_TAGS
        from data import TAG2ID as CWS_TAG2ID
        from piece_tokenizer_adapter import PieceTokenizerAdapter
    except Exception as e:                                       # noqa: BLE001
        print(f"  旧 MT 模块导入失败({e}),跳过。")
        return 0
    finally:
        if shadowed is not None:
            sys.modules["data"] = shadowed
    if not (PD98 / "cws.pd98.jsonl").exists():
        print("  pd98 jsonl 不存在,跳过。")
        return 0

    pos2id = build_pos_vocab()
    tok = PieceTokenizerAdapter(str(TOKENIZER))

    # 不截断地建一份,用来生成预编码文件
    full = OldMT(PD98 / "cws.pd98.jsonl", PD98 / "pos.pd98.jsonl",
                 PD98 / "ner.pd98.jsonl", pos2id, max_chars=10 ** 9)
    items = full.items[:N_SAMPLES]

    coll_old = OldCollator(tok)
    encoded = []
    for it in items:
        ids = [coll_old._char_to_id(c) for c in it["chars"]]
        encoded.append({"input_ids": ids, "cws_tags": it["cws_tags"],
                        "pos_tags": it["pos_tags"], "ner_tags": it["ner_tags"]})

    cws_vocab = [t for t, _ in sorted(CWS_TAG2ID.items(), key=lambda kv: kv[1])]
    pos_vocab = [t for t, _ in sorted(pos2id.items(), key=lambda kv: kv[1])]
    ner_vocab = [t for t, _ in sorted(NER_TAGS.items(), key=lambda kv: kv[1])]
    blob = _pack(encoded, new_data.MTDataset.FIELDS,
                 {"format": "bertc-mt-v1", "cws_vocab": cws_vocab,
                  "pos_vocab": pos_vocab, "ner_vocab": ner_vocab,
                  "pad_token_id": tok.pad_token_id})
    path = tmp / "mt.pt"
    torch.save(blob, path)

    failures = 0
    for max_chars in (254, 64):
        old_ds = OldMT(PD98 / "cws.pd98.jsonl", PD98 / "pos.pd98.jsonl",
                       PD98 / "ner.pd98.jsonl", pos2id, max_chars=max_chars)
        old_batch = coll_old([old_ds.items[i] for i in range(N_SAMPLES)])

        new_ds = new_data.MTDataset(path, max_chars=max_chars)
        new_coll = new_data.MTCollator(tok.pad_token_id)
        new_batch = new_coll([new_ds[i] for i in range(N_SAMPLES)])

        for key in old_batch:
            if not torch.equal(old_batch[key], new_batch[key]):
                failures += 1
                d = (old_batch[key] != new_batch[key]).sum().item()
                print(f"  ✗ MT max_chars={max_chars}: {key} 有 {d} 个位置不同")
                break
        else:
            n_trunc = sum(1 for it in items if len(it["chars"]) > max_chars)
            print(f"  ✓ MT max_chars={max_chars}: 5 个张量全等 "
                  f"({N_SAMPLES} 条,其中 {n_trunc} 条被截断)")
    return failures


def test_csc(tmp: Path) -> int:
    sys.path.insert(0, str(OLD_MT))
    try:
        from piece_tokenizer_adapter import PieceTokenizerAdapter
    except Exception as e:                                       # noqa: BLE001
        print(f"  tokenizer 导入失败({e}),跳过。")
        return 0
    if not CSC_PKL.exists():
        print("  all_pairs.pkl 不存在,跳过。")
        return 0

    tok = PieceTokenizerAdapter(str(TOKENIZER))
    cache = {}

    def char_to_id(c):
        if c not in cache:
            ids = tok.encode(c, add_special_tokens=False)
            cache[c] = ids[0] if ids else tok.unk_token_id
        return cache[c]

    with open(CSC_PKL, "rb") as f:
        pairs = pickle.load(f)
    # 挑长短混合的一批,保证覆盖截断
    pairs = sorted(pairs[:5000], key=lambda p: -len(p[0]))[:N_SAMPLES]

    encoded = []
    for src, tgt in pairs:
        L = min(len(src), len(tgt))
        encoded.append({
            "input_ids": [char_to_id(c) for c in src[:L]],
            "cor_labels": [char_to_id(c) for c in tgt[:L]],
            "det_labels": [1 if src[i] != tgt[i] else 0 for i in range(L)],
        })
    blob = _pack(encoded, new_data.CSCDataset.FIELDS,
                 {"format": "bertc-csc-v1", "pad_token_id": tok.pad_token_id})
    path = tmp / "csc.pt"
    torch.save(blob, path)

    failures = 0
    for max_len in (128, 32):
        # 旧路径:CSCDataset.__getitem__ 里现算
        old_items = []
        for src, tgt in pairs:
            L = min(len(src), len(tgt), max_len)
            old_items.append({
                "input_ids": [char_to_id(c) for c in src[:L]],
                "cor_labels": [char_to_id(c) for c in tgt[:L]],
                "det_labels": [1.0 if src[i] != tgt[i] else 0.0 for i in range(L)],
                "length": L,
            })
        B = len(old_items)
        max_l = max(it["length"] for it in old_items)
        old_batch = {
            "input_ids": torch.full((B, max_l), tok.pad_token_id, dtype=torch.long),
            "attention_mask": torch.zeros((B, max_l), dtype=torch.long),
            "cor_labels": torch.full((B, max_l), -100, dtype=torch.long),
            "det_labels": torch.zeros((B, max_l), dtype=torch.float),
        }
        for i, it in enumerate(old_items):
            n = it["length"]
            old_batch["input_ids"][i, :n] = torch.tensor(it["input_ids"])
            old_batch["attention_mask"][i, :n] = 1
            old_batch["cor_labels"][i, :n] = torch.tensor(it["cor_labels"])
            old_batch["det_labels"][i, :n] = torch.tensor(it["det_labels"])

        new_ds = new_data.CSCDataset(path, max_len=max_len)
        new_batch = new_data.CSCCollator(tok.pad_token_id)(
            [new_ds[i] for i in range(len(pairs))])

        for key in old_batch:
            if not torch.equal(old_batch[key], new_batch[key]):
                failures += 1
                print(f"  ✗ CSC max_len={max_len}: {key} 不同")
                break
        else:
            n_err = int(new_batch["det_labels"].sum().item())
            print(f"  ✓ CSC max_len={max_len}: 4 个张量全等 "
                  f"({len(pairs)} 条,{n_err} 个错字位置)")
    return failures


def test_memmap() -> int:
    if not Path(str(CORPUS) + ".meta").exists():
        print("  data3 语料不存在,跳过。")
        return 0
    try:
        import numpy as np
    except ImportError:
        print("  numpy 未装,跳过(src/ 本来就不依赖它)。")
        return 0
    import json

    failures = 0
    for suffix, np_dtype in [("", np.int32), (".wid", np.int32), (".seg", np.uint8)]:
        p = str(CORPUS) + suffix
        meta = json.loads(Path(p + ".meta").read_text())
        ref = np.memmap(p, dtype=np_dtype, mode="r", shape=tuple(meta["shape"]))
        mine = new_data._memmap(p)

        if tuple(mine.shape) != tuple(meta["shape"]):
            failures += 1
            print(f"  ✗ memmap{suffix}: 形状 {tuple(mine.shape)} != {meta['shape']}")
            continue
        # 抽查头尾和中间几个 chunk
        idxs = [0, 1, 12345, meta["shape"][0] // 2, meta["shape"][0] - 1]
        bad = [i for i in idxs
               if not torch.equal(mine[i], torch.from_numpy(np.array(ref[i])))]
        if bad:
            failures += 1
            print(f"  ✗ memmap{suffix}: chunk {bad} 与 numpy 读出的不一致")
        else:
            print(f"  ✓ memmap{suffix}: {meta['dtype']} {tuple(mine.shape)},"
                  f"抽查 {len(idxs)} 个 chunk 与 numpy 逐值相等")
    return failures


def main() -> int:
    tmp = Path(__file__).resolve().parent / "_tmp"
    tmp.mkdir(exist_ok=True)
    failures = 0
    print("=== MT ===")
    failures += test_mt(tmp)
    print("\n=== CSC ===")
    failures += test_csc(tmp)
    print("\n=== _memmap(torch.from_file 取代 numpy.memmap)===")
    failures += test_memmap()

    for f in tmp.glob("*.pt"):
        f.unlink()
    tmp.rmdir()

    if failures:
        print(f"\n{failures} 项不一致")
        return 1
    print("\ndata 对拍全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
