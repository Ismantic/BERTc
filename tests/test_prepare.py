"""prepare/ 的 builder 与旧管线的对拍。

test_data.py 验的是 src/data.py 的读取端,用的编码器是测试里现写的。
这里验的是 prepare/build_{mt,csc}.py 这两个**真正的写入端** ——
标签构造、字→id、打包三步合起来,产出是否跟旧的 MTDataset / CSCDataset 一致。

    python tests/test_prepare.py
"""
import pickle
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import data as new_data                                  # noqa: E402

OLD_MT = ROOT / "finetune" / "NLP_BERT_CRF"
PD98 = OLD_MT / "data"
DATASETS = ROOT / "prepare" / "datasets"
CSC_PKL = ROOT / "csc" / "data" / "all_pairs.pkl"
CSC_TSV = ROOT / "csc" / "data" / "test" / "sighan2015_test_official.tsv"

N = 2000


def test_mt() -> int:
    path = DATASETS / "mt_train.pt"
    if not path.exists():
        print(f"  {path} 不存在,先跑 python -m prepare.build_mt")
        return 0

    sys.path.insert(0, str(OLD_MT))
    shadowed = sys.modules.pop("data", None)
    try:
        from data_mt import MTDataset as OldMT
        from data_pos_ner import build_pos_vocab
    except Exception as e:                                        # noqa: BLE001
        print(f"  旧 MT 模块导入失败({e}),跳过。")
        return 0
    finally:
        if shadowed is not None:
            sys.modules["data"] = shadowed

    from prepare.tokenizer import load_tokenizer
    tok = load_tokenizer()

    old = OldMT(PD98 / "cws.pd98.jsonl", PD98 / "pos.pd98.jsonl",
                PD98 / "ner.pd98.jsonl", build_pos_vocab(), max_chars=10 ** 9)
    new = new_data.MTDataset(path, max_chars=10 ** 9)

    if len(old.items) != len(new):
        print(f"  ✗ 条数不同:旧 {len(old.items):,} vs 新 {len(new):,}")
        return 1

    bad = 0
    for i in range(min(N, len(new))):
        o, n = old.items[i], new[i]
        exp_ids = torch.tensor([tok.char_to_id(c) for c in o["chars"]],
                               dtype=torch.int32)
        checks = [
            ("input_ids", exp_ids, n["input_ids"]),
            ("cws_tags", torch.tensor(o["cws_tags"], dtype=torch.int32), n["cws_tags"]),
            ("pos_tags", torch.tensor(o["pos_tags"], dtype=torch.int32), n["pos_tags"]),
            ("ner_tags", torch.tensor(o["ner_tags"], dtype=torch.int32), n["ner_tags"]),
        ]
        for name, a, b in checks:
            if not torch.equal(a, b):
                bad += 1
                if bad <= 2:
                    print(f"  ✗ 第 {i} 条 {name} 不同")
                    print(f"      旧 {a[:15].tolist()}")
                    print(f"      新 {b[:15].tolist()}")
                break

    if bad:
        print(f"  ✗ 前 {min(N, len(new)):,} 条里 {bad} 条不一致")
        return 1
    print(f"  ✓ {len(new):,} 条(逐条比对前 {min(N, len(new)):,} 条):"
          f"input_ids / cws / pos / ner 全等")
    print(f"    标签表 cws={len(new.cws_vocab)} pos={len(new.pos_vocab)} "
          f"ner={len(new.ner_vocab)}")
    return 0


def test_csc() -> int:
    train_p, test_p = DATASETS / "csc_train.pt", DATASETS / "csc_test.pt"
    if not train_p.exists():
        print(f"  {train_p} 不存在,先跑 python -m prepare.build_csc")
        return 0

    from prepare.tokenizer import load_tokenizer
    tok = load_tokenizer()
    with open(CSC_PKL, "rb") as f:
        pairs = pickle.load(f)
    ds = new_data.CSCDataset(train_p, max_len=10 ** 9)

    if len(pairs) != len(ds):
        print(f"  ✗ 条数不同:pkl {len(pairs):,} vs 数据集 {len(ds):,}")
        return 1

    bad = 0
    for i in range(min(N, len(ds))):
        src, tgt = pairs[i]
        n = min(len(src), len(tgt))
        item = ds[i]
        exp_in = torch.tensor([tok.char_to_id(c) for c in src[:n]], dtype=torch.int32)
        exp_cor = torch.tensor([tok.char_to_id(c) for c in tgt[:n]], dtype=torch.int32)
        exp_det = torch.tensor([1 if src[j] != tgt[j] else 0 for j in range(n)],
                               dtype=torch.uint8)
        if not (torch.equal(exp_in, item["input_ids"])
                and torch.equal(exp_cor, item["cor_labels"])
                and torch.equal(exp_det, item["det_labels"])):
            bad += 1
            if bad <= 2:
                print(f"  ✗ 第 {i} 条不同:{src!r} → {tgt!r}")
    if bad:
        print(f"  ✗ 前 {min(N, len(ds)):,} 条里 {bad} 条不一致")
        return 1
    print(f"  ✓ 训练集 {len(ds):,} 条(逐条比对前 {min(N, len(ds)):,} 条):"
          f"input_ids / cor / det 全等")

    # det 必须按字比对,不是按 id —— 这条是 UNK 冲突时唯一的防线
    n_id_diff = n_char_diff = 0
    for i in range(min(N, len(ds))):
        src, tgt = pairs[i]
        m = min(len(src), len(tgt))
        item = ds[i]
        n_char_diff += sum(1 for j in range(m) if src[j] != tgt[j])
        n_id_diff += int((item["input_ids"] != item["cor_labels"]).sum())
    print(f"    错字位置:按字 {n_char_diff:,} 处 / 按 id {n_id_diff:,} 处"
          + ("(一致)" if n_char_diff == n_id_diff else
             f" ← 差 {n_char_diff - n_id_diff} 处,按 id 会漏"))

    # 测试集:707 条 + id_to_char 反查表
    blob = torch.load(test_p, map_location="cpu", weights_only=False)
    n_test = blob["offsets"].numel() - 1
    ok = n_test == 707 and "id_to_char" in blob
    print(f"  {'✓' if ok else '✗'} 测试集 {n_test} 条(SIGHAN-15 官方 707 条),"
          f"反查表 {len(blob.get('id_to_char', {})):,} 个字")
    return 0 if ok else 1


def main() -> int:
    print("=== build_mt vs 旧 MTDataset ===")
    f = test_mt()
    print("\n=== build_csc vs all_pairs.pkl ===")
    f += test_csc()
    if f:
        print(f"\n{f} 项不一致")
        return 1
    print("\nprepare builder 对拍全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
