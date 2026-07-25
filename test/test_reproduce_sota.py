"""用新代码复现已记录的 SOTA 数字。

这是整个重构最终的正确性依据:前面所有对拍都只证明"新旧实现逐值相等",
但那是在随机权重或小模型上比的。这里拿**真实的 SOTA checkpoint** +
prepare/ 产出的真实数据集,跑完整评测,看能不能还原 finetune/sota/README.md
里记的数。

    python test/test_reproduce_sota.py            # 两条都跑
    python test/test_reproduce_sota.py --only mt

慢:MT 要在 21,143 句 dev 上解码,CSC 要跑 707 条,单卡几分钟。
"""
import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data as bertc_data                               # noqa: E402
from src.evaluate import evaluate_csc, evaluate_mt               # noqa: E402
from src.finetune_csc import ModernBertCSC                       # noqa: E402
from src.finetune_mt import ModernBertMT                         # noqa: E402

DATASETS = ROOT / "prepare" / "datasets"
SOTA = ROOT / "save" / "sota"
BACKBONE_LARGE = ROOT / "prepare" / "backbones" / "BERTc-315M"

# finetune/sota/README.md 记录的数
EXPECT_MT = {"cws_f1": 0.9840, "pos_acc": 0.9800, "ner_f1": 0.9660, "score": 1.4712}
EXPECT_CSC_F1 = 0.8346
TOL = 0.005          # 允许的偏差
DEV_LIMIT = 2000     # 原训练脚本 --eval_dev_limit 的默认值


def report(name: str, got: float, want: float) -> bool:
    d = got - want
    ok = abs(d) <= TOL
    print(f"    {name:<12} {got:.4f}  (记录 {want:.4f},差 {d:+.4f})  "
          f"{'✓' if ok else '✗ 超出容差'}")
    return ok


def test_mt(device: str) -> int:
    ckpt = SOTA / "sota_mt_v4large_fgm_5ep_best.pt"
    dev_path = DATASETS / "mt_dev.pt"
    if not ckpt.exists() or not dev_path.exists():
        print("  checkpoint 或 dev 数据集缺失,跳过")
        return 0

    # 原训练脚本的 --eval_dev_limit 默认 2000:记录的数字是在 dev 前 2000 句上
    # 测的,不是全部 21,143 句。口径要对齐,否则数字对不上。
    full = bertc_data.MTDataset(dev_path, max_chars=254)
    ds = torch.utils.data.Subset(full, range(min(DEV_LIMIT, len(full))))
    for attr in ("cws_vocab", "pos_vocab", "ner_vocab", "pad_token_id",
                 "num_cws_tags", "num_pos_tags", "num_ner_tags"):
        setattr(ds, attr, getattr(full, attr))
    collator = bertc_data.MTCollator(ds.pad_token_id)
    print(f"  dev 全量 {len(full):,} 句,按原口径取前 {len(ds):,} 句")

    model = ModernBertMT(BACKBONE_LARGE, ds.num_cws_tags, ds.num_pos_tags,
                         ds.num_ner_tags).to(device)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        print(f"  ✗ 权重不匹配:缺 {missing},多 {unexpected}")
        return 1
    print(f"  加载 {ckpt.name} 成功(严格模式)")

    t0 = time.time()
    m = evaluate_mt(model, ds, collator, device)
    print(f"  评测用时 {time.time() - t0:.0f}s")
    ok = all([report("CWS F1", m["cws_f1"], EXPECT_MT["cws_f1"]),
              report("POS 准确率", m["pos_acc"], EXPECT_MT["pos_acc"]),
              report("NER F1", m["ner_f1"], EXPECT_MT["ner_f1"]),
              report("joint score", m["score"], EXPECT_MT["score"])])
    return 0 if ok else 1


def test_csc(device: str) -> int:
    ckpt = SOTA / "sota_csc_v4large_v8_best.pt"
    test_path = DATASETS / "csc_test.pt"
    if not ckpt.exists() or not test_path.exists():
        print("  checkpoint 或测试集缺失,跳过")
        return 0

    blob = torch.load(test_path, map_location="cpu", weights_only=False)
    id_to_char = {int(k): v for k, v in blob["id_to_char"].items()}
    ds = bertc_data.CSCDataset(test_path, max_len=128)
    collator = bertc_data.CSCCollator(ds.pad_token_id)
    print(f"  测试 {len(ds):,} 条,反查表 {len(id_to_char):,} 个字")

    model = ModernBertCSC(BACKBONE_LARGE, blob["vocab_size"]).to(device)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = state.get("model", state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  ✗ 权重不匹配:缺 {missing},多 {unexpected}")
        return 1
    print(f"  加载 {ckpt.name} 成功")

    t0 = time.time()
    m = evaluate_csc(model, ds, collator, device, id_to_char, threshold=0.7,
                     src_texts=blob.get("src_texts"),
                     tgt_texts=blob.get("tgt_texts"))
    print(f"  评测用时 {time.time() - t0:.0f}s")
    print(f"    TP={m['TP']} FP={m['FP']} FN={m['FN']} TN={m['TN']}  n={m['n']}")
    print(f"    P={m['precision']:.4f}  R={m['recall']:.4f}  acc={m['acc']:.4f}")
    return 0 if report("句级 F1", m["f1"], EXPECT_CSC_F1) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=("mt", "csc"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fails = 0
    if args.only in (None, "mt"):
        print("=== MT:CWS + POS + NER 联合(v4-Large + FGM 5ep)===")
        fails += test_mt(device)
    if args.only in (None, "csc"):
        print("\n=== CSC:SIGHAN-15(v4-Large v8)===")
        fails += test_csc(device)

    if fails:
        print(f"\n{fails} 条链没能复现记录的数字")
        return 1
    print("\n新代码复现 SOTA 通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
