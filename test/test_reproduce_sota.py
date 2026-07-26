"""复现 save/sota/README.md 里记录的 SOTA 数字。

改了 src/ 或 prepare/ 之后跑这个 —— 拿**真实的 SOTA checkpoint** 和 prepare/
产出的真实数据集跑完整评测,对照记录的数字。这是判断有没有改坏的硬标准:
单元级的对拍只能说明某个函数没写错,说明不了整条链对。

    python test/test_reproduce_sota.py            # 两条都跑
    python test/test_reproduce_sota.py --only mt

MT 在 dev 前 2000 句上解码,CSC 跑 707 条,单卡几分钟。
"""
import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data as bertc_data                               # noqa: E402
from src.checkpoint import load_safetensors                      # noqa: E402
from src.evaluate import evaluate_csc, evaluate_mt               # noqa: E402
from src.finetune_csc import ModernBertCSC                       # noqa: E402
from src.finetune_mt import ModernBertMT                         # noqa: E402

DATASETS = ROOT / "prepare" / "datasets"
SOTA = ROOT / "save" / "sota"

# save/sota/README.md 记录的数
EXPECT_MT = {"cws_f1": 0.9840, "pos_acc": 0.9800, "ner_f1": 0.9660, "score": 1.4712}
EXPECT_CSC_F1 = 0.8346
TOL = 0.005          # 允许的偏差
DEV_LIMIT = 2000     # 原训练脚本 --eval_dev_limit 的默认值


def resolve_backbone(name: str = "BERTc-315M") -> Path | None:
    """找一个能提供架构 config 的骨干目录。

    权重随后会被微调 checkpoint 整个覆盖,所以这里要的只是 config.json 加一份
    能让 load_state_dict 走通的初始权重。三个来源任取其一 —— 全新 clone 上
    最省事的是从 HF 下发布包。
    """
    for p in (ROOT / "save" / "releases" / name,
              ROOT / "models" / name,
              ROOT / "prepare" / "output" / name / "checkpoint-8500"):
        if (p / "config.json").exists():
            return p
    return None


def resolve_finetuned(sota_name: str, release_name: str) -> Path | None:
    """微调 checkpoint:优先本地训练产物,退回已发布的 safetensors。

    两者权重相同 —— 发布包就是从 save/sota/*.pt 导出的,唯一的差别是
    CSC 的 cor_head.weight 与词嵌入绑权重、导出时去了重,加载时会自动绑回。
    """
    pt = SOTA / sota_name
    if pt.exists():
        return pt
    st = ROOT / "save" / "releases" / release_name / "model.safetensors"
    return st if st.exists() else None


def load_finetuned(path: Path) -> dict:
    if path.suffix == ".safetensors":
        return load_safetensors(path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    return state.get("model", state) if isinstance(state, dict) else state


def missing_inputs(what: str, backbone: Path | None, ckpt: Path | None,
                   data: Path) -> bool:
    """缺东西时说清楚缺哪个、怎么补,而不是让它在半路报 config.json 不存在。"""
    lack = []
    if backbone is None:
        lack.append("骨干(要 config.json):"
                    "huggingface-cli download Ismantic/BERTc-315M "
                    "--local-dir models/BERTc-315M")
    if ckpt is None:
        lack.append(f"{what} checkpoint:"
                    f"huggingface-cli download Ismantic/BERTc-315M-{what.upper()} "
                    f"--local-dir save/releases/BERTc-315M-{what.upper()}")
    if not data.exists():
        lack.append(f"数据集 {data.name}:make -C data all && make -C prepare datasets")
    for line in lack:
        print(f"  缺 {line}")
    return bool(lack)


def report(name: str, got: float, want: float) -> bool:
    d = got - want
    ok = abs(d) <= TOL
    print(f"    {name:<12} {got:.4f}  (记录 {want:.4f},差 {d:+.4f})  "
          f"{'✓' if ok else '✗ 超出容差'}")
    return ok


def test_mt(device: str) -> int:
    dev_path = DATASETS / "mt_dev.pt"
    backbone = resolve_backbone()
    ckpt = resolve_finetuned("sota_mt_v4large_fgm_5ep_best.pt", "BERTc-315M-MT")
    if missing_inputs("mt", backbone, ckpt, dev_path):
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

    model = ModernBertMT(backbone, ds.num_cws_tags, ds.num_pos_tags,
                         ds.num_ner_tags).to(device)
    missing, unexpected = model.load_state_dict(load_finetuned(ckpt), strict=True)
    if missing or unexpected:
        print(f"  ✗ 权重不匹配:缺 {missing},多 {unexpected}")
        return 1
    print(f"  骨干 {backbone.name} + 权重 {ckpt.name}(严格模式)")

    t0 = time.time()
    m = evaluate_mt(model, ds, collator, device)
    print(f"  评测用时 {time.time() - t0:.0f}s")
    ok = all([report("CWS F1", m["cws_f1"], EXPECT_MT["cws_f1"]),
              report("POS 准确率", m["pos_acc"], EXPECT_MT["pos_acc"]),
              report("NER F1", m["ner_f1"], EXPECT_MT["ner_f1"]),
              report("joint score", m["score"], EXPECT_MT["score"])])
    return 0 if ok else 1


def test_csc(device: str) -> int:
    test_path = DATASETS / "csc_test.pt"
    backbone = resolve_backbone()
    ckpt = resolve_finetuned("sota_csc_v4large_v8_best.pt", "BERTc-315M-CSC")
    if missing_inputs("csc", backbone, ckpt, test_path):
        return 0

    blob = torch.load(test_path, map_location="cpu", weights_only=False)
    id_to_char = {int(k): v for k, v in blob["id_to_char"].items()}
    ds = bertc_data.CSCDataset(test_path, max_len=128)
    collator = bertc_data.CSCCollator(ds.pad_token_id)
    print(f"  测试 {len(ds):,} 条,反查表 {len(id_to_char):,} 个字")

    model = ModernBertCSC(backbone, blob["vocab_size"]).to(device)
    # cor_head.weight 与 bert.embed.weight 绑权重,发布包里去了重 —— 加载
    # bert.embed.weight 就等于同时设好了它,所以这一个 key 缺失是预期的。
    missing, unexpected = model.load_state_dict(load_finetuned(ckpt), strict=False)
    missing = [k for k in missing if k != "cor_head.weight"]
    if missing or unexpected:
        print(f"  ✗ 权重不匹配:缺 {missing},多 {unexpected}")
        return 1
    print(f"  骨干 {backbone.name} + 权重 {ckpt.name}")

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
