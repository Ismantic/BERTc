"""把 checkpoint 导出成可以直接上传 HF 的发布目录。

发布目录里的推理代码不是模板字符串,而是从 src/ 和 save/assets/ 拷过去的
**真实文件** —— 所以 tests/test_save.py 能直接 import 它们跑一遍,
而不是发出去了才发现示例代码根本跑不通。

    python -m save.export                       # 全部
    python -m save.export BERTc-315M            # 单个
    python -m save.export --list

产出在 save/releases/<名字>/,用 save/upload.py 上传。
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

from . import cards
from .releases import ALL, BACKBONES, FINETUNES

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
ASSETS = Path(__file__).resolve().parent / "assets"
TOKENIZER_DIR = ROOT / "pretrain" / "modern_bertc" / "tokenizer"
DEFAULT_OUT = Path(__file__).resolve().parent / "releases"


def _copy(src: Path, dst_dir: Path) -> None:
    if not src.exists():
        sys.exit(f"缺少 {src}")
    shutil.copy2(src, dst_dir / src.name)


def _common_assets(out: Path, backbone_dir: Path, extra_code=()) -> None:
    """所有发布目录都要带的东西:骨干定义、tokenizer、config。"""
    _copy(SRC / "model.py", out)
    _copy(ASSETS / "tokenizer.py", out)
    _copy(TOKENIZER_DIR / "piece.model", out)
    _copy(backbone_dir / "config.json", out)
    mask_file = backbone_dir / "mask_token_id.txt"
    if mask_file.exists():
        _copy(mask_file, out)
    else:
        (out / "mask_token_id.txt").write_text(
            str(json.loads((backbone_dir / "config.json").read_text())["mask_token_id"]))
    for name in extra_code:
        src = (SRC if (SRC / name).exists() else ASSETS) / name
        _copy(src, out)


def export_backbone(name: str, spec: dict, out_root: Path) -> Path:
    ckpt_dir = Path(spec["checkpoint"])
    if not (ckpt_dir / "model.pt").exists():
        sys.exit(f"{name}: 没有 {ckpt_dir / 'model.pt'}")
    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(ckpt_dir / "model.pt", map_location="cpu", weights_only=False)
    state = ckpt.get("ema") or ckpt["model"]
    save_file(state, out / "model.safetensors")

    _common_assets(out, ckpt_dir, extra_code=["example_load.py"])
    (out / "README.md").write_text(cards.backbone_card(name, spec), encoding="utf-8")
    (out / "release_metadata.json").write_text(json.dumps({
        "name": name,
        "source_checkpoint": str(ckpt_dir.relative_to(ROOT)),
        "step": ckpt.get("step"),
        "metrics": {"mt": spec["mt"], "csc": spec["csc"]},
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out


def export_finetune(name: str, spec: dict, out_root: Path) -> Path:
    ckpt_path = Path(spec["checkpoint"])
    backbone_dir = Path(spec["backbone"])
    if not ckpt_path.exists():
        sys.exit(f"{name}: 没有 {ckpt_path}")
    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)

    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = blob.get("model", blob)
    # 绑权重的那一份不重复存,safetensors 不接受共享内存的张量
    state = {k: v for k, v in state.items() if k != "cor_head.weight"}
    save_file({k: v.contiguous() for k, v in state.items()},
              out / "model.safetensors")

    task_code = (["mt_model.py", "crf.py", "example_decode.py"] if spec["task"] == "mt"
                 else ["csc_model.py", "example_correct.py"])
    _common_assets(out, backbone_dir, extra_code=task_code)
    (out / "README.md").write_text(cards.finetune_card(name, spec), encoding="utf-8")
    (out / f"{spec['task']}_config.json").write_text(json.dumps({
        "task": spec["task"], "base_model": f"Ismantic/{spec['base']}",
        "metrics": spec["metrics"], "recipe": spec["recipe"], "data": spec["data"],
        "source_checkpoint": str(ckpt_path.relative_to(ROOT)),
        "epoch": blob.get("epoch"),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="要导出的发布名,默认全部")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for n, s in ALL.items():
            ok = Path(s["checkpoint"]).exists()
            print(f"  {'✓' if ok else '✗'} {n:<18} {s['checkpoint']}")
        return

    names = args.names or list(ALL)
    unknown = [n for n in names if n not in ALL]
    if unknown:
        sys.exit(f"未知发布名 {unknown},可用:{list(ALL)}")

    for name in names:
        spec = ALL[name]
        if not Path(spec["checkpoint"]).exists():
            print(f"  跳过 {name}:{spec['checkpoint']} 不存在")
            continue
        out = (export_backbone(name, spec, args.out) if name in BACKBONES
               else export_finetune(name, spec, args.out))
        size = sum(f.stat().st_size for f in out.iterdir() if f.is_file()) / 1e6
        print(f"  ✓ {name:<18} → {out}  ({size:.0f} MB, "
              f"{len(list(out.iterdir()))} 个文件)")


if __name__ == "__main__":
    main()
