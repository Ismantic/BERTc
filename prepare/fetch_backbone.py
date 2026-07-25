"""从 Hugging Face 拉预训练骨干,转成微调脚本认的格式。

HF 发布包是 model.safetensors,而 src/finetune_*.py 读的是预训练产出的
model.pt(格式 {"model": state_dict, "config": {...}})。这一步做转换,
顺带把 tokenizer 资产一并放好。

放在 prepare/ 而不是 src/:src/ 只依赖 torch,不引入 huggingface_hub 和
safetensors —— 下载和格式转换是"准备"的事,不是训练的事。

    python -m prepare.fetch_backbone                       # 默认 Ismantic/BERTc-315M
    python -m prepare.fetch_backbone --repo Ismantic/BERTc-165M
    python -m prepare.fetch_backbone --local save/releases/BERTc-315M   # 已经下好了

产出目录可以直接喂给:
    python -m src.finetune_mt  --ckpt_dir <产出目录> ...
    python -m src.finetune_csc --ckpt_dir <产出目录> ...
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "Ismantic/BERTc-315M"
DEFAULT_OUT = ROOT / "prepare" / "backbones"
NEEDED = ("model.safetensors", "config.json")
OPTIONAL = ("mask_token_id.txt", "piece.model")


def download(repo: str, dst: Path) -> Path:
    """从 HF 拉发布包。走 hf-mirror 时必须清代理,跟 data/download.py 同一个坑。"""
    if os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"):
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    for k in [k for k in os.environ if "proxy" in k.lower()]:
        del os.environ[k]
    from huggingface_hub import snapshot_download

    print(f"从 {repo} 下载(endpoint={os.environ.get('HF_ENDPOINT') or '官方'})...")
    path = snapshot_download(repo_id=repo, repo_type="model", local_dir=str(dst),
                             allow_patterns=list(NEEDED + OPTIONAL))
    return Path(path)


def convert(src_dir: Path, out_dir: Path) -> Path:
    """model.safetensors → model.pt。"""
    from safetensors.torch import load_file

    st = src_dir / "model.safetensors"
    cfg_path = src_dir / "config.json"
    if not st.exists():
        sys.exit(f"{src_dir} 下没有 model.safetensors")
    if not cfg_path.exists():
        sys.exit(f"{src_dir} 下没有 config.json")

    cfg = json.loads(cfg_path.read_text())
    state = load_file(str(st))
    n_bert = sum(1 for k in state if k.startswith("bert."))
    if n_bert == 0:
        sys.exit(f"{st} 里没有 bert.* 前缀的张量 —— 这是骨干包吗?"
                 f"(微调要从骨干开始,不是从另一个微调结果)")

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": state, "config": cfg, "step": None},
               out_dir / "model.pt")
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    mask_file = src_dir / "mask_token_id.txt"
    if mask_file.exists():
        shutil.copy2(mask_file, out_dir / "mask_token_id.txt")
    else:
        (out_dir / "mask_token_id.txt").write_text(str(cfg["mask_token_id"]))
    if (src_dir / "piece.model").exists():
        shutil.copy2(src_dir / "piece.model", out_dir / "piece.model")

    size = (out_dir / "model.pt").stat().st_size / 1e6
    print(f"  {n_bert} 个骨干张量 → {out_dir / 'model.pt'}  ({size:.0f} MB)")
    print(f"  架构 {cfg['num_hidden_layers']}L / {cfg['hidden_size']}H / "
          f"{cfg['intermediate_size']}I / {cfg['num_attention_heads']}h,"
          f"词表 {cfg['vocab_size']}")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=DEFAULT_REPO, help="HF 仓库名")
    ap.add_argument("--local", type=Path, default=None,
                    help="已经下好的发布目录,跳过下载")
    ap.add_argument("--out", type=Path, default=None,
                    help="默认 prepare/backbones/<名字>")
    args = ap.parse_args()

    name = (args.local.name if args.local else args.repo.split("/")[-1])
    out = args.out or (DEFAULT_OUT / name)

    if args.local:
        src = args.local
        if not src.exists():
            sys.exit(f"{src} 不存在")
        print(f"用本地发布目录 {src}")
    else:
        src = download(args.repo, DEFAULT_OUT / f"_hf_{name}")

    convert(src, out)
    print(f"\n可以微调了:")
    print(f"  python -m src.finetune_mt  --ckpt_dir {out} \\")
    print(f"      --train_data prepare/datasets/mt_train.pt \\")
    print(f"      --dev_data prepare/datasets/mt_dev.pt --output_dir <输出目录>")


if __name__ == "__main__":
    main()
