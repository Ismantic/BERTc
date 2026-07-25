"""把发布目录传到 Hugging Face。

    python -m save.upload --namespace Ismantic --dry-run    # 先看要传什么
    python -m save.upload --namespace Ismantic --code-only  # 只传代码和文档
    python -m save.upload --namespace Ismantic              # 全部(含权重)
    python -m save.upload --namespace Ismantic BERTc-315M   # 单个

--code-only 跳过 model.safetensors 和 tokenizer 资产,只传推理代码、示例、
模型卡。权重没变时用它 —— 六个仓库合计 4.8GB,重传一遍纯属浪费,而且
覆盖已发布权重本身就是不必要的风险。

需要有目标 namespace 写权限的 token(`huggingface-cli login`)。
"""
import argparse
import sys
from pathlib import Path

from .releases import ALL

DEFAULT_DIR = Path(__file__).resolve().parent / "releases"
IGNORE = ["**/__pycache__/**", "**/*.pyc"]
# --code-only 时跳过的大文件。config.json / mask_token_id.txt 是配置不是权重,
# 很小而且改架构描述时需要跟着更新,所以照传。
WEIGHTS = ["model.safetensors", "piece.model"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="要上传的发布名,默认全部已导出的")
    ap.add_argument("--namespace", required=True)
    ap.add_argument("--release-dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--code-only", action="store_true",
                    help="只传推理代码 / 示例 / 模型卡,跳过权重和 tokenizer")
    ap.add_argument("--dry-run", action="store_true", help="只打印,不上传")
    args = ap.parse_args()

    names = args.names or [n for n in ALL if (args.release_dir / n).exists()]
    if not names:
        sys.exit(f"{args.release_dir} 下没有已导出的发布目录,先跑 python -m save.export")

    missing = [n for n in names if not (args.release_dir / n).exists()]
    if missing:
        sys.exit(f"没导出:{missing}")

    ignore = IGNORE + (WEIGHTS if args.code_only else [])
    for name in names:
        folder = args.release_dir / name
        repo_id = f"{args.namespace}/{name}"
        skip = set(WEIGHTS) if args.code_only else set()
        files = [f for f in sorted(folder.iterdir())
                 if f.is_file() and f.name not in skip and f.suffix != ".pyc"]
        size = sum(f.stat().st_size for f in files) / 1e6
        print(f"  {repo_id}  ←  {folder}  ({size:.1f} MB, {len(files)} 个文件)"
              + ("  [私有]" if args.private else ""))
        for f in files:
            print(f"      {f.name}")
        if args.dry_run:
            continue

        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo_id=repo_id, repo_type="model",
                        private=args.private, exist_ok=True)
        api.upload_folder(repo_id=repo_id, repo_type="model",
                          folder_path=str(folder), ignore_patterns=ignore,
                          commit_message=("Update inference code and model card"
                                          if args.code_only else f"Upload {name}"))
        print(f"    → https://huggingface.co/{repo_id}")

    if args.dry_run:
        print("\n--dry-run:什么都没传。去掉这个参数才会真上传。")
    elif args.code_only:
        print("\n--code-only:权重未改动,只更新了代码和模型卡。")


if __name__ == "__main__":
    main()
