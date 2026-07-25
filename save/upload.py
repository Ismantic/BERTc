"""把发布目录传到 Hugging Face。

    python -m save.upload --namespace Ismantic --dry-run   # 先看要传什么
    python -m save.upload --namespace Ismantic             # 全部
    python -m save.upload --namespace Ismantic BERTc-315M  # 单个

需要有目标 namespace 写权限的 token(`huggingface-cli login`)。
"""
import argparse
import sys
from pathlib import Path

from .releases import ALL

DEFAULT_DIR = Path(__file__).resolve().parent / "releases"
IGNORE = ["**/__pycache__/**", "**/*.pyc"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="要上传的发布名,默认全部已导出的")
    ap.add_argument("--namespace", required=True)
    ap.add_argument("--release-dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="只打印,不上传")
    args = ap.parse_args()

    names = args.names or [n for n in ALL if (args.release_dir / n).exists()]
    if not names:
        sys.exit(f"{args.release_dir} 下没有已导出的发布目录,先跑 python -m save.export")

    missing = [n for n in names if not (args.release_dir / n).exists()]
    if missing:
        sys.exit(f"没导出:{missing}")

    for name in names:
        folder = args.release_dir / name
        repo_id = f"{args.namespace}/{name}"
        size = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file()) / 1e6
        n_files = sum(1 for f in folder.iterdir() if f.is_file())
        print(f"  {repo_id}  ←  {folder}  ({size:.0f} MB, {n_files} 个文件)"
              + ("  [私有]" if args.private else ""))
        if args.dry_run:
            continue

        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo_id=repo_id, repo_type="model",
                        private=args.private, exist_ok=True)
        api.upload_folder(repo_id=repo_id, repo_type="model",
                          folder_path=str(folder), ignore_patterns=IGNORE,
                          commit_message=f"Upload {name}")
        print(f"    → https://huggingface.co/{repo_id}")

    if args.dry_run:
        print("\n--dry-run:什么都没传。去掉这个参数才会真上传。")


if __name__ == "__main__":
    main()
