#!/usr/bin/env python3
"""Upload prepared BERTc Hugging Face release folders."""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--release-dir", default=str(ROOT / "hf_release"))
    parser.add_argument("--private", action="store_true")
    parser.add_argument("models", nargs="*", default=["BERTc-165M", "BERTc-315M"])
    args = parser.parse_args()

    api = HfApi()
    release_dir = Path(args.release_dir)
    for model_name in args.models:
        folder = release_dir / model_name
        if not folder.exists():
            raise FileNotFoundError(folder)
        repo_id = f"{args.namespace}/{model_name}"
        api.create_repo(repo_id=repo_id, repo_type="model", private=args.private, exist_ok=True)
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(folder),
            ignore_patterns=["**/__pycache__/**", "**/*.pyc"],
            commit_message=f"Upload {model_name} release",
        )
        print(f"uploaded {repo_id}")


if __name__ == "__main__":
    main()
