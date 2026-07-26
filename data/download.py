"""下载 BERTc 用到的全部数据源。

用法:
    python data/download.py --list                # 看现状,不下载
    python data/download.py skypile               # 下一个源(默认用量)
    python data/download.py skypile --n-parts 42  # 覆盖用量
    python data/download.py --pretrain            # 预训练那 7 个源
    python data/download.py --all

part 数默认 = v4-Large 实跑用量(见 source.py)。SkyPile / CCI3-HQ /
FineWeb-Edu 在 HF 上都是几百 GB 全量,**不要**无参数 snapshot_download。

已存在的文件跳过,可反复运行续传。
"""
import argparse
import os
import re
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import source

# 必须在 import huggingface_hub 之前设 HF_ENDPOINT。
# 代理要清掉,否则走不通 hf-mirror —— 但 GitHub 反过来**需要**代理,
# 所以先存下来,download_github 里再临时恢复。
if source.HF_ENDPOINT:
    os.environ["HF_ENDPOINT"] = source.HF_ENDPOINT
SAVED_PROXY = {k: v for k, v in os.environ.items() if "proxy" in k.lower()}
for _k in SAVED_PROXY:
    del os.environ[_k]


def _glob_to_re(pat: str) -> re.Pattern:
    """glob → regex。* 不跨 /,** 跨 /。"""
    out, i = [], 0
    while i < len(pat):
        c = pat[i]
        if c == "*":
            if pat[i:i + 2] == "**":
                out.append(".*")
                i += 2
                if pat[i:i + 1] == "/":
                    i += 1
            else:
                out.append("[^/]*")
                i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(c))
            i += 1
    return re.compile("^" + "".join(out) + "$")


def download_hf(src: source.Source, n_parts, workers: int, dry: bool) -> None:
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    dest = source.DATA_ROOT / src.subdir
    api = HfApi(endpoint=source.HF_ENDPOINT or None)

    print(f"  列 {src.repo_id} 文件表 ...")
    info = api.dataset_info(src.repo_id, files_metadata=False)
    all_files = sorted(s.rfilename for s in info.siblings)

    pats = [_glob_to_re(p) for p in (src.allow_patterns or [src.part_glob])]
    picked = [f for f in all_files if any(p.match(f) for p in pats)]
    if n_parts is not None:
        picked = picked[:n_parts]
    if not picked:
        print(f"  !! 没有文件匹配 {src.allow_patterns or src.part_glob}")
        return

    todo = [f for f in picked if not (dest / f).exists()]
    print(f"  选中 {len(picked)} 个,已有 {len(picked) - len(todo)},待下 {len(todo)}")
    if dry or not todo:
        return

    def one(path):
        try:
            hf_hub_download(repo_id=src.repo_id, repo_type="dataset",
                            filename=path, local_dir=str(dest),
                            endpoint=source.HF_ENDPOINT or None)
            return path, None
        except Exception as e:                      # noqa: BLE001
            return path, str(e)

    done, failed = 0, []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(one, p) for p in todo]
        for fut in as_completed(futs):
            path, err = fut.result()
            done += 1
            if err:
                failed.append(path)
            print(f"    [{done}/{len(todo)}] {os.path.basename(path)}: "
                  f"{'OK' if not err else 'FAIL ' + err[:80]}", flush=True)
    if failed:
        print(f"  !! {len(failed)} 个失败,重跑本命令即可续传")


def download_hf_snapshot(src: source.Source, dry: bool) -> None:
    """整仓下载。用于文件不多的小数据集。"""
    from huggingface_hub import snapshot_download

    dest = source.DATA_ROOT / src.subdir
    print(f"  snapshot_download {src.repo_id} → {dest}")
    if dry:
        return
    snapshot_download(repo_id=src.repo_id, repo_type="dataset",
                      local_dir=str(dest))


def download_github_repo(src: source.Source, dry: bool) -> None:
    """git clone --depth 1。数据集仓库常带 LFS,交给 git 处理。"""
    import subprocess

    dest = source.DATA_ROOT / src.subdir
    url = f"https://github.com/{src.repo_id}.git"
    print(f"  git clone --depth 1 {url} → {dest}")
    if dry:
        return
    if (dest / ".git").exists():
        print("    已存在,git pull")
        cmd = ["git", "-C", str(dest), "pull", "--ff-only"]
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["git", "clone", "--depth", "1", url, str(dest)]
    # GitHub 要走代理(跟 hf-mirror 相反)
    env = {**os.environ, **SAVED_PROXY}
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        print(f"    !! 失败:{(r.stdout + r.stderr)[-400:]}")
        return
    n = sum(1 for _ in dest.rglob("*") if _.is_file())
    size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()) / 1e6
    print(f"    {n} 个文件,{size:.0f} MB")


def download_github(src: source.Source, dry: bool) -> None:
    import urllib.request

    dest = source.DATA_ROOT / src.subdir
    # 多文件用 paths 列清单;单文件时 part_glob 就是文件名
    fnames = src.paths or [src.part_glob]
    # GitHub 需要代理(跟 hf-mirror 相反),用回模块加载时存下的那份
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler(
            {k.replace("_proxy", "").replace("_PROXY", "").lower(): v
             for k, v in SAVED_PROXY.items()}))

    for fname in fnames:
        url = f"https://raw.githubusercontent.com/{src.repo_id}/master/{fname}"
        target = dest / fname
        print(f"  {url} → {target}")
        if dry:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            with opener.open(url, timeout=120) as resp, open(target, "wb") as f:
                f.write(resp.read())
            print(f"    下载完成 {target.stat().st_size / 1e6:.1f} MB")
        else:
            print("    已存在,跳过下载")

        if target.suffix == ".zip":
            with zipfile.ZipFile(target) as zf:
                names = zf.namelist()
                if all((dest / n).exists() for n in names):
                    print("    已解压,跳过")
                else:
                    zf.extractall(dest)
                    print(f"    解压 {len(names)} 项 → {dest}")


def unpack_dep(src: source.Source, dry: bool) -> None:
    """kind=dep-file:文件已经随 deps/ 下的依赖仓库 clone 下来了,只需解压。

    不拷贝 zip 到 DATA_ROOT —— deps/ 是 git clone,产物落在这边就够了,
    也别去弄脏那个 clone。
    """
    zip_path = src.dep_src()
    dest = src.dir()
    print(f"  {zip_path} → {dest}")
    if not zip_path.exists():
        print(f"    !! 找不到 {zip_path}")
        print(f"       {src.repo_id} 还没 clone。只做微调的话:")
        print("         bash prepare/install_deps.sh wapic-data")
        print("       要跑预训练的话装全套:make -C prepare deps")
        return
    if dry:
        return
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        if all((dest / n).exists() for n in names):
            print("    已解压,跳过")
        else:
            zf.extractall(dest)
            print(f"    解压 {len(names)} 项 → {dest}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="源名,见 --list")
    ap.add_argument("--list", action="store_true", help="只打印现状")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--pretrain", action="store_true", help="仅预训练语料")
    ap.add_argument("--finetune", action="store_true", help="仅下游任务数据")
    ap.add_argument("--csc", action="store_true", help="仅 CSC 数据源")
    ap.add_argument("--n-parts", type=int, default=None,
                    help="覆盖 part 数(只在下单个源时有意义);0 表示全量")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.list or not (args.names or args.all or args.pretrain
                     or args.finetune or args.csc):
        print(source.describe())
        if not args.list:
            print("\n用 --all / --pretrain / <源名> 开始下载。")
        return

    if args.all:
        names = list(source.ALL_SOURCES)
    elif args.pretrain:
        names = list(source.PRETRAIN_SOURCES)
    elif args.finetune:
        names = list(source.FINETUNE_SOURCES) + list(source.CSC_SOURCES)
    elif args.csc:
        names = list(source.CSC_SOURCES)
    else:
        names = args.names

    unknown = [n for n in names if n not in source.ALL_SOURCES]
    if unknown:
        sys.exit(f"未知源: {unknown}\n可用: {list(source.ALL_SOURCES)}")

    for name in names:
        src = source.ALL_SOURCES[name]
        n_parts = src.n_parts
        if args.n_parts is not None and len(names) == 1:
            n_parts = None if args.n_parts == 0 else args.n_parts
        print(f"\n=== {name} ({src.repo_id}) ===")
        if src.note:
            print(f"  {src.note}")
        if src.kind == "hf":
            download_hf(src, n_parts, args.workers, args.dry_run)
        elif src.kind == "hf-snapshot":
            download_hf_snapshot(src, args.dry_run)
        elif src.kind == "github-file":
            download_github(src, args.dry_run)
        elif src.kind == "github-repo":
            download_github_repo(src, args.dry_run)
        elif src.kind == "dep-file":
            unpack_dep(src, args.dry_run)
        else:
            print(f"  !! 未知 kind {src.kind}")

    print("\n完成。当前状态:\n")
    print(source.describe())


if __name__ == "__main__":
    main()
