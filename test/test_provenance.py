"""检查每个中间产物都有生产者 —— 不允许存在只能靠手工放上去的文件。

这个仓库声称"只靠 Hugging Face 和 GitHub 就能从零重训 BERTc"。声称会烂掉:
某个文件当初是手工拷进去的,之后所有流程都读它、都正常,只有全新 clone 的人
会撞墙,而本机上永远发现不了。

所以把 DAG 显式写在这里,逐条检查:

  产物 → 生产它的代码 → 那段代码存在且引用了这个路径

不检查文件在不在(本机有不代表别人能造出来),只检查**有没有代码负责造它**。

    python test/test_provenance.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (产物, 生产它的文件, 该文件里应当出现的标志串, 消费者)
# 标志串用来确认"生产者确实在写这个东西",而不是名字凑巧对上。
DAG = [
    ("data/downloads/*",
     "data/download.py", "def unpack_dep", "process*.py / pretokenize.py"),
    ("data/derived/*.documents.txt",
     "data/process.py", "documents.txt", "prepare/pretokenize.py"),
    ("data/derived/pd98/*.jsonl",
     "data/process_cws.py", "cws.pd98.jsonl", "prepare/build_mt.py"),
    ("data/derived/csc/sighan_wang271k_pairs.pkl",
     "data/process_csc.py", "pickle.dump", "prepare/build_csc.py"),
    ("data/derived/csc/sighan2015_test_official.tsv",
     "data/process_csc.py", "SIGHAN_TEST", "prepare/build_csc.py"),
    ("prepare/datasets/mt_{train,dev}.pt",
     "prepare/build_mt.py", "save(", "src/finetune_mt.py"),
    ("prepare/datasets/csc_{train,test}.pt",
     "prepare/build_csc.py", "save(", "src/finetune_csc.py"),
    ("prepare/corpus/v4.pt{,.wid,.seg}",
     "prepare/pretokenize.py", "wid", "src/pretrain.py"),
    ("prepare/output/<名字>/checkpoint-*",
     "src/pretrain.py", "save_steps", "src/finetune_*.py"),
    ("save/releases/<名字>/",
     "save/export.py", "model.safetensors", "save/upload.py, save/cws.py"),
    ("test/fixtures/tokenizer_baseline.json",
     "test/capture_baseline.py", "tokenizer_baseline.json", "test/test_tokenizer.py"),
    ("deps/PieceTokenizer, deps/Wapic",
     "prepare/install_deps.sh", "git_clone_or_pull", "prepare/tokenizer.py"),
]

# 训练产物:没法逐位复现,但必须能说清"跑哪条命令能造出来",
# 且必须有一条不用重训就能拿到的退路,否则全新 clone 验不了。
TRAINED = [
    ("models/<名字>/", "make -C prepare pretrain",
     "huggingface-cli download Ismantic/BERTc-315M", "save/export.py"),
    ("save/sota/*.pt", "make -C prepare finetune",
     "save/releases/<名字>/model.safetensors", "save/export.py, test_reproduce_sota.py"),
]


def check_dag() -> list[str]:
    bad = []
    for product, producer, marker, consumer in DAG:
        p = ROOT / producer
        if not p.exists():
            bad.append(f"{product}:生产者 {producer} 不存在")
            continue
        if marker not in p.read_text(encoding="utf8"):
            bad.append(f"{product}:{producer} 里找不到 {marker!r},"
                       f"可能已经不再生产它了")
            continue
        print(f"  ✓ {product:<44} ← {producer}")
    return bad


def check_fallbacks() -> list[str]:
    """训练产物必须有不重训就能拿到的退路,且退路要写在代码里。"""
    bad = []
    test_src = (ROOT / "test" / "test_reproduce_sota.py").read_text(encoding="utf8")
    export_src = (ROOT / "save" / "export.py").read_text(encoding="utf8")
    for product, how, fallback, consumer in TRAINED:
        print(f"  ○ {product:<44} ← {how}")
        print(f"    {'退路':<42} ← {fallback}")
    if "resolve_finetuned" not in test_src or "releases" not in test_src:
        bad.append("test_reproduce_sota.py 没有退回已发布权重的路径 —— "
                   "全新 clone 上就跑不了回归")
    if "huggingface-cli download" not in export_src:
        bad.append("save/export.py 缺权重时没告诉用户怎么拿")
    return bad


def check_no_stale_paths() -> list[str]:
    """扫已删目录的残留引用 —— test_reproduce_sota.py 就这么坏过一次:
    它指向 prepare/backbones/,那个目录早删了,而错误直到真跑才暴露。"""
    gone = ["prepare/backbones", "data3/", "NLP_BERT_CRF", "output_v4_",
            "finetune/sota", "hf_release/"]
    # 讲历史的叙述不算残留引用 —— 这几处是在解释"当初为什么这么改"
    allow = {("src/__init__.py", "NLP_BERT_CRF")}
    bad = []
    for f in ROOT.rglob("*.py"):
        if any(x in f.parts for x in ("deps", ".git", "releases", "downloads")):
            continue
        if f.name == "test_provenance.py":      # 自己就列着这些名字
            continue
        txt = f.read_text(encoding="utf8", errors="ignore")
        for g in gone:
            if g in txt and (str(f.relative_to(ROOT)), g) not in allow:
                line = next((i + 1 for i, l in enumerate(txt.splitlines()) if g in l), 0)
                bad.append(f"{f.relative_to(ROOT)}:{line} 引用了已删的 {g}")
    return bad


def main() -> int:
    print("=== 每个中间产物都有生产者 ===")
    bad = check_dag()
    print("\n=== 训练产物:生产命令 + 不重训的退路 ===")
    bad += check_fallbacks()
    print("\n=== 已删目录的残留引用 ===")
    stale = check_no_stale_paths()
    print("  (无)" if not stale else "")
    bad += stale

    if bad:
        print(f"\n{len(bad)} 处问题:")
        for b in bad:
            print(f"  ✗ {b}")
        return 1
    print("\n全部产物都有源头")
    return 0


if __name__ == "__main__":
    sys.exit(main())
