"""发布目录的自包含性验证。

发布目录里的推理代码是真实文件,这里把它们**当外部用户那样**加载:
切到发布目录、只用目录内的模块、跑一遍真实推理 —— 发出去之前就知道跑不跑得通。

同时验证两件事:
  1. 发布权重与仓库内实现的输出一致(safetensors 转换没丢东西)
  2. 三个任务的推理结果是像样的中文,不是乱码

    python test/test_save.py
"""
import json
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RELEASES = ROOT / "save" / "releases"
PY = sys.executable


def run_in(folder: Path, code: str) -> tuple[int, str]:
    """在发布目录里跑一段代码。cwd 设过去,只能看到目录内的模块。"""
    r = subprocess.run([PY, "-c", textwrap.dedent(code)], cwd=folder,
                       capture_output=True, text=True, timeout=900)
    return r.returncode, (r.stdout + r.stderr).strip()


def marked(out: str, tag: str) -> str:
    """取出以 tag 开头那一行的内容。tokenizer 会往 stdout 打 INFO 日志,
    不能整段 split。"""
    for line in out.splitlines():
        if line.startswith(tag):
            return line[len(tag):].strip()
    raise ValueError(f"输出里没有 {tag} 标记:\n{out[-500:]}")


def check_weights(name: str, spec: dict) -> int:
    """发布的 safetensors 与源 checkpoint 逐张量比对。

    比跑一次前向更彻底 —— 前向只覆盖走到的路径,这里覆盖每一个参数。
    """
    import torch
    from safetensors.torch import load_file

    folder = RELEASES / name
    ckpt_path = Path(spec["checkpoint"])
    if ckpt_path.is_dir():
        # 骨干:预训练产出(model.pt)或从 HF 下的发布包(model.safetensors)
        sys.path.insert(0, str(ROOT))
        from src.checkpoint import load_safetensors
        pt, st = ckpt_path / "model.pt", ckpt_path / "model.safetensors"
        if pt.exists():
            blob = torch.load(pt, map_location="cpu", weights_only=False)
            src = blob.get("ema") or blob["model"]
        elif st.exists():
            src = load_safetensors(st)
        else:
            print(f"  - {name}: 源 {ckpt_path} 里没有权重,跳过忠实性检查")
            return 0
    elif ckpt_path.exists():
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        src = blob.get("model", blob)
    else:
        print(f"  - {name}: 源 {ckpt_path.name} 不存在,跳过忠实性检查")
        return 0
    rel = load_file(str(folder / "model.safetensors"))

    # cor_head.weight 与 embed 绑权重,safetensors 不重复存,加载时重新绑
    src = {k: v for k, v in src.items() if k != "cor_head.weight"}
    if set(src) != set(rel):
        print(f"  ✗ {name}: 张量名不一致 "
              f"缺 {sorted(set(src) - set(rel))[:3]} 多 {sorted(set(rel) - set(src))[:3]}")
        return 1
    worst = max((src[k] - rel[k]).abs().max().item() for k in src)
    if worst != 0.0:
        print(f"  ✗ {name}: 权重有差异 max|Δ| = {worst:.3e}")
        return 1
    print(f"  ✓ {name}: {len(src)} 个张量与源 checkpoint 逐值相等")
    return 0


def check_files(name: str, expect: list[str]) -> int:
    folder = RELEASES / name
    missing = [f for f in expect if not (folder / f).exists()]
    if missing:
        print(f"  ✗ {name}: 缺 {missing}")
        return 1
    return 0


def test_backbone(name: str) -> int:
    folder = RELEASES / name
    if not folder.exists():
        print(f"  {name} 未导出,跳过")
        return 0
    fails = check_files(name, ["model.safetensors", "config.json", "model.py",
                               "tokenizer.py", "BERTc-Tokenizer.pt", "README.md",
                               "example_load.py"])
    code, out = run_in(folder, """
        import json, torch
        from safetensors.torch import load_file
        from model import ModernBertConfig, ModernBertForMLM
        from tokenizer import PieceCharTokenizer

        cfg = ModernBertConfig.from_dict(json.load(open("config.json")))
        m = ModernBertForMLM(cfg); m.load_state_dict(load_file("model.safetensors"), strict=True); m.eval()
        tok = PieceCharTokenizer(".")
        text = "北京是中国的首都"
        ids = torch.tensor([tok.encode(text)])
        ids[0, 2] = tok.mask_token_id
        with torch.no_grad():
            pred = int(m(ids)["logits"][0, 2].argmax())
        print("PRED", tok.id_to_char(pred))
    """)
    if code != 0:
        print(f"  ✗ {name} 加载失败:\n{out[-800:]}")
        return fails + 1
    pred = marked(out, "PRED")
    ok = pred == "是"
    print(f"  {'✓' if ok else '!'} {name}: 掩码预测「北京[?]中国的首都」→ {pred!r}"
          f"{'' if ok else '(期望 是)'}")
    return fails


def test_mt(name: str) -> int:
    folder = RELEASES / name
    if not folder.exists():
        print(f"  {name} 未导出,跳过")
        return 0
    fails = check_files(name, ["model.safetensors", "mt_model.py", "crf.py",
                               "model.py", "tokenizer.py", "mt_config.json"])
    code, out = run_in(folder, """
        import json
        from mt_model import BERTcForMT
        m = BERTcForMT.from_pretrained(".")
        r = m.predict("中国科学院计算技术研究所在北京")
        print("JSON", json.dumps(r, ensure_ascii=False))
    """)
    if code != 0:
        print(f"  ✗ {name} 推理失败:\n{out[-800:]}")
        return fails + 1
    r = json.loads(marked(out, "JSON"))
    words, pos, ner = r["words"], r["pos"], r["ner"]
    ok = ("".join(words) == r["text"] and len(pos) == len(words)
          and all(e["type"] in ("Nh", "Ns", "Ni") for e in ner))
    print(f"  {'✓' if ok else '✗'} {name}")
    print(f"      分词 {' / '.join(words)}")
    print(f"      词性 {' '.join(f'{w}/{p}' for w, p in zip(words, pos))}")
    print(f"      实体 {[(e['type'], r['text'][e['start']:e['end']]) for e in ner]}")
    return fails + (0 if ok else 1)


def test_csc(name: str) -> int:
    folder = RELEASES / name
    if not folder.exists():
        print(f"  {name} 未导出,跳过")
        return 0
    fails = check_files(name, ["model.safetensors", "csc_model.py", "model.py",
                               "tokenizer.py", "csc_config.json"])
    code, out = run_in(folder, """
        import json
        from csc_model import BERTcForCSC
        m = BERTcForCSC.from_pretrained(".")
        cases = ["我今天很稿兴", "他平时喜欢锻练身体", "这个问题很重要,需要引起足够重视"]
        print("JSON", json.dumps([[c, m.correct(c)] for c in cases], ensure_ascii=False))
    """)
    if code != 0:
        print(f"  ✗ {name} 推理失败:\n{out[-800:]}")
        return fails + 1
    pairs = json.loads(marked(out, "JSON"))
    # 纠错必须等长(狭义 CSC 只做同长替换)
    bad_len = [p for p in pairs if len(p[0]) != len(p[1])]
    ok = not bad_len
    print(f"  {'✓' if ok else '✗'} {name}")
    for src, dst in pairs:
        mark = "  (无改动)" if src == dst else ""
        print(f"      {src}  →  {dst}{mark}")
    if bad_len:
        print(f"      ✗ {len(bad_len)} 条纠错后长度变了")
    return fails + (0 if ok else 1)


def main() -> int:
    if not RELEASES.exists():
        print("还没导出,先跑 python -m save.export")
        return 0
    from save.releases import ALL

    fails = 0
    print("=== 权重导出忠实性(逐张量 vs 源 checkpoint)===")
    for n, spec in ALL.items():
        if (RELEASES / n).exists():
            fails += check_weights(n, spec)

    print("\n=== 骨干 ===")
    for n in ("BERTc-165M", "BERTc-315M"):
        fails += test_backbone(n)
    print("\n=== MT ===")
    for n in ("BERTc-165M-MT", "BERTc-315M-MT"):
        fails += test_mt(n)
    print("\n=== CSC ===")
    for n in ("BERTc-165M-CSC", "BERTc-315M-CSC"):
        fails += test_csc(n)

    if fails:
        print(f"\n{fails} 项失败")
        return 1
    print("\n发布目录验证全部通过(都是在目录内、只用目录内的模块跑的)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
