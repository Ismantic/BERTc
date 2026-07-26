"""PieceTokenizer / Wapic 重建后的行为校验。

这两个是 C++ 扩展,重新编译后行为变了不会有任何报错,但后果不一样:

  PieceTokenizer  编码变了 → 12536 词表和已发布模型(Ismantic/BERTc-165M /
                  -315M)的 embedding 全部错位。这是**致命**的,必须一字不差。
  Wapic           切词变了 → WWM 的词边界跟着变,只影响预训练掩码粒度,
                  不影响词表。所以这里只报告差异,不判失败。

基线由 test/capture_baseline.py 在重建**之前**抓好,存 test/fixtures/。
重建后 prepare/install_deps.sh 会自动调本脚本比对。

    python test/test_tokenizer.py
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "test" / "fixtures" / "tokenizer_baseline.json"
sys.path.insert(0, str(ROOT))
from prepare.encode_corpus import default_wapic_model   # noqa: E402


def check_piece(base: dict) -> int:
    try:
        import piece_tokenizer as pt
    except ImportError as e:
        print(f"  piece_tokenizer 导入失败:{e}")
        return 1

    # 用和生产代码同一套定位逻辑找词表,**不要**用基线里记下的绝对路径 ——
    # 那是抓基线那台机器上的路径,换台机器就不存在了。基线里的路径只当提示。
    from prepare.tokenizer import default_piece_model
    model = default_piece_model()
    if not model.exists():
        print(f"  找不到词表 {model}。跑 `bash prepare/install_deps.sh piece`")
        return 1

    tok = pt.Tokenizer()
    tok.load(str(model), dict="no")
    failures = 0
    if str(model) != base["model"]:
        print(f"  词表 {model.name}(基线抓自 {Path(base['model']).parent})")

    if tok.vocab_size() != base["vocab_size"]:
        print(f"  ✗ vocab_size {tok.vocab_size()} != 基线 {base['vocab_size']}")
        return 1

    allp = " ".join(tok.id_to_piece(i) for i in range(tok.vocab_size()))
    got = hashlib.sha256(allp.encode("utf-8", "surrogatepass")).hexdigest()
    if got != base["vocab_sha256"]:
        failures += 1
        print(f"  ✗ 全表指纹变了:{got[:24]}... != {base['vocab_sha256'][:24]}...")
        print("    id→piece 的映射整体变了,已发布模型的 embedding 会全部错位")
    else:
        print(f"  ✓ 全表指纹一致({tok.vocab_size()} 个 piece)")

    bad = []
    for s in base["samples"]:
        ids = tok.encode_as_ids(s["text"])
        pieces = tok.encode_as_pieces(s["text"])
        if ids != s["ids"] or pieces != s["pieces"]:
            bad.append((s["text"], s["ids"], ids))
    if bad:
        failures += 1
        print(f"  ✗ {len(bad)}/{len(base['samples'])} 条样例编码结果变了:")
        for text, old, new in bad[:3]:
            print(f"      {text!r}")
            print(f"        基线 {old[:12]}")
            print(f"        现在 {new[:12]}")
    else:
        print(f"  ✓ {len(base['samples'])} 条样例编码逐值一致")
    return failures


def check_wapic(base: dict) -> int:
    """只报告,不判失败 —— 切词变化影响掩码粒度,不影响词表。"""
    try:
        import wapic
    except ImportError as e:
        print(f"  wapic 导入失败:{e}")
        return 1

    model = default_wapic_model()
    if not model.exists():
        model = None
    if model is None:
        print("  找不到 .wac 模型,跳过。"
              "跑 `bash prepare/install_deps.sh wapic` 从 HF 下载")
        return 0

    seg = wapic.Segmenter(str(model))
    same_model = str(model) == base["model"]
    diffs = []
    for s in base["samples"]:
        words = seg.segment(s["text"])
        if words != s["words"]:
            diffs.append((s["text"], s["words"], words))

    print(f"  模型 {model.name}" + ("(与基线同一个)" if same_model else
                                     f"(基线用的是 {Path(base['model']).name})"))
    if not diffs:
        print(f"  ✓ {len(base['samples'])} 条样例切词一致")
    else:
        print(f"  ! {len(diffs)}/{len(base['samples'])} 条切词有变化"
              f"(只影响 WWM 掩码粒度,不影响词表):")
        for text, old, new in diffs[:3]:
            print(f"      {text!r}")
            print(f"        基线 {old}")
            print(f"        现在 {new}")
    return 0


def main() -> int:
    if not BASELINE.exists():
        print(f"没有基线文件 {BASELINE},跳过校验。")
        return 0
    base = json.loads(BASELINE.read_text())

    print("=== PieceTokenizer(编码必须一字不差)===")
    failures = check_piece(base["piece"])
    print("\n=== Wapic(切词变化只影响 WWM 粒度)===")
    failures += check_wapic(base["wapic"])

    if failures:
        print(f"\n{failures} 项不一致 —— 不要在这个状态下重跑 encode_corpus 或微调")
        return 1
    print("\n校验通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
