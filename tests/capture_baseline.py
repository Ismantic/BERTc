"""抓一份 PieceTokenizer / Wapic 的行为基线,给 test_tokenizer.py 当参照。

**重建这两个 C++ 依赖之前**跑一次,重建之后用 test_tokenizer.py 比对。
顺序反了就失去意义 —— 基线是用来发现"重建把行为改了"的。

    python tests/capture_baseline.py            # 写 tests/fixtures/tokenizer_baseline.json
    python tests/capture_baseline.py --force    # 覆盖已有基线

piece 那半是**红线**:编码一旦变了,12536 词表就跟已发布模型的 embedding 错位。
wapic 那半是**参考**:切词变了只影响 WWM 掩码粒度。
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIECE_MODEL = "pretrain/modern_bertc/tokenizer/piece.model"
WAPIC_MODEL_CANDIDATES = [
    "/home/tfbao/Shiyu/Wapic/data/model/wapic-cws.wac",
    "/home/tfbao/Shiyu/wapic_models_backup/wapic-20260602-h19_1-full.wac",
]

SAMPLES = [
    "人民日报社论:坚持改革开放不动摇",
    "The quick brown fox jumps over the lazy dog.",
    "混合 Chinese 和 English 的句子,还有数字 12345 和标点!?",
    "繁體字測試:國語辭典、學術研究",
    "特殊符号 ①②③ ÷×± αβγ",
    "换行\n制表\t和  多个空格",
    "单",
    "重复重复重复重复重复重复重复重复",
    "中国科学院计算技术研究所与北京大学合作",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="覆盖已有基线")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "tests" / "fixtures" / "tokenizer_baseline.json")
    args = ap.parse_args()

    if args.out.exists() and not args.force:
        sys.exit(f"{args.out} 已存在。确认要用当前行为覆盖基线再加 --force")

    out = {"note": "重建 PieceTokenizer / Wapic 前抓的行为基线,"
                   "由 tests/test_tokenizer.py 校验。"}

    import piece_tokenizer as pt
    tok = pt.Tokenizer()
    tok.load(str(ROOT / PIECE_MODEL), dict="no")
    allp = " ".join(tok.id_to_piece(i) for i in range(tok.vocab_size()))
    out["piece"] = {
        "model": PIECE_MODEL,
        "vocab_size": tok.vocab_size(),
        "vocab_sha256": hashlib.sha256(
            allp.encode("utf-8", "surrogatepass")).hexdigest(),
        "samples": [{"text": s, "ids": tok.encode_as_ids(s),
                     "pieces": tok.encode_as_pieces(s)} for s in SAMPLES],
    }

    wac = next((p for p in WAPIC_MODEL_CANDIDATES if Path(p).exists()), None)
    if wac is None:
        sys.exit("找不到 .wac 模型,先跑 bash prepare/install_deps.sh wapic")
    import wapic
    seg = wapic.Segmenter(wac)
    out["wapic"] = {
        "model": wac,
        "api": "segment",          # 旧版叫 cut_smart,2026-07 的 Wapic 改名了
        "samples": [{"text": s, "words": seg.segment(s)} for s in SAMPLES],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"基线已写入 {args.out}")
    print(f"  piece  vocab={out['piece']['vocab_size']}  "
          f"指纹={out['piece']['vocab_sha256'][:24]}...")
    print(f"  wapic  {Path(wac).name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
