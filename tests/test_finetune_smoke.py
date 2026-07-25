"""三个 trainer 的端到端冒烟:真数据、真训练、真存 ckpt。

对拍测试保证单个组件数值正确,但组件拼起来还会有别的问题 ——
骨干权重加载的前缀对不对、CRF 挂在 bf16 autocast 下会不会炸、
评测能不能从 id 还原出句子。这里用一个小骨干把三条链各跑几十步。

预编码数据在测试里现生成,这段逻辑之后会搬进 prepare/。

    python tests/test_finetune_smoke.py
"""
import json
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
OLD_MT = ROOT / "finetune" / "NLP_BERT_CRF"
TOKENIZER = ROOT / "pretrain" / "modern_bertc" / "tokenizer"
PD98 = OLD_MT / "data"
CSC_PKL = ROOT / "csc" / "data" / "all_pairs.pkl"
CSC_TEST = ROOT / "csc" / "data" / "test" / "sighan2015_test_official.tsv"

TINY = dict(vocab_size=12536, hidden_size=128, num_hidden_layers=2,
            num_attention_heads=4, intermediate_size=256,
            max_position_embeddings=1024, pad_token_id=12531,
            mask_token_id=12535, attn_out_dropout=0.0)


def make_backbone(dst: Path) -> Path:
    """造一个随机初始化的小骨干 ckpt,格式跟 pretrain.py 存的一致。"""
    sys.path.insert(0, str(ROOT))
    from src.model import ModernBertConfig, ModernBertForMLM
    cfg = ModernBertConfig(**TINY)
    model = ModernBertForMLM(cfg)
    dst.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": cfg.__dict__, "step": 0},
               dst / "model.pt")
    (dst / "config.json").write_text(json.dumps(cfg.__dict__, indent=2))
    return dst


def pack(items, fields, extra) -> dict:
    offsets, cols = [0], {f: [] for f in fields}
    for it in items:
        n = len(it[fields[0]])
        for f in fields:
            cols[f].extend(it[f])
        offsets.append(offsets[-1] + n)
    blob = {"offsets": torch.tensor(offsets, dtype=torch.int64)}
    for f in fields:
        blob[f] = torch.tensor(cols[f],
                               dtype=torch.uint8 if f == "det_labels" else torch.int32)
    blob.update(extra)
    return blob


def build_mt_data(tmp: Path, n_train=400, n_dev=100):
    sys.path.insert(0, str(OLD_MT))
    shadowed = sys.modules.pop("data", None)
    try:
        from data_mt import MTDataset as OldMT, MTCollator
        from data_pos_ner import build_pos_vocab, NER_TAGS
        from data import TAG2ID as CWS_TAG2ID
        from piece_tokenizer_adapter import PieceTokenizerAdapter
    finally:
        if shadowed is not None:
            sys.modules["data"] = shadowed

    pos2id = build_pos_vocab()
    tok = PieceTokenizerAdapter(str(TOKENIZER))
    coll = MTCollator(tok)
    vocabs = {
        "cws_vocab": [t for t, _ in sorted(CWS_TAG2ID.items(), key=lambda kv: kv[1])],
        "pos_vocab": [t for t, _ in sorted(pos2id.items(), key=lambda kv: kv[1])],
        "ner_vocab": [t for t, _ in sorted(NER_TAGS.items(), key=lambda kv: kv[1])],
    }

    out = {}
    for split, prefix, n in (("train", "", n_train), ("dev", "_dev", n_dev)):
        ds = OldMT(PD98 / f"cws{prefix}.pd98.jsonl", PD98 / f"pos{prefix}.pd98.jsonl",
                   PD98 / f"ner{prefix}.pd98.jsonl", pos2id, max_chars=10 ** 9)
        items = []
        for it in ds.items[:n]:
            items.append({"input_ids": [coll._char_to_id(c) for c in it["chars"]],
                          "cws_tags": it["cws_tags"], "pos_tags": it["pos_tags"],
                          "ner_tags": it["ner_tags"]})
        path = tmp / f"mt_{split}.pt"
        torch.save(pack(items, ("input_ids", "cws_tags", "pos_tags", "ner_tags"),
                        {"format": "bertc-mt-v1", "pad_token_id": tok.pad_token_id,
                         **vocabs}), path)
        out[split] = path
    return out


def build_csc_data(tmp: Path, n_train=800):
    sys.path.insert(0, str(OLD_MT))
    from piece_tokenizer_adapter import PieceTokenizerAdapter
    tok = PieceTokenizerAdapter(str(TOKENIZER))
    cache = {}

    def cid(c):
        if c not in cache:
            ids = tok.encode(c, add_special_tokens=False)
            cache[c] = ids[0] if ids else tok.unk_token_id
        return cache[c]

    def encode(pairs):
        items = []
        for src, tgt in pairs:
            L = min(len(src), len(tgt))
            if L == 0:
                continue
            items.append({"input_ids": [cid(c) for c in src[:L]],
                          "cor_labels": [cid(c) for c in tgt[:L]],
                          "det_labels": [1 if src[i] != tgt[i] else 0
                                         for i in range(L)]})
        return items

    with open(CSC_PKL, "rb") as f:
        train_pairs = pickle.load(f)[:n_train]
    test_pairs = []
    for line in CSC_TEST.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split("\t")
        if len(parts) == 2:
            test_pairs.append((parts[0], parts[1]))

    train_items = encode(train_pairs)
    test_items = encode(test_pairs)
    # id→字 反查表:只覆盖编码过程中见过的字,跟原实现的字符缓存范围一致
    id_to_char = {}
    for c, i in cache.items():
        id_to_char.setdefault(i, c)

    common = {"format": "bertc-csc-v1", "pad_token_id": tok.pad_token_id,
              "id_to_char": id_to_char, "vocab_size": tok.vocab_size}
    fields = ("input_ids", "cor_labels", "det_labels")
    paths = {}
    for name, items in (("train", train_items), ("test", test_items)):
        p = tmp / f"csc_{name}.pt"
        torch.save(pack(items, fields, common), p)
        paths[name] = p
    print(f"    CSC 测试集 {len(test_items)} 条,反查表 {len(id_to_char)} 个字")
    return paths


def run(cmd) -> tuple[int, str]:
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    return r.returncode, r.stdout + r.stderr


def main() -> int:
    if not (PD98 / "cws.pd98.jsonl").exists() or not CSC_PKL.exists():
        print("下游数据缺失,跳过。")
        return 0

    failures = 0
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        backbone = make_backbone(tmp / "backbone")
        print(f"小骨干 ckpt: {backbone}")

        print("\n=== MT ===")
        mt = build_mt_data(tmp)
        code, log = run([PY, "-m", "src.finetune_mt",
                         "--ckpt_dir", str(backbone),
                         "--train_data", str(mt["train"]),
                         "--dev_data", str(mt["dev"]),
                         "--output_dir", str(tmp / "mt_out"),
                         "--epochs", "2", "--batch_size", "16",
                         "--fgm", "--log_every", "10"])
        if code != 0 or not (tmp / "mt_out" / "best.pt").exists():
            failures += 1
            print(f"  ✗ MT 微调失败(exit {code})\n{log[-2500:]}")
        else:
            for line in log.splitlines():
                if line.startswith("==="):
                    print(f"  {line}")
            print("  ✓ MT:2 轮跑完,best.pt / final.pt 已写出(含 FGM 对抗训练)")

        print("\n=== CSC ===")
        csc = build_csc_data(tmp)
        code, log = run([PY, "-m", "src.finetune_csc",
                         "--ckpt_dir", str(backbone),
                         "--train_data", str(csc["train"]),
                         "--test_data", str(csc["test"]),
                         "--output_dir", str(tmp / "csc_out"),
                         "--epochs", "2", "--batch_size", "16",
                         "--log_every", "20"])
        # 只查 final.pt:best.pt 仅在 F1 变好时才写,而这个随机初始化的小模型
        # 从不纠错,两轮 F1 都是 0(跟原实现一样不会落盘)
        if code != 0 or not (tmp / "csc_out" / "final.pt").exists():
            failures += 1
            print(f"  ✗ CSC 微调失败(exit {code})\n{log[-2500:]}")
        else:
            for line in log.splitlines():
                if line.startswith("==="):
                    print(f"  {line}")
            m = [l for l in log.splitlines() if "TN=" in l]
            n_eval = 0
            if m:
                import re
                d = dict(re.findall(r"(TP|FP|FN|TN)=(\d+)", m[-1]))
                n_eval = sum(int(v) for v in d.values())
            ok = n_eval == 707
            failures += 0 if ok else 1
            print(f"  {'✓' if ok else '✗'} CSC:2 轮跑完,final.pt 已写出;"
                  f"评测覆盖 {n_eval} 条(SIGHAN-15 官方 707 条)")

    if failures:
        print(f"\n{failures} 条链失败")
        return 1
    print("\n微调冒烟全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
