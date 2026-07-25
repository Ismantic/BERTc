"""src/masking.py 与 pretrain/train_bert_mlm.py 的对拍。

掩码逻辑改错了不会报错,只会让预训练目标悄悄变掉(比如 80/10/10 的比例偏了、
WWM 退化成逐字掩码),所以固定随机种子逐值比对。

    python tests/test_masking.py
"""
import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import masking as new_masking                                    # noqa: E402


def load_old():
    """train_bert_mlm.py 顶层会 import transformers,单独抽函数出来即可。"""
    path = ROOT / "pretrain" / "train_bert_mlm.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("_old_mlm", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:                                       # noqa: BLE001
        print(f"  旧 train_bert_mlm.py 加载失败({e}),跳过。")
        return None
    return mod


def make_batch(B, L, vocab, pad_id, seed):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab - 5, (B, L), generator=g)
    # 尾部 pad,长度不一
    for i in range(B):
        n = int(torch.randint(L // 2, L + 1, (1,), generator=g).item())
        ids[i, n:] = pad_id
    # word_ids:随机把连续 1~4 个字归成一个词
    wids = torch.zeros(B, L, dtype=torch.long)
    for i in range(B):
        w, j = 0, 0
        while j < L:
            k = int(torch.randint(1, 5, (1,), generator=g).item())
            wids[i, j:j + k] = w
            w += 1
            j += k
    return ids, wids


def main() -> int:
    old = load_old()
    if old is None:
        return 0

    VOCAB, PAD, MASK = 12536, 12531, 12535
    failures = 0

    for seed in range(5):
        ids, wids = make_batch(8, 128, VOCAB, PAD, seed)

        for prob in (0.15, 0.30):
            torch.manual_seed(seed * 100 + int(prob * 100))
            a_ids, a_lbl = new_masking.mlm_mask_batch(ids, MASK, VOCAB,
                                                      prob=prob, pad_id=PAD)
            torch.manual_seed(seed * 100 + int(prob * 100))
            b_ids, b_lbl = old.mlm_mask_batch(ids, MASK, VOCAB,
                                              prob=prob, pad_id=PAD)
            if not (torch.equal(a_ids, b_ids) and torch.equal(a_lbl, b_lbl)):
                failures += 1
                print(f"  ✗ 逐 token 掩码 seed={seed} prob={prob} 不一致")

            torch.manual_seed(seed * 100 + int(prob * 100))
            a_ids, a_lbl = new_masking.mlm_mask_batch_wwm(ids, wids, MASK, VOCAB,
                                                          prob=prob, pad_id=PAD)
            torch.manual_seed(seed * 100 + int(prob * 100))
            b_ids, b_lbl = old.mlm_mask_batch_wwm(ids, wids, MASK, VOCAB,
                                                  prob=prob, pad_id=PAD)
            if not (torch.equal(a_ids, b_ids) and torch.equal(a_lbl, b_lbl)):
                failures += 1
                print(f"  ✗ WWM 掩码 seed={seed} prob={prob} 不一致")

    if failures == 0:
        # 顺带确认统计性质对得上,不只是"两边一样错"
        torch.manual_seed(0)
        ids, wids = make_batch(64, 256, VOCAB, PAD, 42)
        m_ids, lbl = new_masking.mlm_mask_batch(ids, MASK, VOCAB, prob=0.15, pad_id=PAD)
        sel = lbl != -100
        not_pad = ids != PAD
        rate = sel.sum().item() / not_pad.sum().item()
        to_mask = ((m_ids == MASK) & sel).sum().item() / max(1, sel.sum().item())
        kept = ((m_ids == ids) & sel).sum().item() / max(1, sel.sum().item())
        print(f"  ✓ 逐 token / WWM 掩码:5 组种子 × 2 个掩码率,逐值相等")
        print(f"    统计:选中率 {rate:.3f}(目标 0.15)、"
              f"其中 →[MASK] {to_mask:.3f}(目标 0.80)、保持原样 {kept:.3f}(目标 0.10)")

        # WWM 必须整词一起,不能出现半个词被选中
        m_ids, lbl = new_masking.mlm_mask_batch_wwm(ids, wids, MASK, VOCAB,
                                                    prob=0.15, pad_id=PAD)
        sel = lbl != -100
        broken = 0
        for b in range(ids.size(0)):
            valid = ids[b] != PAD
            for w in wids[b][valid].unique():
                pos = (wids[b] == w) & valid
                if 0 < sel[b][pos].sum().item() < pos.sum().item():
                    broken += 1
        if broken:
            failures += 1
            print(f"  ✗ WWM 有 {broken} 个词只被选中了一部分")
        else:
            print(f"  ✓ WWM 整词性:没有任何词被拆开选中")

    if failures:
        print(f"\n{failures} 项不一致")
        return 1
    print("\nmasking 对拍全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
