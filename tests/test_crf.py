"""src/crf.py 与 torchcrf 0.7.2 的数值对拍。

MT 的 CWS / NER head 都挂 CRF,重写必须逐值等价,否则 joint 1.4712 复现不了。
torchcrf 不是 src/ 的依赖,只在这里当参照物 —— 装了才跑,没装就跳过。

    python tests/test_crf.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.crf import CRF as MyCRF                                     # noqa: E402

try:
    from torchcrf import CRF as RefCRF
except ImportError:
    print("torchcrf 未安装,跳过对拍。")
    sys.exit(0)


def make_batch(batch, seq_len, num_tags, seed):
    g = torch.Generator().manual_seed(seed)
    emissions = torch.randn(batch, seq_len, num_tags, generator=g, dtype=torch.float64)
    tags = torch.randint(0, num_tags, (batch, seq_len), generator=g)
    # 变长:每条随机截断,但至少 1 步;padding 只在尾部
    lengths = torch.randint(1, seq_len + 1, (batch,), generator=g)
    lengths[0] = seq_len                       # 保证有满长的一条
    mask = torch.arange(seq_len).unsqueeze(0) < lengths.unsqueeze(1)
    return emissions, tags, mask.to(torch.uint8)


def sync(mine, ref):
    """让两个 CRF 用同一组转移参数。"""
    with torch.no_grad():
        ref.start_transitions.copy_(mine.start_transitions)
        ref.end_transitions.copy_(mine.end_transitions)
        ref.transitions.copy_(mine.transitions)


def main() -> int:
    torch.manual_seed(0)
    failures = 0

    for num_tags in (4, 7, 13):
        for seed in range(4):
            mine = MyCRF(num_tags, batch_first=True).double()
            ref = RefCRF(num_tags, batch_first=True).double()
            sync(mine, ref)

            emissions, tags, mask = make_batch(6, 12, num_tags, seed)

            # --- 对数似然,四种 reduction
            for red in ("none", "sum", "mean", "token_mean"):
                a = mine(emissions, tags, mask=mask, reduction=red)
                b = ref(emissions, tags, mask=mask, reduction=red)
                if not torch.allclose(a, b, atol=1e-10, rtol=0):
                    failures += 1
                    print(f"  ✗ llh 不一致 num_tags={num_tags} seed={seed} "
                          f"reduction={red}  max|Δ|={(a - b).abs().max():.3e}")

            # --- 梯度
            e1 = emissions.clone().requires_grad_(True)
            e2 = emissions.clone().requires_grad_(True)
            mine(e1, tags, mask=mask, reduction="mean").neg().backward()
            ref(e2, tags, mask=mask, reduction="mean").neg().backward()
            dg = (e1.grad - e2.grad).abs().max().item()
            dt = (mine.transitions.grad - ref.transitions.grad).abs().max().item()
            if dg > 1e-10 or dt > 1e-10:
                failures += 1
                print(f"  ✗ 梯度不一致 num_tags={num_tags} seed={seed} "
                      f"d_emis={dg:.3e} d_trans={dt:.3e}")

            # --- Viterbi 解码
            p1 = mine.decode(emissions, mask=mask)
            p2 = ref.decode(emissions, mask=mask)
            if p1 != p2:
                failures += 1
                print(f"  ✗ 解码不一致 num_tags={num_tags} seed={seed}")
                for i, (x, y) in enumerate(zip(p1, p2)):
                    if x != y:
                        print(f"      第 {i} 条: {x} vs {y}")
                        break

            # --- 全长 mask(无 padding)
            full = torch.ones_like(mask)
            if not torch.allclose(mine(emissions, tags, mask=full, reduction="sum"),
                                  ref(emissions, tags, mask=full, reduction="sum"),
                                  atol=1e-10, rtol=0):
                failures += 1
                print(f"  ✗ 全长 mask 不一致 num_tags={num_tags} seed={seed}")

    if failures:
        print(f"\n{failures} 项不一致")
        return 1
    print("CRF 对拍全部通过(llh 4 种 reduction / 梯度 / Viterbi / 全长 mask,"
          "3 种 tag 数 × 4 组随机数据)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
