"""src/optim.py 的 StableAdamW 与 optimi 0.3.3 的数值对拍。

v4-Large(HF 上的 Ismantic/BERTc-315M)就是用 optimi.StableAdamW 训出来的,
重写必须逐步等价 —— 优化器数值行为一变,等于换了个 recipe。
optimi 不是 src/ 的依赖,只在这里当参照物。

    python tests/test_optim.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.optim import StableAdamW as MyOpt, linear_schedule_with_warmup   # noqa: E402

try:
    from optimi import StableAdamW as RefOpt
except ImportError:
    RefOpt = None


def run(opt_cls, shapes, grads_per_step, lr, betas, eps, wd, ref: bool):
    """用同一串梯度跑 N 步,返回最终参数。"""
    g = torch.Generator().manual_seed(1234)
    params = [torch.randn(*s, generator=g, dtype=torch.float64).requires_grad_(True)
              for s in shapes]
    # 两组:一组带 weight decay,一组不带 —— 覆盖 make_param_groups 的实际用法
    groups = [{"params": params[:len(params) // 2], "weight_decay": wd},
              {"params": params[len(params) // 2:], "weight_decay": 0.0}]
    kw = dict(lr=lr, betas=betas, eps=eps)
    if ref:
        kw.update(foreach=False, triton=False, kahan_sum=False)
    opt = opt_cls(groups, **kw)

    for grads in grads_per_step:
        for p, gr in zip(params, grads):
            p.grad = gr.clone()
        opt.step()
        opt.zero_grad()
    return params


def compare_stableadamw() -> int:
    if RefOpt is None:
        print("optimi 未安装,跳过 StableAdamW 对拍。")
        return 0

    shapes = [(64, 32), (128,), (16, 16, 4), (7,)]
    failures = 0

    for case, (lr, betas, eps, wd, n_steps) in enumerate([
        (8e-4, (0.9, 0.95), 1e-6, 0.01, 25),     # v4-Large 实际用的超参
        (1e-3, (0.9, 0.99), 1e-8, 0.1, 15),
        (3e-4, (0.8, 0.999), 1e-6, 0.0, 30),
    ]):
        g = torch.Generator().manual_seed(99 + case)
        # 梯度量级跨几个数量级,专门压 RMS 裁剪那条分支
        grads_per_step = [
            [torch.randn(*s, generator=g, dtype=torch.float64)
             * (10.0 ** (step % 5 - 2)) for s in shapes]
            for step in range(n_steps)
        ]
        mine = run(MyOpt, shapes, grads_per_step, lr, betas, eps, wd, ref=False)
        ref = run(RefOpt, shapes, grads_per_step, lr, betas, eps, wd, ref=True)

        worst = max((a - b).abs().max().item() for a, b in zip(mine, ref))
        scale = max(a.abs().max().item() for a in mine)
        if worst > 1e-12 * max(1.0, scale):
            failures += 1
            print(f"  ✗ case {case} (lr={lr} betas={betas} eps={eps} wd={wd} "
                  f"{n_steps} 步): max|Δ| = {worst:.3e}")
        else:
            print(f"  ✓ case {case}: lr={lr} betas={betas} eps={eps} wd={wd} "
                  f"{n_steps} 步,max|Δ| = {worst:.3e}")
    return failures


def compare_schedule() -> int:
    """linear_schedule_with_warmup 与 transformers 同名函数对拍。"""
    try:
        from transformers import get_linear_schedule_with_warmup as ref_sched
    except ImportError:
        print("transformers 未安装,跳过 LR 调度对拍。")
        return 0

    failures = 0
    for warmup, total in [(0, 100), (10, 100), (50, 200), (1, 3)]:
        p1 = torch.zeros(1, requires_grad=True)
        p2 = torch.zeros(1, requires_grad=True)
        o1 = torch.optim.SGD([p1], lr=0.1)
        o2 = torch.optim.SGD([p2], lr=0.1)
        s1 = linear_schedule_with_warmup(o1, warmup, total)
        s2 = ref_sched(o2, warmup, total)
        for _ in range(total + 5):
            a, b = o1.param_groups[0]["lr"], o2.param_groups[0]["lr"]
            if abs(a - b) > 1e-12:
                failures += 1
                print(f"  ✗ 调度不一致 warmup={warmup} total={total}: {a} vs {b}")
                break
            o1.step(); o2.step(); s1.step(); s2.step()
        else:
            print(f"  ✓ 调度 warmup={warmup} total={total}:{total + 5} 步全等")
    return failures


def main() -> int:
    print("=== StableAdamW ===")
    f = compare_stableadamw()
    print("\n=== linear_schedule_with_warmup ===")
    f += compare_schedule()
    if f:
        print(f"\n{f} 项不一致")
        return 1
    print("\n优化器与调度对拍全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
