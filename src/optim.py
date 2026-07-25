"""优化器与学习率调度。只依赖 torch。

替掉两个外部依赖:
  optimi.StableAdamW                        → StableAdamW
  transformers.get_linear_schedule_with_warmup → linear_schedule_with_warmup

damped_cosine_lr / current_grad_accum 原本就写在 train_modern.py 里,搬过来集中放。

StableAdamW 的实现与 optimi 0.3.3 逐运算对齐(见 tests/test_optim.py 的对拍),
因为 v4-Large(HF 上的 Ismantic/BERTc-315M)就是用它训出来的,换个数值行为
不同的优化器等于换了个 recipe。
"""
import math

import torch
from torch.optim import Optimizer


def _debias_beta(beta: float, step: int) -> float:
    """把 Adam 的 bias correction 折进 beta 本身。

    等价于 beta_hat = beta * (1 - beta^(step-1)) / (1 - beta^step),
    这样滑动平均从第一步起就是无偏的,不需要再对 m / v 做除法修正。
    """
    return (beta ** step - beta) / (beta ** step - 1)


class StableAdamW(Optimizer):
    """AdamW + Adafactor 式的 update 裁剪(Wortsman et al. 2023)。

    与普通 AdamW 的两点区别:
      1. bias correction 折进 beta(见 _debias_beta),不单独修正 m / v
      2. 每个 tensor 算一个 RMS = sqrt(mean(g^2 / max(v, eps^2))),
         用 lr / max(1, RMS) 当有效学习率 —— 梯度相对二阶矩偏大时自动降 lr,
         这是它在 bf16 下比 AdamW 稳的原因

    weight decay 是 decoupled 的,且乘的是**裁剪后**的 lr。
    """

    def __init__(self, params, lr: float, betas=(0.9, 0.99), eps: float = 1e-6,
                 weight_decay: float = 1e-2):
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"beta1 非法: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"beta2 非法: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"eps 非法: {eps}")
        super().__init__(params, dict(lr=lr, beta1=betas[0], beta2=betas[1],
                                      eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["beta1"], group["beta2"]
            lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("StableAdamW 不支持稀疏梯度")

                state = self.state[p]
                if not state:
                    state["step"] = torch.tensor(0, dtype=torch.int32, device=p.device)
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["eps_sq"] = torch.tensor(eps ** 2, dtype=p.dtype,
                                                   device=p.device)

                state["step"] += 1
                step = int(state["step"].item())
                beta1_hat = _debias_beta(beta1, step)
                beta2_hat = _debias_beta(beta2, step)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                exp_avg.lerp_(grad, weight=1 - beta1_hat)
                exp_avg_sq.mul_(beta2_hat).addcmul_(grad, grad, value=1 - beta2_hat)

                # per-tensor RMS 裁剪
                rms = grad.pow(2).div_(exp_avg_sq.maximum(state["eps_sq"])).mean().sqrt()
                eff_lr = lr / max(1.0, rms.item())

                if wd != 0:
                    p.mul_(1 - eff_lr * wd)
                p.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(eps), value=-eff_lr)

        return loss


# ---------------------------------------------------------------- LR 调度

def linear_schedule_with_warmup(optimizer, num_warmup_steps: int,
                                num_training_steps: int, last_epoch: int = -1):
    """线性 warmup 后线性衰减到 0。等价于 transformers 的同名函数。

    MT / CSC 微调和 inline eval 都用这条。
    """
    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return step / max(1, num_warmup_steps)
        remain = num_training_steps - num_warmup_steps
        return max(0.0, (num_training_steps - step) / max(1, remain))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def damped_cosine_lr(step: int, total_steps: int, peak: float, min_lr: float,
                     n_cycles: int = 1, damp_gamma: float = 0.0,
                     warmup_steps: int = 0) -> float:
    """Damped cosine(Smith 2017 cyclical + 余弦退火 + 阻尼),预训练用。

       η(s) = (Peak(p)+Valley(p))/2 + (Peak(p)-Valley(p))/2 · cos(π(2N-1)p)
       其中 p = s / S
            Peak(p)   = peak * (1 - (1 - damp_gamma) * p)
            Valley(p) = peak/2 * (1 - p) + min_lr * p
       N=1 退化成单 cycle 余弦(v4-Large 用的就是 N=1、damp_gamma=0)。

       warmup_steps:前 warmup_steps 步线性 0 → peak。
    """
    if step < warmup_steps:
        return peak * (step + 1) / max(1, warmup_steps)
    s = step - warmup_steps
    S = max(1, total_steps - warmup_steps)
    p = min(1.0, s / S)
    peak_p = peak * (1.0 - (1.0 - damp_gamma) * p)
    valley_p = (peak * 0.5) * (1.0 - p) + min_lr * p
    mid = (peak_p + valley_p) * 0.5
    amp = (peak_p - valley_p) * 0.5
    return mid + amp * math.cos(math.pi * (2 * n_cycles - 1) * p)


def current_grad_accum(step: int, total_steps: int, peak: int,
                       warmup_frac: float = 0.05, min_accum: int = 1) -> int:
    """batch size warmup:前 warmup_frac 的 step 里 grad_accum 从 min_accum 线性升到 peak。

    batch_size 固定,所以 grad_accum 线性 ramp 等价于 effective batch ramp
    (Cramming 论文与 ModernBERT 配置里的 batch_size_warmup_tokens)。
    """
    if total_steps <= 0 or warmup_frac <= 0:
        return peak
    p = step / total_steps
    if p >= warmup_frac:
        return peak
    val = min_accum + (peak - min_accum) * (p / warmup_frac)
    return max(min_accum, int(round(val)))
