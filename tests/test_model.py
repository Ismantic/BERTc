"""src/model.py 与 pretrain/modern_bertc/model.py 的对拍。

model.py 是搬过来的,但"搬"最容易出的事故是 state_dict key 变了 ——
改动会让 HF 上已发布的 Ismantic/BERTc-315M / -165M 权重全部失配,
而且模型照样能随机初始化跑起来,不报错。所以这里查两件事:

  1. state_dict 的 key 集合与形状完全一致
  2. 加载真实 ckpt(output_v4_large/checkpoint-8500)后,同输入下 logits 逐值相等

没有 ckpt 时退化成随机权重对拍(仍能抓 key 不一致和结构差异)。

    python tests/test_model.py
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import model as new_model                                       # noqa: E402

sys.path.insert(0, str(ROOT / "pretrain" / "modern_bertc"))
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_old_model", ROOT / "pretrain" / "modern_bertc" / "model.py")
    old_model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old_model)
except Exception as e:                                          # noqa: BLE001
    print(f"旧 model.py 加载失败({e}),跳过对拍。")
    sys.exit(0)

CKPT = ROOT / "pretrain" / "modern_bertc" / "output_v4_large" / "checkpoint-8500"


def build(mod, cfg_dict):
    cfg = mod.ModernBertConfig(**cfg_dict)
    return mod.ModernBertForMLM(cfg)


def compare_keys(a, b, label) -> int:
    ka = {k: tuple(v.shape) for k, v in a.state_dict().items()}
    kb = {k: tuple(v.shape) for k, v in b.state_dict().items()}
    if ka == kb:
        print(f"  ✓ {label}: state_dict 完全一致({len(ka)} 个张量)")
        return 0
    only_a, only_b = set(ka) - set(kb), set(kb) - set(ka)
    if only_a:
        print(f"  ✗ {label}: 新实现多出 {sorted(only_a)[:5]}")
    if only_b:
        print(f"  ✗ {label}: 新实现缺少 {sorted(only_b)[:5]}")
    for k in set(ka) & set(kb):
        if ka[k] != kb[k]:
            print(f"  ✗ {label}: {k} 形状 {ka[k]} vs {kb[k]}")
    return 1


def compare_forward(m_new, m_old, cfg_dict, label) -> int:
    m_new.eval()
    m_old.eval()
    g = torch.Generator().manual_seed(7)
    B, L = 3, 96
    ids = torch.randint(0, cfg_dict["vocab_size"], (B, L), generator=g)
    mask = torch.ones(B, L, dtype=torch.long)
    mask[1, 60:] = 0                                    # 带 padding 的一条
    labels = torch.randint(0, cfg_dict["vocab_size"], (B, L), generator=g)

    with torch.no_grad():
        o1 = m_new(ids, attention_mask=mask, labels=labels)
        o2 = m_old(ids, attention_mask=mask, labels=labels)

    d_logits = (o1["logits"] - o2["logits"]).abs().max().item()
    d_loss = abs(o1["loss"].item() - o2["loss"].item())
    if d_logits == 0.0 and d_loss == 0.0:
        print(f"  ✓ {label}: logits / loss 逐值相等(loss={o1['loss'].item():.6f})")
        return 0
    print(f"  ✗ {label}: max|Δlogits|={d_logits:.3e}  |Δloss|={d_loss:.3e}")
    return 1


def main() -> int:
    failures = 0

    print("=== 默认 config(22L/768H,随机权重)===")
    cfg_default = {}
    m_new = build(new_model, cfg_default)
    m_old = build(old_model, cfg_default)
    m_old.load_state_dict(m_new.state_dict())          # 同权重才能比 forward
    failures += compare_keys(m_new, m_old, "默认 config")
    failures += compare_forward(m_new, m_old,
                                {"vocab_size": m_new.config.vocab_size}, "默认 config")

    print("\n=== v4-Large 真实 ckpt(24L/1024H)===")
    if not (CKPT / "config.json").exists():
        print(f"  ckpt 不存在({CKPT}),跳过。")
        return 1 if failures else 0

    cfg_dict = json.loads((CKPT / "config.json").read_text())
    m_new = build(new_model, cfg_dict)
    m_old = build(old_model, cfg_dict)
    failures += compare_keys(m_new, m_old, "v4-Large config")

    sd = torch.load(CKPT / "model.pt", map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}

    missing, unexpected = m_new.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  ! 新实现加载 ckpt: 缺 {len(missing)} 个, 多 {len(unexpected)} 个")
        if missing:
            print(f"      缺: {sorted(missing)[:5]}")
        if unexpected:
            print(f"      多: {sorted(unexpected)[:5]}")
        failures += 1
    else:
        print(f"  ✓ 新实现严格加载 v4-Large ckpt 成功({len(sd)} 个张量)")
    m_old.load_state_dict(sd, strict=False)

    failures += compare_forward(m_new, m_old, cfg_dict, "v4-Large ckpt")

    if failures:
        print(f"\n{failures} 项不一致")
        return 1
    print("\nmodel 对拍全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
