"""Modern BERTc v2 MLM 预训练 — release-aligned + 蚂蚁 Chinese ModernBERT 经验。

关键改动 vs v1:
  - StableAdamW(β2=0.95, wd=0.01, eps=1e-6, filter_bias_norm_wd)
  - Damped cosine LR(蚂蚁论文消融:更稳,fewer divergence restarts)
  - Dynamic MLM curriculum 15%→30%→15%(蚂蚁论文核心)
  - Cross-doc attention 隔离:数据加 seg_ids → flex_attention block-diag mask
  - grad_clip=1.0
  - Megatron init(在 model.py 里)
  - LayerNorm no-bias / embed_norm / skip_first_prenorm(在 model.py 里)

数据:
  --train_data .pt        [N, L] int32   token ids
  --train_data + ".wid"   [N, L] int32   word_ids(WWM 用)
  --train_data + ".seg"   [N, L] uint8   doc_ids in chunk(cross-doc 隔离用)

config 通过 CLI 传(默认 22L/768H/1152I/12h release-aligned)。
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_bert_mlm import mlm_mask_batch, mlm_mask_batch_wwm, _load_memmap_or_pt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import ModernBertConfig, ModernBertForMLM


# ============ Dataset(加 seg_ids 支持)============

class ModernBertcDataset(Dataset):
    """memmap 加载 ids + word_ids + seg_ids。getitem 返回 (ids, wids, segs) tuple。"""
    def __init__(self, pt_path, max_chunks=None,
                 word_ids_path=None, seg_ids_path=None):
        data, mode = _load_memmap_or_pt(pt_path)
        if max_chunks is not None and data.shape[0] > max_chunks:
            data = data[:max_chunks]
        self.data = data
        N, L = data.shape
        print(f"Loaded {N:,} chunks × {L} from {pt_path} "
              f"({data.numel()*data.element_size()/1e9:.1f} GB, {data.dtype}, {mode})")

        self.word_ids = None
        if word_ids_path is not None and os.path.exists(word_ids_path):
            wdata, wmode = _load_memmap_or_pt(word_ids_path)
            if max_chunks is not None and wdata.shape[0] > max_chunks:
                wdata = wdata[:max_chunks]
            assert wdata.shape == data.shape, f"wids shape {wdata.shape} != {data.shape}"
            self.word_ids = wdata
            print(f"  + word_ids from {word_ids_path} ({wdata.dtype}, {wmode}) → WWM")

        self.seg_ids = None
        if seg_ids_path is not None and os.path.exists(seg_ids_path):
            # uint8 memmap
            seg_meta_path = seg_ids_path + ".meta"
            if os.path.exists(seg_meta_path):
                meta = json.load(open(seg_meta_path))
                shape = tuple(meta["shape"])
                dtype = np.dtype(meta["dtype"])
            else:
                shape = data.shape
                dtype = np.uint8
            sarr = np.memmap(seg_ids_path, dtype=dtype, mode="r", shape=shape)
            if max_chunks is not None and sarr.shape[0] > max_chunks:
                sarr = sarr[:max_chunks]
            assert sarr.shape == data.shape, f"seg shape {sarr.shape} != {data.shape}"
            self.seg_ids = torch.from_numpy(sarr)
            print(f"  + seg_ids from {seg_ids_path} ({self.seg_ids.dtype}, memmap) "
                  f"→ cross-doc isolation")

    def __len__(self): return self.data.shape[0]

    def __getitem__(self, i):
        ids = self.data[i].to(torch.long)
        wids = self.word_ids[i].to(torch.long) if self.word_ids is not None else None
        segs = self.seg_ids[i].to(torch.int32) if self.seg_ids is not None else None
        return ids, wids, segs


def collate_fn(batch):
    """batch = list of (ids, wids, segs)。stack 并返回 dict。"""
    ids = torch.stack([b[0] for b in batch])
    has_wids = batch[0][1] is not None
    has_segs = batch[0][2] is not None
    wids = torch.stack([b[1] for b in batch]) if has_wids else None
    segs = torch.stack([b[2] for b in batch]) if has_segs else None
    return {"ids": ids, "wids": wids, "segs": segs}


# ============ Dynamic MLM curriculum(蚂蚁论文 Section 3.2)============

def dynamic_mlm_prob(step: int, total_steps: int, warmup_frac: float = 0.05,
                     low: float = 0.15, high: float = 0.30) -> float:
    """Anti-curriculum:warmup 期 low → high(逼模型早期 global reasoning),
    main 期 high → low(后期 local refinement)。
    """
    if total_steps <= 0:
        return high
    p = step / total_steps
    wfrac = max(1e-6, warmup_frac)
    if p < wfrac:
        # warmup: low → high linear
        t = p / wfrac
        return low + (high - low) * t
    else:
        # main: high → low linear
        t = (p - wfrac) / max(1e-6, 1.0 - wfrac)
        t = min(1.0, t)
        return high - (high - low) * t


# ============ Damped Cosine LR(蚂蚁论文 Section 3.3,Eq 1)============

def damped_cosine_lr(step: int, total_steps: int, peak: float, min_lr: float,
                     n_cycles: int = 1, damp_gamma: float = 0.0,
                     warmup_steps: int = 0) -> float:
    """Damped cosine schedule(Smith 2017 cyclical + cosine annealing + 阻尼)。
       η(s) = (Peak(p)+Valley(p))/2 + (Peak(p)-Valley(p))/2 · cos(π(2N-1)p)
       where p = s / S
             Peak(p)   = peak * (1 - (1 - damp_gamma) * p)
             Valley(p) = peak/2 * (1 - p) + min_lr * p
       N=1 退化为单 cycle 余弦。

       warmup_steps:前 warmup_steps 步线性 0 → peak,然后开始 damped cosine。
    """
    if step < warmup_steps:
        return peak * (step + 1) / max(1, warmup_steps)
    s = step - warmup_steps
    S = max(1, total_steps - warmup_steps)
    p = min(1.0, s / S)
    Peak_p = peak * (1.0 - (1.0 - damp_gamma) * p)
    Valley_p = (peak * 0.5) * (1.0 - p) + min_lr * p
    mid = (Peak_p + Valley_p) * 0.5
    amp = (Peak_p - Valley_p) * 0.5
    return mid + amp * math.cos(math.pi * (2 * n_cycles - 1) * p)


# ============ Batch size warmup(linear ramp grad_accum)============

def current_grad_accum(step: int, total_steps: int, peak: int,
                        warmup_frac: float = 0.05, min_accum: int = 1) -> int:
    """前 warmup_frac 比例的 step:grad_accum 从 min_accum 线性 ramp 到 peak。
    Cramming 论文 + ModernBERT yaml 都用类似 schedule(`batch_size_warmup_tokens`)。
    我们 grad_accum 跟 effective batch 是线性关系(batch_size 不变),所以等价 batch warmup。
    """
    if total_steps <= 0 or warmup_frac <= 0:
        return peak
    p = step / total_steps
    if p >= warmup_frac:
        return peak
    t = p / warmup_frac
    val = min_accum + (peak - min_accum) * t
    return max(min_accum, int(round(val)))


# ============ Optimizer 参数分组(filter_bias_norm_wd)============

def make_param_groups(model: torch.nn.Module, weight_decay: float):
    """Bias + Norm.weight 不上 wd(论文 filter_bias_norm_wd=true)。
    其余 trainable param 上 wd。
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # 1D 参数(LayerNorm.weight、head_bias、所有 bias)都不 decay
        if p.ndim < 2 or name.endswith(".bias") or "head_bias" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_data", required=True, help="memmap pretokenized .pt")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--vocab_size", type=int, default=12536)
    p.add_argument("--pad_token_id", type=int, default=12531)
    p.add_argument("--mask_token_id", type=int, default=12535)
    # model config(默认 release-aligned 22L/768H/1152I/12h)
    p.add_argument("--hidden_size", type=int, default=768)
    p.add_argument("--num_layers", type=int, default=22)
    p.add_argument("--num_heads", type=int, default=12)
    p.add_argument("--intermediate_size", type=int, default=1152)
    p.add_argument("--max_position", type=int, default=1024)
    p.add_argument("--rope_theta", type=float, default=10000.0)
    p.add_argument("--layer_norm_eps", type=float, default=1e-5)
    p.add_argument("--embed_dropout", type=float, default=0.0)
    p.add_argument("--mlp_dropout", type=float, default=0.0)
    p.add_argument("--attn_out_dropout", type=float, default=0.0,
                   help="Cramming 论文 Section 4.3:short single-epoch 训练 dropout "
                        "只减少 update 数,无 overfit 风险。默认关掉。")
    # train
    p.add_argument("--max_seq_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8,
                   help="peak grad_accum(eff_batch = batch_size × accum)")
    p.add_argument("--accum_warmup_frac", type=float, default=0.05,
                   help="前 X 比例 step 线性 ramp grad_accum 1→peak(Cramming + ModernBERT 推荐)")
    p.add_argument("--accum_min", type=int, default=1,
                   help="ramp 起点 grad_accum(默认 1,即起步 micro-batch 等于物理 batch)")
    p.add_argument("--max_steps", type=int, default=200000)
    p.add_argument("--warmup_steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=8e-4, help="StableAdamW peak LR")
    p.add_argument("--min_lr", type=float, default=8e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    # MLM dynamic curriculum(论文 Section 3.2)
    p.add_argument("--mlm_low", type=float, default=0.15)
    p.add_argument("--mlm_high", type=float, default=0.30)
    p.add_argument("--mlm_warmup_frac", type=float, default=0.05,
                   help="MLM 前 X 比例:low→high(anti-curriculum)")
    # LR schedule
    p.add_argument("--damp_gamma", type=float, default=0.0,
                   help="Damped cosine 阻尼系数:0 = peak 衰减到 min_lr;1 = 不衰减")
    p.add_argument("--n_cycles", type=int, default=1)
    # data
    p.add_argument("--max_chunks", type=int, default=None)
    p.add_argument("--word_ids_data", default=None, help="default: train_data + .wid")
    p.add_argument("--seg_ids_data", default=None,  help="default: train_data + .seg")
    p.add_argument("--wwm", action="store_true")
    p.add_argument("--no_cross_doc_isolation", action="store_true",
                   help="不用 seg_ids 做 cross-doc 隔离(走 SDPA,失去 flex_attention)")
    # save
    p.add_argument("--save_steps", type=int, default=20000)
    p.add_argument("--logging_steps", type=int, default=50)
    # inline eval hook
    # 注:flex_attention 在 model.py module-level 已经 torch.compile wrap,
    # 无需 CLI flag
    p.add_argument("--inline_eval_cmd", default=None,
                   help="每 save_steps 后 fire 的 shell 命令,{ckpt} 替换为 ckpt dir")
    p.add_argument("--init_from_ckpt", default=None)
    args = p.parse_args()

    device = "cuda"
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "train_args.json"), "w") as f:
        json.dump({"args": vars(args), "cmdline": sys.argv}, f, indent=2, ensure_ascii=False)

    # 模型
    cfg = ModernBertConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.max_position,
        pad_token_id=args.pad_token_id,
        mask_token_id=args.mask_token_id,
        rope_theta=args.rope_theta,
        layer_norm_eps=args.layer_norm_eps,
        embed_dropout=args.embed_dropout,
        mlp_dropout=args.mlp_dropout,
        attn_out_dropout=args.attn_out_dropout,
    )
    model = ModernBertForMLM(cfg)
    if args.init_from_ckpt:
        ckpt_path = os.path.join(args.init_from_ckpt, "model.pt")
        print(f"Loading init weights from {ckpt_path}", flush=True)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=True)
    model = model.to(device, dtype=torch.bfloat16)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ModernBertc: {n_params/1e6:.1f}M params  H={cfg.hidden_size} "
          f"L={cfg.num_hidden_layers} head={cfg.num_attention_heads} I={cfg.intermediate_size}")

    # 保存 config + mask_id
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(cfg.__dict__, f, indent=2)
    with open(os.path.join(args.output_dir, "mask_token_id.txt"), "w") as f:
        f.write(str(cfg.mask_token_id))

    # flex_attention 在 model.py module-level 已经 torch.compile wrap
    # (default mode,first-call 编译 ~几秒)
    print("flex_attention: torch.compile-wrapped at module import")

    # 数据
    if args.wwm:
        if args.word_ids_data is None:
            args.word_ids_data = args.train_data + ".wid"
    seg_path = args.seg_ids_data or (args.train_data + ".seg")
    use_seg = not args.no_cross_doc_isolation and os.path.exists(seg_path)
    ds = ModernBertcDataset(
        args.train_data, args.max_chunks,
        word_ids_path=args.word_ids_data if args.wwm else None,
        seg_ids_path=seg_path if use_seg else None,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, pin_memory=True, drop_last=True,
                        collate_fn=collate_fn)

    # 优化器:StableAdamW
    from optimi import StableAdamW
    param_groups = make_param_groups(model, args.weight_decay)
    optim = StableAdamW(
        param_groups,
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )
    n_decay = sum(p.numel() for g in param_groups for p in g["params"] if g["weight_decay"] > 0)
    n_nodecay = sum(p.numel() for g in param_groups for p in g["params"] if g["weight_decay"] == 0)
    print(f"StableAdamW: lr={args.lr} betas=({args.beta1},{args.beta2}) eps={args.eps}")
    print(f"  decay params: {n_decay:,} | no-decay (bias/norm): {n_nodecay:,}")

    step = 0
    accum = 0
    cur_accum_target = current_grad_accum(0, args.max_steps,
                                           peak=args.gradient_accumulation_steps,
                                           warmup_frac=args.accum_warmup_frac,
                                           min_accum=args.accum_min)
    t0 = time.time()
    loss_acc = 0.0
    n_micro_acc = 0   # 累积 micro-step 数(grad accum 变化时,不能用 logging × N 算)
    correct_acc = 0
    n_masked_acc = 0
    model.train()
    while step < args.max_steps:
        for batch in loader:
            ids = batch["ids"].to(device, non_blocking=True)
            wids = batch["wids"].to(device, non_blocking=True) if batch["wids"] is not None else None
            segs = batch["segs"].to(device, non_blocking=True) if batch["segs"] is not None else None

            # Dynamic MLM rate
            cur_mlm = dynamic_mlm_prob(step, args.max_steps,
                                        warmup_frac=args.mlm_warmup_frac,
                                        low=args.mlm_low, high=args.mlm_high)

            if args.wwm:
                masked_ids, labels = mlm_mask_batch_wwm(
                    ids, wids, cfg.mask_token_id, cfg.vocab_size,
                    prob=cur_mlm, pad_id=cfg.pad_token_id)
            else:
                masked_ids, labels = mlm_mask_batch(
                    ids, cfg.mask_token_id, cfg.vocab_size,
                    prob=cur_mlm, pad_id=cfg.pad_token_id)

            out = model(input_ids=masked_ids, seg_ids=segs, labels=labels)
            loss = out["loss"] / cur_accum_target  # 用当前 grad-step 起点决定的 target
            loss.backward()
            with torch.no_grad():
                preds = out["logits"].argmax(-1)
                mask_pos = labels != -100
                correct_acc += ((preds == labels) & mask_pos).sum().item()
                n_masked_acc += mask_pos.sum().item()
            loss_acc += loss.item() * cur_accum_target  # 还原回 full per-micro loss
            n_micro_acc += 1
            accum += 1
            if accum >= cur_accum_target:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                # 手动设置 LR(damped cosine)
                cur_lr = damped_cosine_lr(step, args.max_steps, peak=args.lr,
                                           min_lr=args.min_lr, n_cycles=args.n_cycles,
                                           damp_gamma=args.damp_gamma,
                                           warmup_steps=args.warmup_steps)
                for g in optim.param_groups:
                    g["lr"] = cur_lr
                optim.step()
                optim.zero_grad(set_to_none=True)
                step += 1
                accum = 0
                # 决定下个 grad-step 的 accum target(batch warmup)
                cur_accum_target = current_grad_accum(
                    step, args.max_steps,
                    peak=args.gradient_accumulation_steps,
                    warmup_frac=args.accum_warmup_frac,
                    min_accum=args.accum_min,
                )

                if step % args.logging_steps == 0:
                    el = time.time() - t0
                    avg_loss = loss_acc / max(1, n_micro_acc)  # 每 micro-step 平均 loss
                    acc = correct_acc / max(1, n_masked_acc)
                    print(f"step {step}/{args.max_steps} | loss {avg_loss:.4f} | "
                          f"mlm_acc {acc:.4f} | lr {cur_lr:.6f} | mlm_p {cur_mlm:.3f} | "
                          f"accum {cur_accum_target} | {el:.1f}s", flush=True)
                    loss_acc = 0.0
                    n_micro_acc = 0
                    correct_acc = 0
                    n_masked_acc = 0

                if step % args.save_steps == 0 or step >= args.max_steps:
                    save = os.path.join(args.output_dir, f"checkpoint-{step}")
                    os.makedirs(save, exist_ok=True)
                    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
                    torch.save({
                        "model": raw.state_dict(),
                        "config": cfg.__dict__,
                        "step": step,
                    }, os.path.join(save, "model.pt"))
                    import shutil
                    shutil.copy2(config_path, os.path.join(save, "config.json"))
                    shutil.copy2(os.path.join(args.output_dir, "mask_token_id.txt"),
                                 os.path.join(save, "mask_token_id.txt"))
                    print(f"Saved checkpoint to {save}", flush=True)

                    if args.inline_eval_cmd:
                        cmd = args.inline_eval_cmd.replace("{ckpt}", save)
                        print(f"[inline_eval] {cmd}", flush=True)
                        try:
                            subprocess.run(cmd, shell=True, check=False, timeout=1800)
                        except subprocess.TimeoutExpired:
                            print("[inline_eval] TIMEOUT 30min", flush=True)
                        except Exception as e:
                            print(f"[inline_eval] failed: {e}", flush=True)

                if step >= args.max_steps:
                    break
        if step >= args.max_steps:
            break

    # final save
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save({"model": raw.state_dict(), "config": cfg.__dict__, "step": step},
               os.path.join(args.output_dir, "model_final.pt"))
    print(f"\nFinal save to {args.output_dir}")
    print(f"Training complete: {step} steps in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
