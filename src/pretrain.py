"""Modern BERTc MLM 预训练。只依赖 torch。

recipe(v4-Mid / v4-Large 实跑,即 HF 上 Ismantic/BERTc-165M 与 -315M):
  - StableAdamW(β=(0.9, 0.95), wd=0.01, eps=1e-6,bias/norm 不上 wd)
  - Damped cosine LR + 线性 warmup
  - 固定 15% MLM(动态 curriculum 的实现在 masking.py,v4 关掉了)
  - 整词掩码(WWM),词边界来自 .wid
  - 跨文档 attention 隔离,文档边界来自 .seg → flex_attention block-diag mask
  - grad_accum 线性 ramp(等价 batch size warmup)
  - grad clip 0.5(v4)/ 1.0(v3)

数据由 prepare/ 预处理好,三个平行文件:
  <train_data>        int32  token id
  <train_data>.wid    int32  词 id(WWM)
  <train_data>.seg    uint8  文档 id(跨文档隔离)

单卡。没有 DDP —— 机器只有一块 4090。
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import PackedMLMDataset                                   # noqa: E402
from .masking import (mlm_mask_batch, mlm_mask_batch_wwm,            # noqa: E402
                     dynamic_mlm_prob)
from .model import ModernBertConfig, ModernBertForMLM                # noqa: E402
from .optim import StableAdamW, damped_cosine_lr, current_grad_accum  # noqa: E402


class EMA:
    """参数的指数滑动平均。

    每个 optimizer step 后 shadow = decay·shadow + (1−decay)·param。
    存 ckpt 时一并写进去,评测优先用 shadow 权重(过滤训练末期噪声)。
    v4 系列实际关掉了(--no_ema),v3 用过。
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {name: p.data.clone().detach()
                       for name, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def merged_state_dict(self, model: torch.nn.Module) -> dict:
        """完整 state_dict:可训练参数用 shadow,buffer(如位置编码表)用原模型的。"""
        out = dict(model.state_dict())
        out.update(self.shadow)
        return out


def make_param_groups(model: torch.nn.Module, weight_decay: float) -> list:
    """bias 和 LayerNorm.weight 不做 weight decay。

    判据是 ndim < 2 —— 所有 1 维参数(norm 权重、各种 bias、PE 的 scale_factor)
    都归到 no-decay。对它们做 decay 会直接把归一化的尺度压掉。
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or name.endswith(".bias") else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def collate(batch):
    """PackedMLMDataset 每条返回 (ids,)、(ids, wids) 或 (ids, wids, segs)。"""
    if isinstance(batch[0], torch.Tensor):
        return {"ids": torch.stack(batch), "wids": None, "segs": None}
    n = len(batch[0])
    return {
        "ids": torch.stack([b[0] for b in batch]),
        "wids": torch.stack([b[1] for b in batch]) if n > 1 else None,
        "segs": torch.stack([b[2] for b in batch]) if n > 2 else None,
    }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train_data", required=True, help="prepare/ 产出的预处理语料")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--vocab_size", type=int, default=12536)
    p.add_argument("--pad_token_id", type=int, default=12531)
    p.add_argument("--mask_token_id", type=int, default=12535)
    # 模型
    p.add_argument("--hidden_size", type=int, default=768)
    p.add_argument("--num_layers", type=int, default=22)
    p.add_argument("--num_heads", type=int, default=12)
    p.add_argument("--intermediate_size", type=int, default=1152)
    p.add_argument("--max_position", type=int, default=1024)
    p.add_argument("--pe_theta", type=float, default=10000.0)
    p.add_argument("--layer_norm_eps", type=float, default=1e-5)
    p.add_argument("--embed_dropout", type=float, default=0.0)
    p.add_argument("--mlp_dropout", type=float, default=0.0)
    p.add_argument("--attn_out_dropout", type=float, default=0.0)
    # 训练
    p.add_argument("--max_seq_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--accum_warmup_frac", type=float, default=0.05)
    p.add_argument("--accum_min", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=200000)
    p.add_argument("--warmup_steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--min_lr", type=float, default=8e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    # MLM
    p.add_argument("--mlm_low", type=float, default=0.15)
    p.add_argument("--mlm_high", type=float, default=0.30)
    p.add_argument("--mlm_warmup_frac", type=float, default=0.05)
    p.add_argument("--wwm", action="store_true")
    # LR 调度
    p.add_argument("--damp_gamma", type=float, default=0.0)
    p.add_argument("--n_cycles", type=int, default=1)
    # 数据
    p.add_argument("--max_chunks", type=int, default=None)
    p.add_argument("--word_ids_data", default=None, help="默认 train_data + .wid")
    p.add_argument("--seg_ids_data", default=None, help="默认 train_data + .seg")
    p.add_argument("--no_cross_doc_isolation", action="store_true",
                   help="不做跨文档隔离。会退回 SDPA + pad mask,比 flex_attention 慢")
    p.add_argument("--num_workers", type=int, default=4)
    # 保存
    p.add_argument("--save_steps", type=int, default=20000)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--use_ema", action="store_true", default=True)
    p.add_argument("--no_ema", dest="use_ema", action="store_false")
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--inline_eval_cmd", default=None,
                   help="每次存 ckpt 后执行的命令,{ckpt} 会替换成 ckpt 目录")
    p.add_argument("--init_from_ckpt", default=None)
    p.add_argument("--resume_step", type=int, default=0,
                   help="从这个 step 续算 LR / MLM / accum 调度。"
                        "optimizer 状态不恢复(没存),动量会重建")
    return p


def save_checkpoint(path: Path, model, cfg, step, ema, output_dir: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    raw = getattr(model, "_orig_mod", model)
    blob = {"model": raw.state_dict(), "config": cfg.__dict__, "step": step}
    if ema is not None:
        blob["ema"] = ema.merged_state_dict(raw)
    torch.save(blob, path / "model.pt")
    for name in ("config.json", "mask_token_id.txt"):
        src = output_dir / name
        if src.exists():
            (path / name).write_bytes(src.read_bytes())


def main() -> None:
    args = build_argparser().parse_args()
    device = "cuda"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_args.json").write_text(
        json.dumps({"args": vars(args), "cmdline": sys.argv},
                   indent=2, ensure_ascii=False))

    cfg = ModernBertConfig(
        vocab_size=args.vocab_size, hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers, num_attention_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.max_position,
        pad_token_id=args.pad_token_id, mask_token_id=args.mask_token_id,
        pe_theta=args.pe_theta, layer_norm_eps=args.layer_norm_eps,
        embed_dropout=args.embed_dropout, mlp_dropout=args.mlp_dropout,
        attn_out_dropout=args.attn_out_dropout,
    )
    model = ModernBertForMLM(cfg)
    if args.init_from_ckpt:
        ckpt_path = Path(args.init_from_ckpt) / "model.pt"
        print(f"从 {ckpt_path} 加载初始权重", flush=True)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=True)
    model = model.to(device, dtype=torch.bfloat16)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ModernBERTc: {n_params / 1e6:.1f}M  H={cfg.hidden_size} "
          f"L={cfg.num_hidden_layers} heads={cfg.num_attention_heads} "
          f"I={cfg.intermediate_size}")

    (out_dir / "config.json").write_text(json.dumps(cfg.__dict__, indent=2))
    (out_dir / "mask_token_id.txt").write_text(str(cfg.mask_token_id))

    # 数据
    word_ids_path = None
    if args.wwm:
        word_ids_path = args.word_ids_data or (args.train_data + ".wid")
        if not os.path.exists(word_ids_path):
            sys.exit(f"--wwm 需要 {word_ids_path},但文件不存在")
    seg_path = args.seg_ids_data or (args.train_data + ".seg")
    use_seg = not args.no_cross_doc_isolation and os.path.exists(seg_path)
    ds = PackedMLMDataset(args.train_data, word_ids_path=word_ids_path,
                          seg_ids_path=seg_path if use_seg else None,
                          max_chunks=args.max_chunks)
    print(f"语料 {len(ds):,} chunk × {ds.seq_len}"
          f"{' + WWM 词边界' if word_ids_path else ''}"
          f"{' + 跨文档隔离' if use_seg else ''}")

    # fork 而非 spawn:memmap 张量 pickle 不过去(Linux 专属做法)
    import multiprocessing as mp
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True,
                        drop_last=True, collate_fn=collate, prefetch_factor=4,
                        persistent_workers=args.num_workers > 0,
                        multiprocessing_context=mp.get_context("fork")
                        if args.num_workers > 0 else None)

    param_groups = make_param_groups(model, args.weight_decay)
    optim = StableAdamW(param_groups, lr=args.lr,
                        betas=(args.beta1, args.beta2), eps=args.eps)
    n_decay = sum(p.numel() for p in param_groups[0]["params"])
    n_nodecay = sum(p.numel() for p in param_groups[1]["params"])
    print(f"StableAdamW lr={args.lr} betas=({args.beta1},{args.beta2}) eps={args.eps}")
    print(f"  decay {n_decay:,} | no-decay(bias/norm) {n_nodecay:,}")

    ema = EMA(model, decay=args.ema_decay) if args.use_ema else None
    if ema is not None:
        print(f"EMA decay={args.ema_decay}")

    step = args.resume_step
    accum = 0
    accum_target = current_grad_accum(step, args.max_steps,
                                      peak=args.gradient_accumulation_steps,
                                      warmup_frac=args.accum_warmup_frac,
                                      min_accum=args.accum_min)
    if step > 0:
        print(f"从 step {step} 续训,LR / MLM / accum 调度自动接上", flush=True)

    t0 = time.time()
    loss_sum, n_micro, n_correct, n_masked = 0.0, 0, 0, 0
    model.train()

    while step < args.max_steps:
        for batch in loader:
            ids = batch["ids"].to(device, non_blocking=True)
            wids = (batch["wids"].to(device, non_blocking=True)
                    if batch["wids"] is not None else None)
            segs = (batch["segs"].to(device, non_blocking=True).to(torch.int32)
                    if batch["segs"] is not None else None)

            mlm_p = dynamic_mlm_prob(step, args.max_steps,
                                     warmup_frac=args.mlm_warmup_frac,
                                     low=args.mlm_low, high=args.mlm_high)
            if args.wwm:
                masked_ids, labels = mlm_mask_batch_wwm(
                    ids, wids, cfg.mask_token_id, cfg.vocab_size,
                    prob=mlm_p, pad_id=cfg.pad_token_id)
            else:
                masked_ids, labels = mlm_mask_batch(
                    ids, cfg.mask_token_id, cfg.vocab_size,
                    prob=mlm_p, pad_id=cfg.pad_token_id)

            out = model(input_ids=masked_ids, seg_ids=segs, labels=labels)
            (out["loss"] / accum_target).backward()

            with torch.no_grad():
                mask_pos = labels != -100
                n_correct += ((out["logits"].argmax(-1) == labels) & mask_pos).sum().item()
                n_masked += mask_pos.sum().item()
            loss_sum += out["loss"].item()
            n_micro += 1
            accum += 1

            if accum < accum_target:
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            lr = damped_cosine_lr(step, args.max_steps, peak=args.lr,
                                  min_lr=args.min_lr, n_cycles=args.n_cycles,
                                  damp_gamma=args.damp_gamma,
                                  warmup_steps=args.warmup_steps)
            for g in optim.param_groups:
                g["lr"] = lr
            optim.step()
            optim.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(getattr(model, "_orig_mod", model))
            step += 1
            accum = 0
            accum_target = current_grad_accum(
                step, args.max_steps, peak=args.gradient_accumulation_steps,
                warmup_frac=args.accum_warmup_frac, min_accum=args.accum_min)

            if step % args.logging_steps == 0:
                print(f"step {step}/{args.max_steps} | "
                      f"loss {loss_sum / max(1, n_micro):.4f} | "
                      f"mlm_acc {n_correct / max(1, n_masked):.4f} | "
                      f"lr {lr:.6f} | mlm_p {mlm_p:.3f} | accum {accum_target} | "
                      f"{time.time() - t0:.1f}s", flush=True)
                loss_sum, n_micro, n_correct, n_masked = 0.0, 0, 0, 0

            if step % args.save_steps == 0 or step >= args.max_steps:
                ckpt = out_dir / f"checkpoint-{step}"
                save_checkpoint(ckpt, model, cfg, step, ema, out_dir)
                print(f"已存 {ckpt}", flush=True)
                if args.inline_eval_cmd:
                    cmd = args.inline_eval_cmd.replace("{ckpt}", str(ckpt))
                    print(f"[inline_eval] {cmd}", flush=True)
                    try:
                        subprocess.run(cmd, shell=True, check=False, timeout=1800)
                    except subprocess.TimeoutExpired:
                        print("[inline_eval] 超时 30 分钟", flush=True)
                    except Exception as e:                      # noqa: BLE001
                        print(f"[inline_eval] 失败: {e}", flush=True)

            if step >= args.max_steps:
                break
        if step >= args.max_steps:
            break

    raw = getattr(model, "_orig_mod", model)
    final = {"model": raw.state_dict(), "config": cfg.__dict__, "step": step}
    if ema is not None:
        final["ema"] = ema.merged_state_dict(raw)
    torch.save(final, out_dir / "model_final.pt")
    print(f"\n训练结束:{step} step,用时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
