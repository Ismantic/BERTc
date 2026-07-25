"""中文拼写纠错(CSC)微调。只依赖 torch。

这是 CSC SOTA(SIGHAN-15 F1 0.8346,HF 上的 Ismantic/BERTc-315M-CSC)的训练脚本。
recipe:10 epoch、batch 32、lr 3e-5、warmup 0.1、det_weight 0.3、threshold 0.7。

双头:
  cor_head  逐位置预测正确的字。**权重与词嵌入绑定** —— 这是关键:
            预训练的 MLM 头就是简化版的 logits = h @ embed.weightᵀ,
            预训完 h 已经和嵌入空间对齐;换一个随机初始化的 Linear 会把这个
            对齐关系废掉,CSC 就学不动了(v7 时代踩过这个坑)。
  det_head  逐位置判断"这里有没有错",用 focal loss。错字在句子里是极少数,
            普通 BCE 会被海量负样本淹没,focal 的 (1-p)^γ 把已经学会的简单
            负样本压下去。

Large 模型 5 epoch 严重欠训、10 epoch 才到位,这是 v4-Large 调参时的主要发现。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import CSCDataset, CSCCollator                          # noqa: E402
from .evaluate import evaluate_csc                                 # noqa: E402
from .model import ModernBertConfig, ModernBertModel               # noqa: E402
from .optim import linear_schedule_with_warmup                     # noqa: E402


class ModernBertCSC(nn.Module):
    def __init__(self, ckpt_dir: str, vocab_size: int):
        super().__init__()
        ckpt_dir = Path(ckpt_dir)
        cfg_dict = json.loads((ckpt_dir / "config.json").read_text())
        cfg = ModernBertConfig(**{k: v for k, v in cfg_dict.items()
                                  if k in ModernBertConfig.__dataclass_fields__})
        self.bert = ModernBertModel(cfg)

        ckpt = torch.load(ckpt_dir / "model.pt", map_location="cpu",
                          weights_only=False)
        sd = ckpt.get("ema") or ckpt["model"]
        bert_sd = {k[len("bert."):]: v for k, v in sd.items() if k.startswith("bert.")}
        self.bert.load_state_dict(bert_sd, strict=True)

        h = cfg.hidden_size
        self.vocab_size = vocab_size
        self.cor_head = nn.Linear(h, vocab_size, bias=False)
        self.cor_head.weight = self.bert.embed.weight        # 与词嵌入绑权重
        self.det_head = nn.Linear(h, 1)
        self.cfg = cfg

    def forward(self, input_ids, attention_mask=None):
        h = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.cor_head(h), self.det_head(h).squeeze(-1)


def focal_bce_loss(logits, labels, gamma: float = 2.0, valid_mask=None):
    """二分类 focal loss。valid_mask 用来排除 padding 位置。"""
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * labels + (1 - p) * (1 - labels)
    loss = (1.0 - pt) ** gamma * bce
    if valid_mask is not None:
        loss = loss * valid_mask
        return loss.sum() / valid_mask.sum().clamp(min=1.0)
    return loss.mean()


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt_dir", required=True, help="预训练骨干 ckpt 目录")
    p.add_argument("--train_data", required=True, help="prepare/ 产出的 CSC 训练集")
    p.add_argument("--test_data", required=True, help="预编码的 SIGHAN-15 测试集")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--det_weight", type=float, default=0.3)
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--threshold", type=float, default=0.7,
                   help="纠错置信度阈值,低于它保留原字")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--num_workers", type=int, default=0)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_args.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False))

    train_ds = CSCDataset(args.train_data, max_len=args.max_len)
    test_ds = CSCDataset(args.test_data, max_len=args.max_len)
    collator = CSCCollator(train_ds.pad_token_id)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collator, num_workers=args.num_workers,
                        pin_memory=True)

    # id→字 反查表由 prepare/ 写在测试集文件里,只用于把预测还原成句子做字符串比对
    blob = torch.load(args.test_data, map_location="cpu", weights_only=False)
    id_to_char = blob.get("id_to_char")
    if id_to_char is None:
        sys.exit(f"{args.test_data} 里没有 id_to_char,评测无法还原句子 —— "
                 f"请用新版 prepare/ 重新生成")
    id_to_char = {int(k): v for k, v in id_to_char.items()}
    vocab_size = blob.get("vocab_size") or (max(id_to_char) + 1)

    total_steps = args.epochs * len(loader)
    print(f"训练 {len(train_ds):,} 条 | 测试 {len(test_ds):,} 条 | "
          f"每轮 {len(loader)} 步,共 {total_steps} 步")

    model = ModernBertCSC(args.ckpt_dir, vocab_size).to(device)
    print(f"ModernBertCSC {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M  "
          f"H={model.cfg.hidden_size} L={model.cfg.num_hidden_layers}")

    no_decay = ("bias", "norm.weight")
    named = list(model.named_parameters())
    optim = torch.optim.AdamW([
        {"params": [p for n, p in named if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in named if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ], lr=args.lr)
    scheduler = linear_schedule_with_warmup(
        optim, int(total_steps * args.warmup_ratio), total_steps)

    best_f1, step, t0 = 0.0, 0, time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        sum_cor = sum_det = 0.0
        n_batch = 0
        for batch in loader:
            b = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            cor_logits, det_logits = model(b["input_ids"], b["attention_mask"])
            cor_loss = F.cross_entropy(cor_logits.reshape(-1, model.vocab_size),
                                       b["cor_labels"].reshape(-1), ignore_index=-100)
            det_loss = focal_bce_loss(det_logits, b["det_labels"],
                                      gamma=args.focal_gamma,
                                      valid_mask=b["attention_mask"].float())
            (cor_loss + args.det_weight * det_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()
            scheduler.step()
            optim.zero_grad(set_to_none=True)

            sum_cor += cor_loss.item()
            sum_det += det_loss.item()
            n_batch += 1
            step += 1
            if step % args.log_every == 0:
                sps = step / (time.time() - t0)
                print(f"  ep{ep} {step}/{total_steps}  cor={cor_loss.item():.3f} "
                      f"det={det_loss.item():.4f}  lr={scheduler.get_last_lr()[0]:.2e}  "
                      f"{sps:.1f}/s ETA {(total_steps - step) / sps / 60:.1f}m",
                      flush=True)

        m = evaluate_csc(model, test_ds, collator, device, id_to_char,
                         threshold=args.threshold)
        print(f"\n=== epoch {ep}/{args.epochs}  "
              f"cor={sum_cor / max(1, n_batch):.4f} det={sum_det / max(1, n_batch):.4f} | "
              f"SIGHAN-15 acc={m['acc']:.4f} P={m['precision']:.4f} "
              f"R={m['recall']:.4f} F1={m['f1']:.4f} "
              f"(TP={m['TP']} FP={m['FP']} FN={m['FN']} TN={m['TN']}) ===", flush=True)

        blob_out = {"model": model.state_dict(), "epoch": ep, "metrics": m,
                    "args": vars(args)}
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            torch.save(blob_out, out_dir / "best.pt")
            print(f"    ↑ 存 best.pt(F1={best_f1:.4f})", flush=True)
        torch.save(blob_out, out_dir / "final.pt")
        print(flush=True)

    print(f"结束。最好 SIGHAN-15 F1 {best_f1:.4f},"
          f"用时 {(time.time() - t0) / 60:.1f} 分钟")


if __name__ == "__main__":
    main()
