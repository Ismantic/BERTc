"""CWS + POS + NER 三任务联合微调。只依赖 torch。

这是 MT SOTA(joint score 1.4712,HF 上的 Ismantic/BERTc-315M-MT)的训练脚本。
recipe:骨干 lr 2e-5 / 头 lr 5e-4、α_pos=2.0、β_ner=0.5、FGM ε=1.0、5 epoch、batch 64。

三个头:
  CWS  Linear → CRF      切分是强序列约束(B 后面只能跟 I / E),CRF 比逐字 softmax 明显好
  POS  Linear → 交叉熵    词性只在词首字上有监督,其余位置是 -100
  NER  Linear → CRF

α_pos=2.0 是 v6.5 时代的关键发现:POS 的 loss 量级天然比 CWS 小,不加权就被压住,
加权后 POS 从 0.96 提到 0.97。β_ner=0.5 反过来压 NER,防止它抢容量。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crf import CRF                                              # noqa: E402
from data import MTDataset, MTCollator                           # noqa: E402
from evaluate import evaluate_mt                                 # noqa: E402
from model import ModernBertConfig, ModernBertModel              # noqa: E402
from optim import linear_schedule_with_warmup                    # noqa: E402


class ModernBertMT(nn.Module):
    """预训练骨干 + 三个任务头。"""

    def __init__(self, ckpt_dir: str, num_cws: int, num_pos: int, num_ner: int,
                 dropout: float = 0.1):
        super().__init__()
        ckpt_dir = Path(ckpt_dir)
        cfg_dict = json.loads((ckpt_dir / "config.json").read_text())
        cfg = ModernBertConfig(**{k: v for k, v in cfg_dict.items()
                                  if k in ModernBertConfig.__dataclass_fields__})
        self.bert = ModernBertModel(cfg)

        ckpt = torch.load(ckpt_dir / "model.pt", map_location="cpu",
                          weights_only=False)
        # 有 EMA 就优先用 shadow 权重(更稳),否则用原始权重
        sd = ckpt.get("ema") or ckpt["model"]
        bert_sd = {k[len("bert."):]: v for k, v in sd.items() if k.startswith("bert.")}
        self.bert.load_state_dict(bert_sd, strict=True)

        h = cfg.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.cws_classifier = nn.Linear(h, num_cws)
        self.cws_crf = CRF(num_cws, batch_first=True)
        self.pos_classifier = nn.Linear(h, num_pos)
        self.ner_classifier = nn.Linear(h, num_ner)
        self.ner_crf = CRF(num_ner, batch_first=True)
        self.cfg = cfg

    def forward(self, input_ids, attention_mask,
                cws_labels=None, pos_labels=None, ner_labels=None):
        # 微调走 SDPA + pad mask(不传 seg_ids,没有跨文档隔离的必要)
        h = self.dropout(self.bert(input_ids, attention_mask=attention_mask))
        cws_emi = self.cws_classifier(h)
        pos_logits = self.pos_classifier(h)
        ner_emi = self.ner_classifier(h)
        mask = attention_mask.bool()

        losses = {}
        if cws_labels is not None:
            losses["cws"] = -self.cws_crf(cws_emi, cws_labels, mask=mask,
                                          reduction="mean")
        if pos_labels is not None:
            losses["pos"] = F.cross_entropy(
                pos_logits.reshape(-1, pos_logits.size(-1)),
                pos_labels.reshape(-1), ignore_index=-100)
        if ner_labels is not None:
            losses["ner"] = -self.ner_crf(ner_emi, ner_labels, mask=mask,
                                          reduction="mean")
        return losses, (cws_emi, pos_logits, ner_emi)

    @torch.no_grad()
    def decode_cws(self, input_ids, attention_mask):
        h = self.dropout(self.bert(input_ids, attention_mask=attention_mask))
        return self.cws_crf.decode(self.cws_classifier(h).float(),
                                   mask=attention_mask.bool())

    @torch.no_grad()
    def decode_ner(self, input_ids, attention_mask):
        h = self.dropout(self.bert(input_ids, attention_mask=attention_mask))
        return self.ner_crf.decode(self.ner_classifier(h).float(),
                                   mask=attention_mask.bool())

    @torch.no_grad()
    def predict_pos(self, input_ids, attention_mask):
        h = self.dropout(self.bert(input_ids, attention_mask=attention_mask))
        return self.pos_classifier(h).argmax(-1)


class FGM:
    """Fast Gradient Method 对抗训练:给词嵌入加一个受 L2 范数约束的扰动,
    在扰动后的输入上再算一次梯度,两次梯度叠加后一起更新。

    v6.5 时代实测 cws / ner 各 +0.005~0.013,是 MT SOTA 的组成部分。
    只扰动 embedding,不碰其他参数。
    """

    def __init__(self, model: nn.Module, eps: float = 1.0):
        self.model = model
        self.eps = eps
        self.backup = {}

    def attack(self) -> None:
        for name, p in self.model.named_parameters():
            if p.requires_grad and "bert.embed.weight" in name:
                self.backup[name] = p.data.clone()
                if p.grad is None:
                    continue
                norm = torch.norm(p.grad)
                if norm and not torch.isnan(norm):
                    p.data.add_(self.eps * p.grad / norm)

    def restore(self) -> None:
        for name, p in self.model.named_parameters():
            if name in self.backup:
                p.data = self.backup[name]
        self.backup = {}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt_dir", required=True, help="预训练骨干 ckpt 目录")
    p.add_argument("--train_data", required=True, help="prepare/ 产出的 MT 训练集")
    p.add_argument("--dev_data", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_chars", type=int, default=254)
    p.add_argument("--bert_lr", type=float, default=2e-5)
    p.add_argument("--head_lr", type=float, default=5e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--alpha_pos", type=float, default=2.0, help="POS loss 权重")
    p.add_argument("--beta_ner", type=float, default=0.5, help="NER loss 权重")
    p.add_argument("--fgm", action="store_true")
    p.add_argument("--fgm_eps", type=float, default=1.0)
    p.add_argument("--dev_limit", type=int, default=None,
                   help="只评前 N 条 dev,加快每轮评测")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--num_workers", type=int, default=0)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    device = "cuda"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_args.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False))

    train_ds = MTDataset(args.train_data, max_chars=args.max_chars)
    dev_ds = MTDataset(args.dev_data, max_chars=args.max_chars)
    if args.dev_limit:
        dev_ds = torch.utils.data.Subset(dev_ds, range(min(args.dev_limit, len(dev_ds))))
        # Subset 不转发这几个属性,评测要用
        for attr in ("cws_vocab", "pos_vocab", "ner_vocab"):
            setattr(dev_ds, attr, getattr(dev_ds.dataset, attr))
    collator = MTCollator(train_ds.pad_token_id)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collator, num_workers=args.num_workers,
                        pin_memory=True)
    print(f"训练 {len(train_ds):,} 条 | dev {len(dev_ds):,} 条 | "
          f"标签数 cws={train_ds.num_cws_tags} pos={train_ds.num_pos_tags} "
          f"ner={train_ds.num_ner_tags}")

    model = ModernBertMT(args.ckpt_dir, train_ds.num_cws_tags,
                         train_ds.num_pos_tags, train_ds.num_ner_tags,
                         dropout=args.dropout).to(device)
    print(f"ModernBertMT {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M  "
          f"H={model.cfg.hidden_size} L={model.cfg.num_hidden_layers}")

    # 骨干和任务头分开设学习率:骨干已经训好,大 lr 会把它冲坏;头是随机初始化的,要大 lr
    no_decay = ("bias", "norm.weight")
    bert_named = list(model.bert.named_parameters())
    head_params = [p for m in (model.cws_classifier, model.cws_crf,
                               model.pos_classifier, model.ner_classifier,
                               model.ner_crf) for p in m.parameters()]
    optim = AdamW([
        {"params": [p for n, p in bert_named
                    if not any(nd in n for nd in no_decay)],
         "lr": args.bert_lr, "weight_decay": args.weight_decay},
        {"params": [p for n, p in bert_named if any(nd in n for nd in no_decay)],
         "lr": args.bert_lr, "weight_decay": 0.0},
        {"params": head_params, "lr": args.head_lr, "weight_decay": 0.0},
    ])
    total_steps = len(loader) * args.epochs
    scheduler = linear_schedule_with_warmup(
        optim, int(total_steps * args.warmup_ratio), total_steps)
    fgm = FGM(model, eps=args.fgm_eps) if args.fgm else None
    print(f"总步数 {total_steps}" + (f" | FGM ε={args.fgm_eps}" if fgm else ""))

    def total_loss(losses):
        return (losses["cws"] + args.alpha_pos * losses["pos"]
                + args.beta_ner * losses["ner"])

    model.train()
    best_score, step, t0 = 0.0, 0, time.time()
    for epoch in range(args.epochs):
        sums = {"cws": 0.0, "pos": 0.0, "ner": 0.0}
        n_batch = 0
        for batch in loader:
            b = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                losses, _ = model(b["input_ids"], b["attention_mask"],
                                  cws_labels=b["cws_labels"],
                                  pos_labels=b["pos_labels"],
                                  ner_labels=b["ner_labels"])
            total_loss(losses).backward()

            if fgm is not None:
                fgm.attack()
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    adv, _ = model(b["input_ids"], b["attention_mask"],
                                   cws_labels=b["cws_labels"],
                                   pos_labels=b["pos_labels"],
                                   ner_labels=b["ner_labels"])
                total_loss(adv).backward()
                fgm.restore()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()
            scheduler.step()
            optim.zero_grad(set_to_none=True)

            for k in sums:
                sums[k] += losses[k].item()
            n_batch += 1
            step += 1
            if step % args.log_every == 0:
                sps = step / (time.time() - t0)
                print(f"  ep{epoch + 1} {step}/{total_steps}  "
                      f"cws={losses['cws'].item():.3f} pos={losses['pos'].item():.3f} "
                      f"ner={losses['ner'].item():.3f}  "
                      f"lr={optim.param_groups[0]['lr']:.2e}  {sps:.1f}/s  "
                      f"ETA {(total_steps - step) / sps / 60:.1f}m", flush=True)

        m = evaluate_mt(model, dev_ds, collator, device)
        avg = {k: v / max(1, n_batch) for k, v in sums.items()}
        print(f"\n=== epoch {epoch + 1}/{args.epochs}  "
              f"loss cws={avg['cws']:.3f} pos={avg['pos']:.3f} ner={avg['ner']:.3f} | "
              f"dev cws_F1={m['cws_f1']:.4f} pos_acc={m['pos_acc']:.4f} "
              f"ner_F1={m['ner_f1']:.4f} score={m['score']:.4f} ===", flush=True)
        if m["score"] > best_score:
            best_score = m["score"]
            torch.save(model.state_dict(), out_dir / "best.pt")
            print(f"    ↑ 存 best.pt(score={best_score:.4f})", flush=True)
        print(flush=True)

    torch.save(model.state_dict(), out_dir / "final.pt")
    print(f"结束。最好 score {best_score:.4f},用时 {(time.time() - t0) / 60:.1f} 分钟")


if __name__ == "__main__":
    main()
