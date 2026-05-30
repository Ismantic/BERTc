"""Char-level RoBERTa-style MLM pretrain — encoder-only,无 NSP/SOP/WWM。

输入数据格式:[N, seqlen] int32/int64 tensor(`encode_char_data.py` 输出),
每个 chunk 内 token id 必须在 vocab 范围内。

实现细节:
- 动态 mask(每 forward 随机采样,RoBERTa-style,而非预先固定 mask)
- 标准 BERT MLM 策略:15% 位置选中,其中 80% → [MASK] / 10% → random / 10% unchanged
- pad_token_id 从模型 config 读,不 hardcode
- mask_token_id 从 model_path/mask_token_id.txt 读(build_bert_init.py 写入)
"""
import argparse, os, sys, json, time, math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import BertForMaskedLM, BertConfig, get_cosine_schedule_with_warmup


def _load_memmap_or_pt(pt_path):
    meta_path = pt_path + ".meta"
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        arr = np.memmap(pt_path, dtype=np.dtype(meta["dtype"]), mode="r",
                        shape=tuple(meta["shape"]))
        return torch.from_numpy(arr), "memmap"
    return torch.load(pt_path, map_location="cpu", weights_only=True), "torch.load"


class IntTensorDataset(Dataset):
    """保留 int32 节省内存。memmap 优先,torch.load 向后兼容。单 chunk 取出时转 long。
    若 word_ids_path 提供,getitem 返回 (input_ids, word_ids) tuple,用于 WWM 训练。
    """
    def __init__(self, pt_path, max_chunks=None, word_ids_path=None):
        data, mode = _load_memmap_or_pt(pt_path)
        if max_chunks is not None and data.shape[0] > max_chunks:
            data = data[:max_chunks]
        self.data = data
        print(f"Loaded {data.shape[0]:,} chunks × {data.shape[1]} from {pt_path} "
              f"({data.numel()*data.element_size()/1e9:.1f} GB, dtype={data.dtype}, {mode})")
        self.word_ids = None
        if word_ids_path is not None:
            wdata, wmode = _load_memmap_or_pt(word_ids_path)
            if max_chunks is not None and wdata.shape[0] > max_chunks:
                wdata = wdata[:max_chunks]
            assert wdata.shape == data.shape, f"word_ids shape {wdata.shape} != data shape {data.shape}"
            self.word_ids = wdata
            print(f"  + word_ids from {word_ids_path} ({wdata.dtype}, {wmode}) → WWM mode")
    def __len__(self): return self.data.shape[0]
    def __getitem__(self, i):
        ids = self.data[i].to(torch.long)
        if self.word_ids is not None:
            wids = self.word_ids[i].to(torch.long)
            return ids, wids
        return ids


def mlm_mask_batch(input_ids, mask_token_id, vocab_size, prob=0.15, pad_id=0):
    """In-place MLM masking — 15% selected,其中 80% → mask,10% → random,10% unchanged。
    所有中间 tensor 都跟 input_ids 同 device(CPU 或 CUDA)。
    """
    dev = input_ids.device
    input_ids = input_ids.clone()
    labels = input_ids.clone()
    not_pad = input_ids != pad_id
    probability_matrix = torch.full(labels.shape, prob, device=dev)
    masked_indices = torch.bernoulli(probability_matrix).bool() & not_pad
    labels[~masked_indices] = -100

    # 80% → [MASK]
    replace_mask = torch.bernoulli(
        torch.full(labels.shape, 0.8, device=dev)).bool() & masked_indices
    input_ids[replace_mask] = mask_token_id
    # 10% → random token
    replace_rand = torch.bernoulli(
        torch.full(labels.shape, 0.5, device=dev)).bool() & masked_indices & ~replace_mask
    random_words = torch.randint(0, vocab_size, labels.shape,
                                 dtype=input_ids.dtype, device=dev)
    input_ids[replace_rand] = random_words[replace_rand]
    return input_ids, labels


def mlm_mask_batch_wwm(input_ids, word_ids, mask_token_id, vocab_size,
                       prob=0.15, pad_id=0):
    """Whole Word Masking — 按 word_id group,15% 选 word,整 word 一起 80/10/10。
    word_ids: [B, L] int64,同词共用 id。pad 位置 word_id 任意(被 not_pad mask 掉)。
    """
    dev = input_ids.device
    B, L = input_ids.shape
    input_ids = input_ids.clone()
    labels = input_ids.clone()
    not_pad = input_ids != pad_id

    # 给 word_ids 跨 batch 加 offset(让不同 sample 的同名 word_id 不冲突)
    max_wid = int(word_ids.max().item()) + 1
    global_wid = word_ids + torch.arange(B, device=dev, dtype=word_ids.dtype).unsqueeze(1) * max_wid
    # pad 位置 set 成一个独占值,避免污染 unique
    SENTINEL = global_wid.max().item() + 1
    global_wid_for_unique = torch.where(not_pad, global_wid, torch.full_like(global_wid, SENTINEL))

    flat = global_wid_for_unique.flatten()
    unique_wids, inverse = torch.unique(flat, return_inverse=True)
    # 找 SENTINEL 在 unique 中的 index,屏蔽
    sentinel_idx_mask = unique_wids != SENTINEL
    n_words = int(sentinel_idx_mask.sum().item())
    if n_words == 0:
        return input_ids, torch.full_like(labels, -100)

    # 在 valid words 里选 prob 比例
    valid_word_indices = torch.nonzero(sentinel_idx_mask, as_tuple=False).squeeze(-1)
    n_to_mask = max(1, int(n_words * prob))
    perm = torch.randperm(n_words, device=dev)
    chosen_in_valid = perm[:n_to_mask]
    chosen_unique_idx = valid_word_indices[chosen_in_valid]

    # per word 决定 80/10/10
    chosen_flag = torch.zeros(len(unique_wids), dtype=torch.bool, device=dev)
    chosen_flag[chosen_unique_idx] = True
    decision = torch.rand(len(unique_wids), device=dev)
    is_mask_word = chosen_flag & (decision < 0.8)
    is_rand_word = chosen_flag & (decision >= 0.8) & (decision < 0.9)
    # else unchanged but labels still kept

    # scatter back to [B, L]
    chosen_per_pos = chosen_flag[inverse].view(B, L) & not_pad
    mask_per_pos = is_mask_word[inverse].view(B, L) & not_pad
    rand_per_pos = is_rand_word[inverse].view(B, L) & not_pad

    labels[~chosen_per_pos] = -100
    input_ids[mask_per_pos] = mask_token_id
    rand_tokens = torch.randint(0, vocab_size, input_ids.shape,
                                dtype=input_ids.dtype, device=dev)
    input_ids[rand_per_pos] = rand_tokens[rand_per_pos]
    return input_ids, labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--train_data", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_seq_length", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--max_steps", type=int, default=20000)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--mlm_prob", type=float, default=0.15)
    p.add_argument("--max_chunks", type=int, default=None,
                   help="只用前 N chunks(等价限制 token 量)")
    p.add_argument("--save_steps", type=int, default=5000)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--wwm", action="store_true",
                   help="Whole Word Masking(需要 --word_ids_data 配套)")
    p.add_argument("--word_ids_data", default=None,
                   help="word_id memmap path(uint16 [N,seq] 同 train_data shape)")
    p.add_argument("--use_muon", action="store_true",
                   help="2D 矩阵走 Muon,embed/head/bias 走 AdamW")
    p.add_argument("--use_aurora", action="store_true",
                   help="Aurora 变体(leverage-uniform polar),隐含 use_muon")
    p.add_argument("--muon_lr", type=float, default=0.02,
                   help="Muon/Aurora 2D 矩阵的 lr(常用 0.02)")
    p.add_argument("--muon_momentum", type=float, default=0.95)
    p.add_argument("--moonshot_scaling", action="store_true",
                   help="Moonshot per-param lr scaling(默认关)")
    args = p.parse_args()

    device = "cuda"
    os.makedirs(args.output_dir, exist_ok=True)

    # dump args 到 train_args.json,避免事后只能从 lr 轨迹反推
    with open(os.path.join(args.output_dir, "train_args.json"), "w") as f:
        json.dump({"args": vars(args), "cmdline": sys.argv}, f, indent=2, ensure_ascii=False)

    # 读 mask_token_id
    with open(os.path.join(args.model_path, "mask_token_id.txt")) as f:
        mask_token_id = int(f.read().strip())
    print(f"mask_token_id = {mask_token_id}")

    # 加载模型
    model = BertForMaskedLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    model.to(device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    vocab_size = model.config.vocab_size
    pad_id = model.config.pad_token_id
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.1f}M params, vocab={vocab_size}, max_pos={model.config.max_position_embeddings}")

    if args.compile:
        model = torch.compile(model)
        print("torch.compile enabled")

    # 数据
    if args.wwm:
        if args.word_ids_data is None:
            args.word_ids_data = args.train_data + ".wid"
        assert os.path.exists(args.word_ids_data), \
            f"--wwm requires word_ids_data at {args.word_ids_data}"
    ds = IntTensorDataset(args.train_data, args.max_chunks,
                          word_ids_path=args.word_ids_data if args.wwm else None)
    print(f"Dataset: {len(ds):,} chunks × {ds.data.shape[1]} = {ds.data.numel():,} tokens"
          f"{' [WWM]' if args.wwm else ''}")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, pin_memory=True, drop_last=True)
    steps_per_epoch = len(loader) // args.gradient_accumulation_steps
    print(f"steps/epoch: {steps_per_epoch}, max_steps: {args.max_steps}")

    # 优化器:Muon/Aurora(2D 矩阵)+ AdamW(embed/head/bias)or 全 AdamW
    use_muon = args.use_muon or args.use_aurora
    if use_muon:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from muon import SingleDeviceMuonWithAuxAdam
        muon_params, adam_params = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad: continue
            if param.ndim >= 2 and "embed" not in name and "lm_head" not in name:
                muon_params.append(param)
            else:
                adam_params.append(param)
        rule = "Aurora" if args.use_aurora else "Muon"
        if args.moonshot_scaling: rule += "+MoonshotLR"
        print(f"{rule} params: {sum(p.numel() for p in muon_params):,} | "
              f"AdamW params: {sum(p.numel() for p in adam_params):,}")
        param_groups = [
            dict(params=muon_params, lr=args.muon_lr, momentum=args.muon_momentum,
                 weight_decay=args.weight_decay, use_muon=True),
            dict(params=adam_params, lr=args.lr, betas=(0.9, 0.95), eps=1e-10,
                 weight_decay=args.weight_decay, use_muon=False),
        ]
        update_fn = None
        if args.use_aurora:
            from aurora import aurora_update
            update_fn = aurora_update
        optim = SingleDeviceMuonWithAuxAdam(param_groups, update_fn=update_fn,
                                             moonshot_scaling=args.moonshot_scaling)
    else:
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.999), eps=1e-8,
                                  weight_decay=args.weight_decay)
    # per-group min_lr 必须在 scheduler 创建前取(scheduler.__init__ 会 set lr=base×lambda(0)=0,
    # 在 warmup 期间该值是 0,后取就拿到 0 → min_lrs 全为 0 → clip 永远不激活,bug)
    initial_lrs = [g["lr"] for g in optim.param_groups]
    min_lrs = [lr * args.min_lr_ratio for lr in initial_lrs]
    scheduler = get_cosine_schedule_with_warmup(
        optim,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
        num_cycles=0.5,
    )

    step = 0
    accum = 0
    t0 = time.time()
    loss_acc = 0.0
    correct_acc = 0
    n_masked_acc = 0
    model.train()
    while step < args.max_steps:
        for batch in loader:
            if args.wwm:
                input_ids, word_ids = batch
                input_ids = input_ids.to(device, non_blocking=True)
                word_ids = word_ids.to(device, non_blocking=True)
                masked_ids, labels = mlm_mask_batch_wwm(
                    input_ids, word_ids, mask_token_id, vocab_size,
                    prob=args.mlm_prob, pad_id=pad_id)
            else:
                input_ids = batch
                input_ids = input_ids.to(device, non_blocking=True)
                masked_ids, labels = mlm_mask_batch(input_ids, mask_token_id, vocab_size,
                                                    prob=args.mlm_prob, pad_id=pad_id)
            out = model(input_ids=masked_ids, labels=labels)
            loss = out.loss / args.gradient_accumulation_steps
            loss.backward()
            # accuracy on masked
            with torch.no_grad():
                preds = out.logits.argmax(-1)
                mask_pos = labels != -100
                correct_acc += ((preds == labels) & mask_pos).sum().item()
                n_masked_acc += mask_pos.sum().item()
            # loss.item() 已经除过 grad_accum,直接累(每 forward 累一次)
            loss_acc += loss.item()
            accum += 1
            if accum >= args.gradient_accumulation_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optim.step()
                scheduler.step()
                # per-group min_lr clip
                for g, floor in zip(optim.param_groups, min_lrs):
                    if g["lr"] < floor: g["lr"] = floor
                optim.zero_grad(set_to_none=True)
                step += 1
                accum = 0

                if step % args.logging_steps == 0:
                    el = time.time() - t0
                    lrs = [f"{g['lr']:.6f}" for g in optim.param_groups]
                    # loss_acc 累了 args.logging_steps × grad_accum 个 forward
                    avg_loss = loss_acc / (args.logging_steps * args.gradient_accumulation_steps)
                    acc = correct_acc / max(1, n_masked_acc)
                    print(f"step {step}/{args.max_steps} | loss {avg_loss:.4f} | "
                          f"mlm_acc {acc:.4f} | lr [{', '.join(lrs)}] | {el:.1f}s", flush=True)
                    loss_acc = 0.0
                    correct_acc = 0
                    n_masked_acc = 0
                if step % args.save_steps == 0 or step >= args.max_steps:
                    save = os.path.join(args.output_dir, f"checkpoint-{step}")
                    os.makedirs(save, exist_ok=True)
                    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
                    raw.save_pretrained(save, safe_serialization=True)
                    # 拷 mask_token_id 给 inference 用
                    import shutil
                    shutil.copy2(os.path.join(args.model_path, "mask_token_id.txt"),
                                 os.path.join(save, "mask_token_id.txt"))
                    print(f"Saved checkpoint to {save}")
                if step >= args.max_steps:
                    break
        if step >= args.max_steps:
            break

    # 最终 save
    final = args.output_dir
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    raw.save_pretrained(final, safe_serialization=True)
    print(f"\nFinal save to {final}")
    print(f"Training complete: {step} steps in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
