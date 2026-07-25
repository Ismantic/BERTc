"""MLM 掩码。只依赖 torch。

两种:逐 token 随机(mlm_mask_batch)和整词(mlm_mask_batch_wwm)。
v4-Large 用的是 WWM —— 词边界由 prepare/ 用 Wapic 切好、写在 .wid 里,
同一个词的所有字共享一个 word id。

两者都遵循 BERT 原文的 80/10/10:选中的位置里 80% 换成 [MASK]、
10% 换成随机 token、10% 保持原样(但仍然算 loss)。

掩码逻辑改错了不会报错,只会让预训练目标悄悄变掉(80/10/10 比例偏了、
WWM 退化成逐字掩码),训练照跑、loss 照降。改这里要格外小心。
"""
import torch


def mlm_mask_batch(input_ids: torch.Tensor, mask_token_id: int, vocab_size: int,
                   prob: float = 0.15, pad_id: int = 0):
    """逐 token 独立采样。返回 (masked_ids, labels),labels 里 -100 表示不算 loss。"""
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
    # 剩下 20% 里再取一半 → 随机 token(即总体 10%)
    replace_rand = torch.bernoulli(
        torch.full(labels.shape, 0.5, device=dev)).bool() & masked_indices & ~replace_mask
    random_words = torch.randint(0, vocab_size, labels.shape,
                                 dtype=input_ids.dtype, device=dev)
    input_ids[replace_rand] = random_words[replace_rand]
    return input_ids, labels


def mlm_mask_batch_wwm(input_ids: torch.Tensor, word_ids: torch.Tensor,
                       mask_token_id: int, vocab_size: int,
                       prob: float = 0.15, pad_id: int = 0):
    """整词掩码:按 word id 分组,选中的词整体一起 80/10/10。

    word_ids: [B, L] int64,同一个词的字共享 id。不同样本的 word id 会撞,
    所以先按 batch 加偏移;pad 位置统一映射到一个哨兵值,免得污染 unique。
    """
    dev = input_ids.device
    B, L = input_ids.shape
    input_ids = input_ids.clone()
    labels = input_ids.clone()
    not_pad = input_ids != pad_id

    max_wid = int(word_ids.max().item()) + 1
    offset = torch.arange(B, device=dev, dtype=word_ids.dtype).unsqueeze(1) * max_wid
    global_wid = word_ids + offset
    sentinel = global_wid.max().item() + 1
    wid_for_unique = torch.where(not_pad, global_wid,
                                 torch.full_like(global_wid, sentinel))

    unique_wids, inverse = torch.unique(wid_for_unique.flatten(), return_inverse=True)
    valid = unique_wids != sentinel
    n_words = int(valid.sum().item())
    if n_words == 0:
        return input_ids, torch.full_like(labels, -100)

    valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
    n_to_mask = max(1, int(n_words * prob))
    chosen_unique_idx = valid_idx[torch.randperm(n_words, device=dev)[:n_to_mask]]

    # 每个被选中的词独立决定 80/10/10
    chosen_flag = torch.zeros(len(unique_wids), dtype=torch.bool, device=dev)
    chosen_flag[chosen_unique_idx] = True
    decision = torch.rand(len(unique_wids), device=dev)
    is_mask_word = chosen_flag & (decision < 0.8)
    is_rand_word = chosen_flag & (decision >= 0.8) & (decision < 0.9)

    chosen_per_pos = chosen_flag[inverse].view(B, L) & not_pad
    mask_per_pos = is_mask_word[inverse].view(B, L) & not_pad
    rand_per_pos = is_rand_word[inverse].view(B, L) & not_pad

    labels[~chosen_per_pos] = -100
    input_ids[mask_per_pos] = mask_token_id
    rand_tokens = torch.randint(0, vocab_size, input_ids.shape,
                                dtype=input_ids.dtype, device=dev)
    input_ids[rand_per_pos] = rand_tokens[rand_per_pos]
    return input_ids, labels


def dynamic_mlm_prob(step: int, total_steps: int, warmup_frac: float = 0.05,
                     low: float = 0.15, high: float = 0.30) -> float:
    """动态掩码率(anti-curriculum)。

    前 warmup_frac 的 step:low → high,逼模型早期做全局推理;
    之后:high → low,后期转向局部精修。

    注意 v4-Mid / v4-Large 实际上**关掉了**这个 curriculum(run_v4_*.sh 里
    --mlm_low 和 --mlm_high 都传 0.15,即恒定 15%),因为 v4-Mid 的消融显示
    固定 15% 就拿到了 MT/CSC 双 SOTA。保留实现是为了能复现 v3 以及后续实验。
    """
    if total_steps <= 0:
        return high
    p = step / total_steps
    wfrac = max(1e-6, warmup_frac)
    if p < wfrac:
        return low + (high - low) * (p / wfrac)
    t = min(1.0, (p - wfrac) / max(1e-6, 1.0 - wfrac))
    return high - (high - low) * t
