"""下游任务评测。只依赖 torch。

口径跟重构前一致:
  CWS  micro span F1
  POS  per-word 准确率(只在该句 CWS 完全切对时才计入 —— 切错了词,词级 POS 无意义)
  NER  micro span F1
  MT joint score = cws_f1 + 0.3 · pos_acc + 0.2 · ner_f1(选 best.pt 用)
  CSC  pycorrector 口径的句级 P / R / F1(SIGHAN-15 707 条)

CWS / NER 的 span 是直接从 BIES 标签算的,不需要原文 —— 原实现先把标签还原成词串
再比,但同一句里预测和标准答案的字完全相同,比词串等价于比 (start, end) 区间。
所以这里不碰文本。

CSC 那条要还原成字符串才能跟 pycorrector 对齐,用的是预编码文件里带的
id_to_char 表(prepare/ 写入,只覆盖编码时真正出现过的字,跟原实现的
字符缓存范围一致 —— 表更大会让本该回退成原字的未知 id 变成真解码,口径就偏了)。
"""
import torch
from torch.utils.data import DataLoader


# ---------------------------------------------------------------- span 工具

def bies_to_spans(tags, cws_vocab: list[str]) -> set:
    """BIES 标签序列 → {(start, end)}。对不合法序列的处理跟原实现一致:
    遇到 B / S 就起新词,I / E 一律并入当前词。"""
    id2tag = cws_vocab
    spans, start = set(), None
    for i, t in enumerate(tags):
        tag = id2tag[int(t)]
        if tag in ("B", "S"):
            if start is not None:
                spans.add((start, i))
            start = i
        elif start is None:                 # 句首就是 I / E,当作起点
            start = i
    if start is not None:
        spans.add((start, len(tags)))
    return spans


def ner_tags_to_spans(tags, ner_vocab: list[str]) -> set:
    """BIES-类型 标签序列 → {(类型, start, end)}。B 后面没等到 E 的残缺片段丢弃。"""
    id2tag = ner_vocab
    spans, i, n = set(), 0, len(tags)
    while i < n:
        t = id2tag[int(tags[i])]
        if t.startswith("S-"):
            spans.add((t[2:], i, i + 1))
            i += 1
        elif t.startswith("B-"):
            et, j = t[2:], i + 1
            while j < n:
                tj = id2tag[int(tags[j])]
                if tj == f"I-{et}":
                    j += 1
                elif tj == f"E-{et}":
                    j += 1
                    spans.add((et, i, j))
                    break
                else:
                    break
            i = max(j, i + 1)
        else:
            i += 1
    return spans


def _f1(tp: int, n_pred: int, n_gold: int) -> float:
    p = tp / max(1, n_pred)
    r = tp / max(1, n_gold)
    return 2 * p * r / max(1e-9, p + r)


# ---------------------------------------------------------------- MT

@torch.no_grad()
def evaluate_mt(model, dataset, collator, device, batch_size: int = 64) -> dict:
    """返回 cws_f1 / pos_acc / ner_f1 / score。"""
    was_training = model.training
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collator, num_workers=0)

    cws_tp = cws_np = cws_ng = 0
    ner_tp = ner_np = ner_ng = 0
    pos_ok = pos_total = 0
    idx = 0

    for batch in loader:
        b = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            cws_preds = model.decode_cws(b["input_ids"], b["attention_mask"])
            ner_preds = model.decode_ner(b["input_ids"], b["attention_mask"])
            pos_preds = model.predict_pos(b["input_ids"], b["attention_mask"]).cpu()

        for i in range(len(cws_preds)):
            item = dataset[idx]
            idx += 1
            gold_cws = item["cws_tags"]
            n = min(gold_cws.numel(), len(cws_preds[i]), len(ner_preds[i]))

            gold_spans = bies_to_spans(gold_cws[:n], dataset.cws_vocab)
            pred_spans = bies_to_spans(cws_preds[i][:n], dataset.cws_vocab)
            cws_tp += len(gold_spans & pred_spans)
            cws_np += len(pred_spans)
            cws_ng += len(gold_spans)

            # 只有整句切分完全正确时才评 POS,跟原口径一致
            if pred_spans == gold_spans:
                gold_pos = item["pos_tags"]
                for s, _ in sorted(gold_spans):
                    if s >= n:
                        continue
                    gp = int(gold_pos[s])
                    if gp < 0:                      # -100:该词无 POS 监督
                        continue
                    pos_total += 1
                    if int(pos_preds[i, s]) == gp:
                        pos_ok += 1

            gold_ner = ner_tags_to_spans(item["ner_tags"][:n], dataset.ner_vocab)
            pred_ner = ner_tags_to_spans(ner_preds[i][:n], dataset.ner_vocab)
            ner_tp += len(gold_ner & pred_ner)
            ner_np += len(pred_ner)
            ner_ng += len(gold_ner)

    if was_training:
        model.train()

    cws_f1 = _f1(cws_tp, cws_np, cws_ng)
    pos_acc = pos_ok / max(1, pos_total)
    ner_f1 = _f1(ner_tp, ner_np, ner_ng)
    return {"cws_f1": cws_f1, "pos_acc": pos_acc, "ner_f1": ner_f1,
            "score": cws_f1 + 0.3 * pos_acc + 0.2 * ner_f1}


# ---------------------------------------------------------------- CSC

@torch.no_grad()
def correct_ids(model, input_ids, attention_mask, threshold: float = 0.7):
    """纠错一个 batch,返回修正后的 id。

    跟 pycorrector 的 MacBertCorrector 对齐:取 argmax 候选,但 softmax 概率
    低于 threshold 时保留原字 —— 低置信度的"纠正"绝大多数是误伤,
    这个阈值是 CSC 精确率的主要来源。
    """
    cor_logits, _ = model(input_ids, attention_mask)
    probs = torch.softmax(cor_logits.float(), dim=-1)
    top_prob, top_id = probs.max(dim=-1)
    keep = top_prob < threshold
    return torch.where(keep, input_ids, top_id)


@torch.no_grad()
def evaluate_csc(model, dataset, collator, device, id_to_char: dict,
                 threshold: float = 0.7, batch_size: int = 32,
                 src_texts=None, tgt_texts=None) -> dict:
    """pycorrector 口径的句级评测。

    句级判定:
      原句==答案 且 预测==答案 → TN(本来就对,没被改坏)
      原句==答案 但 预测!=答案 → FP(误伤)
      原句!=答案 且 预测==答案 → TP(改对了)
      原句!=答案 但 预测!=答案 → FN(没改对)
    整句必须完全一致才算对,改对一半不给分。

    **参照必须用 src_texts / tgt_texts 里的原文**(prepare/ 写进数据集文件)。
    id→字 的往返是有损的:不同的字可能撞到同一个 id,而且截断会丢尾巴。
    拿还原出来的文本当参照,分数会虚高(实测 SIGHAN-15 上 +0.006)。
    预测那一侧只能走还原,这跟原实现一致。
    """
    was_training = model.training
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collator, num_workers=0)

    def to_text(ids, fallback):
        """id → 字。表里没有的 id 保留原字,跟原实现一致。"""
        return "".join(id_to_char.get(int(t), id_to_char.get(int(f), ""))
                       for t, f in zip(ids, fallback))

    tp = fp = fn = tn = 0
    idx = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attn = batch["attention_mask"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred_ids = correct_ids(model, input_ids, attn, threshold=threshold)
        pred_ids = pred_ids.cpu()
        src_ids = batch["input_ids"]
        cor_ids = batch["cor_labels"]

        for i in range(src_ids.size(0)):
            n = int(batch["attention_mask"][i].sum())
            pred = to_text(pred_ids[i, :n], src_ids[i, :n])
            if src_texts is not None:
                src, tgt = src_texts[idx], tgt_texts[idx]
                pred = pred + src[n:]          # 超出 max_len 的尾巴原样接回
            else:
                src = to_text(src_ids[i, :n], src_ids[i, :n])
                tgt = to_text(cor_ids[i, :n], src_ids[i, :n])
            idx += 1
            if src == tgt:
                tn += (tgt == pred)
                fp += (tgt != pred)
            else:
                tp += (tgt == pred)
                fn += (tgt != pred)

    if was_training:
        model.train()

    n = tp + fp + fn + tn
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return {"acc": (tp + tn) / max(1, n), "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / max(1e-9, prec + rec),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn, "n": n}
