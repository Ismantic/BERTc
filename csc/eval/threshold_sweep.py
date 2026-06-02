"""在训好的 ckpt 上扫 threshold,看不重训能涨多少 F1。

我们 v1: P=0.95 R=0.69 F1=0.80。Precision 太高 = 模型太保守 = 阈值太高。
降低 threshold → 更多预测 → P 略降 / R 涨 → F1 可能涨。
"""
import argparse
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/tfbao/Shiyu/Summer/BERT/NLP_BERT_CRF")
sys.path.insert(0, "/home/tfbao/Shiyu/BERTc/csc/train")
from piece_tokenizer_adapter import PieceTokenizerAdapter
from train_csc import BERTcForCSC, CharToId


@torch.no_grad()
def get_predictions(model, char_to_id, texts, device, max_len=128):
    """一次性跑出所有 (top1_id, top1_prob) 序列,后续可用任意 threshold 过滤。"""
    model.eval()
    pad_id = char_to_id.tok.pad_token_id
    results = []
    for text in texts:
        L = min(len(text), max_len)
        ids = [char_to_id(c) for c in text[:L]]
        input_ids = torch.tensor([ids], dtype=torch.long).to(device)
        attn = torch.ones((1, L), dtype=torch.long).to(device)
        cor_logits, _ = model(input_ids, attn)
        probs = F.softmax(cor_logits[0], dim=-1)
        top_probs, top_ids = probs.max(dim=-1)
        results.append((text, L, top_ids.cpu().tolist(), top_probs.cpu().tolist()))
    return results


def evaluate_with_threshold(samples, model_predictions, id_to_char, threshold):
    TP = FP = FN = TN = 0
    for (src, tgt), (orig_text, L, top_ids, top_probs) in zip(samples, model_predictions):
        assert src == orig_text
        pred = []
        for j in range(L):
            tid = top_ids[j]
            p = top_probs[j]
            if p < threshold:
                pred.append(src[j])
            elif tid in id_to_char:
                pred.append(id_to_char[tid])
            else:
                pred.append(src[j])
        if len(src) > L:
            pred.extend(list(src[L:]))
        pred = "".join(pred)
        if src == tgt:
            if tgt == pred: TN += 1
            else: FP += 1
        else:
            if tgt == pred: TP += 1
            else: FN += 1
    n = len(samples)
    acc = (TP + TN) / n
    prec = TP / max(1, TP + FP)
    rec = TP / max(1, TP + FN)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return acc, prec, rec, f1, TP, FP, FN, TN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help=".pt file")
    ap.add_argument("--backbone_path", required=True)
    ap.add_argument("--test_tsv",
                    default="/home/tfbao/Shiyu/BERTc/csc/data/test/sighan2015_test_official.tsv")
    ap.add_argument("--max_len", type=int, default=128)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # tokenizer
    tok = PieceTokenizerAdapter(args.backbone_path)
    char_to_id = CharToId(tok)
    # 预热 cache(从 test 集)
    samples = []
    with open(args.test_tsv) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = line.split("\t")
            if len(parts) != 2: continue
            samples.append((parts[0], parts[1]))
    for src, tgt in samples:
        for c in src + tgt:
            char_to_id(c)
    # 还要从训练数据加(模型可能预测的字符)
    print(f"Cache warm from test only: {len(char_to_id.cache)}")
    import pickle
    with open("/home/tfbao/Shiyu/BERTc/csc/data/sighan_wang271k_pairs.pkl", "rb") as f:
        pairs = pickle.load(f)
    for src, tgt in pairs:
        for c in src + tgt:
            char_to_id(c)
    print(f"Cache warm from train+test: {len(char_to_id.cache)}")

    # model
    model = BERTcForCSC(args.backbone_path, tok.vocab_size).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    print(f"Loaded ckpt: epoch={state.get('epoch')}, metrics={state.get('metrics')}")

    # id_to_char
    id_to_char = {}
    for c, tid in char_to_id.cache.items():
        id_to_char.setdefault(tid, c)

    # 一次性推理出每个位置的 (top1_id, top1_prob)
    srcs = [s for s, _ in samples]
    print(f"Running inference on {len(srcs)} samples...")
    import time
    t0 = time.time()
    preds = get_predictions(model, char_to_id, srcs, device, max_len=args.max_len)
    print(f"  done in {time.time()-t0:.1f}s")

    # 扫 threshold
    print("\nThreshold sweep:")
    print(f"  {'thresh':>7} {'Acc':>6} {'P':>6} {'R':>6} {'F1':>6}   TP/FP/FN/TN")
    best = (0, 0)
    for th in [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        acc, prec, rec, f1, TP, FP, FN, TN = evaluate_with_threshold(
            samples, preds, id_to_char, th
        )
        marker = " ←" if f1 > best[0] else ""
        print(f"  {th:>7.2f} {acc:.4f} {prec:.4f} {rec:.4f} {f1:.4f}   "
              f"{TP}/{FP}/{FN}/{TN}{marker}")
        if f1 > best[0]:
            best = (f1, th)
    print(f"\nBest: threshold={best[1]:.2f} → F1={best[0]:.4f}")


if __name__ == "__main__":
    main()
