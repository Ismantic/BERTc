"""跑 MacBERT4CSC 在 SIGHAN-15 test 上的 baseline.
CPU 推理,batch=16,~5-10 min 全跑完。
"""
import time, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import torch
from transformers import BertForMaskedLM, BertTokenizer

from eval_ctc import load_dataset, evaluate, print_results

MODEL = "/home/tfbao/Shiyu/data/bertc_ctc/macbert4csc_model"

print("Loading MacBERT4CSC model + tokenizer...", flush=True)
tok = BertTokenizer.from_pretrained(MODEL)
model = BertForMaskedLM.from_pretrained(MODEL)
model.eval()
device = "cpu"
model.to(device)
print(f"  Model on {device}, vocab={tok.vocab_size}, params={sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)

def correct_batch(texts, max_len=128, threshold=0.7):
    """跟 pycorrector 完全对齐:
       - argmax 取候选
       - softmax 概率 > threshold 才采纳改正,否则用原字"""
    enc = tok(texts, return_tensors='pt', padding=True, truncation=True, max_length=max_len)
    with torch.no_grad():
        out = model(**{k: v.to(device) for k, v in enc.items()})
    pred_ids = out.logits.argmax(-1)
    probs = torch.max(torch.softmax(out.logits, dim=-1), dim=-1)[0]   # [B, T]
    preds = []
    for i, text in enumerate(texts):
        orig_chars = list(text)
        L = len(orig_chars)
        pred_seq = pred_ids[i, 1:L+1].tolist()
        prob_seq = probs[i, 1:L+1].tolist()
        decoded = tok.convert_ids_to_tokens(pred_seq, skip_special_tokens=False)
        decoded = [c.replace('##', '') for c in decoded]
        if len(decoded) != L:
            preds.append(text)
            continue
        final = []
        for j, c in enumerate(decoded):
            # special / 多字 / 低概率 → 用原字
            if c.startswith('[') and c.endswith(']'):
                final.append(orig_chars[j])
            elif len(c) != 1:
                final.append(orig_chars[j])
            elif prob_seq[j] < threshold:
                final.append(orig_chars[j])
            else:
                final.append(c)
        preds.append(''.join(final))
    return preds


# 跑 SIGHAN-15 test
data = load_dataset('sighan15/test')
print(f"\nLoaded sighan15/test: {len(data)} samples", flush=True)

BATCH = 16
preds = []
t0 = time.time()
for i in range(0, len(data), BATCH):
    chunk = data[i:i+BATCH]
    texts = [s for s, _, _ in chunk]
    batch_preds = correct_batch(texts)
    preds.extend(batch_preds)
    if (i // BATCH) % 5 == 0:
        elapsed = time.time() - t0
        rate = (i + BATCH) / elapsed
        eta = (len(data) - i - BATCH) / max(rate, 0.1)
        print(f"  [{i+BATCH}/{len(data)}] {rate:.1f} samples/s, ETA {eta:.0f}s", flush=True)

print(f"\nInference done in {time.time()-t0:.0f}s ({len(data) / (time.time()-t0):.1f} samples/s CPU)", flush=True)

# 评估
r = evaluate(data, preds)
print_results("MacBERT4CSC on SIGHAN-15 test (CPU)", r)

# 抽 5 个例子手工看
print("\n=== 抽 5 个例子 ===")
for i in [0, 200, 500, 800, 1099]:
    s, t, ty = data[i]
    p = preds[i]
    mark = "✓" if p == t else "✗"
    print(f"\n[{i}] {ty} {mark}")
    print(f"  src:  {s}")
    print(f"  tgt:  {t}")
    print(f"  pred: {p}")
