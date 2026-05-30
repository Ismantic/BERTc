"""批量 micro F1 重测所有 single-CWS ckpt 在 raw + clean dev,产出完整对比表。"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from torch.utils.data import DataLoader
from data import CWSDataset, Collator, bies_to_words
from model import BertCRF


def _spans(ws):
    out, pos = set(), 0
    for w in ws: out.add((pos, pos+len(w))); pos += len(w)
    return out


def eval_micro(model, ds, collator, device, batch_size=64):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collator, num_workers=0)
    tp, p_total, g_total = 0, 0, 0
    idx = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                preds = model.decode(batch["input_ids"], batch["attention_mask"])
            for pred_tags in preds:
                item = ds.items[idx]
                chars = item["chars"]
                n = min(len(chars), len(pred_tags))
                pred_words = bies_to_words(chars[:n], pred_tags[:n])
                gold_words = bies_to_words(chars[:n], item["tags"][:n])
                P, G = _spans(pred_words), _spans(gold_words)
                tp += len(P & G); p_total += len(P); g_total += len(G)
                idx += 1
    p = tp/max(1,p_total); r = tp/max(1,g_total)
    return 2*p*r/max(1e-9, p+r)


def get_tok(model_path):
    if os.path.exists(os.path.join(model_path, "piece.model")):
        from piece_tokenizer_adapter import PieceTokenizerAdapter
        return PieceTokenizerAdapter(model_path)
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_path, use_fast=False)


CKPTS = [
    ("v4 (BERT-mid)",      "output_v4_pd98_crf/best.pt",   "/home/tfbao/Shiyu/Summer/BERT/bert_train_v6_mid"),
    ("v5 (BERT-mid)",      "output_v5_pd98_crf/best.pt",   "/home/tfbao/Shiyu/Summer/BERT/bert_train_v6_mid"),
    ("v6 (BERT-mid)",      "output_v6_pd98_crf/best.pt",   "/home/tfbao/Shiyu/Summer/BERT/bert_train_v6_mid"),
    ("v6+FGM 3ep",         "output_v6_fgm_crf/best.pt",    "/home/tfbao/Shiyu/Summer/BERT/bert_train_v6_mid"),
    ("v6+FGM 5ep",         "output_v6_fgm5_crf/best.pt",   "/home/tfbao/Shiyu/Summer/BERT/bert_train_v6_mid"),
    ("MacBERT-large",      "output_macbert_pd98_crf/best.pt", "./macbert-large"),
    ("RoBERTa-wwm-ext",    "output_roberta_pd98_crf/best.pt", "./roberta-wwm-ext"),
]

device = torch.device("cuda")
results = []
for tag, ckpt, model_path in CKPTS:
    if not os.path.exists(ckpt):
        print(f"SKIP {tag}: ckpt missing")
        continue
    t0 = time.time()
    tok = get_tok(model_path)
    model = BertCRF(model_path, num_tags=4).to(device)
    sd = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(sd)
    model.eval()
    collator = Collator(tok)

    raw_ds = CWSDataset("./data/cws_dev.pd98.jsonl", max_chars=254)
    f1_raw = eval_micro(model, raw_ds, collator, device)
    clean_ds = CWSDataset("./data/cws_dev.pd98.cleanjudge.jsonl", max_chars=254)
    f1_clean = eval_micro(model, clean_ds, collator, device)
    el = time.time() - t0
    results.append((tag, f1_raw, f1_clean))
    print(f"{tag:20s}  raw={f1_raw:.4f}  clean={f1_clean:.4f}  ({el:.0f}s)", flush=True)
    del model, sd
    torch.cuda.empty_cache()

print("\n=== Final table (CWS micro F1) ===")
print(f"{'Model':22s} {'raw 21143':>10s} {'clean 18221':>12s}")
for tag, fr, fc in results:
    print(f"{tag:22s} {fr:>10.4f} {fc:>12.4f}")
print(f"{'LTP/base1 zero-shot':22s} {'0.9782':>10s} {'0.9783':>12s}")
