"""加载 BertMT ckpt,在 PD dev 上用 LTP-style micro/per-word metric 评测,
直接 fair compare LTP zero-shot 数。
"""
import argparse, json, os, sys, time
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import bies_to_words
from data_pos_ner import build_pos_vocab
from data_mt import MTDataset, MTCollator
from model_mt import BertMT
from ltp_label_align import bies_tags_to_spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="best.pt path")
    ap.add_argument("--model_path", required=True, help="backbone init dir(同 train 时)")
    ap.add_argument("--cws_dev", default="./data/cws_dev.pd98.jsonl")
    ap.add_argument("--pos_dev", default="./data/pos_dev.pd98.jsonl")
    ap.add_argument("--ner_dev", default="./data/ner_dev.pd98.jsonl")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda")
    if os.path.exists(os.path.join(args.model_path, "piece.model")):
        from piece_tokenizer_adapter import PieceTokenizerAdapter
        tokenizer = PieceTokenizerAdapter(args.model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)

    pos2id = build_pos_vocab()
    ds = MTDataset(args.cws_dev, args.pos_dev, args.ner_dev, pos2id, max_chars=254)
    if args.limit: ds.items = ds.items[:args.limit]
    print(f"Dev: {len(ds)} samples")

    model = BertMT(args.model_path, num_pos=len(pos2id)).to(device)
    sd = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(sd)
    model.eval()

    collator = MTCollator(tokenizer)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collator, num_workers=0)

    cws_tp, cws_p, cws_g = 0, 0, 0
    ner_tp, ner_p, ner_g = 0, 0, 0
    pos_word_correct, pos_word_total = 0, 0
    pos_char_correct, pos_char_total = 0, 0
    idx = 0
    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            b = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                cws_preds = model.decode_cws(b["input_ids"], b["attention_mask"])
                ner_preds = model.decode_ner(b["input_ids"], b["attention_mask"])
                pos_pred = model.predict_pos(b["input_ids"], b["attention_mask"]).cpu().numpy()
            pos_labels = b["pos_labels"].cpu().numpy()
            valid = pos_labels != -100
            pos_char_correct += int(((pos_pred == pos_labels) & valid).sum())
            pos_char_total += int(valid.sum())
            for i in range(len(cws_preds)):
                item = ds.items[idx]
                chars = item["chars"]
                n = min(len(chars), len(cws_preds[i]), len(ner_preds[i]))
                gold_words = bies_to_words(chars[:n], item["cws_tags"][:n])
                pred_words = bies_to_words(chars[:n], cws_preds[i][:n])
                def _spans(ws):
                    out, p = set(), 0
                    for w in ws: out.add((p, p+len(w))); p += len(w)
                    return out
                P, G = _spans(pred_words), _spans(gold_words)
                cws_tp += len(P & G); cws_p += len(P); cws_g += len(G)
                if pred_words == gold_words:
                    cpos = 0
                    pp = pos_pred[i]
                    for w in gold_words:
                        gp = int(item['pos_tags'][cpos]) if cpos < len(item['pos_tags']) else -100
                        pr = int(pp[cpos]) if cpos < len(pp) else -1
                        if gp >= 0:
                            pos_word_total += 1
                            if pr == gp: pos_word_correct += 1
                        cpos += len(w)
                gold_ner_t = [(t,s,e) for t,s,e in bies_tags_to_spans(item["ner_tags"][:n])]
                pred_ner_t = [(t,s,e) for t,s,e in bies_tags_to_spans(ner_preds[i][:n])]
                ner_tp += len(set(gold_ner_t) & set(pred_ner_t))
                ner_p += len(set(pred_ner_t))
                ner_g += len(set(gold_ner_t))
                idx += 1
    cp, cr = cws_tp/max(1,cws_p), cws_tp/max(1,cws_g)
    cf = 2*cp*cr/max(1e-9, cp+cr)
    np_, nr = ner_tp/max(1,ner_p), ner_tp/max(1,ner_g)
    nf = 2*np_*nr/max(1e-9, np_+nr)
    print(f"\n=== {os.path.basename(args.ckpt)} on PD-1998 dev (n={len(ds)}) ===")
    print(f"  CWS micro F1:     {cf:.4f}  (P={cp:.4f} R={cr:.4f})")
    print(f"  POS per-word acc: {pos_word_correct/max(1,pos_word_total):.4f}  (only when CWS==gold,n={pos_word_total:,})")
    print(f"  POS per-char acc: {pos_char_correct/max(1,pos_char_total):.4f}")
    print(f"  NER micro F1:     {nf:.4f}  (P={np_:.4f} R={nr:.4f},n_gold={ner_g:,})")
    print(f"  time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
