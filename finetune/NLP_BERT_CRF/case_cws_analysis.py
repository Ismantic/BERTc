"""分析 v6 MT CWS 错误类型: split / merge / shift 错误,抽 case。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from collections import Counter
from torch.utils.data import DataLoader
from data import bies_to_words
from data_pos_ner import build_pos_vocab
from data_mt import MTDataset, MTCollator
from model_mt import BertMT
from piece_tokenizer_adapter import PieceTokenizerAdapter


def categorize_error(pred_words, gold_words):
    """对每个错的 pred span(不在 gold)分类:
    - split: pred 一个词,gold 是多个连续词(over-segment)
    - merge: pred 跨多个 gold 词(under-segment)
    - shift: 跟 gold 部分重叠但边界不同
    返回 errs: [(类型, pred_word, gold_words_span)]
    """
    def _spans(ws):
        out, p = [], 0
        for w in ws: out.append((p, p+len(w), w)); p += len(w)
        return out
    PS = _spans(pred_words)
    GS = _spans(gold_words)
    G_starts = set(g[0] for g in GS)
    G_ends = set(g[1] for g in GS)
    errs = []
    for ps, pe, pw in PS:
        if (ps, pe) in [(g[0],g[1]) for g in GS]: continue
        # 跟哪些 gold span overlap
        ov = [g for g in GS if not (g[1]<=ps or g[0]>=pe)]
        if not ov: continue
        # 分类
        if ps in G_starts and pe in G_ends:
            # 边界正确但 pred span 包含多 gold span
            errs.append(("merge", pw, "/".join(g[2] for g in ov)))
        elif ps in G_starts:
            errs.append(("shift_right", pw, "/".join(g[2] for g in ov)))
        elif pe in G_ends:
            errs.append(("shift_left", pw, "/".join(g[2] for g in ov)))
        else:
            errs.append(("misalign", pw, "/".join(g[2] for g in ov)))
    return errs


def main():
    model_path = "/home/tfbao/Shiyu/Summer/BERT/bert_train_v6_mid"
    ckpt = "output_v6_mt_crf/best.pt"
    device = torch.device("cuda")
    tok = PieceTokenizerAdapter(model_path)
    pos2id = build_pos_vocab()
    ds = MTDataset("./data/cws_dev.pd98.jsonl", "./data/pos_dev.pd98.jsonl",
                   "./data/ner_dev.pd98.jsonl", pos2id, max_chars=254)
    print(f"Dev: {len(ds)} samples")
    model = BertMT(model_path, num_pos=len(pos2id)).to(device)
    sd = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(sd)
    model.eval()
    collator = MTCollator(tok)
    loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collator, num_workers=0)

    err_type_count = Counter()
    err_cases = {"merge": [], "shift_right": [], "shift_left": [], "misalign": []}
    miss_gold = Counter()  # gold word 漏切(pred 没识别为独立 word)
    over_pred = Counter()  # pred 多出的 word(gold 不含)
    idx = 0
    with torch.no_grad():
        for batch in loader:
            b = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                cws_preds = model.decode_cws(b["input_ids"], b["attention_mask"])
            for i in range(len(cws_preds)):
                item = ds.items[idx]; idx += 1
                chars = item["chars"]
                n = min(len(chars), len(cws_preds[i]))
                gold_words = bies_to_words(chars[:n], item["cws_tags"][:n])
                pred_words = bies_to_words(chars[:n], cws_preds[i][:n])
                if pred_words == gold_words: continue
                # 计算错
                errs = categorize_error(pred_words, gold_words)
                for et, pw, gw in errs:
                    err_type_count[et] += 1
                    if len(err_cases[et]) < 30:
                        text = "".join(chars[:n])
                        err_cases[et].append((pw, gw, text[:80]))

    print(f"\n=== CWS 错误类型统计 ===")
    total_err = sum(err_type_count.values())
    for et, c in err_type_count.most_common():
        print(f"  {et:15s}: {c:>6d} ({100*c/max(1,total_err):.1f}%)")
    print(f"  total errors  : {total_err:>6d}")

    print(f"\n=== merge 错误抽样(pred 一词 = gold 多词) ===")
    for pw, gw, ctx in err_cases["merge"][:15]:
        print(f"  pred [{pw}] vs gold [{gw}]")
    print(f"\n=== shift_right 抽样(pred 起点对,终点扩展) ===")
    for pw, gw, ctx in err_cases["shift_right"][:10]:
        print(f"  pred [{pw}] vs gold [{gw}]")
    print(f"\n=== shift_left 抽样 ===")
    for pw, gw, ctx in err_cases["shift_left"][:10]:
        print(f"  pred [{pw}] vs gold [{gw}]")
    print(f"\n=== misalign(完全错位)抽样 ===")
    for pw, gw, ctx in err_cases["misalign"][:10]:
        print(f"  pred [{pw}] vs gold [{gw}]")


if __name__ == "__main__":
    main()
