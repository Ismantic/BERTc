"""分析 v6 MT POS 错误分布:per-tag 错率 + 抽 case。"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from collections import Counter, defaultdict
from torch.utils.data import DataLoader
from data import bies_to_words
from data_pos_ner import build_pos_vocab
from data_mt import MTDataset, MTCollator
from model_mt import BertMT
from ltp_label_align import LTP_ID2POS, map_pd_pos
from piece_tokenizer_adapter import PieceTokenizerAdapter


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

    # per-word level: 仅当 CWS pred==gold 时算(跟训练 eval 一致)
    # 也算 per-char (不依赖 CWS pred 对错)
    word_correct = Counter()
    word_total = Counter()
    confusion = defaultdict(Counter)  # gold_tag -> Counter(pred_tag -> count)
    case_samples = defaultdict(list)  # gold_tag -> 错例 [(word, pred_tag, context)]
    idx = 0
    with torch.no_grad():
        for batch in loader:
            b = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                cws_preds = model.decode_cws(b["input_ids"], b["attention_mask"])
                pos_pred = model.predict_pos(b["input_ids"], b["attention_mask"]).cpu().numpy()
            for i in range(len(cws_preds)):
                item = ds.items[idx]
                chars = item["chars"]
                n = min(len(chars), len(cws_preds[i]))
                gold_words = bies_to_words(chars[:n], item["cws_tags"][:n])
                pred_words = bies_to_words(chars[:n], cws_preds[i][:n])
                if pred_words != gold_words:
                    idx += 1; continue
                # word-level POS:取每个 word 第 1 char 的 pos_pred 跟 gold pos_tag
                cpos = 0
                pp = pos_pred[i]
                text = "".join(chars[:n])
                for w in gold_words:
                    if cpos >= len(item['pos_tags']) or cpos >= len(pp):
                        cpos += len(w); continue
                    gp = int(item['pos_tags'][cpos])
                    pr = int(pp[cpos])
                    if gp >= 0:
                        gtag = LTP_ID2POS.get(gp, "?")
                        ptag = LTP_ID2POS.get(pr, "?")
                        word_total[gtag] += 1
                        confusion[gtag][ptag] += 1
                        if gtag == ptag:
                            word_correct[gtag] += 1
                        else:
                            if len(case_samples[gtag]) < 5:
                                # 截取 context
                                ws = max(0, cpos - 10); we = min(len(text), cpos + len(w) + 10)
                                ctx = text[ws:cpos] + f"【{w}】" + text[cpos+len(w):we]
                                case_samples[gtag].append((w, ptag, ctx))
                    cpos += len(w)
                idx += 1

    print(f"\n=== Per-tag 错误率 (按总数排序) ===")
    print(f"{'tag':6s} {'gold count':>10s} {'correct':>10s} {'acc':>8s} {'top 错→':<30s}")
    rows = []
    for t in sorted(word_total, key=lambda x: -word_total[x]):
        tot = word_total[t]
        cor = word_correct[t]
        acc = cor / max(1, tot)
        # top 错预测
        errs = [(p, c) for p, c in confusion[t].most_common() if p != t][:3]
        err_str = ", ".join([f"{p}({c})" for p, c in errs])
        rows.append((t, tot, cor, acc, err_str))
    for t, tot, cor, acc, errs in rows:
        print(f"{t:6s} {tot:>10d} {cor:>10d} {acc:>8.4f}  {errs}")

    print(f"\n=== 错率最高的 5 个 tag 的 case 样本 ===")
    # 按错率排
    sorted_by_errrate = sorted(rows, key=lambda r: r[3])
    for t, tot, cor, acc, errs in sorted_by_errrate[:5]:
        print(f"\n--- tag '{t}' (gold={tot}, acc={acc:.4f}, top错={errs}) ---")
        for w, ptag, ctx in case_samples[t][:5]:
            print(f"   {w} (pred {ptag}): ...{ctx}...")

    # 整体
    total = sum(word_total.values())
    correct = sum(word_correct.values())
    print(f"\n=== 整体 per-word acc = {correct/max(1,total):.4f} ({correct}/{total}) ===")
    print(f"(注:仅 CWS pred==gold 的 word 才算,排除了 CWS 错的部分)")


if __name__ == "__main__":
    main()
