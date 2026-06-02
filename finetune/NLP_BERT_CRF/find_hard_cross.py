"""Find TRULY unacceptable boundary-cross errors in a model's predictions.

Definition of 'unacceptable':
  - A common short word (length 2, in dict) appears in gold but pred boundary cuts through it,
    AND the resulting pred span(s) glue part of this word with adjacent function/content words.
  - Filter out: cases where pred's alt span is also a common word (= dictionary merge ambiguity).
"""
import argparse
import os
import sys
import torch
from collections import Counter
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import CWSDataset, Collator, bies_to_words  # noqa: E402
from model import BertCRF  # noqa: E402
from strict_f1 import load_dict, words_to_spans  # noqa: E402


FUNCTION_WORDS = set("""
应 在 是 把 的 了 和 与 及 还 也 又 都 既 已 将 要 才 就 让 使 给 由 从 自 由于 因为
不 没 没有 别 莫 未
而 但 且 或 然 因 因此 不过 然而 因为 所以
这 那 些 各 每 某 此
向 对 为 被 比 跟 同 关于 至于 通过 根据 按 依 沿 替 朝 趁
但是 何 怎 哪 多少 几
之 其 而 则 即 乃 凡 故
我 你 他 她 它 您 我们 你们 他们 她们 它们
吗 呢 啊 哦 嘛 啦 哈 吧 哟 喽
也 还 也是 都是
""".split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--name", default="model")
    ap.add_argument("--dev_jsonl", default="./data/cws_dev.jsonl")
    ap.add_argument("--dict", default="/home/tfbao/Shiyu/IsCut/dict.txt")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_show", type=int, default=40)
    args = ap.parse_args()

    print(f"=== {args.name} ===")
    print("Loading dict ...")
    dict_set = load_dict(args.dict)
    print(f"  {len(dict_set)} dict words")
    print(f"  {len(FUNCTION_WORDS)} function words")

    print(f"Loading dev ...")
    dev = CWSDataset(args.dev_jsonl)
    if args.limit:
        dev.items = dev.items[:args.limit]
    print(f"  {len(dev)} samples")

    print(f"Loading {args.name} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    device = torch.device("cuda")
    model = BertCRF(args.model_path, num_tags=4).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    collator = Collator(tokenizer)

    from torch.utils.data import DataLoader
    loader = DataLoader(dev, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collator, num_workers=2)

    bad_cases = []
    cat_counts = Counter()
    idx = 0
    print("Scanning ...\n")
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                preds = model.decode(batch["input_ids"], batch["attention_mask"])
            for pred_tags in preds:
                item = dev.items[idx]
                idx += 1
                n = min(len(item["chars"]), len(pred_tags))
                pred_words = bies_to_words(item["chars"][:n], pred_tags[:n])
                gold_words = bies_to_words(item["chars"][:n], item["tags"][:n])
                gold_spans = words_to_spans(gold_words)
                pred_spans = words_to_spans(pred_words)
                gold_set = {(s, e): w for s, e, w in gold_spans}
                pred_set = {(s, e): w for s, e, w in pred_spans}

                # char position → pred span
                p_at = {}
                for s, e, w in pred_spans:
                    for i in range(s, e):
                        p_at[i] = (s, e, w)

                text = "".join(item["chars"][:n])
                local_bad = []

                # Pattern A: function word + content entity glued
                #   gold has [fn][entity], pred has [fn+ first chars of entity] OR [last chars of prev][fn+entity]
                for s, e, w in gold_spans:
                    if w not in FUNCTION_WORDS:
                        continue
                    # check pred's span at position s
                    ps = p_at.get(s)
                    if not ps:
                        continue
                    ps_s, ps_e, ps_w = ps
                    if ps_w == w:
                        continue  # correctly segmented
                    if len(ps_w) >= 2:
                        # function word got absorbed
                        local_bad.append(("A: 功能词被吞", s, e, w, ps_w))
                        cat_counts["A 功能词被吞"] += 1

                # Pattern B: common 2-char word split by pred where part is glued to adjacent
                for s, e, w in gold_spans:
                    if len(w) != 2:
                        continue
                    if w not in dict_set:
                        continue
                    # both chars covered by same pred span = good
                    p1 = p_at.get(s)
                    p2 = p_at.get(s + 1)
                    if not p1 or not p2:
                        continue
                    if p1 == p2:
                        # gold has [c1 c2] as word, pred has same boundary
                        if (p1[0], p1[1]) == (s, e):
                            continue  # exact match
                        # pred span covers both chars but spans more = pred merged with neighbor
                        # → "拆开"了 gold 这个词
                        merged_w = p1[2]
                        if merged_w in dict_set:
                            continue  # merged into another valid word — ambiguous, skip
                        local_bad.append(("B: 常用词被吞入更大乱串", s, e, w, merged_w))
                        cat_counts["B 常用词被吞入虚假串"] += 1
                    elif (p1[0], p1[1]) != (p2[0], p2[1]):
                        # gold's two chars are in DIFFERENT pred spans → pred cut the word!
                        # one of them is merged with adjacent
                        # check if either pred span is a valid word
                        v1 = p1[2] in dict_set or len(p1[2]) == 1
                        v2 = p2[2] in dict_set or len(p2[2]) == 1
                        if not (v1 and v2):
                            # at least one merged side is invalid → definitely bad
                            local_bad.append(("C: 词被劈+乱黏", s, e, w, f"{p1[2]} | {p2[2]}"))
                            cat_counts["C 词被劈开+乱黏"] += 1

                if local_bad and len(bad_cases) < args.max_show * 3:
                    bad_cases.append({
                        "text": text,
                        "gold": gold_words,
                        "pred": pred_words,
                        "issues": local_bad,
                    })

    print(f"\n=== Summary ===")
    print(f"  Total egregious cases found: {len(bad_cases)}")
    for cat, n in cat_counts.most_common():
        print(f"  {cat}: {n}")

    print(f"\n=== Examples (showing first {min(args.max_show, len(bad_cases))}) ===")
    for ex in bad_cases[:args.max_show]:
        print()
        print(f"  text  : {ex['text']}")
        print(f"  gold  : {' / '.join(ex['gold'])}")
        print(f"  pred  : {' / '.join(ex['pred'])}")
        for cat, s, e, w, info in ex["issues"]:
            print(f"     ❌ {cat}  gold='{w}' → pred='{info}'  (pos {s}-{e})")


if __name__ == "__main__":
    main()
