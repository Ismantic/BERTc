"""用 LTP/base1 在 PeopleDaily.sentences 上一次出 CWS+POS+NER 三任务弱监督数据。

输出 jsonl 每行:
  {text, words, pos, entities: [{type, start, end, text}], source}
  - pos 已是 LTP scheme(27 tag,直接是 LTP 输出)
  - entities type ∈ {Nh, Ns, Ni},char-level start/end (半开区间)
"""
import argparse, json, time, os, random


def filter_sentence(s):
    s = s.strip()
    if not s: return None
    if len(s) < 8 or len(s) > 200: return None
    n_chinese = sum(1 for c in s if '一' <= c <= '鿿')
    if n_chinese < len(s) * 0.5: return None
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n", type=int, default=1000000)
    ap.add_argument("--model", default="LTP/base1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shuffle", action="store_true")
    args = ap.parse_args()

    from ltp import LTP
    print(f"Loading LTP {args.model}")
    ltp = LTP(args.model)
    if args.device == "cuda":
        ltp.to("cuda")
    print(f"  device: {args.device}")

    print(f"Reading from {args.src}...")
    sentences = []
    with open(args.src, encoding="utf8") as f:
        if args.shuffle:
            random.seed(args.seed)
            for i, line in enumerate(f):
                s = filter_sentence(line)
                if not s: continue
                if len(sentences) < args.n:
                    sentences.append(s)
                else:
                    j = random.randint(0, i)
                    if j < args.n:
                        sentences[j] = s
                if i % 1000000 == 0 and i > 0:
                    print(f"  scanned {i:,} lines, kept {len(sentences):,}", flush=True)
        else:
            for line in f:
                s = filter_sentence(line)
                if s:
                    sentences.append(s)
                    if len(sentences) >= args.n:
                        break
    print(f"Collected {len(sentences):,} sentences", flush=True)

    print(f"LTP pipeline cws+pos+ner batch_size={args.batch_size}...")
    t0 = time.time()
    n_written = 0
    n_skip = 0
    with open(args.output, "w", encoding="utf8") as fout:
        for i in range(0, len(sentences), args.batch_size):
            batch = sentences[i:i + args.batch_size]
            try:
                out = ltp.pipeline(batch, tasks=["cws", "pos", "ner"])
                cws_list, pos_list, ner_list = out.cws, out.pos, out.ner
            except Exception:
                # 单条重试
                cws_list, pos_list, ner_list = [], [], []
                for s in batch:
                    try:
                        o = ltp.pipeline([s], tasks=["cws", "pos", "ner"])
                        cws_list.append(o.cws[0])
                        pos_list.append(o.pos[0])
                        ner_list.append(o.ner[0])
                    except Exception:
                        cws_list.append([s]); pos_list.append(['x']); ner_list.append([])
            for sent, words, pos, ents_w in zip(batch, cws_list, pos_list, ner_list):
                if not words or "".join(words) != sent:
                    n_skip += 1
                    continue
                if len(pos) != len(words):
                    n_skip += 1
                    continue
                # NER: ents_w 每项 (type, text, word_start, word_end) — word idx 闭区间
                # 转 char-level [start, end) 半开
                word_offsets = []
                pos_c = 0
                for w in words:
                    word_offsets.append(pos_c)
                    pos_c += len(w)
                word_offsets.append(pos_c)  # sentinel
                entities = []
                for e in ents_w:
                    ent_type, ent_text, ws, we = e[0], e[1], e[2], e[3]
                    if ws >= len(words) or we >= len(words): continue
                    cs = word_offsets[ws]
                    ce = word_offsets[we + 1]
                    if "".join(words[ws:we+1]) != sent[cs:ce]:
                        continue
                    entities.append({"type": ent_type, "start": cs, "end": ce, "text": sent[cs:ce]})
                rec = {
                    "text": sent,
                    "words": list(words),
                    "pos": list(pos),
                    "entities": entities,
                    "source": f"ltp-{args.model.replace('/', '-')}-distill-mt",
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_written += 1
            if (i // args.batch_size) % 20 == 0:
                el = time.time() - t0
                done = i + len(batch)
                rate = done / max(1, el)
                eta = (len(sentences) - done) / rate / 60
                print(f"  {done:,}/{len(sentences):,} ({100*done/len(sentences):.1f}%) "
                      f"{el:.0f}s ({rate:.0f}/s) ETA {eta:.0f}m", flush=True)
                fout.flush()
    print(f"\nDone in {(time.time()-t0)/60:.1f}m")
    print(f"  wrote {n_written:,} samples,skipped {n_skip:,} → {args.output}")


if __name__ == "__main__":
    main()
