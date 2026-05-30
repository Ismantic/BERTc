"""用 LTP/base1 在 PeopleDaily.sentences.txt 上标 CWS 弱监督数据,输出 jsonl
(跟 cws.pd98.jsonl 同 schema)用于 Stage 1 pre-fine-tune。

用法:
  python distill_with_ltp.py \
      --src /home/tfbao/Shiyu/Data/data/PeopleDaily.sentences.txt \
      --output ./data/cws.ltp_distill.jsonl \
      --n 1000000 \
      --model LTP/base1 \
      --device cuda \
      --batch_size 64
"""
import argparse, json, time, os, random


def filter_sentence(s):
    """过滤太短 / 太长 / 含异常字符的 sentence。"""
    s = s.strip()
    if not s: return None
    if len(s) < 8 or len(s) > 200: return None  # 过短无用,过长 LTP 慢
    # 跳过含大量 ASCII / 数字的(网址 / 表格残留)
    n_chinese = sum(1 for c in s if '一' <= c <= '鿿')
    if n_chinese < len(s) * 0.5: return None  # 中文比例 >= 50%
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="PeopleDaily.sentences.txt path")
    ap.add_argument("--output", required=True)
    ap.add_argument("--n", type=int, default=1000000, help="目标抽取 N 个句子")
    ap.add_argument("--model", default="LTP/base1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shuffle", action="store_true",
                    help="先 shuffle file 行再抽 N(否则取前 N)")
    args = ap.parse_args()

    from ltp import LTP
    print(f"Loading LTP {args.model}")
    ltp = LTP(args.model)
    if args.device == "cuda":
        ltp.to("cuda")
    print(f"  device: {args.device}")

    # 抽 sentences
    print(f"Reading from {args.src}...")
    sentences = []
    with open(args.src, encoding="utf8") as f:
        if args.shuffle:
            # reservoir sampling N 个
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
                    print(f"  scanned {i:,} lines, kept {len(sentences):,}")
        else:
            for line in f:
                s = filter_sentence(line)
                if s:
                    sentences.append(s)
                    if len(sentences) >= args.n:
                        break
    print(f"Collected {len(sentences):,} sentences")

    # batch inference
    print(f"LTP inference batch_size={args.batch_size}...")
    t0 = time.time()
    n_written = 0
    with open(args.output, "w", encoding="utf8") as fout:
        for i in range(0, len(sentences), args.batch_size):
            batch = sentences[i:i + args.batch_size]
            try:
                out = ltp.pipeline(batch, tasks=["cws"])
                preds = out.cws
            except Exception:
                # 单条重试(批失败回退)
                preds = []
                for s in batch:
                    try:
                        preds.append(ltp.pipeline([s], tasks=["cws"]).cws[0])
                    except Exception:
                        preds.append([s])
            for sent, words in zip(batch, preds):
                if not words: continue
                # gold must match raw sentence(LTP 不应该改字)
                joined = "".join(words)
                if joined != sent:
                    continue  # skip mismatched
                rec = {
                    "task": "cws",
                    "lang": "zh",
                    "messages": [
                        {"role": "user",
                         "content": f"任务: 中文分词\n请将下面这句中文分词,词与词之间用单个空格隔开,原文不增不减:\n原文: {sent}"},
                        {"role": "assistant", "content": " ".join(words)},
                    ],
                    "gold": words,
                    "source": f"ltp-{args.model.replace('/', '-')}-distill",
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_written += 1
            if i % (args.batch_size * 20) == 0:
                el = time.time() - t0
                done = i + len(batch)
                rate = done / max(1, el)
                eta = (len(sentences) - done) / rate / 60
                print(f"  {done:,}/{len(sentences):,} ({100*done/len(sentences):.1f}%) "
                      f"{el:.0f}s ({rate:.0f}/s) ETA {eta:.0f}m", flush=True)
    print(f"\nDone in {time.time()-t0:.0f}s")
    print(f"  wrote {n_written:,} samples → {args.output}")


if __name__ == "__main__":
    main()
