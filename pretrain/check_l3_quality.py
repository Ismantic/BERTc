"""检查 L3-mix 各 config 一个样本 parquet,看:
- 字段结构 / 平均 doc 长度 / 中英比例 / 噪声(URL/code/特殊字符)
- 抽 5 个 doc 看人类感受
"""
import os, glob, json
import pyarrow.parquet as pq
import random

ROOT = "/home/tfbao/Shiyu/data/L3-mix"
configs = [
    "data/ultrafineweb_zh_l3/qa",
    "data/ultrafineweb_zh_l3/multi_style",
    "data/ultrafineweb_en_l3/qa",
    "data/ultrafineweb_en_l3/multi_style",
]

def stats(text):
    n = len(text)
    n_zh = sum(1 for c in text if '一' <= c <= '鿿')
    n_en = sum(1 for c in text if c.isascii() and c.isalpha())
    return n, n_zh, n_en

for cfg in configs:
    full = os.path.join(ROOT, cfg)
    if not os.path.exists(full):
        print(f"\n=== {cfg}:目录不存在,跳过 ===")
        continue
    parts = sorted(glob.glob(os.path.join(full, "*.parquet")))
    if not parts:
        print(f"\n=== {cfg}:无 parquet 文件(可能还在下载)===")
        continue
    print(f"\n=== {cfg} ({len(parts)} parts 已下) ===")
    # 读第一个 parquet 看 schema + 抽样
    pf = pq.ParquetFile(parts[0])
    schema = pf.schema_arrow
    print(f"  schema: {schema}")
    print(f"  num_rows: {pf.metadata.num_rows:,}")
    print(f"  file size: {os.path.getsize(parts[0])/1e6:.1f} MB")
    # 抽样
    tbl = pf.read_row_group(0)
    df = tbl.to_pylist()[:200]
    # 找 text 字段
    text_key = None
    for k in ['text', 'content', 'document', 'data']:
        if k in df[0]: text_key = k; break
    if text_key is None:
        # 列名第一个 string 字段
        text_key = [k for k in df[0] if isinstance(df[0][k], str)][0]
    print(f"  text field: '{text_key}'")
    # 统计
    lens = [len(d[text_key]) for d in df if d.get(text_key)]
    zh_ratios = []
    for d in df[:50]:
        t = d.get(text_key, "")
        if len(t) < 10: continue
        n, nz, ne = stats(t)
        zh_ratios.append(nz/max(1,n))
    print(f"  avg doc len (chars): {sum(lens)/max(1,len(lens)):.0f}")
    print(f"  median doc len: {sorted(lens)[len(lens)//2] if lens else 0}")
    print(f"  max doc len: {max(lens) if lens else 0}")
    print(f"  avg zh ratio: {sum(zh_ratios)/max(1,len(zh_ratios)):.2%}")
    # 抽 2 个看
    print("  ---- 样本 ----")
    sample = random.Random(42).sample(df, min(2, len(df)))
    for i, d in enumerate(sample):
        t = d.get(text_key, "")[:300]
        print(f"  [{i}] (len={len(d.get(text_key,''))}) {t[:200]}...")
