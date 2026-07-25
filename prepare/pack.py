"""预编码数据的打包格式。prepare/ 写,src/data.py 读。

变长序列用「扁平数组 + offsets」存:N 条样本只存 N+1 个 offset,
比存 N 个小 tensor 省内存也快。格式细节见 src/data.py 的模块文档。
"""
from pathlib import Path

import torch

# det_labels 是 0/1,用 uint8;其余是 id 或标签,int32 够(-100 也放得下)
_DTYPE = {"det_labels": torch.uint8}


def pack(items: list[dict], fields: tuple[str, ...], extra: dict) -> dict:
    """items[i][f] 是等长的 int 列表 → 扁平 blob。"""
    offsets = [0]
    cols = {f: [] for f in fields}
    for it in items:
        n = len(it[fields[0]])
        for f in fields:
            col = it[f]
            if len(col) != n:
                raise ValueError(f"同一条样本里 {f} 长度 {len(col)} != {fields[0]} 的 {n}")
            cols[f].extend(col)
        offsets.append(offsets[-1] + n)

    blob = {"offsets": torch.tensor(offsets, dtype=torch.int64)}
    for f in fields:
        blob[f] = torch.tensor(cols[f], dtype=_DTYPE.get(f, torch.int32))
    blob.update(extra)
    return blob


def save(blob: dict, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, path)
    n = blob["offsets"].numel() - 1
    total = int(blob["offsets"][-1])
    size_mb = path.stat().st_size / 1e6
    print(f"  写入 {path}  {n:,} 条 / {total:,} token / {size_mb:.1f} MB")
    return path
