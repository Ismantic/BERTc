"""下载 Ultra-FineWeb-L3 多样性 mix:4 子集各取前 N parts,凑 20B tokens。
中:英 = 2.2:1。下完到 ~/Shiyu/data/L3-mix。"""
import os, sys
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
for k in list(os.environ):
    if 'proxy' in k.lower(): del os.environ[k]
from huggingface_hub import HfApi, hf_hub_download
from concurrent.futures import ThreadPoolExecutor, as_completed

LOCAL = "/home/tfbao/Shiyu/data/L3-mix"
os.makedirs(LOCAL, exist_ok=True)
api = HfApi(endpoint='https://hf-mirror.com')

# 列 4 子集每个,排序后取前 N
PLAN = {
    'data/ultrafineweb_zh_l3/qa': 20,
    'data/ultrafineweb_zh_l3/multi_style': 20,
    'data/ultrafineweb_en_l3/qa': 10,
    'data/ultrafineweb_en_l3/multi_style': 10,
}

print("Listing repo files...", flush=True)
info = api.dataset_info(repo_id='openbmb/Ultra-FineWeb-L3', files_metadata=False)
all_files = sorted(s.rfilename for s in info.siblings if s.rfilename.endswith('.parquet'))

to_download = []
for prefix, n in PLAN.items():
    sub = sorted(f for f in all_files if f.startswith(prefix + '/'))[:n]
    print(f"  {prefix}: {len(sub)} parts selected")
    to_download.extend(sub)
print(f"Total: {len(to_download)} parts")

def dl(path):
    try:
        hf_hub_download(
            repo_id='openbmb/Ultra-FineWeb-L3', repo_type='dataset',
            filename=path, local_dir=LOCAL,
            endpoint='https://hf-mirror.com',
        )
        return (path, True, None)
    except Exception as e:
        return (path, False, str(e))

done = 0
with ThreadPoolExecutor(max_workers=16) as pool:
    futs = [pool.submit(dl, p) for p in to_download]
    for fut in as_completed(futs):
        path, ok, err = fut.result()
        done += 1
        msg = "OK" if ok else f"FAIL {err}"
        print(f"  [{done}/{len(to_download)}] {os.path.basename(path)}: {msg}", flush=True)

print("All done.")
