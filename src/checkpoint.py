"""加载骨干权重。只依赖 torch。

支持两种目录:

  预训练产出   model.pt          {"model": state_dict, "config": {...}, ...}
  HF 发布包    model.safetensors + config.json

后者不走 safetensors 库 —— 那个格式简单到不值得为它加一个依赖:

    8 字节小端 uint64:JSON 头的长度
    JSON 头:{张量名: {dtype, shape, data_offsets:[起, 止]}, "__metadata__": ...}
    其余:裸张量数据,offsets 相对于头之后的位置

这样 --ckpt_dir 可以直接指向从 HF 下下来的目录,不用先转格式。
"""
import json
import struct
from pathlib import Path

import torch

# safetensors 的 dtype 名 → torch dtype
_ST_DTYPE = {
    "BOOL": torch.bool, "U8": torch.uint8, "I8": torch.int8,
    "I16": torch.int16, "U16": torch.uint16,
    "I32": torch.int32, "U32": torch.uint32, "I64": torch.int64,
    "F16": torch.float16, "BF16": torch.bfloat16,
    "F32": torch.float32, "F64": torch.float64,
}


def load_safetensors(path, device=None) -> dict:
    """读 .safetensors → state_dict。纯 torch 实现。

    device 给了就把张量搬过去,省得调用方再遍历一遍(发布包里的推理代码
    直接这么用)。
    """
    path = Path(path)
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        blob = f.read()

    out = {}
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        dtype = _ST_DTYPE.get(spec["dtype"])
        if dtype is None:
            raise ValueError(f"{path} 里未知的 dtype {spec['dtype']}(张量 {name})")
        lo, hi = spec["data_offsets"]
        # 先 bytearray 拷一份再 frombuffer:直接映射 bytes 会得到只读缓冲区上的
        # 张量,torch 会警告"可以写但不该写";而且零拷贝视图会把整个 blob
        # 一直拖在内存里不释放。
        buf = bytearray(blob[lo:hi])
        t = torch.frombuffer(buf, dtype=dtype, count=(hi - lo) // dtype.itemsize)
        t = t.view(*spec["shape"]) if spec["shape"] else t
        out[name] = t.to(device) if device is not None else t
    return out


def load_backbone(ckpt_dir) -> tuple[dict, dict]:
    """返回 (骨干 state_dict, config dict)。

    骨干权重在两种格式里都以 bert.* 为前缀(HF 发布包是从 ModernBertForMLM
    存的,预训练 ckpt 同理),这里统一剥掉前缀返回,调用方直接喂给
    ModernBertModel.load_state_dict。
    """
    ckpt_dir = Path(ckpt_dir)
    cfg_path = ckpt_dir / "config.json"
    if not cfg_path.exists():
        raise SystemExit(f"{ckpt_dir} 下没有 config.json")
    config = json.loads(cfg_path.read_text())

    pt, st = ckpt_dir / "model.pt", ckpt_dir / "model.safetensors"
    if pt.exists():
        ckpt = torch.load(pt, map_location="cpu", weights_only=False)
        # 有 EMA 就优先用 shadow 权重(更稳),否则用原始权重
        state = ckpt.get("ema") or ckpt["model"]
    elif st.exists():
        state = load_safetensors(st)
    else:
        raise SystemExit(f"{ckpt_dir} 下既没有 model.pt 也没有 model.safetensors")

    bert = {k[len("bert."):]: v for k, v in state.items() if k.startswith("bert.")}
    if not bert:
        raise SystemExit(
            f"{ckpt_dir} 里没有 bert.* 前缀的张量 —— 这是骨干吗?"
            f"(微调要从骨干开始,不能从另一个微调结果开始)")
    return bert, config
