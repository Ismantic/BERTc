"""发布清单:每个 HF 仓库对应哪个 checkpoint、什么指标、模型卡怎么写。

数字改了要同步 finetune/sota/README.md —— 那份是 SOTA 的源头,这里只是抄过去
写进模型卡。
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 骨干:预训练产出,可继续微调
BACKBONES = {
    "BERTc-165M": {
        "checkpoint": ROOT / "pretrain" / "modern_bertc" / "output_v4_mid" / "checkpoint-8500",
        "params": "165M",
        "arch": "12L / 1024H / 2752I / 16 heads",
        "mt": "score 1.4689(CWS 0.9836 / POS 0.9753 / NER 0.9632)",
        "csc": "SIGHAN-15 句级 F1 0.8308",
        "notes": "首个在 165M 规模同时拿到 MT / CSC SOTA 的 Modern BERTc 骨干。",
    },
    "BERTc-315M": {
        "checkpoint": ROOT / "pretrain" / "modern_bertc" / "output_v4_large" / "checkpoint-8500",
        "params": "315M",
        "arch": "24L / 1024H / 2752I / 16 heads",
        "mt": "score 1.4712(CWS 0.9840 / POS 0.9800 / NER 0.9660)",
        "csc": "SIGHAN-15 句级 F1 0.8346",
        "notes": "当前最强的 Modern BERTc 骨干,用完整 17.65B token 语料训练。",
    },
}

# 下游微调:任务头 + 推理代码
FINETUNES = {
    "BERTc-315M-MT": {
        "task": "mt",
        "base": "BERTc-315M",
        "backbone": BACKBONES["BERTc-315M"]["checkpoint"],
        "checkpoint": ROOT / "finetune" / "sota" / "sota_mt_v4large_fgm_5ep_best.pt",
        "metrics": {"joint score": "1.4712", "CWS F1": "0.9840",
                    "POS 准确率": "0.9800", "NER F1": "0.9660"},
        "recipe": "5 epoch,batch 64,骨干 lr 2e-5 / 头 lr 5e-4,warmup 0.1,"
                  "α_pos 2.0,β_ner 0.5,FGM ε 1.0",
        "data": "PD-1998(人民日报 1998 年 1-6 月 PFR 标注),"
                "train 102,739 句 / dev 21,143 句",
    },
    "BERTc-165M-MT": {
        "task": "mt",
        "base": "BERTc-165M",
        "backbone": BACKBONES["BERTc-165M"]["checkpoint"],
        "checkpoint": ROOT / "finetune" / "sota" / "sota_mt_v4mid_fgm_5ep_best.pt",
        "metrics": {"joint score": "1.4689", "CWS F1": "0.9836",
                    "POS 准确率": "0.9753", "NER F1": "0.9632"},
        "recipe": "同 315M-MT",
        "data": "同 315M-MT",
    },
    "BERTc-315M-CSC": {
        "task": "csc",
        "base": "BERTc-315M",
        "backbone": BACKBONES["BERTc-315M"]["checkpoint"],
        "checkpoint": ROOT / "finetune" / "sota" / "sota_csc_v4large_v8_best.pt",
        "metrics": {"句级 F1": "0.8346", "准确率": "0.8430",
                    "精确率": "0.9396", "召回率": "0.7507"},
        "recipe": "10 epoch,batch 32,lr 3e-5,warmup 0.1,"
                  "det_weight 0.3,纠错阈值 0.7,max_len 128",
        "data": "SIGHAN + Wang271K + MCSCSet + CSCD-NS 等,去重后 826,097 对",
    },
    "BERTc-165M-CSC": {
        "task": "csc",
        "base": "BERTc-165M",
        "backbone": BACKBONES["BERTc-165M"]["checkpoint"],
        "checkpoint": ROOT / "finetune" / "sota" / "sota_csc_v4mid_5ep_best.pt",
        "metrics": {"句级 F1": "0.8308"},
        "recipe": "5 epoch,batch 64,lr 5e-5,其余同 315M-CSC",
        "data": "同 315M-CSC",
    },
}

ALL = {**BACKBONES, **FINETUNES}
