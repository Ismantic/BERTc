"""BERTc 的模型与训练代码。只依赖 torch。

这是一个 package,模块之间用相对 import,不再靠 sys.path.insert ——
`data` / `model` / `optim` 这几个名字太通用,平铺在 sys.path 上会跟别的模块
互相遮蔽(重构时就撞过一次:旧的 NLP_BERT_CRF/data.py 和 src/data.py)。

所以带 __main__ 的三个训练脚本要用 -m 从仓库根目录跑:

    python -m src.pretrain     --train_data ... --output_dir ...
    python -m src.finetune_mt  --ckpt_dir ... --train_data ... --dev_data ...
    python -m src.finetune_csc --ckpt_dir ... --train_data ... --test_data ...

直接 `python src/pretrain.py` 会因为相对 import 失败。
"""
