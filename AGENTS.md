# 仓库约定

字级中文 Modern BERT 的预训练与微调。详细说明见 [`CLAUDE.md`](CLAUDE.md),
这里只列日常需要的。

## 结构

```
data/       下载 + 加工。source.py 是数据源注册表
src/        模型 + 训练。只依赖 torch,且不碰文本
prepare/    编排:词表、预编码、语料切块、调训练
save/       导出 HF 发布包 + 上传
deps/ docs/ test/
```

**只依赖 Hugging Face 和 GitHub。** 加数据源必须登记公开出处,不要指向本机
既有目录 —— 一旦允许"本地有就跳过下载",别人克隆下来跑不通而自己发现不了。

## 常用命令

```bash
bash prepare/install_deps.sh            # clone + 编译 PieceTokenizer / Wapic
python data/download.py --list          # 数据源状态
huggingface-cli download Ismantic/BERTc-315M --local-dir models/BERTc-315M
make -C prepare finetune   # MT + CSC 微调
python test/test_reproduce_sota.py      # 复现 MT 1.4712 / CSC 0.8346
```

训练脚本用 `-m` 从仓库根目录跑(`src/` 是 package,内部相对 import):

```bash
python -m src.pretrain --train_data ... --output_dir ...
```

## 环境

`/home/tfbao/.venv/bin/python`,**用 `uv pip install`,没有 pip**。
单张 RTX 4090,没有多卡代码路径。

## 代码风格

4 空格缩进,snake_case,CLI 参数走 `argparse`。注释解释**为什么**,不复述
代码在做什么;把踩过的坑写下来(哪种写法会静默出错),这比描述控制流有用。
优先扩展现有脚本,不要新开平行入口。

## 测试

没有单元测试套件,靠基准驱动。改了 `src/` 或 `prepare/` 就跑
`python test/test_reproduce_sota.py` —— 它拿真 checkpoint 复现已记录的数字,
是唯一的硬标准。报告结果时带上 checkpoint、命令、指标。

不要为了让数字好看去改评测代码。

## 提交

短标题,说清结果。commit message **不要带 `Co-Authored-By: Claude ...`**
或任何 AI 署名。一次提交只做一件事。大文件不进 git。

删目录前先 `du -sh` 看看里面有没有 gitignored 的数据 —— `git ls-files`
看不到的东西才是危险的。
