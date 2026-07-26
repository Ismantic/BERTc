# save/

把 checkpoint 打包成可直接上传 Hugging Face 的目录,并上传。

```bash
python -m save.export --list                          # 哪些 checkpoint 就位
python -m save.export                                 # 导出全部六个
python test/test_save.py                              # 验证发布目录能独立跑
python -m save.upload --namespace Ismantic --dry-run  # 先看要传什么
```

## 六个仓库

| 仓库 | 内容 | 指标 |
|---|---|---|
| `BERTc-165M` / `BERTc-315M` | Backbone + MLM 头 | — |
| `BERTc-165M-MT` / `BERTc-315M-MT` | CWS + POS + NER | joint 1.4689 / 1.4712 |
| `BERTc-165M-CSC` / `BERTc-315M-CSC` | Correction | 句级 F1 0.8333 / 0.8388 |

Backbone 和微调分开发布。两个任务的头、评测口径、推理入口都不同,合并成一个
仓库会让使用者难以判断该用哪个。

## 权重从哪来

`save/` 消费两类权重,都不在 git 里:

| | 位置 | 生产者 | 拿现成的 |
|---|---|---|---|
| 骨干 | `models/<名字>/` | `make -C prepare pretrain` 的产物拷过来 | `huggingface-cli download Ismantic/BERTc-315M --local-dir models/BERTc-315M` |
| 微调 | `save/sota/*.pt` | `make -C prepare finetune` | 见下 |

`save/sota/*.pt` 是训练产物,重训能到同等水平但不逐位相同。需要**已发布的
那一份**权重时,直接从 HF 下发布包:它与 `.pt` 里的张量逐个相同,只差
CSC 的 `cor_head.weight`(与词嵌入绑权重,导出时去重)。

`test/test_reproduce_sota.py` 接受这两种来源,先找 `save/sota/*.pt`,没有则用
`save/releases/<名字>/model.safetensors`。全新 clone 下载发布包即可跑回归,
无需先训练。

`save/sota/README.md` 另记录了几个消融用的 checkpoint(v6 / v6.5 / MacBERT
对照组),只用于那份消融表,不参与发布。

## 推理代码是真实文件

`save/assets/` 下是真实的 `.py`,而非导出脚本里的字符串模板。模板在发布前
不会被执行,示例失效或 API 变更都只能等使用者反馈。骨干定义 `model.py`、
`crf.py` 和 safetensors 读取 `checkpoint.py` 直接从 `src/` 拷贝,不维护第二份。

发布包只依赖 torch 和 PieceTokenizer。safetensors 的读取用 `checkpoint.py`
(纯 torch,85 行),不需要 safetensors 库。

`test/test_save.py` 查三件事:

1. **权重忠实性** —— 发布的 safetensors 与源 checkpoint 逐张量比对。前向
   只覆盖执行到的路径,逐张量比对覆盖每一个参数
2. **自包含性** —— 切到发布目录内 import 并推理,只能看到目录内的模块
3. **结果合理性** —— 分词能拼回原文、词性数与词数一致、实体类型合法、
   纠错前后等长

第 2 条针对的情况是:开发环境的 `sys.path` 上有全部模块,发布目录少一个文件
在本机不会暴露。

## 交互式脚本

```bash
python -m save.cws                                  # CWS + POS + NER
python -m save.csc                                  # Correction
python -m save.cws "中国科学院计算技术研究所在北京"      # 单句
cat corpus.txt | python -m save.cws -q              # 批量,一行一句空格分词
```

`cws.py` 把 LTP 那 27 个词性代号翻成中文,整句视图里词间用 `·` 分隔、
实体整体染色:

```
  中国·科学院·计算技术·研究所·在·北京
  分词  中国 / 科学院 / 计算技术 / 研究所 / 在 / 北京
  词性  中国/地名  科学院/名词  计算技术/名词  研究所/名词  在/介词  北京/地名
  实体
        [机构名] 中国科学院计算技术研究所  0-12
        [地名] 北京  13-15
```

`csc.py` 除了改动还会报「存疑」—— 检测头报警但纠错头没选出别的字的位置:

```
  我今天很稿兴
  没有改动
  存疑  检测头报警但纠错头没选出别的字,调阈值无效
    第 5 字  稿  检测分 0.982  候选 稿0.22  感0.15  搞0.12
```

推理只用纠错头,因此会出现「检测到错误但选不出正确的字」。不显示这一项,
使用者只看到「没有改动」,容易转而调阈值,而**阈值只能否决改动,不能产生
改动**。背景见 [`docs/WHY.md`](../docs/WHY.md#correct-不使用检测头)。

`-v` 显示置信度,`-q` 只输出结果(接管道用),不是 tty 时自动关掉颜色。

## 文件

```
releases.py    发布清单:哪个 checkpoint、什么指标、模型卡写什么
cards.py       生成 HF 模型卡
export.py      checkpoint → 发布目录
upload.py      发布目录 → HF
cws.py         交互式:CWS + POS + NER
csc.py         交互式:Correction
assets/        随模型发出去的推理代码(真实文件)
  tokenizer.py     字级 tokenizer
  mt_model.py      BERTcForMT
  csc_model.py     BERTcForCSC
  example_*.py     三个任务的示例
sota/          SOTA checkpoint 与消融记录(进 git)
```

导出产物在 `save/releases/`,不进 git。
