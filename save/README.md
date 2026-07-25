# save/ — 发布层

把 checkpoint 打包成可以直接上传 Hugging Face 的目录,并上传。

```bash
python -m save.export --list                          # 看哪些 checkpoint 就位
python -m save.export                                 # 导出全部六个
python test/test_save.py                             # 验证发布目录能独立跑
python -m save.upload --namespace Ismantic --dry-run  # 先看要传什么
python -m save.upload --namespace Ismantic            # 真传
```

## 六个发布仓库

| 仓库 | 来源 | 指标 |
|---|---|---|
| `BERTc-165M` | `output_v4_mid/checkpoint-8500` | MT 1.4689 / CSC 0.8308 |
| `BERTc-315M` | `output_v4_large/checkpoint-8500` | MT 1.4712 / CSC 0.8346 |
| `BERTc-315M-MT` | `sota_mt_v4large_fgm_5ep_best.pt` | joint 1.4712 |
| `BERTc-165M-MT` | `sota_mt_v4mid_fgm_5ep_best.pt` | joint 1.4689 |
| `BERTc-315M-CSC` | `sota_csc_v4large_v8_best.pt` | 句级 F1 0.8346 |
| `BERTc-165M-CSC` | `sota_csc_v4mid_5ep_best.pt` | 句级 F1 0.8308 |

骨干和微调分开发:骨干加载成 `ModernBertForMLM` 可以继续微调,两个任务的头、
评测口径、推理入口都不一样,混在一个仓库里下载的人会分不清该用哪个。

## 交互式体验

导出发布目录之后,两个脚本可以直接上手玩:

```bash
python -m save.cws                                  # 分词 + 词性 + 实体
python -m save.csc                                  # 拼写纠错
python -m save.cws "中国科学院计算技术研究所在北京"     # 单句
cat corpus.txt | python -m save.cws -q              # 批量,一行一句空格分词
```

`cws.py` 把 LTP 那 27 个词性代号翻成中文(`ns` / `ni` / `nh` 没人记得住),
整句视图里词间用 `·` 分隔、实体整体染色,一眼看出切分和实体的关系:

```
  中国·科学院·计算技术·研究所·在·北京
  分词  中国 / 科学院 / 计算技术 / 研究所 / 在 / 北京
  词性  中国/地名  科学院/名词  计算技术/名词  研究所/名词  在/介词  北京/地名
  实体
        [机构名] 中国科学院计算技术研究所  0-12
        [地名] 北京  13-15
```

`csc.py` 默认除了改动还会报"存疑"——**检测头报警但纠错头没选出别的字**的位置:

```
  我今天很稿兴
  没有改动
  存疑  检测头报警但纠错头没选出别的字,调阈值无效
    第 5 字  稿  检测分 0.982  候选 稿0.22  感0.15  搞0.12
```

这条是必要的。检测和纠错是两个独立的头、推理只用纠错头,所以经常出现
"模型知道这里有错但选不出正确的字"。不显示的话使用者只会看到"没有改动",
然后去反复调阈值 —— 而阈值只能否决改动,不能凭空造出改动,怎么调都没反应。

`-v` 额外显示置信度,`-q` 只输出结果(接管道用),不是 tty 时自动关掉颜色。

## 文件

```
releases.py    发布清单:哪个 checkpoint、什么指标、模型卡写什么
cards.py       生成 HF 模型卡
export.py      checkpoint → 发布目录
upload.py      发布目录 → HF
cws.py         交互式体验:分词 + 词性 + 实体
csc.py         交互式体验:拼写纠错
assets/        随模型一起发出去的推理代码(真实文件,不是模板字符串)
  tokenizer.py     字级 tokenizer
  mt_model.py      BERTcForMT:分词 / 词性 / 实体
  csc_model.py     BERTcForCSC:拼写纠错
  example_*.py     三个任务的示例
```

导出产物在 `save/releases/`,不进 git。

## 为什么推理代码是真实文件

推理代码如果写成导出脚本里的字符串模板,发出去之前就从来没被执行过 ——
示例跑不通、API 改了没跟上,都要等用户来报。

所以 `assets/` 下是真实的 `.py`,`test/test_save.py` 会切到发布目录、
只用目录内的模块跑一遍真实推理,跟外部用户的处境一样。骨干定义 `model.py` 和
`crf.py` 直接从 `src/` 拷,不再维护第二份。

## 验证

`test/test_save.py` 查三件事:

1. **权重忠实性** —— 发布的 safetensors 与源 checkpoint 逐张量比对。
   比跑一次前向彻底:前向只覆盖走到的路径,这里覆盖每一个参数。
2. **自包含性** —— 切到发布目录里 import 并推理,只能看到目录内的模块。
3. **结果合理性** —— 分词能拼回原文、词性数与词数一致、实体类型合法、
   纠错前后等长。

## 已知行为

`BERTcForCSC.correct()` 只用纠错头的 argmax,**不用检测头**。检测头是训练时的
辅助信号,推理不参与 —— 这跟训出 F1 0.8346 的口径一致,别"顺手"改成用 det
过滤,那会改变报告的指标。

一个例子:`我今天很稿兴` 这句,det 给 0.98(知道有错),但 cor 的 top-1 仍是
`稿`(0.224),`高` 排第 4(0.115),所以不会改。降阈值也没用 —— top-1 是原字时
阈值不起作用。
