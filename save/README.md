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

## 文件

```
releases.py    发布清单:哪个 checkpoint、什么指标、模型卡写什么
cards.py       生成 HF 模型卡
export.py      checkpoint → 发布目录
upload.py      发布目录 → HF
assets/        随模型一起发出去的推理代码(真实文件,不是模板字符串)
  tokenizer.py     字级 tokenizer
  mt_model.py      BERTcForMT:分词 / 词性 / 实体
  csc_model.py     BERTcForCSC:拼写纠错
  example_*.py     三个任务的示例
```

导出产物在 `save/releases/`,不进 git。

## 为什么推理代码是真实文件

旧实现把 `mt_model.py` / `csc_model.py` 当**字符串模板**存在导出脚本里
(200+ 行)。这样的代码在发出去之前从来没被执行过 —— 示例跑不通、API 改了没跟上,
都要等用户来报。

现在 `assets/` 下是真实的 `.py`,`test/test_save.py` 会切到发布目录、
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
