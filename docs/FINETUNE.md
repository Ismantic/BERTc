# 微调 BERTc

从 Hugging Face 上的骨干开始,在自己的任务上微调。**不需要预训练** —— 那要
单张 4090 跑 3 到 5 天,而微调只要几小时。

全程只依赖 Hugging Face 和 GitHub。想从零预训练见
[`PRETRAIN.md`](PRETRAIN.md)。

- [装依赖](#装依赖)
- [拿骨干](#拿骨干)
- [分词 + 词性 + 实体](#分词--词性--实体)
- [拼写纠错](#拼写纠错)
- [换成自己的数据](#换成自己的数据)
- [导出与发布](#导出与发布)
- [常见问题](#常见问题)

## 装依赖

BERTc 有两个 C++ 依赖:PieceTokenizer(字级分词器,同时提供词表)和 Wapic
(CRF 分词器,只有预训练做整词掩码时才用到)。一条命令搞定 —— 仓库不在本地
就从 GitHub clone,然后编译安装:

```bash
bash prepare/install_deps.sh
```

需要 `cmake`、C++17 编译器、`git`。仓库落在 `deps/`,想换位置设
`BERTC_DEPS_DIR`。

装完会自动比对词表行为,输出 `校验通过` 才算成功。这一步不能跳:**编码行为
一旦变了,词表就和已发布模型的 embedding 对不上,而代码不会报错**,只会
悄悄训出垃圾结果。

只做微调的话,Wapic 可以不装:

```bash
bash prepare/install_deps.sh piece
```

## 拿骨干

```bash
huggingface-cli download Ismantic/BERTc-315M --local-dir models/BERTc-315M
```

下下来直接当 `--ckpt_dir` 用,**不需要任何格式转换**。微调脚本认两种目录:
预训练产出的 `model.pt`,和 HF 发布包的 `model.safetensors`。

> 顺带一提,`src/` 读 safetensors 没有用 `safetensors` 库 —— 那个格式简单到
> 不值得为它加依赖(8 字节头长度 + JSON 头 + 裸张量数据),`src/checkpoint.py`
> 里三十行纯 torch 就够了。`src/` 至今只依赖 torch。

两个规格:

| | 参数 | 层数 | 显存(batch 64 微调) |
|---|---|---|---|
| `BERTc-315M` | 315M | 24L / 1024H | 约 22 GB |
| `BERTc-165M` | 165M | 12L / 1024H | 约 12 GB |

24GB 卡跑 315M 的 MT 微调会比较满,显存不够就调小 `--batch_size`,或者用
165M。

## 分词 + 词性 + 实体

三个任务联合微调,共用骨干,各挂一个头。

### 准备数据

```bash
python data/download.py pd1998        # 从 GitHub 拉 PD-1998 标注语料
python data/process_cws.py            # PFR 格式 → cws/pos/ner jsonl
python -m prepare.build_mt            # jsonl → 预编码数据集
```

产出:

```
prepare/datasets/mt_train.pt   102,739 句
prepare/datasets/mt_dev.pt      21,143 句
```

PD-1998 是人民日报 1998 年 1–6 月的 PFR 标注语料。前 5 个月做训练,199806
做 dev。压缩包名字叫 `199801` 但里面其实是六个月,不要被骗。

### 训练

```bash
python -m src.finetune_mt \
    --ckpt_dir models/BERTc-315M \
    --train_data prepare/datasets/mt_train.pt \
    --dev_data prepare/datasets/mt_dev.pt \
    --output_dir output/mt \
    --epochs 5 --batch_size 64 \
    --bert_lr 2e-5 --head_lr 5e-4 \
    --alpha_pos 2.0 --beta_ner 0.5 \
    --fgm --fgm_eps 1.0 \
    --dev_limit 2000
```

单张 4090 约 **135 分钟**。每个 epoch 结束评一次 dev,score 变好就存
`best.pt`。

参数里值得说的:

| 参数 | 为什么 |
|---|---|
| `--alpha_pos 2.0` | 词性 loss 的量级天然比分词小,不加权只占 1.6% 的梯度,学不动。加权后词性 +0.02,追平 MacBERT |
| `--beta_ner 0.5` | 反过来压实体,防止它抢容量 |
| `--fgm --fgm_eps 1.0` | 对抗训练:给词嵌入加受 L2 约束的扰动再算一次梯度。分词和实体各 +0.005~0.013 |
| `--bert_lr` ≪ `--head_lr` | 骨干已经训好,大 lr 会冲坏;头是随机初始化的,要大 lr |
| `--dev_limit 2000` | 只评前 2000 句。全量 21,143 句慢 10 倍,而且报告的指标就是这个口径 |

预期结果:

```
分词 F1 0.9840 / 词性 0.9800 / 实体 F1 0.9660 / joint 1.4712
```

### 推理

```python
import sys; sys.path.insert(0, "src")
from finetune_mt import ModernBertMT
```

或者用发布包里的自包含实现(见[导出与发布](#导出与发布)):

```python
from mt_model import BERTcForMT

model = BERTcForMT.from_pretrained(".")
print(model.predict("中国科学院计算技术研究所在北京"))
# {'words': ['中国', '科学院', '计算技术', '研究所', '在', '北京'],
#  'pos': ['ns', 'n', 'n', 'n', 'p', 'ns'],
#  'ner': [{'type': 'Ni', 'start': 0, 'end': 12},
#          {'type': 'Ns', 'start': 13, 'end': 15}]}
```

标签体系:分词是 BIES;词性是 LTP base1 的 27 个标签(PD-1998 的 43 个映射
过来);实体是 BIES × {人名 Nh / 地名 Ns / 机构 Ni},PD 的 MISC 丢弃。

## 拼写纠错

双头:cor 逐位置预测正确的字,det 判断该位置有没有错。**只做等长替换**,
不处理多字少字。

### 准备数据

```bash
python data/download.py --csc         # 5 个公开源
python data/process_csc.py            # 合并去重 → 句对 pkl
python -m prepare.build_csc           # → 预编码数据集
```

产出:

```
prepare/datasets/csc_train.pt   826,200 对
prepare/datasets/csc_test.pt        707 条(SIGHAN-15 官方)
```

数据来自 5 个源:`zejunwang1/CTCDataset`、`yzhihao/MCSCSet`、
`shibing624/CSC`、`shibing624/chinese_text_correction`,测试集来自
`shibing624/pycorrector`。合并规则见 `data/process_csc.py` —— 只保留
**等长**的句对(狭义 CSC),语法纠错类的源整体排除。

### 训练

```bash
python -m src.finetune_csc \
    --ckpt_dir models/BERTc-315M \
    --train_data prepare/datasets/csc_train.pt \
    --test_data prepare/datasets/csc_test.pt \
    --output_dir output/csc \
    --epochs 10 --batch_size 32 --lr 3e-5 \
    --warmup_ratio 0.1 --det_weight 0.3 --threshold 0.7
```

单张 4090 约 **264 分钟**。

| 参数 | 为什么 |
|---|---|
| `--epochs 10` | Large 模型 5 epoch 严重欠训,10 epoch 才到位。这是调 315M 时最大的发现 |
| `--det_weight 0.3` | 检测头用 focal loss。错字在句子里是极少数,普通 BCE 会被海量负样本淹没 |
| `--threshold 0.7` | 纠错置信度低于它就保留原字。低置信度的"纠正"绝大多数是误伤,这个阈值是精确率的主要来源 |

预期结果:

```
F1 0.8346  P 0.9396  R 0.7507  (TP=280 FP=18 FN=93 TN=316)
```

### 推理

```python
from csc_model import BERTcForCSC

model = BERTcForCSC.from_pretrained(".")
print(model.correct("他平时喜欢锻练身体"))   # 他平时喜欢锻炼身体
```

`threshold` 可以在调用时调:调低提召回,调高提精确率。

> **注意**:`correct()` 只用纠错头的 argmax,**不用检测头**。检测头是训练时的
> 辅助信号,推理不参与 —— 这跟训出 0.8346 的口径一致,别"顺手"改成用 det
> 过滤,那会改变报告的指标。

## 换成自己的数据

`src/` 读的是预编码好的 id,格式定义在 `src/data.py` 的模块文档里。最省事的
做法是仿照 `prepare/build_mt.py` / `build_csc.py` 写一个自己的 builder:

```python
from prepare.pack import pack, save
from prepare.tokenizer import load_tokenizer

tok = load_tokenizer()
items = [{"input_ids": tok.encode(text), "cor_labels": ..., "det_labels": ...}
         for text in my_data]
save(pack(items, ("input_ids", "cor_labels", "det_labels"),
          {"format": "bertc-csc-v1", "pad_token_id": tok.pad_token_id,
           "vocab_size": tok.vocab_size, "id_to_char": tok.id_to_char()}),
     "my_train.pt")
```

三条容易踩的:

- **不要在这一步截断**。`max_chars`(MT 254)和 `max_len`(CSC 128)是训练
  超参,在 `src/data.py` 里截,换长度不用重跑预编码。
- **CSC 的 `det_labels` 要按字比对,不能按 id**。字级 tokenizer 是多对一的,
  两个不同的字可能落到同一个 id,按 id 比会漏掉那处错误。实测 82 万对里
  确实存在这种情况。
- **CSC 的测试集要额外存原文**(`src_texts` / `tgt_texts`)。评测靠字符串比对,
  而 id→字 的往返是有损的 —— SIGHAN-15 的 707 条里就有 16 条还原不回去,
  拿还原文本当参照会让 F1 虚高 0.006。

## 导出与发布

把训练结果打包成可以直接传 HF 的目录:

```bash
python -m save.export --list          # 看哪些 checkpoint 就位
python -m save.export BERTc-315M-MT
python test/test_save.py              # 验证发布目录能独立跑
python -m save.upload --namespace <你的账号> --dry-run
```

发布目录是自包含的:骨干定义、CRF、tokenizer 封装、推理入口、示例都是**真实
文件**(从 `src/` 和 `save/assets/` 拷过去),不是模板字符串。`test/test_save.py`
会切进目录、只用目录内的模块跑一遍真实推理,跟外部用户的处境一样。

要发布自己的 checkpoint,在 `save/releases.py` 里加一条,指向你的 `.pt` 和
骨干目录即可。

## 常见问题

**`TypeError: load(): incompatible function arguments ... cn_dict`**

PieceTokenizer 在 2026-07 把 `load(model, cn_dict=)` 改成了 `load(model, dict=)`。
更新到新版即可。语义没变,编码结果完全一致。

**`AttributeError: 'Segmenter' object has no attribute 'cut_smart'`**

同期 Wapic 的 API 也重设计了:`cut_smart` → `segment`,`cut` → `segment_raw`,
新增 `word_starts`。重跑 `bash prepare/install_deps.sh wapic`。

**`CUDA out of memory`**

315M + batch 64 在 24GB 卡上很满。调小 `--batch_size`,或换 165M 骨干。

**找不到词表**

词表来自 PieceTokenizer 仓库的 `save/BERTc-Tokenizer.pt`,通过已安装的
`piece_tokenizer` 模块位置反查。装的是非 editable 版本时找不到,用
`BERTC_PIECE_MODEL` 指定路径,或重跑 `bash prepare/install_deps.sh piece`。

**下载失败**

Hugging Face 走 `hf-mirror.com` 时**必须清代理**,GitHub 反过来**需要代理**。
`data/download.py` 和 `install_deps.sh` 已经处理了这个矛盾;手工下载时注意。
换官方源:`HF_ENDPOINT= python data/download.py ...`。

**微调结果对不上文档里的数字**

先跑 `python test/test_reproduce_sota.py` —— 它拿已发布的 SOTA checkpoint
复现 MT 1.4712 和 CSC 0.8346。这条过了说明评测链路没问题,再查训练配方。
注意 MT 的指标是在 dev **前 2000 句**上测的。
