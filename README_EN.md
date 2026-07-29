# BERTc

English | [中文](README.md)

A character-level Modern BERT for Chinese, pretrained from scratch and
implemented in pure PyTorch.

BERTc covers the complete pipeline from raw corpora to Hugging Face releases:
data acquisition and processing, vocabulary construction, pre-encoding,
pretraining, fine-tuning, and export. The model and training code in `src/`
depends only on PyTorch. CRFs, the optimizer, LR scheduling, whole-word
masking, and safetensors loading are implemented locally to keep the entire
training stack readable.

Six checkpoints are available on Hugging Face: two model sizes, each released
as a pretrained backbone and with two task-specific variants.

```python
from mt_model import BERTcForMT

BERTcForMT.from_pretrained(".").predict(
    "中国科学院计算技术研究所在北京"
)
# words: 中国 / 科学院 / 计算技术 / 研究所 / 在 / 北京
# pos:   ns  n  n  n  p  ns
# ner:   [organization] 中国科学院计算技术研究所
#        [location] 北京
```

```python
from csc_model import BERTcForCSC

BERTcForCSC.from_pretrained(".").correct("他平时喜欢锻练身体")
# 他平时喜欢锻炼身体
```

## Models

| Hugging Face | Parameters | Task |
|---|---:|---|
| [`BERTc-315M`](https://huggingface.co/Ismantic/BERTc-315M) · [`BERTc-165M`](https://huggingface.co/Ismantic/BERTc-165M) | 315M / 165M | Pretrained backbone |
| [`BERTc-315M-MT`](https://huggingface.co/Ismantic/BERTc-315M-MT) · [`BERTc-165M-MT`](https://huggingface.co/Ismantic/BERTc-165M-MT) | — | CWS + POS + NER |
| [`BERTc-315M-CSC`](https://huggingface.co/Ismantic/BERTc-315M-CSC) · [`BERTc-165M-CSC`](https://huggingface.co/Ismantic/BERTc-165M-CSC) | — | Chinese spelling correction |

Download a task-specific release and run its bundled example:

```bash
huggingface-cli download Ismantic/BERTc-315M-MT --local-dir BERTc-MT
pip install git+https://github.com/Ismantic/PieceTokenizer
cd BERTc-MT
python example_decode.py
```

Each Hugging Face repository includes self-contained inference code and an
example. Runtime inference requires only PyTorch and PieceTokenizer. This
repository also provides two interactive entry points:

```bash
python -m save.cws        # CWS + POS + NER
python -m save.csc        # Chinese spelling correction
```

## Results

CWS + POS + NER on PD-1998, fine-tuned for five epochs with FGM:

| Model | Parameters | CWS | POS | NER | Joint |
|---|---:|---:|---:|---:|---:|
| **BERTc-315M + FGM** | 315M | 0.9840 | **0.9800** | 0.9660 | **1.4712** |
| BERTc-165M + FGM | 165M | 0.9836 | 0.9753 | 0.9632 | 1.4689 |
| MacBERT-Large | 326M | **0.9856** | 0.9629 | **0.9664** | 1.4677 |
| RoBERTa-wwm-ext | 102M | 0.9828 | 0.9562 | 0.9629 | 1.4623 |

The joint score is `CWS F1 + 0.3 × POS accuracy + 0.2 × NER F1`. Results are
measured on the first 2,000 development sentences, matching the subset used to
select `best.pt` during training. The score on the complete 21,143-sentence
development set is 1.4646.

Chinese spelling correction on the official 707-example SIGHAN-15 test set,
using the PyCorrector evaluation protocol:

| Model | Parameters | F1 | Precision | Recall |
|---|---:|---:|---:|---:|
| **BERTc-315M** | 315M | **0.8388** | 0.9461 | 0.7534 |
| MacBERT4CSC | 110M | 0.8314 | 0.9274 | 0.7534 |
| MacBERT-Large | 326M | 0.8309 | 0.9302 | 0.7507 |
| BERTc-165M | 165M | 0.8333 | 0.9582 | 0.7373 |

## Architecture and training recipe

The 315M model uses 24 layers, a hidden size of 1,024, an intermediate size of
2,752, and 16 attention heads. The 165M model uses the same dimensions with 12
layers. The 12,536-entry character-level vocabulary assigns one piece to each
covered Chinese character while using BPE subwords for English.

- Scaled sinusoidal positional embeddings, computed once at the embedding layer
- GeGLU feed-forward blocks, bias-free LayerNorm, bias-free linear layers, and
  pre-norm with the first layer norm skipped
- A minimal MLM head: `logits = h @ embed.weight.T`, without an extra dense
  layer, normalization, activation, or bias
- Megatron-style initialization, scaling residual branches by `1 / sqrt(2L)`
- No dropout, with tied input and output embeddings

Pretraining uses StableAdamW (`beta2 = 0.95`), damped cosine decay from `8e-4`
to `8e-5`, fixed 15% whole-word masking, and FlexAttention-based isolation
between documents packed into the same sequence. Training runs for 8,500 steps
at an effective batch size of 4,096, or approximately 17.4B tokens.

## Repository layout

The pipeline is split into four layers:

```text
data/       Download and normalize raw corpora and labeled datasets
prepare/    Build the vocabulary, pre-encode data, pack corpora, launch training
src/        Model definitions and training loops
save/       Export and upload Hugging Face packages; interactive inference
```

Two boundaries are deliberate:

- `src/` depends only on PyTorch. CRFs, StableAdamW, LR scheduling, whole-word
  masking, safetensors loading, and memory mapping are implemented locally.
- `src/` never processes text. Tokenization, character-to-ID conversion, and
  label construction live in `prepare/`; training consumes pre-encoded IDs.

Additional directories include `deps/` for cloned C++ dependencies, `docs/`
for design and workflow documentation, and `test/` for benchmark-driven
regression checks.

## Training

All corpora, labeled datasets, vocabularies, and C++ dependencies are fetched
from public sources on Hugging Face or GitHub. No pre-existing local data is
required.

```bash
make -C prepare deps        # Clone and build PieceTokenizer and Wapic
make -C data status         # Check downloaded data sources
make -C prepare status      # Check each generated artifact
```

| Workflow | Time on one RTX 4090 | Starting point | Guide |
|---|---:|---|---|
| Fine-tuning | A few hours | Hugging Face backbone | [`docs/FINETUNE.md`](docs/FINETUNE.md) |
| Pretraining | 2–4 days plus ~8 hours of preparation | Random initialization | [`docs/PRETRAIN.md`](docs/PRETRAIN.md) |

Fine-tuning is the recommended first run: it gives feedback within hours and
exercises the complete data, tokenizer, model, and evaluation pipeline.

Training modules use relative imports because `src/` is a package. Run them
from the repository root with `-m`:

```bash
python -m src.pretrain --train_data ... --output_dir ...
python -m src.finetune_mt --ckpt_dir ... --train_data ... --dev_data ...
python -m src.finetune_csc --ckpt_dir ... --train_data ... --test_data ...
```

The full published configurations are encoded in `prepare/Makefile`; run
`make -C prepare help` for the available targets.

## Documentation

The detailed guides are currently written in Chinese:

| Document | Contents |
|---|---|
| [`docs/PRETRAIN.md`](docs/PRETRAIN.md) | End-to-end pretraining, including time and disk estimates |
| [`docs/FINETUNE.md`](docs/FINETUNE.md) | End-to-end fine-tuning for MT and CSC |
| [`docs/WHY.md`](docs/WHY.md) | Design rationale, failed experiments, and silent failure modes |
| [`src/README.md`](src/README.md) | A guided reading order for the implementation |

## Environment

The current reference setup uses Python 3.11, PyTorch 2.11 with CUDA 13.0,
bf16, and a single RTX 4090 with 24 GB of VRAM. There is no multi-GPU code
path.

```bash
uv pip install -r requirements.txt
```

Only PyTorch is required by `src/`; the other Python packages support data
download and processing, pre-encoding, export, and reference tests.

`make -C prepare deps` clones and builds two C++17 dependencies:

- [PieceTokenizer](https://github.com/Ismantic/PieceTokenizer), the
  character-level tokenizer and vocabulary provider
- [Wapic](https://github.com/Ismantic/Wapic), a CRF segmenter used to obtain
  word boundaries for whole-word masking. Its repository also provides the
  PD-1998 corpus used for fine-tuning.

Building them requires CMake, a C++17 compiler, and Git.

## Reproducing the published results

After downloading the published checkpoints and preparing the dependencies,
run the benchmark-driven regression test:

```bash
python test/test_reproduce_sota.py
```

It reproduces the recorded MT joint score of 1.4712 and CSC F1 of 0.8388 from
the real checkpoints. These benchmark numbers are the project's primary
regression standard.

## License

Code is released under the Apache License 2.0. Training corpora retain the
licenses documented in their respective dataset cards.
