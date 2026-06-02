# NLP_BERT_CRF — Chinese Word Segmentation with BERT + CRF

Standalone fine-tuning project. Trains `MacBERT-large` / `RoBERTa-wwm-ext` + linear + CRF head
on People's Daily 1998 (PD-1998) for Chinese word segmentation.

## Results (PD-06 dev, 20,973 sentences)

| Backbone | Params | Train time (1× RTX 4090) | Overall F1 | Strict F1 (style-tolerant) |
|---|---|---|---|---|
| MacBERT-large + CRF  | 330M | 46 min | **0.9747** | **0.9962** |
| RoBERTa-wwm-ext + CRF | 110M | 25 min | 0.9720 | 0.9952 |

## Directory

```
NLP_BERT_CRF/
├── data.py              PD jsonl → char + BIES tags Dataset + Collator
├── model.py             BertCRF (AutoModel + Linear(4) + torchcrf.CRF)
├── train.py             AdamW (BERT lr=2e-5, CRF lr=5e-4), bf16, eval per epoch
├── eval.py              Full PD-06 dev F1
├── run.sh               One-shot: MacBERT-large + CRF
├── run_roberta.sh       One-shot: RoBERTa-wwm-ext + CRF
│
├── strict_f1.py         Strict F1 (separate hard errors from style choices)
├── compare_all.py       3-way vs Wapic + IsCut
├── compare_4way.py      4-way vs RoBERTa + Wapic + IsCut
├── case_dump.py         Categorised case examples (perfect / only-X / errors)
├── find_hard_cross.py   Hunt for truly unacceptable boundary errors
│
├── data/                Training/dev data (~180MB)
│   ├── cws.jsonl        102,205 sentences (PD-1998 Jan–May)
│   └── cws_dev.jsonl    20,973 sentences (PD-1998 Jun)
│
├── macbert-large/       hfl/chinese-macbert-large (1.3GB)
└── roberta-wwm-ext/     hfl/chinese-roberta-wwm-ext (1.2GB)
```

## Setup

```bash
# 1. Activate Python env with torch / transformers
#    (or pass your own via PY=… env var)
PY=/path/to/python

# 2. Install runtime deps
$PY -m pip install torch transformers pytorch-crf seqeval huggingface_hub

# 3. Download backbones (if not bundled in directory)
$PY -c "from huggingface_hub import snapshot_download; \
    snapshot_download('hfl/chinese-macbert-large', local_dir='./macbert-large')"
$PY -c "from huggingface_hub import snapshot_download; \
    snapshot_download('hfl/chinese-roberta-wwm-ext', local_dir='./roberta-wwm-ext')"
```

## Train

```bash
# Default: hardcoded venv path
bash run.sh

# Override venv
PY=/usr/bin/python3 bash run.sh
PY=/usr/bin/python3 bash run_roberta.sh
```

## Data format

`data/cws.jsonl` schema (one JSON per line):

```json
{
  "task": "cws",
  "lang": "zh",
  "messages": [{"role":"user","content":"切分: …"},{"role":"assistant","content":"… …"}],
  "gold": ["北京", "是", "首都"],
  "source": "pd1998-02"
}
```

Only the `gold` field (list of words) is used. `messages` is for LLM CWS training.

Source corpus: People's Daily 1998 (北大 PKU annotation style).
Train: 1998-01 to 1998-05 (102,205 sentences).
Dev: 1998-06 (20,973 sentences).

## External dependencies (only for comparison scripts)

These are OPTIONAL — only needed if running `compare_all.py` / `compare_4way.py` / `case_dump.py` / `find_hard_cross.py`:

- **IsCut** (Unigram segmenter, 35w dict): default path `/home/tfbao/Shiyu/IsCut/dict.txt`
  Pass `--iscut_dict /your/path/dict.txt` to override.
- **Wapic** (Wapiti-style CRF binary): default `/home/tfbao/Shiyu/Wapic/build/wapic` + model `data/cut.wac`
  Pass `--wapic_bin /your/path/wapic --wapic_model /your/path/cut.wac` to override.

For pure training/eval, none of these are needed.

## Architecture

```
[CLS] not added — pure char sequence:
  chars       = ['北', '京', '是', '首', '都']
  ↓ tokenizer (BertTokenizer, 1-char→1-token for CJK)
  input_ids   = [1266, 776, 3221, 7674, 6963]
  ↓ MacBERT-large (24 layers, 1024-dim)
  hidden      = [N, 1024]
  ↓ Dropout(0.1) + Linear(1024, 4)
  emissions   = [N, 4]   (B/I/E/S logits per char)
  ↓ CRF (4-tag transition matrix learned)
  loss        = -log P(tag_seq | emissions)
  decode      = Viterbi best path → BIES tags → word list
```

BIES tagging: `B`=begin (multi-char), `I`=inside, `E`=end, `S`=single-char word.
