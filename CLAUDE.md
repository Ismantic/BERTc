# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working in this repository.

## What this project is

**BERTc** is a char-level Chinese BERT-mid (165M, 12L/1024H) trained **from scratch** plus its modern successor. Three workstreams under one repo:

1. **Char-level BERTc (legacy, v4→v7)** — `pretrain/` + `finetune/NLP_BERT_CRF/`. Standard BERT pre-norm + char-level piece tokenizer (vocab 12536). Goal: beat `hfl/chinese-roberta-wwm-ext` (102M) and `hfl/chinese-macbert-large` (326M) on PD-1998 CWS / POS / NER. Current SOTA: `v6.5+FGM 5ep` MT joint (see `finetune/sota/README.md`).

2. **Modern BERTc (current, v3)** — `pretrain/modern_bertc/`. ModernBERT-aligned redesign: 22L/768H/1152I (~120M), LayerNorm no-bias, RoPE, GeGLU, Megatron init, StableAdamW, Damped Cosine LR, Dynamic MLM curriculum, `flex_attention` with cross-doc isolation. Same piece tokenizer (12536). Target: surpass `v7+FGM` SOTA and close the gap to RoBERTa-wwm-large on a smaller param budget.

3. **BERTc-CSC** — `csc/`. Chinese Spelling Correction (dual-head: MLM correction + focal detection). Production user: the `Sime` input-method project. Best so far: MacBERT-large baseline 0.8309 F1; BERTc-v7 stuck at 0.7994; goal of v3 backbone is to push BERTc-CSC ≥ 0.83.

## Environment & external paths

Absolute, assumed by nearly every script:
- Python venv: `/home/tfbao/.venv/bin/python` (Python 3.14, torch 2.11+cu13, transformers, vllm). **Use `uv pip install`, not `pip` — there is no `pip` in this venv.**
- `piece_tokenizer` — built from sibling repo `/home/tfbao/Shiyu/PieceTokenizer` (`pip install -e .`). Loaded via `_PIECE.load(piece_path, cn_dict="no")` (char-mode, no segmentation dict).
- `wapic` — built from `/home/tfbao/Shiyu/Wapic`. CRF Chinese segmenter for WWM data prep. **Use `cut_smart`** (mixed CN/EN: whitespace split → EN intact → CN goes through CRF), not `cut` (CRF over English explodes char-by-char). Model file: `/home/tfbao/Shiyu/Wapic/data/wapic-20260602-h19_1-full.wac`.
- Char-level piece tokenizer: `pretrain/modern_bertc/tokenizer/piece.model` (vocab 12536, pad=12531, mask=12535). Came from the old `bert_train_v6_mid/`; migrated here when Summer/BERT was removed.
- Baselines & backbones (under `finetune/`):
  - `NLP_BERT_CRF/macbert-large/` (3.8GB, 326M) — MacBERT baseline
  - `NLP_BERT_CRF/roberta-wwm-ext/` (1.2GB, 102M) — RoBERTa baseline
  - `backbones/bert_train_v7_mid/` (317MB) — char BERTc v7 backbone (CSC + comparison)
  - `backbones/bert_train_v6_5_mid/` (633MB) — char BERTc v6.5 backbone (MT SOTA backbone)
- Pretraining corpora live on `a6000/` (external SSD): SkyPile, CCI3-HQ, Chinese-FineWeb-Edu, Wikipedia_{cn,en}. PeopleDaily on `/home/tfbao/Shiyu/Data/`.
- **`.gitignore` excludes**: `data/`, `output_*/`, `*.pt`, `*.pt.wid`, `*.pt.seg`, `*.bin`, `*.safetensors`, `*.parquet`, `*.wac`, `bert_train*/`, `*_init*/`. Big files stay on disk but never enter git.

## Workstream layout

```
pretrain/
├── modern_bertc/              # **current**: Modern BERTc v3 (ModernBERT-aligned)
│   ├── model.py               # 22L/768H + flex_attention + cross-doc isolation
│   ├── train_modern.py        # StableAdamW + Damped Cosine + Dynamic MLM
│   ├── pretokenize_modern.py  # writes .pt/.pt.wid/.pt.seg (uint8 doc ids)
│   ├── run_v3.sh              # entry point: batch=32, accum=8, 111K steps
│   ├── eval_modern_{cws,csc}.py  # inline-eval probes
│   ├── tokenizer/             # piece.model + mask_token_id.txt (vocab 12536)
│   ├── data3/                 # current 20B-target pretokenized corpus
│   └── output_v*/             # checkpoints
├── train_bert_mlm.py          # legacy char-BERTc trainer (v4–v7)
├── pretokenize_v6_anneal.py   # v6.5 anneal data prep (legacy)
├── chain_v6_anneal.sh         # legacy chain (path-broken after Summer rm)
├── aurora.py / muon.py        # legacy optimizers (Modern BERTc uses StableAdamW)
├── inline_eval_modern_{both,cws,csc}.sh  # called by train_modern.py at each save
├── inline_track{,_csc}.tsv    # legacy inline-eval logs (v7 era)
└── CORPUS.md                  # corpus & mixing notes

finetune/
├── NLP_BERT_CRF/              # migrated from Summer/BERT/NLP_BERT_CRF/
│   ├── train.py / train_mt.py # single-task / joint CWS+POS+NER
│   ├── data*.py / model*.py
│   ├── piece_tokenizer_adapter.py  # HF-compatible shim over piece_tokenizer
│   ├── macbert-large/ roberta-wwm-ext/  # HF baselines
│   └── data/                  # PD-1998 jsonl + LTP distilled labels
├── backbones/
│   ├── bert_train_v7_mid/     # char BERTc v7 (used by csc/)
│   └── bert_train_v6_5_mid/   # char BERTc v6.5 (MT SOTA backbone)
├── sota/                      # hardlinked SOTA + secondbest checkpoints
│   ├── README.md              # **the source of truth for SOTA numbers**
│   ├── sota_mt_v65_fgm_5ep_best.pt      # MT joint SOTA: score 1.4636
│   ├── sota_cws_v6_fgm_5ep_best.pt      # CWS SOTA: clean F1 0.9819
│   ├── secondbest_mt_v65_3ep_best.pt    # MT v65 no-FGM (FGM ablation)
│   └── secondbest_mt_macbert_3ep_best.pt# MT MacBERT (ceiling)
└── (other legacy chain_*.sh / eval_*.py with broken Summer/BERT paths — historical)

csc/
├── train/
│   ├── train_csc.py            # BERTc dual-head (cor + det focal)
│   ├── train_csc_hf.py         # HF version (RoBERTa / MacBERT)
│   └── chain_csc_experiments.sh
├── eval/
│   ├── threshold_sweep.py      # detection-threshold tuning
│   ├── eval_pycorrector_baseline.py  # SIGHAN-15 official 707 test
│   └── eval_ctc.py
├── baseline/                   # pycorrector MacBERT4CSC CPU run
└── output_*/                   # 13GB of CSC fine-tune checkpoints
```

## Commands

```bash
# --- Modern BERTc v3 (current) ---
bash pretrain/modern_bertc/run_v3.sh         # full training (3–5 days on 1× 4090)
# Single-GPU only — there is no DDP wrap. eff_batch=256 via grad_accum=8.

# --- Legacy char-BERTc MT / CWS training ---
cd finetune/NLP_BERT_CRF
python train_mt.py --backbone_path ../backbones/bert_train_v7_mid \
                   --alpha_pos 2.0 --beta_ner 0.5 --fgm --fgm_eps 1.0 \
                   --epochs 5 --batch_size 64

# --- BERTc-CSC ---
bash csc/train/chain_csc_experiments.sh      # 4-way (BERTc / RoBERTa / MacBERT / dual)
python csc/eval/threshold_sweep.py --ckpt <path>

# --- Inline eval at each ckpt save (called by train_modern.py) ---
bash pretrain/inline_eval_modern_both.sh <ckpt_dir>   # runs cws + csc, ~30 min
```

## Critical conventions

- **GPU is 1× RTX 4090 (24GB, bf16).** There is no DDP / multi-GPU code path. Older `train_bert_mlm.py` references 2× A6000 in comments — that machine is gone.
- **Modern BERTc v1 (RMSNorm + 12L/1024H) and v3 (LayerNorm no-bias + 22L/768H) checkpoints are NOT interchangeable** — the state-dict keys diverge (RMSNorm vs LayerNorm, embed_norm/skip_first_prenorm only in v3). Eval scripts auto-detect via `config.json` but `--init_from_ckpt` across architectures will fail.
- **`flex_attention` must be `torch.compile`-wrapped** to get FlashAttention speed; `model.py` already does this at module import. Calling raw `flex_attention` produces an unfused materialized-scores kernel (warning emitted) and is ~3× slower.
- **Pretokenize must write `.seg` alongside `.pt` / `.pt.wid`** for cross-doc attention isolation. `--no_cross_doc_isolation` falls back to SDPA + plain pad mask (also slower because non-trivial attn_mask falls off the flash kernel).
- **Wapic `cut` mangles English** (every char becomes a "word"). Use `cut_smart` (in C++ binding `_core.so`) which whitespace-splits first, keeps EN segments intact, runs CRF only on CN. If you rebuild Wapic, copy the new `_core.cpython-*.so` into the venv site-packages.
- **`dict.txt` is NOT used by the BERTc piece tokenizer** (char-mode loads with `cn_dict="no"`). It is only needed by the Summer ReTok piece tokenizer.
- **Eval ground truth: do not edit `eval_modern_*.py`, `threshold_sweep.py`, `eval_pycorrector_baseline.py` to make numbers look better.** SIGHAN-15 test (707 samples, pycorrector vendored) is the canonical CSC benchmark.
- **`gsm8k` and other arithmetic benchmarks are not measured here** — that was a Qwen ReTok concern (sibling project `Summer/`); BERTc is encoder-only.

## Migration history (what you should know)

- This repo was carved out of `Summer/BERT/` over multiple sessions. Summer/BERT/ has been **deleted** (commit `116dbf4` in `Summer`); all code/data the BERTc workstreams need is now under `BERTc/` (verified via path grep — see `2026-06-02` session).
- The legacy `finetune/` top-level scripts (`eval_cws_micro_batch.py`, `case_*.py`, `chain_mt_*.sh`) and `pretrain/{chain_v6_anneal,pretokenize_v6_anneal,inline_eval_quick_mt}` still hard-code `/home/tfbao/Shiyu/Summer/BERT/...` paths. Those are **historical comparison scripts** kept for archival; they don't run anymore. Don't try to "fix" them unless you actually need to re-run that chain.
- The two Modern BERTc data attempts before v3:
  - `data/` was buggy (wapic `cut` over EN); deleted.
  - `data2/` was correct cut_smart but no `.seg` (no cross-doc isolation); kept as backup; may be deleted later.
  - `data3/` is the current good one (cut_smart + .seg).

## Autonomous-loop guidance

If asked to "iterate on Modern BERTc Phase 2" / "run the loop": this repo has no `program.md` yet. Modern BERTc v3 is itself the first non-trivial run; only consider Phase-2 search after seeing v3's downstream numbers (CWS / CSC inline-eval over 11 checkpoints). Until then, **finish v3 and report numbers** rather than spawning new experiments.
