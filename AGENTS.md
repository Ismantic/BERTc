# Repository Guidelines

## Project Structure & Module Organization

BERTc is organized around three experiment workstreams. `pretrain/` contains legacy char-level BERTc pretraining plus the current Modern BERTc code in `pretrain/modern_bertc/`; tokenizer assets live under `pretrain/modern_bertc/tokenizer/`. `finetune/` contains CWS/POS/NER fine-tuning, with the migrated main code in `finetune/NLP_BERT_CRF/` and SOTA notes/checkpoint pointers in `finetune/sota/`. `csc/` contains Chinese spelling correction training, evaluation, and baselines under `csc/train/`, `csc/eval/`, and `csc/baseline/`. Large local data, checkpoints, and generated outputs are intentionally excluded from git.

## Build, Test, and Development Commands

Use the project venv at `/home/tfbao/.venv/bin/python`; install dependencies with `uv pip install`, not `pip`.

- `bash pretrain/modern_bertc/run_v3.sh`: launch Modern BERTc pretraining.
- `bash pretrain/inline_eval_modern_both.sh <ckpt_dir>`: run CWS and CSC inline evaluation for a checkpoint.
- `cd finetune/NLP_BERT_CRF && python train_mt.py --backbone_path ../backbones/bert_train_v7_mid --alpha_pos 2.0 --beta_ner 0.5 --fgm --fgm_eps 1.0 --epochs 5 --batch_size 64`: run joint CWS/POS/NER fine-tuning.
- `bash csc/train/chain_csc_experiments.sh`: run CSC experiment chain.
- `python csc/eval/threshold_sweep.py --ckpt <path>`: tune CSC detection threshold.

## Coding Style & Naming Conventions

Python is the primary language. Follow existing style: 4-space indentation, snake_case for functions and files, lowercase experiment script names, and explicit CLI arguments via `argparse`. Keep experiment outputs in `output_*`, `data/`, or checkpoint directories that stay out of git. Prefer extending existing scripts over adding parallel entry points.

## Testing Guidelines

There is no separate unit-test suite; validation is benchmark-driven. For CSC, use `csc/eval/eval_pycorrector_baseline.py` and the canonical SIGHAN-15 707-sample pycorrector protocol. For Modern BERTc checkpoints, use `pretrain/inline_eval_modern_both.sh <ckpt_dir>`. Report the exact checkpoint, command, threshold, and metrics when changing training or evaluation code.

## Commit & Pull Request Guidelines

Recent commits use short, result-oriented subjects, often bilingual Chinese/English, such as `README: 更新到 v4-Large Modern BERTc SOTA` or `CSC SOTA: ...`. Keep commits scoped to one experiment, fix, or documentation update. Pull requests should include the motivation, changed scripts, commands run, key metrics, and any required external paths or checkpoint assumptions. Do not commit large model/data artifacts.

## Security & Configuration Tips

Many scripts assume local absolute paths for `PieceTokenizer`, `Wapic`, corpora, and checkpoints. Document any new path assumptions in README-style notes, keep secrets out of scripts, and avoid modifying benchmark ground-truth files to improve reported numbers.
