# Hugging Face Release Organization

## Recommended Repositories

Publish pretrained backbones as the primary reusable models:

- `Ismantic/BERTc-165M`: Modern BERTc v4-Mid backbone, 12L/1024H, about 165M parameters.
- `Ismantic/BERTc-315M`: Modern BERTc v4-Large backbone, 24L/1024H, about 315M parameters.

Keep task-specific fine-tuned weights in separate repositories instead of mixing them
with the backbone weights:

- `Ismantic/BERTc-315M-CSC`: Chinese spelling correction checkpoint and CSC inference code.
- `Ismantic/BERTc-315M-MT`: PD-1998 CWS/POS/NER multi-task checkpoint and decoding code.
- Optional historical variants: `Ismantic/BERTc-165M-CSC` and `Ismantic/BERTc-165M-MT`.

## Why Separate Fine-Tunes

The backbone checkpoints load as `ModernBertForMLM` and are useful for continued
fine-tuning. The CSC and PD98-MT checkpoints use different task heads, evaluation
protocols, and inference entry points. Separate repos make the default download
unambiguous and allow each model card to document task-specific metrics, datasets,
thresholds, and usage.

## Local Release Folders

Backbone release folders are generated under:

```bash
hf_release/BERTc-165M
hf_release/BERTc-315M
hf_release/BERTc-315M-CSC
hf_release/BERTc-315M-MT
```

Regenerate them with:

```bash
/home/tfbao/.venv/bin/python scripts/prepare_hf_release.py
/home/tfbao/.venv/bin/python scripts/prepare_hf_csc_release.py
/home/tfbao/.venv/bin/python scripts/prepare_hf_mt_release.py
```

Upload after the Hugging Face token has write access to the target organization:

```bash
/home/tfbao/.venv/bin/python scripts/upload_hf_release.py --namespace Ismantic
```
