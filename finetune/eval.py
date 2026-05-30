"""Eval BertCRF ckpt on full PD-06 dev set → boundary F1."""
import argparse
import os
import sys
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import CWSDataset, Collator  # noqa: E402
from model import BertCRF  # noqa: E402
from train import evaluate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model_path", default="./macbert-large")
    ap.add_argument("--dev_jsonl", default="./data/cws_dev.jsonl")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import os
    if os.path.exists(os.path.join(args.model_path, "piece.model")):
        from piece_tokenizer_adapter import PieceTokenizerAdapter
        tokenizer = PieceTokenizerAdapter(args.model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    dev_ds = CWSDataset(args.dev_jsonl)
    if args.limit:
        dev_ds.items = dev_ds.items[:args.limit]
    print(f"Dev: {len(dev_ds)} samples")

    device = torch.device("cuda")
    model = BertCRF(args.model_path, num_tags=4).to(device)
    sd = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(sd)

    collator = Collator(tokenizer)
    f1 = evaluate(model, dev_ds, collator, device, batch_size=args.batch_size)
    print(f"\nPD-06 dev F1: {f1:.4f}  (n={len(dev_ds)})")


if __name__ == "__main__":
    main()
