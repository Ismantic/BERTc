"""Char-level BERT init (RoBERTa-style:无 NSP,纯 MLM)。

从 PieceTokenizer 的 piece.model 读 vocab_size,在 BERT 端 +1 给 [MASK]:
  - piece vocab id ∈ [0, piece_vocab)
  - BERT vocab_size = piece_vocab + 1
  - mask_token_id = piece_vocab(在 piece vocab 外,数据不会冲突)

用法:
  python build_bert_init.py --piece_model sp_char_v1/piece.model \
      --output_dir bert_init --size base|large
"""
import argparse, os, shutil, torch
from transformers import BertConfig, BertForMaskedLM


SIZE_CFG = {
    "base":  dict(hidden_size=768,  num_layers=12, num_heads=12,  intermediate_size=3072),
    # mid: BERT-base 的层数,但 hidden / head / ffn 跟 large 同等(1024×12L)
    "mid":   dict(hidden_size=1024, num_layers=12, num_heads=16,  intermediate_size=4096),
    "large": dict(hidden_size=1024, num_layers=24, num_heads=16,  intermediate_size=4096),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--piece_model", required=True, help="PieceTokenizer 训出的 piece.model")
    ap.add_argument("--output_dir", default="bert_init")
    ap.add_argument("--size", choices=list(SIZE_CFG.keys()), default="base")
    ap.add_argument("--max_position", type=int, default=512)
    ap.add_argument("--pad_token_id", type=int, default=-1,
                    help="piece tokenizer 里 <pad> 的 id(-1 = 自动从 piece 读)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # 读 piece tokenizer 拿 vocab_size
    import piece_tokenizer as pt
    tok = pt.Tokenizer()
    tok.load(args.piece_model, cn_dict="no")
    piece_vocab = tok.vocab_size()
    mask_token_id = piece_vocab          # BERT vocab 末位
    total_vocab = piece_vocab + 1

    # pad_token_id: -1 = 自动从 piece <pad> 读
    pad_id = args.pad_token_id
    if pad_id < 0:
        pad_id = tok.piece_to_id("<pad>")
        if pad_id <= 0 or pad_id >= piece_vocab:
            raise RuntimeError(f"<pad> not found in piece vocab (got id {pad_id})")

    cfg = SIZE_CFG[args.size]
    config = BertConfig(
        vocab_size=total_vocab,
        hidden_size=cfg["hidden_size"],
        num_hidden_layers=cfg["num_layers"],
        num_attention_heads=cfg["num_heads"],
        intermediate_size=cfg["intermediate_size"],
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=args.max_position,
        type_vocab_size=1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=pad_id,
    )

    torch.manual_seed(args.seed)
    model = BertForMaskedLM(config)
    model.init_weights()

    total = sum(p.numel() for p in model.parameters())
    embed = model.bert.embeddings.word_embeddings.weight.numel()
    transformer = total - embed - (config.hidden_size if model.cls.predictions.bias is not None else 0)
    print(f"\nBERT-{args.size} init:")
    print(f"  piece vocab:   {piece_vocab}")
    print(f"  BERT vocab:    {total_vocab}  (+1 for [MASK] at id {mask_token_id})")
    print(f"  hidden/layers/heads/ffn: {cfg['hidden_size']} / {cfg['num_layers']} / "
          f"{cfg['num_heads']} / {cfg['intermediate_size']}")
    print(f"  max_position:  {args.max_position}")
    print(f"  pad_token_id:  {pad_id}")
    print(f"  total params:  {total/1e6:.1f}M (embed {embed/1e6:.1f}M)")

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    shutil.copy2(args.piece_model, os.path.join(args.output_dir, "piece.model"))
    with open(os.path.join(args.output_dir, "mask_token_id.txt"), "w") as f:
        f.write(str(mask_token_id))
    print(f"\nSaved to {args.output_dir}/")


if __name__ == "__main__":
    main()
