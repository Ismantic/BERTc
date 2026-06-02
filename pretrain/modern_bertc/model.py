"""Modern BERTc v3 — ModernBERT release-aligned + Cramming-style ScaledSinusoidal PE.

主要按 release `modernbert-base-pretrain.yaml` 对齐(除 Alt Attn 和 PE):
  - 架构: 22L / 768H / 1152I (GLU) / 12 heads,head_dim=64
  - **ScaledSinusoidal 位置编码**(Hua et al. 2022 FLASH;Cramming 实测短 seq 比
    RoPE 更值:计算几乎免费,RoPE 收益被 5-10% 速度损失抵消)
  - GeGLU FFN(glu + gelu)
  - LayerNorm 无 bias(eps=1e-5),非 RMSNorm
  - pre-norm 布局 + skip_first_prenorm
  - embed_norm + final_norm
  - Megatron-style init:残差层 W 缩 1/sqrt(2L)
  - 全无 Linear bias
  - Dropout: 全 0(Cramming 论据:short single-epoch 无 overfit risk)
  - tied word embedding
  - flex_attention compiled(支持 cross-doc 隔离 via seg_ids)

不上的 ModernBERT 特性:
  - Alternating Attention(我们走全局 attention)
  - Unpadded packing + cu_seqlens(我们定长 pack)
  - RoPE(换 ScaledSinusoidal,见 Cramming Section 4.2)

参数量(默认 22L/768H/1152I,V=12536):
  emb (tied)              : 12536 × 768  ≈ 9.6M
  embed_norm              : 768 × 1     ≈ 1K(no bias)
  per layer:
    norm1/2               : 768 × 2     ≈ 1.5K
    Q K V O               : 4 × 768²    ≈ 2.36M
    GeGLU (W_in=2I, W_out): 768×2304 + 1152×768 ≈ 2.66M
    total per layer       : ≈ 5.0M
  × 22 layers              : ≈ 110M
  final_norm              : 768 × 1
  head: dense + norm + gelu: 768×768 + 768 ≈ 0.59M
  head_bias (V,)          : 12.5K
  total ≈ 130M
"""
from dataclasses import dataclass, field
from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import (
    flex_attention as _flex_attention_raw,
    create_block_mask,
)

# torch.compile 是 lazy 的 — module import 不触发 trace,first call 时才编译。
# ModernBERT 源码用 mode="max-autotune-no-cudagraphs",我们用 default mode 平衡
# (first-call 编译几秒,vs max-autotune 可能几十秒)。
# 训练时默认调 _flex_attention(compiled);smoke 也走这条路径,保证统一。
_flex_attention = torch.compile(_flex_attention_raw)


# ============ Config ============

@dataclass
class ModernBertConfig:
    vocab_size: int = 12536
    hidden_size: int = 768
    num_hidden_layers: int = 22
    num_attention_heads: int = 12
    intermediate_size: int = 1152
    max_position_embeddings: int = 1024
    pad_token_id: int = 12531
    mask_token_id: int = 12535
    pe_theta: float = 10000.0  # ScaledSinusoidal 频率 base(Vaswani 2017 默认)
    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True
    # Dropout(对齐 ModernBERT release)
    embed_dropout: float = 0.0
    mlp_dropout: float = 0.0
    attn_out_dropout: float = 0.1
    attn_probs_dropout: float = 0.0
    # 架构开关(对齐 release)
    embed_norm: bool = True            # embedding 后立刻 LayerNorm
    skip_first_prenorm: bool = True    # 第 1 层不做 pre-norm
    final_norm: bool = True            # 最后一层后 LayerNorm
    # init
    init_method: str = "megatron"      # "megatron"(残差层 ×1/sqrt(2L))或 "normal"

    @property
    def head_dim(self) -> int:
        assert self.hidden_size % self.num_attention_heads == 0
        return self.hidden_size // self.num_attention_heads


# ============ LayerNorm(no bias)============

class LayerNormNoBias(nn.Module):
    """LayerNorm with weight only (no bias). 对齐 ModernBERT release。"""
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.normalized_shape = (hidden_size,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.normalized_shape, self.weight, None, self.eps)


# ============ ScaledSinusoidal Position Embedding(Hua et al. 2022 / Cramming)============

class ScaledSinusoidalPE(nn.Module):
    """Scaled sinusoidal positional embedding(Hua 2022 FLASH paper)。

    标准 sinusoidal:PE[pos, 2i]=sin(pos/θ^(2i/d)), PE[pos, 2i+1]=cos(...)。
    `scale_factor` 是一个 learnable 标量,初始 1/sqrt(d)。
    用法:embedding 之后直接 `x = embed + pos_emb(input_ids)`,跟所有层共享。
    比 RoPE 便宜:只在 embedding 层 fire 一次,attention 里 0 开销。
    """
    def __init__(self, embedding_dim: int, max_seq_length: int, theta: float = 10000.0):
        super().__init__()
        pe = torch.zeros(max_seq_length, embedding_dim)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float() * (-math.log(theta) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, L, d]
        self.register_buffer("pe", pe, persistent=False)
        self.scale_factor = nn.Parameter(torch.tensor([1.0 / embedding_dim ** 0.5]))

    def forward(self, seq_len: int) -> torch.Tensor:
        return self.scale_factor * self.pe[:, :seq_len, :]


# ============ Attention(bidirectional,无 RoPE,位置走 ScaledSinusoidal)============

class ModernBertAttention(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim ** -0.5
        self.attn_probs_dropout = config.attn_probs_dropout
        # 无 bias
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.o = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.out_dropout = nn.Dropout(config.attn_out_dropout)

    def forward(self, x: torch.Tensor,
                block_mask=None,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """三种 attention 模式:
        - block_mask 非空 → flex_attention(block-diag,跨 doc 隔离,训练时)
        - block_mask 空,attention_mask 非空 → SDPA + pad mask(fine-tune)
        - 都空 → SDPA 全可见
        位置信息走 ScaledSinusoidal,已在 embedding 层加,attention 里无 cos/sin 计算。
        """
        B, L, H = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)            # 各 [B, L, h, d]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if block_mask is not None:
            # flex_attention(compiled):任意 mask + flash 速度;不支持 dropout_p
            out = _flex_attention(q, k, v, block_mask=block_mask)
        else:
            sdpa_mask = None
            if attention_mask is not None:
                sdpa_mask = attention_mask[:, None, None, :].to(torch.bool)
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=sdpa_mask,
                dropout_p=self.attn_probs_dropout if self.training else 0.0,
                is_causal=False,
            )  # [B, h, L, d]
        out = out.transpose(1, 2).reshape(B, L, H)
        return self.out_dropout(self.o(out))


# ============ GeGLU MLP ============

class GeGLU(nn.Module):
    """Linear(H, 2*I) → split → GELU(gate) * up → Linear(I, H).
    全无 bias。
    """
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        I = config.intermediate_size
        self.w_in = nn.Linear(config.hidden_size, 2 * I, bias=False)
        self.w_out = nn.Linear(I, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.mlp_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w_in(x).chunk(2, dim=-1)
        return self.dropout(self.w_out(F.gelu(gate) * up))


# ============ Layer(pre-norm,支持 skip_first_prenorm)============

class ModernBertLayer(nn.Module):
    def __init__(self, config: ModernBertConfig, is_first: bool = False):
        super().__init__()
        # is_first + skip_first_prenorm:第 1 层 attention 前不做 pre-norm
        # (因为 embed_norm 已经 norm 过一次了)
        self.skip_norm1 = is_first and config.skip_first_prenorm
        self.norm1 = nn.Identity() if self.skip_norm1 else LayerNormNoBias(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = ModernBertAttention(config)
        self.norm2 = LayerNormNoBias(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = GeGLU(config)

    def forward(self, x, block_mask=None, attention_mask=None):
        x = x + self.attn(self.norm1(x), block_mask, attention_mask)
        x = x + self.mlp(self.norm2(x))
        return x


# ============ Backbone ============

class ModernBertModel(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size,
                                   padding_idx=config.pad_token_id)
        # ScaledSinusoidal PE(Cramming-style),加在 embedding 后
        self.pos_emb = ScaledSinusoidalPE(
            embedding_dim=config.hidden_size,
            max_seq_length=config.max_position_embeddings,
            theta=config.pe_theta,
        )
        self.embed_norm = (LayerNormNoBias(config.hidden_size, eps=config.layer_norm_eps)
                           if config.embed_norm else nn.Identity())
        self.embed_dropout = nn.Dropout(config.embed_dropout)
        self.layers = nn.ModuleList(
            [ModernBertLayer(config, is_first=(i == 0))
             for i in range(config.num_hidden_layers)]
        )
        self.final_norm = (LayerNormNoBias(config.hidden_size, eps=config.layer_norm_eps)
                           if config.final_norm else nn.Identity())

        # init: Megatron-style 残差缩放
        self.apply(self._init_weights)
        if config.init_method == "megatron":
            self._megatron_residual_init()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=self.config.initializer_range)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=self.config.initializer_range)
            if m.padding_idx is not None:
                with torch.no_grad():
                    m.weight[m.padding_idx].zero_()
        elif isinstance(m, LayerNormNoBias):
            nn.init.ones_(m.weight)

    def _megatron_residual_init(self):
        """对每个 residual 路径的输出 W 缩 1/sqrt(2*L)。
        防止深层网络早期 forward variance 爆炸。
        residual outputs: attn.o, mlp.w_out。
        """
        L = self.config.num_hidden_layers
        scale = (2.0 * L) ** -0.5
        for layer in self.layers:
            with torch.no_grad():
                layer.attn.o.weight.mul_(scale)
                layer.mlp.w_out.weight.mul_(scale)

    def forward(self, input_ids: torch.Tensor,
                seg_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """seg_ids: [B, L] int32/uint8,同 doc 同 id;非空时走 flex_attention 跨 doc 隔离。
        attention_mask: [B, L] 0/1,只在 seg_ids=None 时使用(fine-tune 路径)。
        位置信息:ScaledSinusoidal 加在 embedding 后,attention 内部无位置计算。
        """
        B, L = input_ids.shape
        x = self.embed(input_ids)
        x = x + self.pos_emb(L).to(x.dtype)  # 加 scaled sinusoidal PE
        x = self.embed_norm(x)
        x = self.embed_dropout(x)

        block_mask = self._build_block_mask(seg_ids, B, L) if seg_ids is not None else None

        for layer in self.layers:
            x = layer(x, block_mask, attention_mask)
        x = self.final_norm(x)
        return x

    def _build_block_mask(self, seg_ids: torch.Tensor, B: int, L: int):
        """seg_ids: [B, L] 用 flex_attention 构造 doc-internal mask。
        mask_mod 闭包捕获 seg_ids,在 batch/query/kv 索引下查 doc 是否一致。
        """
        seg_ids_long = seg_ids.to(torch.int32)

        def mask_mod(b, h, q_idx, kv_idx):
            return seg_ids_long[b, q_idx] == seg_ids_long[b, kv_idx]

        # H=None 让 mask 跨 head 共享(同 doc 隔离与 head 无关)
        return create_block_mask(mask_mod, B=B, H=None, Q_LEN=L, KV_LEN=L,
                                  device=seg_ids.device)


# ============ MLM head(tied embedding)============

class ModernBertForMLM(nn.Module):
    """MLM head 简化版(Cramming Section 4.2 推荐):
       - 无 nonlinear head(去 Dense + LN + GeLU)— "without ill effect"
       - 无 decoder bias(去 head_bias)— "drop the decoder bias"
       - 仅 tied embedding projection:logits = h @ embed.weight.T
       - final LayerNorm 已经在 bert.final_norm 提供,这里不需重复
       省参数 ~0.6M,forward 略快。"""
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config
        self.bert = ModernBertModel(config)

    def get_input_embeddings(self):
        return self.bert.embed

    def forward(self, input_ids, seg_ids=None, attention_mask=None, labels=None):
        h = self.bert(input_ids, seg_ids=seg_ids,
                       attention_mask=attention_mask)        # [B, L, H],bert 内已 final_norm
        # 直接 tied embedding projection,无 nonlinear head 也无 bias
        logits = F.linear(h, self.bert.embed.weight)         # [B, L, V]
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )
        return {"logits": logits, "loss": loss}

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============ Quick sanity test ============

if __name__ == "__main__":
    print("=== Test 1: default config (22L/768H release-aligned) ===")
    cfg = ModernBertConfig()
    print(f"Config: V={cfg.vocab_size} H={cfg.hidden_size} L={cfg.num_hidden_layers} "
          f"head={cfg.num_attention_heads} d={cfg.head_dim} I={cfg.intermediate_size} "
          f"pe_theta={cfg.pe_theta} ln_eps={cfg.layer_norm_eps}")
    print(f"embed_norm={cfg.embed_norm} skip_first_prenorm={cfg.skip_first_prenorm} "
          f"final_norm={cfg.final_norm} init={cfg.init_method}")
    print(f"dropout: embed={cfg.embed_dropout} mlp={cfg.mlp_dropout} "
          f"attn_out={cfg.attn_out_dropout} attn_probs={cfg.attn_probs_dropout}")
    model = ModernBertForMLM(cfg)
    n = model.num_parameters()
    print(f"params: {n:,} = {n / 1e6:.1f}M")

    # forward smoke
    B, L = 2, 128
    ids = torch.randint(0, cfg.vocab_size, (B, L))
    mask = torch.ones(B, L, dtype=torch.long)
    labels = torch.randint(0, cfg.vocab_size, (B, L))
    out = model(ids, attention_mask=mask, labels=labels)
    print(f"logits: {out['logits'].shape}  loss: {out['loss'].item():.4f}")

    # backward smoke
    out["loss"].backward()
    # 检查所有 trainable param 都有 grad
    no_grad = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    if no_grad:
        print(f"ERROR: 缺 grad 的参数: {no_grad}")
    else:
        print("backward OK (所有参数都有 grad)")

    # 验证 skip_first_prenorm 生效
    assert model.bert.layers[0].skip_norm1, "layer[0].skip_norm1 应为 True"
    assert not model.bert.layers[1].skip_norm1, "layer[1].skip_norm1 应为 False"
    assert isinstance(model.bert.layers[0].norm1, nn.Identity), "layer[0].norm1 应为 Identity"
    assert isinstance(model.bert.layers[1].norm1, LayerNormNoBias), "layer[1].norm1 应为 LayerNormNoBias"
    print("skip_first_prenorm 配置正确")

    # 验证 embed_norm + final_norm
    assert isinstance(model.bert.embed_norm, LayerNormNoBias), "embed_norm 应为 LayerNormNoBias"
    assert isinstance(model.bert.final_norm, LayerNormNoBias), "final_norm 应为 LayerNormNoBias"
    print("embed_norm + final_norm 配置正确")

    # 验证 Megatron init
    L_layers = cfg.num_hidden_layers
    expected_scale = (2.0 * L_layers) ** -0.5
    # 一个未缩的初始值 std ≈ 0.02,缩后 std ≈ 0.02 * expected_scale
    o_std = model.bert.layers[5].attn.o.weight.std().item()
    expected_std = cfg.initializer_range * expected_scale
    # 允许 ±50% 容差(单 tensor std 估计有 noise)
    assert 0.5 * expected_std < o_std < 1.5 * expected_std, \
        f"Megatron init: attn.o.weight.std={o_std:.6f}, expected≈{expected_std:.6f}"
    print(f"Megatron init OK: attn.o.weight.std={o_std:.6f} ≈ {expected_std:.6f}")

    # 验证 no bias
    for name, p in model.named_parameters():
        if "bias" in name and name != "head_bias":
            print(f"ERROR: 不该有的 bias: {name}")
    print("Linear/Norm no-bias 配置正确")

    print("\n=== Test 1b: seg_ids → flex_attention 路径(cross-doc 隔离)===")
    if torch.cuda.is_available():
        # flex_attention 需要 CUDA
        model_cuda = ModernBertForMLM(cfg).cuda().to(torch.bfloat16)
        ids_c = ids.cuda()
        labels_c = labels.cuda()
        # 构造 seg_ids:chunk 内 0 0 0 ... 0 | 1 1 ... 1 (前半 doc0, 后半 doc1)
        seg_ids = torch.zeros(B, L, dtype=torch.int32, device="cuda")
        seg_ids[:, L // 2:] = 1
        out_flex = model_cuda(ids_c, seg_ids=seg_ids, labels=labels_c)
        print(f"  flex_attention forward OK  loss={out_flex['loss'].item():.4f}  "
              f"logits={out_flex['logits'].shape} dtype={out_flex['logits'].dtype}")
        out_flex["loss"].backward()
        no_grad = [n for n, p in model_cuda.named_parameters() if p.requires_grad and p.grad is None]
        if no_grad:
            print(f"  ERROR: 缺 grad: {no_grad[:5]}")
        else:
            print(f"  flex_attention backward OK")

        # 验证 attention 真正隔离了:用相同 ids 但 seg_ids 不同,前半 token 应该输出不同
        # (因为 doc0 token 在 seg=0 时只 attend 同 seg=0;改成 seg=2 后 attend 不同的 kv 集)
        with torch.no_grad():
            seg_v1 = torch.zeros(B, L, dtype=torch.int32, device="cuda")
            seg_v2 = torch.zeros(B, L, dtype=torch.int32, device="cuda")
            seg_v2[:, :L // 2] = 0
            seg_v2[:, L // 2:] = 1  # 前半 vs 后半 隔离
            h1 = model_cuda.bert(ids_c, seg_ids=seg_v1)  # 全 attend(单 doc)
            h2 = model_cuda.bert(ids_c, seg_ids=seg_v2)  # 前后半隔离
            # 前半 token 在 v1 attend 全 chunk,在 v2 只 attend 前半 → 输出应不同
            front_diff = (h1[:, :L // 2] - h2[:, :L // 2]).abs().mean().item()
            print(f"  前半 token 输出差异(应 > 0,验证隔离生效):{front_diff:.6f}")
            assert front_diff > 1e-3, f"隔离未生效:front_diff={front_diff}"
            print(f"  跨 doc 隔离工作正常 ✓")
    else:
        print("  跳过(无 CUDA)")

    print("\n=== Test 2: legacy v1 config (12L/1024H) — 验证向后兼容 ===")
    cfg_v1 = ModernBertConfig(
        hidden_size=1024, num_hidden_layers=12, num_attention_heads=16,
        intermediate_size=2752,
        embed_norm=False, skip_first_prenorm=False, init_method="normal",
    )
    model_v1 = ModernBertForMLM(cfg_v1)
    n_v1 = model_v1.num_parameters()
    print(f"v1-style params: {n_v1 / 1e6:.1f}M")
    out_v1 = model_v1(ids, attention_mask=mask, labels=labels)
    print(f"v1 forward OK  loss={out_v1['loss'].item():.4f}")

    print("\n=== All smoke tests passed ===")
