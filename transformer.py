from __future__ import annotations

from dataclasses import replace
from functools import partial
from typing import Any, Callable, Optional, Sequence, Tuple

import math
import jax
import jax.numpy as jnp
from jax import lax


# from: https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/tpu/flash_attention.py
from flash_attention import (
    flash_attention as fa_flash_attention,
    BlockSizes,
)

import flax
import flax.linen as nn
from flax import struct
from flax.core.frozen_dict import FrozenDict

@struct.dataclass
class TransformerConfig:
    """
    Configuration for the Transformer.
    """
    vocab_size: int
    emb_dim: int
    num_heads: int
    num_layers: int
    mlp_hidden_dim: int
    max_seq_len: int

    # Regularization
    dropout_rate: float = 0.0
    attn_dropout_rate: float = 0.0

    # Rotary positional embedding (RoPE)
    rope_theta: float = 10000.0  # Base frequency (theta) commonly 10_000
    rope_fraction: float = 1.0   # If <1, apply RoPE to a fraction of head_dim

    # Numerics & dtypes
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    logits_dtype: Any = jnp.float32
    norm_epsilon: float = 1e-6
    use_bias: bool = False

    # Init
    kernel_init: Callable[..., Any] = nn.initializers.lecun_normal()
    bias_init: Callable[..., Any] = nn.initializers.zeros


@struct.dataclass
class LayerCache:
    """
    KV cache for a single self-attention layer.
    Shapes:
      key:   (batch, heads, max_seq_len, head_dim)
      value: (batch, heads, max_seq_len, head_dim)
      index: scalar int32 - number of tokens already cached (next write index)
    """
    key: jnp.ndarray
    value: jnp.ndarray
    index: jnp.int32


@struct.dataclass
class KVCache:
    """
    KV caches for all layers, stored as a tuple of LayerCache.
    """
    layers: Tuple[LayerCache, ...]

class RMSNorm(nn.Module):
    """
    RMSNorm (pre-norm) as in most of GPT variants(like LLaMa). No bias by default.
    """
    features: int
    eps: float = 1e-6
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        scale = self.param(
            "scale",
            nn.initializers.ones,
            (self.features,),
            self.param_dtype,
        )
        x = x.astype(self.dtype)
        # Normalize by root mean square over last dimension
        rms = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + self.eps)
        x_norm = x / rms
        return (scale.astype(self.dtype) * x_norm).astype(self.dtype)


def make_causal_mask(q_len: int, k_len: int, dtype: Any = jnp.float32) -> jnp.ndarray:
    """
    Causal mask of shape (1, 1, q_len, k_len) with 0 for allowed, -inf for disallowed.
    """
    # True for keep, False for mask-out
    mask = jnp.tril(jnp.ones((q_len, k_len), dtype=bool), k=0)
    mask = mask.reshape(1, 1, q_len, k_len)
    # Convert to large negative additive bias for logits
    neg_inf = jnp.array(-1e10, dtype=dtype)
    return jnp.where(mask, jnp.array(0.0, dtype=dtype), neg_inf)


def rope_cos_sin(
    position_ids: jnp.ndarray,  # (batch, seq) or (seq,)
    head_dim: int,
    rope_theta: float,
    dtype: Any,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute cos/sin caches for RoPE at given positions.

    Returns:
      cos: (batch, seq, 1, head_dim/2)
      sin: (batch, seq, 1, head_dim/2)
    """
    assert head_dim % 2 == 0, "RoPE requires even head_dim."
    # Inverse frequencies: theta^{-2i/d}
    idx = jnp.arange(0, head_dim, 2, dtype=dtype)
    inv_freq = 1.0 / (rope_theta ** (idx / head_dim))

    # positions can be (seq,) or (batch, seq)
    if position_ids.ndim == 1:
        position_ids = position_ids[None, :]
    angles = position_ids.astype(dtype)[..., None] * inv_freq[None, None, :]
    return jnp.cos(angles)[..., None, :], jnp.sin(angles)[..., None, :]


def apply_rope(
    x: jnp.ndarray,    # (batch, seq, heads, head_dim)
    cos: jnp.ndarray,  # (batch, seq, 1, head_dim/2)
    sin: jnp.ndarray,  # (batch, seq, 1, head_dim/2)
    rope_fraction: float = 1.0,
) -> jnp.ndarray:
    """
    Apply rotary position embedding to the last dimension in pairs.

    If rope_fraction < 1, only the first rope_fraction of the head_dim are rotated,
    the remainder is left unchanged (helpful for some variants).
    """
    b, t, h, d = x.shape
    rot_d = int(d * rope_fraction)
    rot_d -= rot_d % 2  # even
    if rot_d == 0:
        return x

    x_rot, x_pass = x[..., :rot_d], x[..., rot_d:]
    x1 = x_rot[..., ::2]
    x2 = x_rot[..., 1::2]
    # Broadcast cos/sin: (b, t, 1, rot_d/2)
    cos = cos[..., : rot_d // 2]
    sin = sin[..., : rot_d // 2]
    # Rotate
    rotated = jnp.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)  # (..., rot_d/2, 2)
    x_rotated = rotated.reshape(b, t, h, rot_d)
    return jnp.concatenate([x_rotated, x_pass], axis=-1)


class MultiHeadSelfAttention(nn.Module):
    """
    Causal self-attention with RoPE and optional KV cache.
    """
    config: TransformerConfig

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,                              # (batch, seq, emb)
        *,
        deterministic: bool,
        position_ids: Optional[jnp.ndarray] = None,  # (batch, seq) or (seq,)
        layer_cache: Optional[LayerCache] = None,
        decode: bool = False,
    ) -> Tuple[jnp.ndarray, Optional[LayerCache]]:
        cfg = self.config
        B, T, D = x.shape
        H = cfg.num_heads
        assert D % H == 0, "emb_dim must be divisible by num_heads"
        head_dim = D // H

        # Project once to qkv for efficiency
        qkv = nn.DenseGeneral(
            features=(H, 3 * head_dim),
            axis=-1,
            dtype=cfg.dtype,
            param_dtype=cfg.param_dtype,
            use_bias=cfg.use_bias,
            kernel_init=cfg.kernel_init,
            bias_init=cfg.bias_init,
            name="qkv",
        )(x)  # (B, T, H, 3*head_dim)
        q, k, v = jnp.split(qkv, 3, axis=-1)  # each (B, T, H, head_dim)

        # Positions for RoPE
        if position_ids is None:
            position_ids = jnp.arange(T, dtype=jnp.int32)[None, :]  # (1, T)
        cos, sin = rope_cos_sin(position_ids, head_dim, cfg.rope_theta, cfg.dtype)
        q = apply_rope(q, cos, sin, cfg.rope_fraction)
        k = apply_rope(k, cos, sin, cfg.rope_fraction)

        sm_scale: float = 1.0 / math.sqrt(head_dim)

        # Helper: reshape to (B, H, L, Dh)
        def _to_bhld(t: jnp.ndarray) -> jnp.ndarray:
            return jnp.transpose(t, (0, 2, 1, 3))

        # Prepare keys/values, maybe update/use cache
        # cache tensors are kept as (B, H, max_len, Dh)
        if decode:
            assert layer_cache is not None, "`decode=True` requires a LayerCache."
            assert T == 1, "During decoding, pass one token at a time (seq=1)."

            # Write the new k,v at current index
            k_t = _to_bhld(k)  # (B, H, 1, Dh)
            v_t = _to_bhld(v)  # (B, H, 1, Dh)
            updated_k = lax.dynamic_update_slice_in_dim(
                layer_cache.key, k_t, layer_cache.index, axis=2
            )
            updated_v = lax.dynamic_update_slice_in_dim(
                layer_cache.value, v_t, layer_cache.index, axis=2
            )
            new_index = layer_cache.index + jnp.int32(T)  # scalar

            # Build an attention-bias that masks out K positions >= new_index.
            # We cannot use `causal=True` here because q_len=1 (its index is 0),
            # so we provide the correct geometry via `ab` and set causal=False.
            K_full = updated_k.shape[2]
            pos = jnp.arange(K_full, dtype=jnp.int32)[None, None, None, :]  # (1,1,1,K)
            valid = pos <= (new_index - jnp.int32(1))
            neg = jnp.array(-1e30, dtype=jnp.float32)
            ab = jnp.where(valid, jnp.array(0.0, dtype=jnp.float32), neg)  # (B?,H?,1,K)
            # Broadcast to batch/heads lazily by relying on JAX broadcasting rules:
            ab = jnp.broadcast_to(ab, (B, cfg.num_heads, 1, K_full))

            # Flash Attention call
            q_bhqd = _to_bhld(q)  # (B, H, 1, Dh)
            y_bhqd = fa_flash_attention(
                q_bhqd,
                updated_k,
                updated_v,
                ab=ab,
                segment_ids=None,
                causal=False,          # mask geometry provided by `ab`
                sm_scale=sm_scale,
                block_sizes=BlockSizes(
                   block_q=1,         # q_len == 1
                    block_k_major=K_full,
                    block_k=K_full,    # single-step kernel path
                    block_b=1,
                ),
                debug=False,
            )  # (B, H, 1, Dh)

            y = jnp.transpose(y_bhqd, (0, 2, 1, 3)).reshape(B, T, H * head_dim)
            y = nn.Dropout(rate=cfg.attn_dropout_rate)(y, deterministic=deterministic)
            y = nn.Dense(
                features=D,
                dtype=cfg.dtype,
                param_dtype=cfg.param_dtype,
                use_bias=cfg.use_bias,
                kernel_init=cfg.kernel_init,
                bias_init=cfg.bias_init,
                name="out",
            )(y)

            new_cache = LayerCache(key=updated_k, value=updated_v, index=new_index)
            return y, new_cache

        else:
            # Standard full-sequence flash attention (with causal mask)
            q_bhqd = _to_bhld(q)               # (B, H, T, Dh)
            k_bhkd = _to_bhld(k)               # (B, H, T, Dh)
            v_bhkd = _to_bhld(v)               # (B, H, T, Dh)

            # Kernel requires K to be divisible by 128 for the tiled path (and for bwd).
            # Pad K/V to next multiple of 128; causal=True already prevents using padded tail.
            pad_k = (-T) % 128
            if pad_k:
                pad_cfg = ((0, 0), (0, 0), (0, pad_k), (0, 0))
                k_bhkd = jnp.pad(k_bhkd, pad_cfg)
                v_bhkd = jnp.pad(v_bhkd, pad_cfg)
            K_eff = k_bhkd.shape[2]  # multiple of 128

            # Backward (DKV/DQ) on TPU requires the last two block dims for the involved
            # arrays to be divisible by (8, 128) respectively OR equal to the array dims.
            # D == head_dim may be 64 (<128) so we rely on the "equal to dim" clause there.
            # For Q, pick a tile that always satisfies the first clause: 128 if possible,
            # otherwise exactly T (equal to dimension).
            q_tile = 128 if T >= 128 else T
            y_bhqd = fa_flash_attention(
                q_bhqd,
                k_bhkd,
                v_bhkd,
                ab=None,
                segment_ids=None,
                causal=True,
                sm_scale=sm_scale,
                block_sizes=BlockSizes(
                    block_q=min(128, T),       # q tiles; q_len need not be a multiple
                    block_k_major=128,         # K tiles
                    block_k=128,
                    block_b=1,
                    # Backward tile sizes (required if differentiated)
                    block_q_major_dkv=q_tile,  # major must be >= minor
                    block_k_major_dkv=128,
                    block_k_dkv=128,
                    block_q_dkv=q_tile,        # satisfies TPU constraint
                    block_k_major_dq=128,
                    block_k_dq=128,
                    block_q_dq=q_tile,         # mirror the same logic for dq kernel
                ),
                debug=False,
            )  # (B, H, T, Dh)

            y = jnp.transpose(y_bhqd, (0, 2, 1, 3)).reshape(B, T, H * head_dim)
            y = nn.Dropout(rate=cfg.attn_dropout_rate)(y, deterministic=deterministic)
            y = nn.Dense(
                features=D,
                dtype=cfg.dtype,
                param_dtype=cfg.param_dtype,
                use_bias=cfg.use_bias,
                kernel_init=cfg.kernel_init,
                bias_init=cfg.bias_init,
                name="out",
            )(y)
            return y, None


class MLP(nn.Module):
    """
    Position-wise FFN Network using SwiGLU:
      hidden = (SiLU(x @ W_gate + b_gate) * (x @ W_up + b_up))
      out    = hidden @ W_down + b_down
    """
    config: TransformerConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, deterministic: bool) -> jnp.ndarray:
        cfg = self.config

        # Up (value) projection — keep original name for checkpoint compatibility
        up = nn.Dense(
            features=cfg.mlp_hidden_dim,
            dtype=cfg.dtype,
            param_dtype=cfg.param_dtype,
            use_bias=cfg.use_bias,
            kernel_init=cfg.kernel_init,
            bias_init=cfg.bias_init,
            name="fc1",
        )(x)

        # Gate projection (new parameter set)
        gate = nn.Dense(
            features=cfg.mlp_hidden_dim,
            dtype=cfg.dtype,
            param_dtype=cfg.param_dtype,
            use_bias=cfg.use_bias,
            kernel_init=cfg.kernel_init,
            bias_init=cfg.bias_init,
            name="fc_gate",
        )(x)

        # SwiGLU activation: SiLU(gate) * up
        hidden = jax.nn.silu(gate) * up

        hidden = nn.Dropout(rate=cfg.dropout_rate)(hidden, deterministic=deterministic)

        # Down projection — keep original name for checkpoint compatibility
        out = nn.Dense(
            features=cfg.emb_dim,
            dtype=cfg.dtype,
            param_dtype=cfg.param_dtype,
            use_bias=cfg.use_bias,
            kernel_init=cfg.kernel_init,
            bias_init=cfg.bias_init,
            name="fc2",
        )(hidden)
        out = nn.Dropout(rate=cfg.dropout_rate)(out, deterministic=deterministic)
        return out


class DecoderBlock(nn.Module):
    """
    Transformer decoder block (pre-norm)s
    """
    config: TransformerConfig
    layer_idx: int

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        *,
        deterministic: bool,
        position_ids: Optional[jnp.ndarray] = None,
        layer_cache: Optional[LayerCache] = None,
        decode: bool = False,
    ) -> Tuple[jnp.ndarray, Optional[LayerCache]]:
        cfg = self.config

        # Self-attention (pre-norm)
        y = RMSNorm(cfg.emb_dim, eps=cfg.norm_epsilon, dtype=cfg.dtype, param_dtype=cfg.param_dtype, name="rms_1")(x)
        y, new_cache = MultiHeadSelfAttention(cfg, name="self_attn")(
            y,
            deterministic=deterministic,
            position_ids=position_ids,
            layer_cache=layer_cache,
            decode=decode,
        )
        x = x + y  # residual

        # MLP (pre-norm)
        z = RMSNorm(cfg.emb_dim, eps=cfg.norm_epsilon, dtype=cfg.dtype, param_dtype=cfg.param_dtype, name="rms_2")(x)
        z = MLP(cfg, name="mlp")(z, deterministic=deterministic)
        x = x + z  # residual
        return x, new_cache


class TransformerLM(nn.Module):
    """
    Decoder-only Transformer language model with RoPE, RMSNorm, and KV cache.

    Methods:
      - __call__(token_ids): logits for all positions (training/inference, no cache)
      - init_kv_cache(batch_size): allocate zero-initialized KV cache
      - decode_step(token_ids, cache): single autoregressive step that updates cache
    """
    config: TransformerConfig

    def setup(self) -> None:
        cfg = self.config
        self.token_embed = nn.Embed(
            num_embeddings=cfg.vocab_size,
            features=cfg.emb_dim,
            embedding_init=nn.initializers.normal(stddev=1.0),
            dtype=cfg.dtype,
            param_dtype=cfg.param_dtype,
            name="token_embed",
        )
        self.blocks = [
            DecoderBlock(cfg, layer_idx=i, name=f"block_{i}") for i in range(cfg.num_layers)
        ]
        self.final_norm = RMSNorm(
            cfg.emb_dim, eps=cfg.norm_epsilon, dtype=cfg.dtype, param_dtype=cfg.param_dtype, name="rms_final"
        )

    # Forward (no cache)
    @nn.compact
    def __call__(
        self,
        token_ids: jnp.ndarray,  # (batch, seq)
        *,
        deterministic: bool = True,
        position_ids: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Full forward pass (no KV cache). Returns logits (batch, seq, vocab).
        """
        cfg = self.config
        B, T = token_ids.shape

        # Embedding
        x = self.token_embed(token_ids)  # (B, T, D)
        x = nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=deterministic)

        # Default positions if not provided
        if position_ids is None:
            position_ids = jnp.arange(T, dtype=jnp.int32)[None, :]

        # Transformer blocks
        cache_placeholder = None
        for blk in self.blocks:
            x, _ = blk(
                x,
                deterministic=deterministic,
                position_ids=position_ids,
                layer_cache=cache_placeholder,
                decode=False,
            )

        # Final norm + tied output projection
        x = self.final_norm(x)
        # Weight tying: logits = x @ E^T
        E = self.variables["params"]["token_embed"]["embedding"]  # (vocab, D)
        logits = jnp.einsum("btd,vd->btv", x.astype(cfg.logits_dtype), E.astype(cfg.logits_dtype))
        return logits.astype(cfg.logits_dtype)
    
    # KV cache helpers for decoding
    def init_kv_cache(self, batch_size: int) -> KVCache:
        """
        Allocate zero-initialized KV caches for all layers.
        """
        cfg = self.config
        head_dim = cfg.emb_dim // cfg.num_heads
        k_shape = (batch_size, cfg.num_heads, cfg.max_seq_len, head_dim)
        v_shape = (batch_size, cfg.num_heads, cfg.max_seq_len, head_dim)

        def new_layer_cache():
            return LayerCache(
                key=jnp.zeros(k_shape, dtype=cfg.dtype),
                value=jnp.zeros(v_shape, dtype=cfg.dtype),
                index=jnp.int32(0),
            )

        layers = tuple(new_layer_cache() for _ in range(cfg.num_layers))
        return KVCache(layers=layers)

    @nn.nowrap
    def _one_step_core(
        self,
        x: jnp.ndarray,  # (B, 1, D) embedded current token
        position_ids: jnp.ndarray,  # (B, 1)
        caches: KVCache,
        *,
        deterministic: bool,
    ) -> Tuple[jnp.ndarray, KVCache]:
        """
        Pass through blocks for a single time step with KV cache.
        Returns:
          h: hidden (B, 1, D)
          new_caches: updated KVCache (index advanced by 1)
        """
        new_layers = []
        h = x
        for i, blk in enumerate(self.blocks):
            h, new_cache = blk(
                h,
                deterministic=deterministic,
                position_ids=position_ids,
                layer_cache=caches.layers[i],
                decode=True,
            )
            assert new_cache is not None
            new_layers.append(new_cache)
        h = self.final_norm(h)
        return h, KVCache(layers=tuple(new_layers))

    @nn.compact
    def decode_step(
        self,
        token_ids: jnp.ndarray,  # (batch, 1)
        caches: KVCache,
        *,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray, KVCache]:
        """
        Single autoregressive step:
          - embeds the current token(s)
          - updates KV caches through all blocks
          - returns logits for the next token at the last position
        """
        cfg = self.config
        B, T = token_ids.shape
        assert T == 1, "decode_step expects a single token per batch."

        # Position is the same across layers: use caches.layers[0].index
        step_pos = caches.layers[0].index  # scalar
        position_ids = jnp.full((B, 1), step_pos, dtype=jnp.int32)

        # Embed current tokens
        x = self.token_embed(token_ids)  # (B, 1, D)
        x = nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=deterministic)

        # One-step pass through the stack
        h, new_caches = self._one_step_core(
            x, position_ids, caches, deterministic=deterministic
        )

        # Tied projection to logits
        E = self.variables["params"]["token_embed"]["embedding"]  # (vocab, D)
        logits = jnp.einsum("btd,vd->btv", h.astype(cfg.logits_dtype), E.astype(cfg.logits_dtype))
        return logits.astype(cfg.logits_dtype), new_caches


def init_model_and_params(
    rng: jax.random.PRNGKey,
    config: TransformerConfig,
    batch_size: int,
    seq_len: int,
) -> Tuple[TransformerLM, FrozenDict]:
    """
    Initialize model and parameters with a dummy batch.
    """
    model = TransformerLM(config)
    dummy_tokens = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
    variables = model.init(rng, dummy_tokens, deterministic=True)
    params = variables["params"]
    return model, params


def jit_forward_apply(
    model: TransformerLM,
    params: FrozenDict,
) -> Callable[[jnp.ndarray, bool], jnp.ndarray]:
    """
    Returns a jitted function for full forward (no cache).
    """

    @partial(jax.jit, static_argnames=("deterministic",))
    def _apply(tokens: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        return model.apply({"params": params}, tokens, deterministic=deterministic)

    return _apply


def jit_greedy_generate(
    model: TransformerLM,
    params: FrozenDict,
    max_new_tokens: int,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Returns a JIT-compiled greedy generator (pure JAX).
    The prompt is consumed token-by-token, updating KV cache.
    """
    
    @partial(jax.jit, static_argnames=("prompt_len",))
    def _generate(prompt: jnp.ndarray, prompt_len: int) -> jnp.ndarray:
        """
        Args:
            prompt: (B, L0) initial tokens (L0 <= config.max_seq_len)
            prompt_len: length of prompt L0
        Returns:
            tokens: (B, L0 + max_new_tokens)
        """
        B, L0 = prompt.shape
        cfg = model.config
        assert L0 <= cfg.max_seq_len, "Prompt length must be <= max_seq_len."

        # Initialize caches and prefill with the entire prompt:
        caches = model.apply({"params": params}, B, method=TransformerLM.init_kv_cache)

        # Prefill loop over the prompt (JIT-friendly scan).
        # IMPORTANT to myself: do not pass a Python list (from jnp.split) to lax.scan, what the fuck are you doing?
        # Make the data time-major so xs is a single JAX array, not a list.
        def _prefill_step(carry, token_t):
            caches_t = carry
            logits, caches_t = model.apply(
                {"params": params},
                token_t,
                method=TransformerLM.decode_step,
                deterministic=True,
                caches=caches_t,
            )
            return caches_t, logits

        init_caches = caches
        init_rng = jax.random.PRNGKey(0)
        prompt_tokens = prompt

        # Step through prompt to build cache.
        # time_major_tokens: (L0, B, 1) so each scan step gets a (B, 1) token slice.
        time_major_tokens = jnp.swapaxes(prompt_tokens, 0, 1)[..., None]  # (L0, B, 1)
        init_caches, _ = lax.scan(_prefill_step, init_caches, time_major_tokens)

        # Run greedy decoding for max_new_tokens with fixed-size carry.
        # Preallocate output buffer and keep only the last token in the carry.
        total_len = L0 + max_new_tokens
        out_tokens = jnp.pad(prompt_tokens, ((0, 0), (0, max_new_tokens)))  # (B, total_len)

        def _body_fn(carry, _unused_t):
            last_tok, caches_t, step_idx, rng, tokens_acc = carry
            logits, new_caches = model.apply(
                {"params": params},
                last_tok,  # (B, 1)
                method=TransformerLM.decode_step,
                deterministic=True,
                caches=caches_t,
            )
            next_token = jnp.argmax(logits[:, -1, :], axis=-1).astype(jnp.int32)  # (B,)
            next_token = next_token[:, None]  # (B, 1)
            write_pos = L0 + step_idx  # scalar
            tokens_acc = lax.dynamic_update_slice_in_dim(tokens_acc, next_token, write_pos, axis=1)
            return (next_token, new_caches, step_idx + jnp.int32(1), rng, tokens_acc), next_token

        # Initial carry: last prompt token, caches, step index, rng, and accumulator.
        init_carry = (prompt_tokens[:, -1:], init_caches, jnp.int32(0), init_rng, out_tokens)
        (final_last_tok, final_caches, _final_step, _final_rng, final_tokens), _ = lax.scan(
            _body_fn, init_carry, xs=jnp.arange(max_new_tokens)
        )
        return final_tokens

    return _generate
