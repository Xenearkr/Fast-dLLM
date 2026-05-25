from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import nn
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
except Exception:
    flex_attention = None
    create_block_mask = None

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
try:
    from transformers.integrations import (
        use_kernel_forward_from_hub,
        use_kernel_func_from_hub,
        use_kernelized_func,
    )
except ImportError:
    try:
        from transformers.integrations import use_kernel_forward_from_hub
    except ImportError:
        def use_kernel_forward_from_hub(*args, **kwargs):
            def decorator(obj):
                return obj
            return decorator

    def use_kernel_func_from_hub(*args, **kwargs):
        def decorator(obj):
            return obj
        return decorator

    def use_kernelized_func(*args, **kwargs):
        def decorator(obj):
            return obj
        return decorator
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
try:
    from transformers.modeling_layers import (
        GenericForQuestionAnswering,
        GenericForSequenceClassification,
        GenericForTokenClassification,
        GradientCheckpointingLayer,
    )
except ImportError:
    from transformers.modeling_layers import GradientCheckpointingLayer

    class GenericForSequenceClassification:
        pass

    class GenericForTokenClassification:
        pass

    class GenericForQuestionAnswering:
        pass
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
try:
    from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple
except ImportError:
    try:
        from typing_extensions import TypedDict
    except ImportError:
        from typing import TypedDict

    class TransformersKwargs(TypedDict, total=False):
        pass

    def auto_docstring(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        def decorator(obj):
            return obj
        return decorator

    def can_return_tuple(func=None, *args, **kwargs):
        if callable(func):
            return func
        def decorator(f):
            return f
        return decorator
try:
    from transformers.utils.generic import maybe_autocast, merge_with_config_defaults
except ImportError:
    from contextlib import nullcontext

    def maybe_autocast(device_type=None, enabled=True):
        if enabled:
            return torch.autocast(device_type=device_type)
        return nullcontext()

    def merge_with_config_defaults(func=None, *args, **kwargs):
        if callable(func):
            return func
        def decorator(f):
            return f
        return decorator
try:
    from transformers.utils.output_capturing import capture_outputs
except ImportError:
    def capture_outputs(func=None, *args, **kwargs):
        if callable(func):
            return func
        def decorator(f):
            return f
        return decorator
try:
    from .configuration import Fast_dLLM_Qwen3Config
except ImportError:
    from configuration import Fast_dLLM_Qwen3Config


# ---------------------------------------------------------------------
# Fast-dLLM v2 block-diffusion utilities.
# Added in Milestone 4B. These utilities are not connected to forward()
# until later milestones.
# ---------------------------------------------------------------------

@dataclass
class CausalLMOutputWithPastAndBlockCache(CausalLMOutputWithPast):
    block_past_key_values: Optional[Cache] = None


@dataclass
class BaseModelOutputWithPastAndBlockCache(BaseModelOutputWithPast):
    block_past_key_values: Optional[Cache] = None


def fused_flex_attention(q, k, v, mask=None):
    """
    Thin wrapper around torch flex_attention used by Fast-dLLM training.

    The actual forward path is not connected in Milestone 4B. If the current
    PyTorch build does not provide flex_attention, this function raises a clear
    error when called.
    """
    if flex_attention is None:
        raise RuntimeError(
            "torch.nn.attention.flex_attention is unavailable in this environment. "
            "Install a PyTorch version with flex_attention support before enabling "
            "Fast-dLLM training attention."
        )
    return flex_attention(q, k, v, block_mask=mask, enable_gqa=True)


def block_diff_mask(b, h, q_idx, kv_idx, block_size=None, n=None):
    """
    Construct the Fast-dLLM block-diffusion attention mask for training.

    The concatenated training sequence is [x_t ; x_0], where each half has
    length n. The mask is composed of:

    1. Block Diagonal Mask:
       tokens attend bidirectionally within the same block of the same half.

    2. Offset Block-Causal Mask:
       noised tokens x_t^b attend to previous clean blocks x_0^{<b}.

    3. Block-Causal Mask:
       clean tokens x_0^b attend to clean blocks x_0^{<=b}.

    Args:
        b, h:
            Batch and head indices. They are ignored by the logical mask but
            kept for create_block_mask compatibility.
        q_idx:
            Query indices.
        kv_idx:
            Key/value indices.
        block_size:
            Block size D.
        n:
            Length of each half sequence. Total concatenated length is 2n.

    Returns:
        Boolean tensor indicating allowed attention positions.
    """
    if block_size is None:
        raise ValueError("block_size must be provided.")
    if n is None:
        raise ValueError("n must be provided.")

    x0_flag_q = q_idx >= n
    x0_flag_kv = kv_idx >= n

    block_q = torch.where(
        x0_flag_q,
        (q_idx - n) // block_size,
        q_idx // block_size,
    )
    block_kv = torch.where(
        x0_flag_kv,
        (kv_idx - n) // block_size,
        kv_idx // block_size,
    )

    # M_BD: within-block bidirectional attention inside the same half.
    block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)

    # M_OBC: noised x_t block attends to previous clean x_0 blocks.
    offset_block_causal = (
        (block_q > block_kv)
        & (x0_flag_kv == 1)
        & (x0_flag_q == 0)
    )

    # M_BC: clean x_0 block attends to current and previous clean blocks.
    block_causal = (
        (block_q >= block_kv)
        & (x0_flag_kv == 1)
        & (x0_flag_q == 1)
    )

    return block_diagonal | offset_block_causal | block_causal


def eval_block_diff_mask(q_idx, kv_idx, block_size=None):
    """
    Construct the block-causal mask used by Fast-dLLM inference.

    This mask is block-level causal: tokens in a query block may attend to
    key/value tokens from the same block and all earlier blocks.

    Args:
        q_idx:
            Query indices.
        kv_idx:
            Key/value indices.
        block_size:
            Block size D.

    Returns:
        Boolean tensor indicating allowed attention positions.
    """
    if block_size is None:
        raise ValueError("block_size must be provided.")

    block_q = q_idx // block_size
    block_kv = kv_idx // block_size
    return block_q >= block_kv



# ---------------------------------------------------------------------
# Fast-dLLM v2 cache / mask compatibility helpers.
# Added in Milestone 4C.
# ---------------------------------------------------------------------

def _new_dynamic_cache_compat(config=None):
    try:
        return DynamicCache(config=config)
    except TypeError:
        return DynamicCache()


def _cache_get_seq_length_compat(cache, layer_idx=None):
    if cache is None:
        return 0

    if layer_idx is not None:
        try:
            if len(cache) <= layer_idx:
                return 0
            layer_cache = cache[layer_idx]
            if layer_cache is None or layer_cache[0] is None:
                return 0
            return layer_cache[0].shape[-2]
        except Exception:
            pass

    try:
        return cache.get_seq_length()
    except Exception:
        return 0


def _cache_has_layer_compat(cache, layer_idx):
    if cache is None:
        return False

    try:
        if len(cache) <= layer_idx:
            return False
        layer_cache = cache[layer_idx]
        return layer_cache is not None and layer_cache[0] is not None
    except Exception:
        return _cache_get_seq_length_compat(cache, layer_idx=layer_idx) > 0


def _cache_update_compat(cache, key_states, value_states, layer_idx, cache_kwargs=None):
    if cache_kwargs is None:
        cache_kwargs = {}

    try:
        return cache.update(key_states, value_states, layer_idx, cache_kwargs)
    except TypeError:
        return cache.update(key_states, value_states, layer_idx)


def _allowed_bool_mask_to_additive(allowed_mask, dtype, device, batch_size=None):
    """
    Convert a boolean allow-mask to an additive attention mask.

    allowed_mask:
        True means attention is allowed.
        False means attention is blocked.

    Returns:
        Float additive mask where blocked positions are set to finfo(dtype).min.
    """
    if not isinstance(allowed_mask, torch.Tensor):
        return allowed_mask

    if allowed_mask.dtype != torch.bool:
        return allowed_mask.to(device=device, dtype=dtype)

    mask = allowed_mask.to(device=device)

    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(1)
    elif mask.dim() == 4:
        pass
    else:
        raise ValueError(f"Unsupported boolean attention mask rank: {mask.dim()}")

    if batch_size is not None and mask.shape[0] == 1 and batch_size > 1:
        mask = mask.expand(batch_size, -1, -1, -1)

    additive = torch.zeros(mask.shape, dtype=dtype, device=device)
    additive = additive.masked_fill(~mask, torch.finfo(dtype).min)
    return additive


def _is_block_training_layout(inputs_embeds, labels):
    """
    Fast-dLLM training layout is [x_t ; x_0].
    In 4C we only detect and support this layout; 4D will construct it.
    """
    return (
        labels is not None
        and inputs_embeds is not None
        and inputs_embeds.shape[1] == labels.shape[1] * 2
    )






@use_kernel_forward_from_hub("RMSNorm")
class Fast_dLLM_Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        Fast_dLLM_Qwen3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Fast_dLLM_Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj



class Fast_dLLM_Qwen3RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config: Fast_dLLM_Qwen3Config, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config

        rope_parameters = getattr(config, "rope_parameters", None)
        rope_scaling = getattr(config, "rope_scaling", None)

        if isinstance(rope_parameters, dict):
            self.rope_type = rope_parameters.get("rope_type", rope_parameters.get("type", "default"))
        elif rope_parameters is not None:
            self.rope_type = getattr(rope_parameters, "rope_type", "default")
        elif isinstance(rope_scaling, dict):
            self.rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))
        else:
            self.rope_type = "default"

        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_default_rope_parameters(
        config: Optional[Fast_dLLM_Qwen3Config] = None,
        device: Optional["torch.device"] = None,
        seq_len: Optional[int] = None,
    ) -> tuple["torch.Tensor", float]:
        if config is None:
            raise ValueError("config must be provided for RoPE parameter initialization.")

        rope_parameters = getattr(config, "rope_parameters", None)
        if isinstance(rope_parameters, dict) and "rope_theta" in rope_parameters:
            base = rope_parameters["rope_theta"]
        else:
            base = getattr(config, "rope_theta", 1000000.0)

        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        attention_factor = 1.0

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@use_kernel_func_from_hub("rotary_pos_emb")
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    scaling: Optional[float] = None,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    if scaling is None:
        scaling = module.scaling

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights



def _build_4d_causal_mask_compat(
    inputs_embeds,
    attention_mask=None,
    past_key_values=None,
    position_ids=None,
):
    """
    Compatibility fallback for older Transformers versions whose
    create_causal_mask() signature does not accept Qwen3's newer kwargs.

    Returns an additive 4D causal mask of shape [batch, 1, q_len, kv_len].
    """
    batch_size, q_len, _ = inputs_embeds.shape
    device = inputs_embeds.device
    dtype = inputs_embeds.dtype

    min_dtype = torch.finfo(dtype).min
    past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
    kv_len = past_seen_tokens + q_len

    if position_ids is None:
        query_positions = torch.arange(
            past_seen_tokens,
            past_seen_tokens + q_len,
            device=device,
            dtype=torch.long,
        ).unsqueeze(0).expand(batch_size, -1)
    else:
        query_positions = position_ids.to(device=device)

        if query_positions.dim() == 1:
            query_positions = query_positions.unsqueeze(0)

        if query_positions.shape[0] == 1 and batch_size > 1:
            query_positions = query_positions.expand(batch_size, -1)

        if query_positions.shape[0] != batch_size:
            raise ValueError(
                f"position_ids batch dimension mismatch: "
                f"position_ids.shape={tuple(query_positions.shape)}, batch_size={batch_size}"
            )

        if query_positions.shape[1] != q_len:
            raise ValueError(
                f"position_ids sequence length mismatch: "
                f"position_ids.shape={tuple(query_positions.shape)}, q_len={q_len}"
            )

    key_positions = torch.arange(kv_len, device=device, dtype=torch.long)

    causal_mask = key_positions.view(1, 1, 1, kv_len) > query_positions.view(batch_size, 1, q_len, 1)
    additive_mask = torch.zeros(
        (batch_size, 1, q_len, kv_len),
        dtype=dtype,
        device=device,
    )
    additive_mask = additive_mask.masked_fill(causal_mask, min_dtype)

    if attention_mask is not None:
        if isinstance(attention_mask, dict):
            return attention_mask

        if attention_mask.dim() == 4:
            return attention_mask

        if attention_mask.dim() == 2:
            key_attention_mask = attention_mask.to(device=device)

            if key_attention_mask.shape[-1] < kv_len:
                prefix_len = kv_len - key_attention_mask.shape[-1]
                prefix = torch.ones(
                    (batch_size, prefix_len),
                    dtype=key_attention_mask.dtype,
                    device=device,
                )
                key_attention_mask = torch.cat([prefix, key_attention_mask], dim=-1)
            elif key_attention_mask.shape[-1] > kv_len:
                key_attention_mask = key_attention_mask[:, -kv_len:]

            padding_mask = key_attention_mask[:, None, None, :] == 0
            additive_mask = additive_mask.masked_fill(padding_mask, min_dtype)

    return additive_mask


def _create_causal_mask_mapping_compat(
    config,
    inputs_embeds,
    attention_mask,
    past_key_values,
    position_ids,
    has_sliding_layers=False,
):
    """
    Try the installed Transformers mask utility first. If the installed
    version is older and rejects Qwen3-style kwargs, fall back to a local
    additive causal mask.
    """
    mask_kwargs = {
        "config": config,
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "position_ids": position_ids,
    }

    try:
        full_attention_mask = create_causal_mask(**mask_kwargs)
    except TypeError:
        full_attention_mask = _build_4d_causal_mask_compat(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

    causal_mask_mapping = {
        "full_attention": full_attention_mask,
    }

    if has_sliding_layers:
        try:
            causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
        except TypeError:
            # Qwen3-8B currently has no sliding layers in our config. If a
            # sliding config is used later, implement a real sliding fallback.
            causal_mask_mapping["sliding_attention"] = full_attention_mask

    return causal_mask_mapping



@use_kernelized_func(apply_rotary_pos_emb)
class Fast_dLLM_Qwen3Attention(nn.Module):
    """Qwen3 attention with Fast-dLLM block-cache hooks."""

    def __init__(self, config: Fast_dLLM_Qwen3Config, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

        # Qwen3-specific QK normalization. This must be preserved for checkpoint compatibility.
        self.q_norm = Fast_dLLM_Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Fast_dLLM_Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        update_past_key_values: Optional[bool] = False,
        use_block_cache: Optional[bool] = False,
        block_past_key_values: Optional[Cache] = None,
        replace_position: Optional[int] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings

        # In Fast-dLLM training, the model sees [x_t ; x_0].
        # position_embeddings are built for length n, while q/k length is 2n.
        if self.training and query_states.shape[-2] == cos.shape[1] * 2:
            half = query_states.shape[-2] // 2

            q_1 = query_states[:, :, :half]
            q_2 = query_states[:, :, half:]
            k_1 = key_states[:, :, :half]
            k_2 = key_states[:, :, half:]

            q_1, k_1 = apply_rotary_pos_emb(q_1, k_1, cos, sin)
            q_2, k_2 = apply_rotary_pos_emb(q_2, k_2, cos, sin)

            query_states = torch.cat([q_1, q_2], dim=-2)
            key_states = torch.cat([k_1, k_2], dim=-2)
        else:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

        # Current-block/sub-block cache. This is used for intra-block refinement.
        if use_block_cache and block_past_key_values is not None:
            if not _cache_has_layer_compat(block_past_key_values, self.layer_idx):
                key_states, value_states = _cache_update_compat(
                    block_past_key_values,
                    key_states,
                    value_states,
                    self.layer_idx,
                    cache_kwargs,
                )
            else:
                block_cache_key_states = block_past_key_values[self.layer_idx][0]
                block_cache_value_states = block_past_key_values[self.layer_idx][1]

                if replace_position is not None:
                    start = replace_position
                    end = replace_position + key_states.shape[-2]
                    block_cache_key_states[:, :, start:end] = key_states
                    block_cache_value_states[:, :, start:end] = value_states

                key_states = block_cache_key_states
                value_states = block_cache_value_states

        # Prefix cache. update_past_key_values=True appends the current block.
        # update_past_key_values=False reuses the prefix as read-only context.
        if past_key_values is not None:
            if update_past_key_values:
                key_states, value_states = _cache_update_compat(
                    past_key_values,
                    key_states,
                    value_states,
                    self.layer_idx,
                    cache_kwargs,
                )
            elif _cache_has_layer_compat(past_key_values, self.layer_idx):
                prefix_key_states = past_key_values[self.layer_idx][0]
                prefix_value_states = past_key_values[self.layer_idx][1]
                key_states = torch.cat([prefix_key_states, key_states], dim=-2)
                value_states = torch.cat([prefix_value_states, value_states], dim=-2)

        attn_weights = None

        # Dense boolean masks mean "allowed". Convert them to additive masks
        # for eager / SDPA-style attention.
        if isinstance(attention_mask, torch.Tensor) and attention_mask.dtype == torch.bool:
            attention_mask = _allowed_bool_mask_to_additive(
                attention_mask,
                dtype=query_states.dtype,
                device=query_states.device,
                batch_size=query_states.shape[0],
            )

        # flex_attention BlockMask path. This is expected for real Fast-dLLM training.
        if self.training and attention_mask is not None and not isinstance(attention_mask, torch.Tensor):
            attn_output = fused_flex_attention(query_states, key_states, value_states, mask=attention_mask)
            attn_output = attn_output.transpose(1, 2).contiguous()
        else:
            attn_implementation = getattr(self.config, "_attn_implementation", "eager")

            if hasattr(ALL_ATTENTION_FUNCTIONS, "get_interface"):
                attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
                    attn_implementation, eager_attention_forward
                )
            else:
                try:
                    attention_interface: Callable = ALL_ATTENTION_FUNCTIONS[attn_implementation]
                except Exception:
                    try:
                        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get(
                            attn_implementation, eager_attention_forward
                        )
                    except Exception:
                        attention_interface: Callable = eager_attention_forward

            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,
                **kwargs,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Fast_dLLM_Qwen3DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Fast_dLLM_Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Fast_dLLM_Qwen3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Fast_dLLM_Qwen3MLP(config)
        self.input_layernorm = Fast_dLLM_Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Fast_dLLM_Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        update_past_key_values: Optional[bool] = False,
        use_block_cache: Optional[bool] = False,
        block_past_key_values: Optional[Cache] = None,
        replace_position: Optional[int] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            update_past_key_values=update_past_key_values,
            use_block_cache=use_block_cache,
            block_past_key_values=block_past_key_values,
            replace_position=replace_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


@auto_docstring
class Fast_dLLM_Qwen3PreTrainedModel(PreTrainedModel):
    config: Fast_dLLM_Qwen3Config
    config_class = Fast_dLLM_Qwen3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Fast_dLLM_Qwen3DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Fast_dLLM_Qwen3DecoderLayer,
        "attentions": Fast_dLLM_Qwen3Attention,
    }


@auto_docstring
class Fast_dLLM_Qwen3Model(Fast_dLLM_Qwen3PreTrainedModel):
    def __init__(self, config: Fast_dLLM_Qwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.bd_size = config.bd_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Fast_dLLM_Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Fast_dLLM_Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Fast_dLLM_Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def eval_mask(self, seqlen, block_size, cache_seq_len, device=None):
        q_indices = torch.arange(seqlen, device=device) + cache_seq_len
        kv_indices = torch.arange(seqlen + cache_seq_len, device=device)
        return eval_block_diff_mask(
            q_idx=q_indices[:, None],
            kv_idx=kv_indices[None, :],
            block_size=block_size,
        )

    def gen_mask(self, seqlen, block_size, B, H, device=None):
        # Use flex_attention BlockMask only on CUDA. On CPU smoke tests,
        # flex_attention either is unavailable or may create a CUDA BlockMask
        # by default, causing q/k/v and block_mask device mismatch.
        block_mask_device = torch.device(device) if device is not None else torch.device("cpu")

        if create_block_mask is not None and block_mask_device.type == "cuda":
            try:
                return create_block_mask(
                    lambda b, h, q_idx, kv_idx: block_diff_mask(
                        b=b,
                        h=h,
                        q_idx=q_idx,
                        kv_idx=kv_idx,
                        block_size=block_size,
                        n=seqlen,
                    ),
                    B=B,
                    H=H,
                    Q_LEN=seqlen * 2,
                    KV_LEN=seqlen * 2,
                    device=device,
                )
            except TypeError:
                return create_block_mask(
                    lambda b, h, q_idx, kv_idx: block_diff_mask(
                        b=b,
                        h=h,
                        q_idx=q_idx,
                        kv_idx=kv_idx,
                        block_size=block_size,
                        n=seqlen,
                    ),
                    B=B,
                    H=H,
                    Q_LEN=seqlen * 2,
                    KV_LEN=seqlen * 2,
                )

        q_idx = torch.arange(seqlen * 2, device=device)[:, None]
        kv_idx = torch.arange(seqlen * 2, device=device)[None, :]
        return block_diff_mask(
            b=None,
            h=None,
            q_idx=q_idx,
            kv_idx=kv_idx,
            block_size=block_size,
            n=seqlen,
        )

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        update_past_key_values: Optional[bool] = False,
        block_size: Optional[int] = 32,
        use_block_cache: Optional[bool] = False,
        block_past_key_values: Optional[Cache] = None,
        replace_position: Optional[int] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPastAndBlockCache:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache is None:
            use_cache = self.config.use_cache

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = _new_dynamic_cache_compat(config=self.config)

        if use_block_cache and block_past_key_values is None:
            block_past_key_values = _new_dynamic_cache_compat(config=self.config)

        is_block_training = _is_block_training_layout(inputs_embeds, labels)

        if cache_position is None:
            past_seen_tokens = _cache_get_seq_length_compat(past_key_values)

            if is_block_training:
                effective_seq_len = labels.shape[1]
                cache_position = torch.arange(
                    past_seen_tokens,
                    past_seen_tokens + effective_seq_len,
                    device=inputs_embeds.device,
                )
            elif use_block_cache:
                block_start_position = past_seen_tokens + (replace_position if replace_position is not None else 0)
                cache_position = torch.arange(
                    block_start_position,
                    block_start_position + inputs_embeds.shape[1],
                    device=inputs_embeds.device,
                )
            else:
                cache_position = torch.arange(
                    past_seen_tokens,
                    past_seen_tokens + inputs_embeds.shape[1],
                    device=inputs_embeds.device,
                )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if is_block_training:
            causal_mask_mapping = {
                "full_attention": self.gen_mask(
                    seqlen=labels.shape[1],
                    block_size=self.bd_size,
                    B=labels.shape[0],
                    H=self.config.num_attention_heads,
                    device=inputs_embeds.device,
                )
            }
        else:
            causal_mask_mapping = attention_mask
            if not isinstance(causal_mask_mapping, dict):
                if use_block_cache and _cache_get_seq_length_compat(block_past_key_values) > 0:
                    causal_mask_mapping = {"full_attention": None}
                else:
                    cache_seq_len = _cache_get_seq_length_compat(past_key_values)
                    allowed_mask = self.eval_mask(
                        seqlen=inputs_embeds.shape[1],
                        block_size=block_size,
                        cache_seq_len=cache_seq_len,
                        device=inputs_embeds.device,
                    )
                    causal_mask_mapping = {
                        "full_attention": _allowed_bool_mask_to_additive(
                            allowed_mask,
                            dtype=inputs_embeds.dtype,
                            device=inputs_embeds.device,
                            batch_size=inputs_embeds.shape[0],
                        )
                    }

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                update_past_key_values=update_past_key_values,
                use_block_cache=use_block_cache,
                block_past_key_values=block_past_key_values,
                replace_position=replace_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPastAndBlockCache(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            block_past_key_values=block_past_key_values if use_block_cache else None,
        )


@auto_docstring
class Fast_dLLM_Qwen3ForCausalLM(Fast_dLLM_Qwen3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_rep"}
    _sp_plan = {"lm_head": "colwise"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    _fsdp_plan = {"lm_head": "keep_full_weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = Fast_dLLM_Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def _prepare_block_diffusion_training_batch(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        mask_id: Optional[int] = None,
    ):
        if input_ids is None:
            raise ValueError("Fast-dLLM training requires input_ids, not only inputs_embeds.")
        if labels is None:
            raise ValueError("Fast-dLLM training requires labels.")

        if input_ids.shape != labels.shape:
            raise ValueError(
                f"Fast-dLLM training expects input_ids and labels to have the same shape, "
                f"got input_ids={tuple(input_ids.shape)}, labels={tuple(labels.shape)}."
            )

        bd_size = self.model.bd_size
        if input_ids.shape[1] % bd_size != 0:
            raise ValueError(
                f"Sequence length must be divisible by bd_size during Fast-dLLM training: "
                f"seq_len={input_ids.shape[1]}, bd_size={bd_size}."
            )

        if mask_id is None:
            mask_id = getattr(self.config, "mask_token_id", None)
        if mask_id is None:
            raise ValueError("mask_token_id must be set in config for Fast-dLLM training.")

        original_input_ids = input_ids
        original_labels = labels

        batch_size, seq_len = input_ids.shape
        flat_input_ids = input_ids.reshape(batch_size * seq_len // bd_size, bd_size)
        num_blocks, block_len = flat_input_ids.shape

        t = torch.rand((num_blocks,), device=input_ids.device)
        eps = 1e-3
        p_mask = ((1.0 - eps) * t + eps).unsqueeze(1).expand(num_blocks, block_len)

        mask_indices = torch.rand((num_blocks, block_len), device=input_ids.device) < p_mask
        mask_token_block = torch.full_like(flat_input_ids, mask_id)

        x_t = torch.where(mask_indices, mask_token_block, flat_input_ids).reshape_as(input_ids)

        trainable_positions = original_labels != -100

        noisy_input_ids = original_input_ids.clone()
        noisy_input_ids[trainable_positions] = x_t[trainable_positions]

        masked_labels = original_labels.clone()
        masked_labels[noisy_input_ids != mask_id] = -100

        clean_input_ids = original_input_ids.clone()
        model_input_ids = torch.cat([noisy_input_ids, clean_input_ids], dim=1)

        complementary_enabled = bool(
            getattr(self.config, "complementary_mask", getattr(self.config, "conplemenrary_mask", True))
        )

        if complementary_enabled:
            complementary_mask_indices = ~mask_indices
            complementary_x_t = torch.where(
                complementary_mask_indices,
                mask_token_block,
                flat_input_ids,
            ).reshape_as(input_ids)

            complementary_noisy_input_ids = original_input_ids.clone()
            complementary_noisy_input_ids[trainable_positions] = complementary_x_t[trainable_positions]

            complementary_labels = original_labels.clone()
            complementary_labels[complementary_noisy_input_ids != mask_id] = -100

            complementary_model_input_ids = torch.cat(
                [complementary_noisy_input_ids, clean_input_ids],
                dim=1,
            )

            model_input_ids = torch.cat([model_input_ids, complementary_model_input_ids], dim=0)
            masked_labels = torch.cat([masked_labels, complementary_labels], dim=0)

        return model_input_ids, masked_labels

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        update_past_key_values: Optional[bool] = False,
        block_size: Optional[int] = 32,
        use_block_cache: Optional[bool] = False,
        block_past_key_values: Optional[Cache] = None,
        replace_position: Optional[int] = None,
        mask_id: Optional[int] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPastAndBlockCache:
        block_diffusion_training = False

        if self.training and labels is not None:
            if inputs_embeds is not None:
                raise ValueError("Fast-dLLM block-diffusion training currently requires input_ids, not inputs_embeds.")

            if mask_id is None:
                mask_id = getattr(self.config, "mask_token_id", None)

            input_ids, labels = self._prepare_block_diffusion_training_batch(
                input_ids=input_ids,
                labels=labels,
                mask_id=mask_id,
            )

            attention_mask = None
            position_ids = None
            past_key_values = None
            cache_position = None
            block_past_key_values = None
            use_cache = False
            use_block_cache = False
            update_past_key_values = False
            logits_to_keep = 0
            block_diffusion_training = True

        outputs: BaseModelOutputWithPastAndBlockCache = self.model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            update_past_key_values=update_past_key_values,
            block_size=block_size,
            use_block_cache=use_block_cache,
            block_past_key_values=block_past_key_values,
            replace_position=replace_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state

        if block_diffusion_training:
            hidden_states = hidden_states[:, : hidden_states.shape[1] // 2, :]

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None and logits.shape[1] == labels.shape[1]:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPastAndBlockCache(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
            block_past_key_values=outputs.block_past_key_values,
        )

    def _normalize_eos_token_ids(self, eos_token_id=None):
        if eos_token_id is None:
            generation_config = getattr(self, "generation_config", None)
            eos_token_id = getattr(generation_config, "eos_token_id", None)
        if eos_token_id is None:
            eos_token_id = getattr(self.config, "eos_token_id", None)

        if eos_token_id is None:
            return []

        if isinstance(eos_token_id, torch.Tensor):
            return eos_token_id.detach().cpu().view(-1).tolist()

        if isinstance(eos_token_id, (list, tuple, set)):
            return list(eos_token_id)

        return [int(eos_token_id)]

    def _get_pad_token_id(self, pad_token_id=None):
        if pad_token_id is None:
            generation_config = getattr(self, "generation_config", None)
            pad_token_id = getattr(generation_config, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(self.config, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = 0
        return int(pad_token_id)

    def _update_finished_from_eos(self, sequences, original_input_length, eos_token_ids, finished):
        if not eos_token_ids:
            return finished

        eos_tensor = torch.tensor(eos_token_ids, device=sequences.device, dtype=sequences.dtype)
        generated = sequences[:, original_input_length:]
        if generated.numel() == 0:
            return finished

        eos_mask = torch.isin(generated, eos_tensor)
        return finished | eos_mask.any(dim=1)

    def _pad_after_first_eos(self, sequences, original_input_length, eos_token_ids, pad_token_id):
        if not eos_token_ids:
            return sequences

        eos_tensor = torch.tensor(eos_token_ids, device=sequences.device, dtype=sequences.dtype)
        generated = sequences[:, original_input_length:]
        if generated.numel() == 0:
            return sequences

        eos_mask = torch.isin(generated, eos_tensor)
        for row_idx in range(sequences.shape[0]):
            eos_positions = torch.nonzero(eos_mask[row_idx], as_tuple=False)
            if eos_positions.numel() == 0:
                continue
            first_eos = int(eos_positions[0].item()) + original_input_length
            if first_eos + 1 < sequences.shape[1]:
                sequences[row_idx, first_eos + 1 :] = pad_token_id

        return sequences

    def sample_with_top_p(self, logits, top_p=0.95, temperature=1.0):
        """
        Batch-wise top-p sampling.

        Args:
            logits: Tensor [B, T, V].
        Returns:
            x_1: sampled token ids [B, T].
            p_1t: filtered probability distribution [B, T, V].
        """
        if temperature is None:
            temperature = 0.0

        if temperature <= 0:
            p_1t = torch.softmax(logits, dim=-1)
            x_1 = p_1t.argmax(dim=-1)
            return x_1, p_1t

        scaled_logits = logits / temperature
        probs = torch.softmax(scaled_logits, dim=-1)

        if top_p is None or top_p >= 1.0:
            flat_probs = probs.reshape(-1, probs.shape[-1])
            sampled = torch.multinomial(flat_probs, num_samples=1).view(*probs.shape[:-1])
            return sampled, probs

        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = (cumulative_probs - sorted_probs) > top_p
        sorted_probs = sorted_probs.masked_fill(sorted_indices_to_remove, 0.0)

        denom = sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        sorted_probs = sorted_probs / denom

        sampled_sorted = torch.multinomial(
            sorted_probs.reshape(-1, sorted_probs.shape[-1]),
            num_samples=1,
        ).view(*sorted_probs.shape[:-1])

        sampled = sorted_indices.gather(-1, sampled_sorted.unsqueeze(-1)).squeeze(-1)

        filtered_probs = torch.zeros_like(probs)
        filtered_probs.scatter_(-1, sorted_indices, sorted_probs)

        return sampled, filtered_probs

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens=None,
        max_length=None,
        tokenizer=None,
        mask_id=None,
        threshold=1.0,
        small_block_size=8,
        block_size=32,
        eos_token_id=None,
        stop_token=None,
        pad_token_id=None,
        stopping_criteria=None,
        top_p=0.95,
        temperature=0.0,
        use_block_cache=False,
        return_dict_in_generate=False,
        output_scores=False,
        output_hidden_states=False,
        **kwargs,
    ):
        """
        Fast-dLLM block-wise diffusion generation.

        This is adapted from Fast-dLLM v2 but made Qwen3-compatible:
        - mask_id defaults to config.mask_token_id.
        - eos_token_id supports int/list.
        - max_new_tokens uses ceil-style block allocation and final cropping.
        - sampling and stopping are batch-wise.
        """
        if input_ids is None:
            raise ValueError("generate requires input_ids.")

        if max_new_tokens is None and max_length is None:
            raise ValueError("Either max_new_tokens or max_length must be specified.")

        if max_new_tokens is None:
            max_new_tokens = max_length - input_ids.shape[1]

        if max_new_tokens <= 0:
            return input_ids if not return_dict_in_generate else self._make_generate_output(
                input_ids,
                scores=None,
                hidden_states=None,
            )

        if block_size <= 0 or small_block_size <= 0:
            raise ValueError("block_size and small_block_size must be positive.")
        if block_size % small_block_size != 0:
            raise ValueError("block_size must be divisible by small_block_size.")

        if mask_id is None:
            mask_id = getattr(self.config, "mask_token_id", None)
        if mask_id is None:
            raise ValueError("mask_id must be provided or config.mask_token_id must be set.")
        mask_id = int(mask_id)

        if stop_token is not None and eos_token_id is None:
            eos_token_id = stop_token

        eos_token_ids = self._normalize_eos_token_ids(eos_token_id=eos_token_id)
        pad_token_id = self._get_pad_token_id(pad_token_id=pad_token_id)

        original_training_state = self.training
        self.eval()

        original_input_length = input_ids.shape[1]
        target_length = original_input_length + max_new_tokens

        input_ids = input_ids.clone()
        device = input_ids.device
        batch_size = input_ids.shape[0]

        scores_list = [] if output_scores else None
        decoder_hidden_states = [] if output_hidden_states else None

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        finished = self._update_finished_from_eos(input_ids, original_input_length, eos_token_ids, finished)

        past_key_values = None

        # Cache complete prompt blocks as clean prefix.
        full_prompt_blocks_len = (input_ids.shape[1] // block_size) * block_size
        if full_prompt_blocks_len > 0 and input_ids.shape[1] > block_size:
            output = self.forward(
                input_ids=input_ids[:, :full_prompt_blocks_len],
                use_cache=True,
                update_past_key_values=True,
                block_size=block_size,
            )
            past_key_values = output.past_key_values

            if output_scores:
                scores_list.append(output.logits)
            if output_hidden_states and hasattr(output, "hidden_states"):
                decoder_hidden_states.append(output.hidden_states)

        num_small_blocks = block_size // small_block_size

        while input_ids.shape[1] < target_length and not finished.all().item():
            prompt_length = input_ids.shape[1]
            tokens_to_block_boundary = block_size - (prompt_length % block_size)
            if tokens_to_block_boundary == 0:
                tokens_to_block_boundary = block_size

            x_init = torch.full(
                (batch_size, tokens_to_block_boundary),
                fill_value=mask_id,
                device=device,
                dtype=torch.long,
            )
            x_t = torch.cat([input_ids, x_init], dim=1)

            block_past_key_values = None

            while True:
                current_block = x_t[:, -block_size:]
                active_mask_idx = (current_block == mask_id) & (~finished[:, None])

                if active_mask_idx.sum() == 0:
                    output = self.forward(
                        input_ids=current_block,
                        use_cache=True,
                        past_key_values=past_key_values,
                        update_past_key_values=True,
                        block_size=block_size,
                    )
                    past_key_values = output.past_key_values

                    if output_scores:
                        scores_list.append(output.logits)
                    if output_hidden_states and hasattr(output, "hidden_states"):
                        decoder_hidden_states.append(output.hidden_states)

                    if x_t.shape[1] < target_length and not finished.all().item():
                        next_token = output.logits[:, -1:, :].argmax(dim=-1)
                        next_token = torch.where(
                            finished[:, None],
                            torch.full_like(next_token, pad_token_id),
                            next_token,
                        )
                        x_t = torch.cat([x_t, next_token], dim=1)

                    break

                progressed_this_pass = False

                for small_block_idx in range(num_small_blocks):
                    small_start = small_block_idx * small_block_size
                    small_end = small_start + small_block_size

                    while True:
                        current_block = x_t[:, -block_size:]
                        mask_idx = (current_block == mask_id) & (~finished[:, None])
                        mask_slice = mask_idx[:, small_start:small_end]

                        if mask_slice.sum() == 0:
                            break

                        if use_block_cache:
                            if block_past_key_values is None or (current_block[:, small_start] == mask_id).any():
                                output = self.forward(
                                    input_ids=current_block,
                                    use_cache=True,
                                    past_key_values=past_key_values,
                                    update_past_key_values=False,
                                    use_block_cache=True,
                                    block_size=block_size,
                                )
                                logits = output.logits
                                block_past_key_values = output.block_past_key_values
                                logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                                logits = logits[:, small_start:small_end, :]
                            else:
                                small_input = current_block[:, small_start:small_end]
                                output = self.forward(
                                    input_ids=small_input,
                                    use_cache=True,
                                    past_key_values=past_key_values,
                                    update_past_key_values=False,
                                    use_block_cache=True,
                                    block_past_key_values=block_past_key_values,
                                    replace_position=small_start,
                                    block_size=block_size,
                                )
                                logits = output.logits
                                logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                        else:
                            output = self.forward(
                                input_ids=current_block,
                                use_cache=True,
                                past_key_values=past_key_values,
                                update_past_key_values=False,
                                block_size=block_size,
                            )
                            logits = output.logits
                            logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                            logits = logits[:, small_start:small_end, :]

                        if output_scores:
                            scores_list.append(logits)
                        if output_hidden_states and hasattr(output, "hidden_states"):
                            decoder_hidden_states.append(output.hidden_states)

                        x_1, p_1t = self.sample_with_top_p(logits, top_p=top_p, temperature=temperature)

                        selected_probs = torch.gather(p_1t, dim=-1, index=x_1.unsqueeze(-1)).squeeze(-1)
                        selected_probs = torch.where(mask_slice, selected_probs, torch.full_like(selected_probs, -torch.inf))

                        unmask_idx = selected_probs > threshold

                        row_has_mask = mask_slice.any(dim=-1)
                        if row_has_mask.any():
                            max_prob_idx = selected_probs.argmax(dim=-1)
                            rows = torch.arange(batch_size, device=device)
                            unmask_idx[rows[row_has_mask], max_prob_idx[row_has_mask]] = True

                        unmask_idx = unmask_idx & mask_slice

                        if unmask_idx.any():
                            current_block = x_t[:, -block_size:].clone()
                            sub_block = current_block[:, small_start:small_end].clone()
                            sub_block[unmask_idx] = x_1[unmask_idx]
                            current_block[:, small_start:small_end] = sub_block
                            x_t[:, -block_size:] = current_block
                            progressed_this_pass = True

                        finished = self._update_finished_from_eos(
                            x_t,
                            original_input_length,
                            eos_token_ids,
                            finished,
                        )

                        if finished.all().item():
                            break

                    if finished.all().item():
                        break

                if not progressed_this_pass:
                    break

            input_ids = x_t[:, : min(x_t.shape[1], target_length)]
            input_ids = self._pad_after_first_eos(input_ids, original_input_length, eos_token_ids, pad_token_id)

            if stopping_criteria is not None:
                try:
                    should_stop = stopping_criteria(input_ids, None)
                    if isinstance(should_stop, torch.Tensor):
                        should_stop = bool(should_stop.all().item())
                    if should_stop:
                        break
                except TypeError:
                    if stopping_criteria(input_ids):
                        break

        input_ids = input_ids[:, :target_length]
        input_ids = self._pad_after_first_eos(input_ids, original_input_length, eos_token_ids, pad_token_id)

        if original_training_state:
            self.train()

        if return_dict_in_generate:
            try:
                from transformers.generation.utils import GenerateDecoderOnlyOutput
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=tuple(scores_list) if output_scores and scores_list else None,
                    hidden_states=tuple(decoder_hidden_states) if output_hidden_states and decoder_hidden_states else None,
                )
            except Exception:
                return {
                    "sequences": input_ids,
                    "scores": tuple(scores_list) if output_scores and scores_list else None,
                    "hidden_states": tuple(decoder_hidden_states) if output_hidden_states and decoder_hidden_states else None,
                }

        return input_ids


class Fast_dLLM_Qwen3ForSequenceClassification(GenericForSequenceClassification, Fast_dLLM_Qwen3PreTrainedModel):
    pass


class Fast_dLLM_Qwen3ForTokenClassification(GenericForTokenClassification, Fast_dLLM_Qwen3PreTrainedModel):
    pass


class Fast_dLLM_Qwen3ForQuestionAnswering(GenericForQuestionAnswering, Fast_dLLM_Qwen3PreTrainedModel):
    base_model_prefix = "transformer"  # For BC, where `transformer` was used instead of `model`


__all__ = [
    "Fast_dLLM_Qwen3ForCausalLM",
    "Fast_dLLM_Qwen3PreTrainedModel",
    "Fast_dLLM_Qwen3Model",
    "Fast_dLLM_Qwen3ForSequenceClassification",
    "Fast_dLLM_Qwen3ForTokenClassification",
    "Fast_dLLM_Qwen3ForQuestionAnswering",
]


# ---------------------------------------------------------------------
# Backward-compatible aliases during migration.
# ---------------------------------------------------------------------
Qwen3Config = Fast_dLLM_Qwen3Config
Qwen3RMSNorm = Fast_dLLM_Qwen3RMSNorm
Qwen3MLP = Fast_dLLM_Qwen3MLP
Qwen3RotaryEmbedding = Fast_dLLM_Qwen3RotaryEmbedding
Qwen3Attention = Fast_dLLM_Qwen3Attention
Qwen3DecoderLayer = Fast_dLLM_Qwen3DecoderLayer
Qwen3PreTrainedModel = Fast_dLLM_Qwen3PreTrainedModel
Qwen3Model = Fast_dLLM_Qwen3Model
Qwen3ForCausalLM = Fast_dLLM_Qwen3ForCausalLM
