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
    """Multi-headed attention from 'Attention Is All You Need' paper"""

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
        self.q_norm = Fast_dLLM_Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Fast_dLLM_Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

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
            sliding_window=self.sliding_window,  # diff with Llama
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
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@auto_docstring
class Fast_dLLM_Qwen3PreTrainedModel(PreTrainedModel):
    config: Fast_dLLM_Qwen3Config
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

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Fast_dLLM_Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Fast_dLLM_Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Fast_dLLM_Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        # Initialize weights and apply final processing
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        causal_mask_mapping = attention_mask
        if not isinstance(causal_mask_mapping, dict):
            causal_mask_mapping = _create_causal_mask_mapping_compat(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                has_sliding_layers=self.has_sliding_layers,
            )

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
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
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

        # Initialize weights and apply final processing
        self.post_init()

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
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Fast_dLLM_Qwen3ForCausalLM

        >>> model = Fast_dLLM_Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-8B")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


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
