# coding=utf-8
"""Fast-dLLM Qwen3 model configuration.

This configuration adapts Qwen3-8B to the Fast-dLLM v2 block-diffusion
framework while preserving Qwen3 architectural parameters.
"""

from transformers.configuration_utils import PretrainedConfig

try:
    from transformers.configuration_utils import layer_type_validation
except Exception:  # pragma: no cover
    def layer_type_validation(layer_types):
        return None

try:
    from transformers.modeling_rope_utils import rope_config_validation
except Exception:  # pragma: no cover
    def rope_config_validation(config):
        return None

try:
    from transformers.utils import logging
except Exception:  # pragma: no cover
    import logging


logger = logging.get_logger(__name__)


class Fast_dLLM_Qwen3Config(PretrainedConfig):
    r"""
    Configuration class for Fast-dLLM Qwen3 models.

    This config keeps the Qwen3 architecture, including QK normalization,
    grouped-query attention settings, RoPE parameters, and Qwen3 token ids,
    while adding Fast-dLLM v2 block-diffusion specific fields.

    Args:
        vocab_size (`int`, *optional*, defaults to 151936):
            Vocabulary size of the Qwen3 model.
        hidden_size (`int`, *optional*, defaults to 4096):
            Hidden size of the transformer.
        intermediate_size (`int`, *optional*, defaults to 12288):
            Intermediate size of the MLP.
        num_hidden_layers (`int`, *optional*, defaults to 36):
            Number of hidden layers.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            Number of key/value heads for grouped-query attention.
        head_dim (`int`, *optional*, defaults to 128):
            Dimension of each attention head. Qwen3 requires this field.
        bd_size (`int`, *optional*, defaults to 32):
            Fast-dLLM block diffusion block size.
        mask_token_id (`int`, *optional*, defaults to 151669):
            Token id of the added Fast-dLLM mask token `|<MASK>|`.
        complementary_mask (`bool`, *optional*, defaults to `True`):
            Whether to use complementary masking during training.
        conplemenrary_mask (`bool`, *optional*):
            Backward-compatible alias for the typo used by some Fast-dLLM
            configs. If provided, it overrides `complementary_mask`.
    """

    model_type = "Fast_dLLM_Qwen3"
    keys_to_ignore_at_inference = ["past_key_values"]

    # Tensor parallel plan, aligned with Qwen3.
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise_allreduce",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise_allreduce",
    }

    # Sequence parallel training plan, aligned with Qwen3.
    base_model_sp_plan = {
        "embed_tokens": "vocab_reduce_scatter",
        "layers.*.input_layernorm": "activation",
        "layers.*.self_attn": "module_allgather_hidden_states",
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.q_norm": "activation_seq_dim_2",
        "layers.*.self_attn.k_norm": "activation_seq_dim_2",
        "layers.*.self_attn.o_proj": "rowwise_reduce_scatter",
        "layers.*.post_attention_layernorm": "activation",
        "layers.*.mlp": "module_allgather",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise_reduce_scatter",
        "norm": "activation",
    }

    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    base_model_fsdp_plan = {
        "embed_tokens": "free_full_weight",
        "layers.*": "free_full_weight",
        "norm": "keep_full_weight",
    }

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=12288,
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=40960,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=1000000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        use_sliding_window=False,
        sliding_window=None,
        max_window_layers=36,
        layer_types=None,
        bd_size=32,
        mask_token_id=151669,
        mask_token="|<MASK>|",
        complementary_mask=True,
        conplemenrary_mask=None,
        pad_token_id=151643,
        bos_token_id=151643,
        eos_token_id=151645,
        **kwargs,
    ):
        # Qwen3 architecture fields.
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim

        # Qwen3 uses grouped-query attention. Keep backward compatibility
        # with configs where num_key_value_heads may be None.
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings

        # RoPE fields. Qwen3-8B uses rope_theta=1000000 and rope_scaling=None.
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

        # Backward compatibility: older configs may use {"type": ...}
        # instead of {"rope_type": ...}.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]

        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        # Sliding-window fields. Qwen3-8B has use_sliding_window=False,
        # sliding_window=None, and all layers are full attention.
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if self.use_sliding_window else None
        self.max_window_layers = max_window_layers

        if layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and i >= self.max_window_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        else:
            self.layer_types = layer_types

        layer_type_validation(self.layer_types)

        # Fast-dLLM v2 block-diffusion fields.
        self.bd_size = bd_size
        self.mask_token_id = mask_token_id
        self.mask_token = mask_token

        # Compatibility with the typo present in some Fast-dLLM configs.
        if conplemenrary_mask is not None:
            complementary_mask = conplemenrary_mask
        self.complementary_mask = complementary_mask
        self.conplemenrary_mask = complementary_mask

        # Validate rotary embedding configuration when available.
        rope_config_validation(self)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


# Compatibility alias. This is useful while migrating imports from Qwen3Config
# to Fast_dLLM_Qwen3Config.
Qwen3Config = Fast_dLLM_Qwen3Config


__all__ = ["Fast_dLLM_Qwen3Config", "Qwen3Config"]