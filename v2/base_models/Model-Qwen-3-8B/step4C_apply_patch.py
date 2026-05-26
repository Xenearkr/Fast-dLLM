from pathlib import Path

MODELING_PATH = Path("modeling.py")
text = MODELING_PATH.read_text(encoding="utf-8")


def replace_between(src, start_marker, end_marker, replacement, label):
    start = src.find(start_marker)
    if start == -1:
        raise RuntimeError(f"Could not find start marker for {label}: {start_marker!r}")
    end = src.find(end_marker, start)
    if end == -1:
        raise RuntimeError(f"Could not find end marker for {label}: {end_marker!r}")
    return src[:start] + replacement + src[end:]


# ---------------------------------------------------------------------
# 1. Insert 4C helper functions after eval_block_diff_mask.
# ---------------------------------------------------------------------
helper_block = r'''

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


'''

if "_cache_update_compat" not in text:
    marker = "\n\n\n\n@use_kernel_forward_from_hub(\"RMSNorm\")"
    if marker not in text:
        marker = "\n@use_kernel_forward_from_hub(\"RMSNorm\")"
    if marker not in text:
        raise RuntimeError("Could not find insertion marker before RMSNorm.")
    text = text.replace(marker, "\n" + helper_block + marker, 1)
    print("[PATCH] Inserted 4C helper functions.")
else:
    print("[INFO] 4C helper functions already exist.")


# ---------------------------------------------------------------------
# 2. Replace Fast_dLLM_Qwen3Attention with block-cache aware version.
# ---------------------------------------------------------------------
new_attention = r'''@use_kernelized_func(apply_rotary_pos_emb)
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
'''

text = replace_between(
    text,
    "@use_kernelized_func(apply_rotary_pos_emb)\nclass Fast_dLLM_Qwen3Attention",
    "\n\nclass Fast_dLLM_Qwen3DecoderLayer",
    new_attention,
    "Fast_dLLM_Qwen3Attention",
)
print("[PATCH] Replaced Fast_dLLM_Qwen3Attention.")


# ---------------------------------------------------------------------
# 3. Replace DecoderLayer.
# ---------------------------------------------------------------------
new_decoder_layer = r'''class Fast_dLLM_Qwen3DecoderLayer(GradientCheckpointingLayer):
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
'''

text = replace_between(
    text,
    "class Fast_dLLM_Qwen3DecoderLayer",
    "\n\n@auto_docstring\nclass Fast_dLLM_Qwen3PreTrainedModel",
    new_decoder_layer,
    "Fast_dLLM_Qwen3DecoderLayer",
)
print("[PATCH] Replaced Fast_dLLM_Qwen3DecoderLayer.")


# ---------------------------------------------------------------------
# 4. Replace Model.
# ---------------------------------------------------------------------
new_model = r'''@auto_docstring
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
        if create_block_mask is not None:
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
'''

text = replace_between(
    text,
    "@auto_docstring\nclass Fast_dLLM_Qwen3Model",
    "\n\n@auto_docstring\nclass Fast_dLLM_Qwen3ForCausalLM",
    new_model,
    "Fast_dLLM_Qwen3Model",
)
print("[PATCH] Replaced Fast_dLLM_Qwen3Model.")


# ---------------------------------------------------------------------
# 5. Replace ForCausalLM.
# ---------------------------------------------------------------------
new_causal_lm = r'''@auto_docstring
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
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPastAndBlockCache:
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

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None and logits.shape[1] == labels.shape[1]:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPastAndBlockCache(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            block_past_key_values=outputs.block_past_key_values,
        )
'''

text = replace_between(
    text,
    "@auto_docstring\nclass Fast_dLLM_Qwen3ForCausalLM",
    "\n\nclass Fast_dLLM_Qwen3ForSequenceClassification",
    new_causal_lm,
    "Fast_dLLM_Qwen3ForCausalLM",
)
print("[PATCH] Replaced Fast_dLLM_Qwen3ForCausalLM.")

MODELING_PATH.write_text(text, encoding="utf-8")

required = [
    "def _cache_update_compat",
    "def _allowed_bool_mask_to_additive",
    "def eval_mask(self, seqlen, block_size, cache_seq_len",
    "def gen_mask(self, seqlen, block_size, B, H",
    "block_past_key_values",
    "update_past_key_values",
    "replace_position",
    "BaseModelOutputWithPastAndBlockCache",
    "CausalLMOutputWithPastAndBlockCache",
]

updated = MODELING_PATH.read_text(encoding="utf-8")
missing = [x for x in required if x not in updated]
if missing:
    raise RuntimeError(f"Missing required 4C markers: {missing}")

print("[OK] Milestone 4C patch applied.")
