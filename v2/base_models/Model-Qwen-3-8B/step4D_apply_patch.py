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

    def _prepare_block_diffusion_training_batch(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        mask_id: Optional[int] = None,
    ):
        """
        Build Fast-dLLM v2 training inputs.

        Given original sequence x_0 and labels, construct:

            input_ids = [x_t ; x_0]
            labels    = masked-token-only labels over x_t positions

        If config.complementary_mask is True, also construct the complementary
        noised view and concatenate it along the batch dimension.

        This method intentionally follows the Fast-dLLM v2 training layout and
        leaves the block-diffusion attention mask to Fast_dLLM_Qwen3Model.forward().
        """
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

            # Training uses the block-diffusion mask generated inside the base model.
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
            # outputs are over [x_t ; x_0]; only x_t positions are used for prediction.
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
'''

text = replace_between(
    text,
    "@auto_docstring\nclass Fast_dLLM_Qwen3ForCausalLM",
    "\n\nclass Fast_dLLM_Qwen3ForSequenceClassification",
    new_causal_lm,
    "Fast_dLLM_Qwen3ForCausalLM",
)

MODELING_PATH.write_text(text, encoding="utf-8")

required = [
    "def _prepare_block_diffusion_training_batch",
    "block_diffusion_training = False",
    "getattr(self.config, \"mask_token_id\", None)",
    "complementary_mask",
    "hidden_states = hidden_states[:, : hidden_states.shape[1] // 2, :]",
    "mask_id: Optional[int] = None",
]

updated = MODELING_PATH.read_text(encoding="utf-8")
missing = [x for x in required if x not in updated]
if missing:
    raise RuntimeError(f"Missing required 4D markers: {missing}")

if "mask_id: Optional[int] = 151665" in updated or "mask_id=151665" in updated:
    raise RuntimeError("Forbidden hard-coded Qwen2.5 Fast mask id 151665 remains in modeling.py.")

print("[OK] Milestone 4D patch applied.")
