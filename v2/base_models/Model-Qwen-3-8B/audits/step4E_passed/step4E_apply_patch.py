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
    "def sample_with_top_p",
    "def generate(",
    "getattr(self.config, \"mask_token_id\", None)",
    "_normalize_eos_token_ids",
    "_pad_after_first_eos",
    "block_size % small_block_size",
    "use_block_cache",
    "return_dict_in_generate",
]

updated = MODELING_PATH.read_text(encoding="utf-8")
missing = [x for x in required if x not in updated]
if missing:
    raise RuntimeError(f"Missing required 4E markers: {missing}")

if "mask_id=151665" in updated or "mask_id: Optional[int] = 151665" in updated:
    raise RuntimeError("Forbidden hard-coded Fast-Qwen2.5 mask id 151665 remains in modeling.py.")

print("[OK] Milestone 4E patch applied.")
