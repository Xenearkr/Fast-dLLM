import json
import sys
from pathlib import Path

import torch

MODEL_DIR = Path(".").resolve()
sys.path.insert(0, str(MODEL_DIR))

import modeling as modeling_module
from configuration import Fast_dLLM_Qwen3Config
from modeling import (
    BaseModelOutputWithPastAndBlockCache,
    CausalLMOutputWithPastAndBlockCache,
    Fast_dLLM_Qwen3ForCausalLM,
    Fast_dLLM_Qwen3Model,
    block_diff_mask,
    eval_block_diff_mask,
)


def make_tiny_config():
    cfg = Fast_dLLM_Qwen3Config(
        vocab_size=320,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        rope_theta=1000000.0,
        attention_bias=False,
        use_sliding_window=False,
        sliding_window=None,
        max_window_layers=2,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        mask_token_id=3,
        bd_size=4,
    )
    try:
        cfg._attn_implementation = "eager"
    except Exception:
        pass
    return cfg


def cache_seq_len(cache):
    if cache is None:
        return 0
    try:
        return cache.get_seq_length()
    except Exception:
        return 0


def check_4a_structure_still_ok():
    print("[CHECK] 4A structure still intact")

    cfg = make_tiny_config()
    model = Fast_dLLM_Qwen3ForCausalLM(cfg)
    attn = model.model.layers[0].self_attn

    assert isinstance(model.model, Fast_dLLM_Qwen3Model)
    assert model.config.model_type == "Fast_dLLM_Qwen3"
    assert hasattr(attn, "q_norm")
    assert hasattr(attn, "k_norm")
    assert attn.q_proj.bias is None
    assert attn.k_proj.bias is None
    assert attn.v_proj.bias is None
    assert attn.o_proj.bias is None

    print("[OK] 4A structure check passed.")


def check_4b_masks_still_ok():
    print("[CHECK] 4B block mask utilities still correct")

    block_size = 4
    n = 12
    total_len = 2 * n

    q_idx = torch.arange(total_len)[:, None]
    kv_idx = torch.arange(total_len)[None, :]

    mask = block_diff_mask(
        b=None,
        h=None,
        q_idx=q_idx,
        kv_idx=kv_idx,
        block_size=block_size,
        n=n,
    )

    assert mask.shape == (total_len, total_len)
    assert mask.dtype == torch.bool

    q_xt_b1 = 4
    assert mask[q_xt_b1, 4:8].all().item()
    assert not mask[q_xt_b1, 0:4].any().item()
    assert mask[q_xt_b1, n:n + 4].all().item()
    assert not mask[q_xt_b1, n + 4:n + 8].any().item()

    eval_mask = eval_block_diff_mask(
        q_idx=torch.arange(12)[:, None],
        kv_idx=torch.arange(12)[None, :],
        block_size=4,
    )
    assert eval_mask.shape == (12, 12)
    assert eval_mask.dtype == torch.bool
    assert eval_mask[0, 0:4].all().item()
    assert not eval_mask[0, 4:12].any().item()
    assert eval_mask[8, 0:12].all().item()

    print("[OK] 4B mask checks passed.")


def check_4b_dataclasses_still_ok():
    print("[CHECK] 4B dataclasses still correct")

    lm_out = CausalLMOutputWithPastAndBlockCache(
        logits=torch.zeros(1, 2, 16),
        block_past_key_values=None,
    )
    base_out = BaseModelOutputWithPastAndBlockCache(
        last_hidden_state=torch.zeros(1, 2, 8),
        block_past_key_values=None,
    )

    assert hasattr(lm_out, "block_past_key_values")
    assert hasattr(base_out, "block_past_key_values")

    print("[OK] 4B dataclass checks passed.")


def check_tiny_forward_still_ok():
    print("[CHECK] Tiny CausalLM forward works after 4C")

    cfg = make_tiny_config()
    model = Fast_dLLM_Qwen3ForCausalLM(cfg)
    model.eval()

    input_ids = torch.randint(4, cfg.vocab_size, (2, 8), dtype=torch.long)

    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=False, block_size=4)

    assert out.logits.shape == (2, 8, cfg.vocab_size)
    assert torch.isfinite(out.logits).all().item()
    assert isinstance(out, CausalLMOutputWithPastAndBlockCache)

    print("[OK] Tiny CausalLM forward check passed.")


def check_model_eval_mask_methods():
    print("[CHECK] Fast_dLLM_Qwen3Model eval_mask / gen_mask methods")

    cfg = make_tiny_config()
    model = Fast_dLLM_Qwen3ForCausalLM(cfg).model

    allowed = model.eval_mask(seqlen=4, block_size=4, cache_seq_len=4, device=torch.device("cpu"))
    assert allowed.shape == (4, 8)
    assert allowed.dtype == torch.bool
    # Query positions are 4..7, i.e. block 1; keys are 0..7, blocks 0 and 1.
    assert allowed.all().item()

    # Force dense fallback for gen_mask so this test is CPU-safe.
    old_create_block_mask = modeling_module.create_block_mask
    modeling_module.create_block_mask = None
    try:
        train_mask = model.gen_mask(seqlen=8, block_size=4, B=2, H=4, device=torch.device("cpu"))
    finally:
        modeling_module.create_block_mask = old_create_block_mask

    assert isinstance(train_mask, torch.Tensor)
    assert train_mask.shape == (16, 16)
    assert train_mask.dtype == torch.bool

    print("[OK] eval_mask / gen_mask method checks passed.")


def check_prefix_cache_forward():
    print("[CHECK] prefix past_key_values cache path")

    cfg = make_tiny_config()
    model = Fast_dLLM_Qwen3ForCausalLM(cfg)
    model.eval()

    prefix_ids = torch.randint(4, cfg.vocab_size, (2, 4), dtype=torch.long)
    block_ids = torch.randint(4, cfg.vocab_size, (2, 4), dtype=torch.long)

    with torch.no_grad():
        out_prefix = model(
            input_ids=prefix_ids,
            use_cache=True,
            update_past_key_values=True,
            block_size=4,
        )

        assert out_prefix.past_key_values is not None
        assert cache_seq_len(out_prefix.past_key_values) == 4

        out_block = model(
            input_ids=block_ids,
            use_cache=True,
            past_key_values=out_prefix.past_key_values,
            update_past_key_values=False,
            block_size=4,
        )

    assert out_block.logits.shape == (2, 4, cfg.vocab_size)
    assert torch.isfinite(out_block.logits).all().item()

    # Read-only prefix reuse should not append the new block.
    assert cache_seq_len(out_block.past_key_values) == 4

    print("[OK] prefix cache forward check passed.")


def check_block_cache_forward():
    print("[CHECK] block_past_key_values cache path")

    cfg = make_tiny_config()
    model = Fast_dLLM_Qwen3ForCausalLM(cfg)
    model.eval()

    block_ids = torch.randint(4, cfg.vocab_size, (2, 4), dtype=torch.long)
    small_ids = torch.randint(4, cfg.vocab_size, (2, 2), dtype=torch.long)

    with torch.no_grad():
        out_block_cache = model(
            input_ids=block_ids,
            use_cache=True,
            update_past_key_values=False,
            use_block_cache=True,
            block_size=4,
        )

        assert out_block_cache.block_past_key_values is not None
        assert cache_seq_len(out_block_cache.block_past_key_values) == 4

        out_replace = model(
            input_ids=small_ids,
            use_cache=True,
            update_past_key_values=False,
            use_block_cache=True,
            block_past_key_values=out_block_cache.block_past_key_values,
            replace_position=1,
            block_size=4,
        )

    assert out_replace.logits.shape == (2, 2, cfg.vocab_size)
    assert torch.isfinite(out_replace.logits).all().item()
    assert out_replace.block_past_key_values is not None
    assert cache_seq_len(out_replace.block_past_key_values) == 4

    print("[OK] block cache forward check passed.")


def check_block_training_layout_model_forward():
    print("[CHECK] base Model forward supports [x_t ; x_0] training layout")

    cfg = make_tiny_config()
    causal_lm = Fast_dLLM_Qwen3ForCausalLM(cfg)
    base_model = causal_lm.model
    base_model.train()

    batch = 2
    n = 8
    concat_len = 2 * n

    input_ids = torch.randint(4, cfg.vocab_size, (batch, concat_len), dtype=torch.long)
    labels = torch.randint(4, cfg.vocab_size, (batch, n), dtype=torch.long)

    # Force dense bool mask fallback for CPU-safe test.
    old_create_block_mask = modeling_module.create_block_mask
    modeling_module.create_block_mask = None
    try:
        out = base_model(
            input_ids=input_ids,
            labels=labels,
            use_cache=False,
            block_size=4,
        )
    finally:
        modeling_module.create_block_mask = old_create_block_mask

    assert isinstance(out, BaseModelOutputWithPastAndBlockCache)
    assert out.last_hidden_state.shape == (batch, concat_len, cfg.hidden_size)
    assert torch.isfinite(out.last_hidden_state).all().item()

    print("[OK] block training-layout base Model forward check passed.")


def check_meta_state_dict_still_aligned():
    print("[CHECK] Meta state_dict key alignment still unchanged")

    try:
        from accelerate import init_empty_weights
    except Exception as e:
        print(f"[SKIP] accelerate.init_empty_weights unavailable: {e}")
        return

    cfg = Fast_dLLM_Qwen3Config.from_pretrained(MODEL_DIR)

    index_path = MODEL_DIR / "model.safetensors.index.json"
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    expected_keys = set(index["weight_map"].keys())

    with init_empty_weights():
        model = Fast_dLLM_Qwen3ForCausalLM(cfg)

    actual_keys = set(model.state_dict().keys())

    missing_in_model = sorted(expected_keys - actual_keys)
    extra_in_model = sorted(actual_keys - expected_keys)

    print(f"expected weight keys: {len(expected_keys)}")
    print(f"actual model keys:    {len(actual_keys)}")
    print(f"missing_in_model:     {len(missing_in_model)}")
    print(f"extra_in_model:       {len(extra_in_model)}")

    if missing_in_model:
        print("[MISSING sample]", missing_in_model[:30])
    if extra_in_model:
        print("[EXTRA sample]", extra_in_model[:30])

    assert not missing_in_model
    assert not extra_in_model

    print("[OK] Meta state_dict alignment check passed.")


def main():
    check_4a_structure_still_ok()
    check_4b_dataclasses_still_ok()
    check_4b_masks_still_ok()
    check_tiny_forward_still_ok()
    check_model_eval_mask_methods()
    check_prefix_cache_forward()
    check_block_cache_forward()
    check_block_training_layout_model_forward()
    check_meta_state_dict_still_aligned()

    print("[OK] Milestone 4C evaluation passed.")


if __name__ == "__main__":
    main()
