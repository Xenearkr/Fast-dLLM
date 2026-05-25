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


def make_tiny_config(complementary_mask=True):
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
        mask_token="|<MASK>|",
        bd_size=4,
        complementary_mask=complementary_mask,
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


def patch_dense_training_mask():
    old_create_block_mask = modeling_module.create_block_mask
    modeling_module.create_block_mask = None
    return old_create_block_mask


def restore_dense_training_mask(old_create_block_mask):
    modeling_module.create_block_mask = old_create_block_mask


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
    print("[CHECK] 4B dataclasses and masks still correct")

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

    block_size = 4
    n = 12
    total_len = 2 * n
    q_idx = torch.arange(total_len)[:, None]
    kv_idx = torch.arange(total_len)[None, :]

    mask = block_diff_mask(None, None, q_idx, kv_idx, block_size=block_size, n=n)
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

    print("[OK] 4B checks passed.")


def check_4c_forward_cache_still_ok():
    print("[CHECK] 4C forward/cache paths still work")

    cfg = make_tiny_config()
    model = Fast_dLLM_Qwen3ForCausalLM(cfg)
    model.eval()

    input_ids = torch.randint(4, cfg.vocab_size, (2, 8), dtype=torch.long)

    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=False, block_size=4)

    assert out.logits.shape == (2, 8, cfg.vocab_size)
    assert torch.isfinite(out.logits).all().item()

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
    assert cache_seq_len(out_block.past_key_values) == 4

    print("[OK] 4C checks passed.")


def check_4d_training_forward_still_ok():
    print("[CHECK] 4D training path still works")

    torch.manual_seed(1234)
    cfg = make_tiny_config(complementary_mask=True)
    model = Fast_dLLM_Qwen3ForCausalLM(cfg)
    model.train()

    input_ids = torch.randint(4, cfg.vocab_size, (2, 8), dtype=torch.long)
    labels = input_ids.clone()
    labels[:, :2] = -100

    old_create_block_mask = patch_dense_training_mask()
    try:
        out = model(
            input_ids=input_ids,
            labels=labels,
            use_cache=True,
            block_size=4,
        )
    finally:
        restore_dense_training_mask(old_create_block_mask)

    assert isinstance(out, CausalLMOutputWithPastAndBlockCache)
    assert out.logits.shape == (4, 8, cfg.vocab_size)
    assert out.past_key_values is None
    assert out.block_past_key_values is None
    assert out.loss is not None
    assert torch.isfinite(out.loss).item()
    assert torch.isfinite(out.logits).all().item()

    print("[OK] 4D training forward check passed.")


def check_sample_with_top_p_batchwise():
    print("[CHECK] 4E sample_with_top_p is batch-wise")

    cfg = make_tiny_config()
    model = Fast_dLLM_Qwen3ForCausalLM(cfg)
    model.eval()

    logits = torch.zeros(2, 3, cfg.vocab_size)
    logits[0, :, 10] = 10.0
    logits[1, :, 20] = 10.0

    sampled, probs = model.sample_with_top_p(logits, top_p=0.95, temperature=0.0)

    assert sampled.shape == (2, 3)
    assert probs.shape == (2, 3, cfg.vocab_size)
    assert (sampled[0] == 10).all().item()
    assert (sampled[1] == 20).all().item()

    print("[OK] sample_with_top_p batch-wise check passed.")


def check_generate_basic_without_block_cache():
    print("[CHECK] 4E generate basic path without block cache")

    torch.manual_seed(1234)
    cfg = make_tiny_config()
    model = Fast_dLLM_Qwen3ForCausalLM(cfg)
    model.eval()

    input_ids = torch.randint(4, cfg.vocab_size, (2, 3), dtype=torch.long)

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=5,
            mask_id=None,
            eos_token_id=999999,
            pad_token_id=0,
            block_size=4,
            small_block_size=2,
            threshold=1.0,
            temperature=0.0,
            top_p=0.95,
            use_block_cache=False,
        )

    assert out.shape == (2, 8), out.shape
    assert torch.equal(out[:, :3], input_ids)
    assert not (out[:, 3:] == cfg.mask_token_id).any().item()
    assert torch.isfinite(out.float()).all().item()

    print("[OK] generate basic no-cache check passed.")


def check_generate_with_block_cache():
    print("[CHECK] 4E generate with block cache")

    torch.manual_seed(1234)
    cfg = make_tiny_config()
    model = Fast_dLLM_Qwen3ForCausalLM(cfg)
    model.eval()

    input_ids = torch.randint(4, cfg.vocab_size, (2, 5), dtype=torch.long)

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=4,
            eos_token_id=999999,
            pad_token_id=0,
            block_size=4,
            small_block_size=2,
            threshold=1.0,
            temperature=0.0,
            use_block_cache=True,
        )

    assert out.shape == (2, 9), out.shape
    assert torch.equal(out[:, :5], input_ids)
    assert not (out[:, 5:] == cfg.mask_token_id).any().item()

    print("[OK] generate block-cache check passed.")


def check_generate_return_dict():
    print("[CHECK] 4E generate return_dict_in_generate")

    torch.manual_seed(1234)
    cfg = make_tiny_config()
    model = Fast_dLLM_Qwen3ForCausalLM(cfg)
    model.eval()

    input_ids = torch.randint(4, cfg.vocab_size, (1, 4), dtype=torch.long)

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=4,
            eos_token_id=999999,
            pad_token_id=0,
            block_size=4,
            small_block_size=2,
            threshold=1.0,
            temperature=0.0,
            use_block_cache=False,
            return_dict_in_generate=True,
            output_scores=True,
        )

    if isinstance(out, dict):
        sequences = out["sequences"]
    else:
        sequences = out.sequences

    assert sequences.shape == (1, 8)
    assert torch.equal(sequences[:, :4], input_ids)

    print("[OK] return_dict generate check passed.")


def check_no_hardcoded_qwen25_mask_id():
    print("[CHECK] no hard-coded Fast-Qwen2.5 mask id remains")

    text = Path("modeling.py").read_text(encoding="utf-8")
    assert "mask_id=151665" not in text
    assert "mask_id: Optional[int] = 151665" not in text
    assert "151665" not in text

    print("[OK] no hard-coded 151665 found.")


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
    check_4b_masks_still_ok()
    check_4c_forward_cache_still_ok()
    check_4d_training_forward_still_ok()
    check_sample_with_top_p_batchwise()
    check_generate_basic_without_block_cache()
    check_generate_with_block_cache()
    check_generate_return_dict()
    check_no_hardcoded_qwen25_mask_id()
    check_meta_state_dict_still_aligned()

    print("[OK] Milestone 4E evaluation passed.")


if __name__ == "__main__":
    main()
