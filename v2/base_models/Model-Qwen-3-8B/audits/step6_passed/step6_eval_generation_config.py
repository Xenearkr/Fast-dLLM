import json
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

MODEL_DIR = Path(".").resolve()
sys.path.insert(0, str(MODEL_DIR))

import modeling as modeling_module
from configuration import Fast_dLLM_Qwen3Config
from modeling import (
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


def patch_dense_training_mask():
    old_create_block_mask = modeling_module.create_block_mask
    modeling_module.create_block_mask = None
    return old_create_block_mask


def restore_dense_training_mask(old_create_block_mask):
    modeling_module.create_block_mask = old_create_block_mask


def check_raw_generation_config():
    print("[CHECK] raw generation_config.json")

    with open(MODEL_DIR / "generation_config.json", "r", encoding="utf-8") as f:
        gen = json.load(f)

    assert gen["bos_token_id"] == 151643
    assert gen["eos_token_id"] == [151645, 151643]
    assert gen["pad_token_id"] == 151643
    assert gen["do_sample"] is True
    assert gen["temperature"] == 0.6
    assert gen["top_k"] == 20
    assert gen["top_p"] == 0.95

    # We intentionally do not set repetition_penalty in Step 6 because the
    # custom Fast generate() currently does not implement it.
    assert "repetition_penalty" not in gen

    print("[OK] raw generation_config.json check passed.")


def check_generation_config_loader():
    print("[CHECK] GenerationConfig.from_pretrained")

    gen = GenerationConfig.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )

    print("bos_token_id:", gen.bos_token_id)
    print("eos_token_id:", gen.eos_token_id)
    print("pad_token_id:", gen.pad_token_id)
    print("temperature:", gen.temperature)
    print("top_k:", gen.top_k)
    print("top_p:", gen.top_p)

    assert gen.bos_token_id == 151643
    assert gen.eos_token_id == [151645, 151643]
    assert gen.pad_token_id == 151643
    assert gen.temperature == 0.6
    assert gen.top_k == 20
    assert gen.top_p == 0.95

    print("[OK] GenerationConfig loader check passed.")


def check_step5_autoconfig_tokenizer_still_ok():
    print("[CHECK] Step 5 AutoConfig/tokenizer still OK")

    cfg = AutoConfig.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        local_files_only=True,
    )

    assert cfg.__class__.__name__ == "Fast_dLLM_Qwen3Config"
    assert cfg.model_type == "Fast_dLLM_Qwen3"
    assert cfg.mask_token_id == 151669
    assert cfg.pad_token_id == 151643
    assert cfg.bd_size == 32

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        local_files_only=True,
    )

    assert tokenizer.convert_tokens_to_ids("|<MASK>|") == 151669
    assert tokenizer.pad_token_id == 151643
    assert tokenizer.eos_token_id == 151645
    assert len(tokenizer) <= cfg.vocab_size

    print("[OK] Step 5 AutoConfig/tokenizer check passed.")


def check_4a_4b_structure_and_masks():
    print("[CHECK] 4A/4B structure and masks still OK")

    cfg = make_tiny_config()
    model = Fast_dLLM_Qwen3ForCausalLM(cfg)
    attn = model.model.layers[0].self_attn

    assert isinstance(model.model, Fast_dLLM_Qwen3Model)
    assert hasattr(attn, "q_norm")
    assert hasattr(attn, "k_norm")
    assert attn.q_proj.bias is None

    block_size = 4
    n = 12
    q_idx = torch.arange(2 * n)[:, None]
    kv_idx = torch.arange(2 * n)[None, :]
    mask = block_diff_mask(None, None, q_idx, kv_idx, block_size=block_size, n=n)
    assert mask.shape == (2 * n, 2 * n)
    assert mask.dtype == torch.bool

    emask = eval_block_diff_mask(
        q_idx=torch.arange(12)[:, None],
        kv_idx=torch.arange(12)[None, :],
        block_size=4,
    )
    assert emask.shape == (12, 12)
    assert emask.dtype == torch.bool

    print("[OK] 4A/4B regression checks passed.")


def check_4d_training_still_ok():
    print("[CHECK] 4D training branch still OK")

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
    assert out.loss is not None
    assert torch.isfinite(out.loss).item()
    assert out.past_key_values is None
    assert out.block_past_key_values is None

    print("[OK] 4D training regression check passed.")


def check_4e_generate_uses_generation_config_tokens():
    print("[CHECK] 4E generate uses generation_config eos/pad fallback")

    torch.manual_seed(1234)
    cfg = make_tiny_config()
    model = Fast_dLLM_Qwen3ForCausalLM(cfg)
    model.eval()

    # Attach real repo GenerationConfig. Its eos/pad ids are outside tiny vocab,
    # but generate should still run because those ids are only used for stopping/padding.
    model.generation_config = GenerationConfig.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )

    input_ids = torch.randint(4, cfg.vocab_size, (2, 3), dtype=torch.long)

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=5,
            block_size=4,
            small_block_size=2,
            threshold=1.0,
            temperature=0.0,
            use_block_cache=False,
        )

    assert out.shape == (2, 8)
    assert torch.equal(out[:, :3], input_ids)
    assert not (out[:, 3:] == cfg.mask_token_id).any().item()

    print("[OK] 4E generate GenerationConfig fallback check passed.")


def check_meta_state_dict_still_aligned():
    print("[CHECK] meta state_dict key alignment still unchanged")

    try:
        from accelerate import init_empty_weights
    except Exception as e:
        print(f"[SKIP] accelerate.init_empty_weights unavailable: {e}")
        return

    cfg = AutoConfig.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        local_files_only=True,
    )

    with open(MODEL_DIR / "model.safetensors.index.json", "r", encoding="utf-8") as f:
        index = json.load(f)

    expected_keys = set(index["weight_map"].keys())

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            cfg,
            trust_remote_code=True,
        )

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

    print("[OK] meta state_dict key alignment check passed.")


def main():
    check_raw_generation_config()
    check_generation_config_loader()
    check_step5_autoconfig_tokenizer_still_ok()
    check_4a_4b_structure_and_masks()
    check_4d_training_still_ok()
    check_4e_generate_uses_generation_config_tokens()
    check_meta_state_dict_still_aligned()

    print("[OK] Step 6 generation_config evaluation passed.")


if __name__ == "__main__":
    main()
