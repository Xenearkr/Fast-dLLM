import argparse
import json
import sys
from pathlib import Path

import torch

MODEL_DIR = Path(".").resolve()
sys.path.insert(0, str(MODEL_DIR))

from configuration import Fast_dLLM_Qwen3Config
from modeling import (
    Fast_dLLM_Qwen3Attention,
    Fast_dLLM_Qwen3DecoderLayer,
    Fast_dLLM_Qwen3ForCausalLM,
    Fast_dLLM_Qwen3Model,
)


def set_eager_attn(config):
    # Qwen3 modeling reads config._attn_implementation.
    try:
        config._attn_implementation = "eager"
    except Exception:
        pass
    return config


def tiny_forward_check():
    print("[CHECK] Tiny model forward check")

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
        bd_size=8,
    )
    cfg = set_eager_attn(cfg)

    model = Fast_dLLM_Qwen3ForCausalLM(cfg)
    model.eval()

    attn = model.model.layers[0].self_attn

    assert isinstance(model, Fast_dLLM_Qwen3ForCausalLM)
    assert isinstance(model.model, Fast_dLLM_Qwen3Model)
    assert isinstance(model.model.layers[0], Fast_dLLM_Qwen3DecoderLayer)
    assert isinstance(attn, Fast_dLLM_Qwen3Attention)

    assert hasattr(attn, "q_norm"), "Qwen3 q_norm is missing"
    assert hasattr(attn, "k_norm"), "Qwen3 k_norm is missing"
    assert attn.q_proj.bias is None, "Qwen3-8B uses attention_bias=False; q_proj.bias should be None"
    assert attn.k_proj.bias is None, "k_proj.bias should be None"
    assert attn.v_proj.bias is None, "v_proj.bias should be None"
    assert attn.o_proj.bias is None, "o_proj.bias should be None"

    input_ids = torch.randint(4, cfg.vocab_size, (2, 8), dtype=torch.long)

    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=False)

    assert out.logits.shape == (2, 8, cfg.vocab_size), out.logits.shape
    assert torch.isfinite(out.logits).all(), "logits contain non-finite values"

    print("[OK] Tiny forward check passed.")


def meta_state_dict_key_check():
    print("[CHECK] Real Qwen3-8B state_dict key compatibility check on meta device")

    try:
        from accelerate import init_empty_weights
    except Exception as e:
        print(f"[SKIP] accelerate.init_empty_weights unavailable: {e}")
        return

    cfg = Fast_dLLM_Qwen3Config.from_pretrained(MODEL_DIR)
    cfg = set_eager_attn(cfg)

    index_path = MODEL_DIR / "model.safetensors.index.json"
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    expected_keys = set(index["weight_map"].keys())

    with init_empty_weights():
        model = Fast_dLLM_Qwen3ForCausalLM(cfg)

    actual_keys = set(model.state_dict().keys())

    missing_in_model = sorted(expected_keys - actual_keys)
    extra_in_model = sorted(actual_keys - expected_keys)

    # Print short diagnostics before asserting.
    print(f"expected weight keys: {len(expected_keys)}")
    print(f"actual model keys:    {len(actual_keys)}")
    print(f"missing_in_model:     {len(missing_in_model)}")
    print(f"extra_in_model:       {len(extra_in_model)}")

    if missing_in_model:
        print("[MISSING sample]", missing_in_model[:30])
    if extra_in_model:
        print("[EXTRA sample]", extra_in_model[:30])

    assert not missing_in_model, "Some checkpoint keys are not present in the Fast_dLLM_Qwen3 skeleton."
    assert not extra_in_model, "The skeleton has parameters not present in the Qwen3 checkpoint."

    # Specific Qwen3 structural checks.
    assert "model.layers.0.self_attn.q_norm.weight" in actual_keys
    assert "model.layers.0.self_attn.k_norm.weight" in actual_keys
    assert "model.layers.0.self_attn.q_proj.bias" not in actual_keys
    assert "model.layers.0.self_attn.k_proj.bias" not in actual_keys
    assert "model.layers.0.self_attn.v_proj.bias" not in actual_keys
    assert "model.layers.0.self_attn.o_proj.bias" not in actual_keys

    print("[OK] Meta state_dict key compatibility check passed.")


def optional_full_weight_load_check():
    print("[CHECK] Optional full checkpoint load on CPU. This can take time and memory.")

    cfg = Fast_dLLM_Qwen3Config.from_pretrained(MODEL_DIR)
    cfg = set_eager_attn(cfg)

    model = Fast_dLLM_Qwen3ForCausalLM.from_pretrained(
        MODEL_DIR,
        config=cfg,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cpu",
        local_files_only=True,
    )
    model.eval()

    attn = model.model.layers[0].self_attn

    print("loaded class:", model.__class__.__name__)
    print("q_norm exists:", hasattr(attn, "q_norm"))
    print("k_norm exists:", hasattr(attn, "k_norm"))
    print("q_proj.bias:", attn.q_proj.bias)

    assert model.__class__.__name__ == "Fast_dLLM_Qwen3ForCausalLM"
    assert hasattr(attn, "q_norm")
    assert hasattr(attn, "k_norm")
    assert attn.q_proj.bias is None

    print("[OK] Full checkpoint load check passed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-load", action="store_true", help="Run full 8B checkpoint load on CPU.")
    args = parser.parse_args()

    tiny_forward_check()
    meta_state_dict_key_check()

    if args.full_load:
        optional_full_weight_load_check()

    print("[OK] Milestone 4A evaluation passed.")


if __name__ == "__main__":
    main()
