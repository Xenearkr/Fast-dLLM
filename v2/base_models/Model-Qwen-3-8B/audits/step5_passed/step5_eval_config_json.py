import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = Path(".").resolve()
sys.path.insert(0, str(MODEL_DIR))


def check_raw_config_json():
    print("[CHECK] raw config.json fields")

    with open(MODEL_DIR / "config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    assert cfg["architectures"] == ["Fast_dLLM_Qwen3ForCausalLM"]
    assert cfg["model_type"] == "Fast_dLLM_Qwen3"

    assert cfg["auto_map"]["AutoConfig"] == "configuration.Fast_dLLM_Qwen3Config"
    assert cfg["auto_map"]["AutoModel"] == "modeling.Fast_dLLM_Qwen3Model"
    assert cfg["auto_map"]["AutoModelForCausalLM"] == "modeling.Fast_dLLM_Qwen3ForCausalLM"

    assert cfg["bd_size"] == 32
    assert cfg["mask_token_id"] == 151669
    assert cfg["mask_token"] == "|<MASK>|"
    assert cfg["complementary_mask"] is True
    assert cfg["conplemenrary_mask"] is True
    assert cfg["pad_token_id"] == 151643

    assert cfg["vocab_size"] == 151936
    assert cfg["hidden_size"] == 4096
    assert cfg["intermediate_size"] == 12288
    assert cfg["num_hidden_layers"] == 36
    assert cfg["num_attention_heads"] == 32
    assert cfg["num_key_value_heads"] == 8
    assert cfg["head_dim"] == 128
    assert cfg["attention_bias"] is False
    assert cfg["use_sliding_window"] is False
    assert cfg["sliding_window"] is None

    assert len(cfg["layer_types"]) == 36
    assert set(cfg["layer_types"]) == {"full_attention"}

    print("[OK] raw config.json check passed.")


def check_autoconfig_and_tokenizer():
    print("[CHECK] AutoConfig and tokenizer")

    cfg = AutoConfig.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        local_files_only=True,
    )

    print("config class:", cfg.__class__.__name__)
    print("model_type:", cfg.model_type)
    print("bd_size:", cfg.bd_size)
    print("mask_token_id:", cfg.mask_token_id)
    print("pad_token_id:", cfg.pad_token_id)

    assert cfg.__class__.__name__ == "Fast_dLLM_Qwen3Config"
    assert cfg.model_type == "Fast_dLLM_Qwen3"
    assert cfg.bd_size == 32
    assert cfg.mask_token_id == 151669
    assert cfg.mask_token == "|<MASK>|"
    assert cfg.complementary_mask is True
    assert cfg.conplemenrary_mask is True
    assert cfg.pad_token_id == 151643

    assert cfg.vocab_size == 151936
    assert cfg.hidden_size == 4096
    assert cfg.intermediate_size == 12288
    assert cfg.num_hidden_layers == 36
    assert cfg.num_attention_heads == 32
    assert cfg.num_key_value_heads == 8
    assert cfg.head_dim == 128
    assert cfg.attention_bias is False
    assert len(cfg.layer_types) == 36
    assert set(cfg.layer_types) == {"full_attention"}

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        local_files_only=True,
    )

    mask_id = tokenizer.convert_tokens_to_ids("|<MASK>|")

    print("len(tokenizer):", len(tokenizer))
    print("|<MASK>| id:", mask_id)
    print("tokenizer.pad_token_id:", tokenizer.pad_token_id)
    print("tokenizer.eos_token_id:", tokenizer.eos_token_id)

    assert mask_id == 151669
    assert len(tokenizer) <= cfg.vocab_size

    print("[OK] AutoConfig/tokenizer check passed.")


def check_automodel_from_config_and_keys():
    print("[CHECK] AutoModel / AutoModelForCausalLM from config and state_dict keys")

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
        base_model = AutoModel.from_config(
            cfg,
            trust_remote_code=True,
        )
        causal_lm = AutoModelForCausalLM.from_config(
            cfg,
            trust_remote_code=True,
        )

    print("AutoModel class:", base_model.__class__.__name__)
    print("AutoModelForCausalLM class:", causal_lm.__class__.__name__)

    assert base_model.__class__.__name__ == "Fast_dLLM_Qwen3Model"
    assert causal_lm.__class__.__name__ == "Fast_dLLM_Qwen3ForCausalLM"

    attn = causal_lm.model.layers[0].self_attn
    assert hasattr(attn, "q_norm")
    assert hasattr(attn, "k_norm")
    assert attn.q_proj.bias is None

    actual_keys = set(causal_lm.state_dict().keys())

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

    print("[OK] AutoModel config/key alignment check passed.")


def check_manual_tiny_forward_via_config():
    print("[CHECK] tiny forward via AutoConfig class")

    cfg = AutoConfig.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        local_files_only=True,
    )

    # Shrink the loaded Fast config for a cheap functional check.
    cfg.vocab_size = 320
    cfg.hidden_size = 64
    cfg.intermediate_size = 128
    cfg.num_hidden_layers = 2
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    cfg.head_dim = 16
    cfg.max_position_embeddings = 128
    cfg.max_window_layers = 2
    cfg.layer_types = ["full_attention"] * 2
    cfg.pad_token_id = 0
    cfg.bos_token_id = 1
    cfg.eos_token_id = 2
    cfg.mask_token_id = 3
    cfg.bd_size = 4
    cfg.attention_bias = False
    cfg.use_sliding_window = False
    cfg.sliding_window = None

    try:
        cfg._attn_implementation = "eager"
    except Exception:
        pass

    model = AutoModelForCausalLM.from_config(
        cfg,
        trust_remote_code=True,
    )
    model.eval()

    input_ids = torch.randint(4, cfg.vocab_size, (2, 8), dtype=torch.long)

    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=False, block_size=4)

    assert out.logits.shape == (2, 8, cfg.vocab_size)
    assert torch.isfinite(out.logits).all().item()

    print("[OK] tiny forward via AutoConfig check passed.")


def optional_full_weight_load():
    print("[CHECK] optional full AutoModelForCausalLM.from_pretrained load")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )

    print("model class:", model.__class__.__name__)

    attn = model.model.layers[0].self_attn

    print("q_norm:", hasattr(attn, "q_norm"))
    print("k_norm:", hasattr(attn, "k_norm"))
    print("q_proj.bias:", attn.q_proj.bias)

    assert model.__class__.__name__ == "Fast_dLLM_Qwen3ForCausalLM"
    assert hasattr(attn, "q_norm")
    assert hasattr(attn, "k_norm")
    assert attn.q_proj.bias is None
    assert model.config.model_type == "Fast_dLLM_Qwen3"
    assert model.config.mask_token_id == 151669

    print("[OK] optional full weight AutoModel load passed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-load", action="store_true")
    args = parser.parse_args()

    check_raw_config_json()
    check_autoconfig_and_tokenizer()
    check_automodel_from_config_and_keys()
    check_manual_tiny_forward_via_config()

    if args.full_load:
        optional_full_weight_load()

    print("[OK] Step 5 config.json evaluation passed.")


if __name__ == "__main__":
    main()
