import json
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

MODEL_DIR = Path(".").resolve()
sys.path.insert(0, str(MODEL_DIR))


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_token_in_tokenizer_json(tokenizer_json, token):
    matches = []
    for item in tokenizer_json.get("added_tokens", []):
        if isinstance(item, dict) and item.get("content") == token:
            matches.append(item)
    return matches


def find_token_in_added_tokens_json(added_tokens_json, token):
    if added_tokens_json is None:
        return None

    if isinstance(added_tokens_json, dict):
        return added_tokens_json.get(token)

    if isinstance(added_tokens_json, list):
        for item in added_tokens_json:
            if isinstance(item, dict) and item.get("content") == token:
                return item.get("id")

    return None


def check_config_generation_tokenizer_loaders():
    print("[CHECK] AutoConfig / GenerationConfig / AutoTokenizer")

    cfg = AutoConfig.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        local_files_only=True,
    )
    gen = GenerationConfig.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )
    tok = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        local_files_only=True,
    )

    assert cfg.__class__.__name__ == "Fast_dLLM_Qwen3Config"
    assert cfg.model_type == "Fast_dLLM_Qwen3"
    assert cfg.vocab_size == 151936
    assert cfg.mask_token_id == 151669
    assert cfg.mask_token == "|<MASK>|"
    assert cfg.pad_token_id == 151643
    assert cfg.bos_token_id == 151643
    assert cfg.eos_token_id == 151645

    assert gen.bos_token_id == 151643
    assert gen.eos_token_id == [151645, 151643]
    assert gen.pad_token_id == 151643

    assert len(tok) == 151670
    assert len(tok) <= cfg.vocab_size
    assert tok.convert_tokens_to_ids("|<MASK>|") == 151669
    assert tok.pad_token_id == 151643
    assert tok.bos_token_id == 151643
    assert tok.eos_token_id == 151645

    print("[OK] loaders check passed.")


def check_raw_tokenizer_files():
    print("[CHECK] raw tokenizer asset files")

    cfg = read_json(MODEL_DIR / "config.json")
    tokenizer_config = read_json(MODEL_DIR / "tokenizer_config.json")
    tokenizer_json = read_json(MODEL_DIR / "tokenizer.json")
    special_tokens_map = read_json(MODEL_DIR / "special_tokens_map.json")
    added_tokens_json = read_json(MODEL_DIR / "added_tokens.json")

    mask_token = "|<MASK>|"
    mask_id = 151669

    assert cfg["mask_token_id"] == mask_id
    assert cfg["mask_token"] == mask_token

    tokenizer_json_matches = find_token_in_tokenizer_json(tokenizer_json, mask_token)
    assert len(tokenizer_json_matches) == 1
    assert tokenizer_json_matches[0]["id"] == mask_id
    assert tokenizer_json_matches[0].get("special") is True

    added_tokens_id = find_token_in_added_tokens_json(added_tokens_json, mask_token)
    assert added_tokens_id == mask_id

    tokenizer_config_text = json.dumps(tokenizer_config, ensure_ascii=False)
    special_tokens_map_text = json.dumps(special_tokens_map, ensure_ascii=False)

    assert mask_token in tokenizer_config_text
    assert mask_token in special_tokens_map_text

    important_expected = {
        "<|endoftext|>": 151643,
        "<|im_start|>": 151644,
        "<|im_end|>": 151645,
        "<tool_call>": 151657,
        "</tool_call>": 151658,
        "<tool_response>": 151665,
        "</tool_response>": 151666,
        "<think>": 151667,
        "</think>": 151668,
        "|<MASK>|": 151669,
    }

    tok = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        local_files_only=True,
    )

    actual = {
        token: tok.convert_tokens_to_ids(token)
        for token in important_expected
    }

    assert actual == important_expected, actual

    print("[OK] raw tokenizer files check passed.")


def check_chat_template_consistency():
    print("[CHECK] chat_template.jinja consistency")

    tokenizer_config = read_json(MODEL_DIR / "tokenizer_config.json")
    chat_template_from_config = tokenizer_config.get("chat_template")

    jinja_path = MODEL_DIR / "chat_template.jinja"
    assert jinja_path.exists(), "chat_template.jinja does not exist."

    jinja_text = jinja_path.read_text(encoding="utf-8")

    assert chat_template_from_config is not None
    assert jinja_text == chat_template_from_config

    # Qwen3 thinking/template-related tokens should be preserved.
    assert "<think>" in json.dumps(tokenizer_config, ensure_ascii=False) or "<think>" in jinja_text
    assert "</think>" in json.dumps(tokenizer_config, ensure_ascii=False) or "</think>" in jinja_text

    print("[OK] chat template consistency check passed.")


def check_no_weight_index_change_required():
    print("[CHECK] model.safetensors.index.json does not need tokenizer edits")

    index = read_json(MODEL_DIR / "model.safetensors.index.json")
    weight_keys = set(index["weight_map"].keys())

    assert "model.embed_tokens.weight" in weight_keys
    assert "lm_head.weight" in weight_keys
    assert "model.layers.0.self_attn.q_norm.weight" in weight_keys
    assert "model.layers.0.self_attn.k_norm.weight" in weight_keys

    # Tokenizer assets must not appear in weight index.
    forbidden_fragments = [
        "tokenizer",
        "added_tokens",
        "special_tokens",
        "chat_template",
        "mask_token",
    ]

    for key in weight_keys:
        for frag in forbidden_fragments:
            assert frag not in key, f"Unexpected tokenizer-like key in safetensors index: {key}"

    print("[OK] weight index sanity check passed.")


def check_automodel_key_alignment():
    print("[CHECK] AutoModelForCausalLM key alignment still unchanged")

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

    index = read_json(MODEL_DIR / "model.safetensors.index.json")
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

    print("[OK] AutoModel key alignment check passed.")


def check_tiny_generate_still_ok():
    print("[CHECK] tiny generate still works")

    from configuration import Fast_dLLM_Qwen3Config
    from modeling import Fast_dLLM_Qwen3ForCausalLM

    cfg = Fast_dLLM_Qwen3Config(
        vocab_size=320,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        max_window_layers=2,
        layer_types=["full_attention"] * 2,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        mask_token_id=3,
        bd_size=4,
        attention_bias=False,
        use_sliding_window=False,
        sliding_window=None,
    )
    try:
        cfg._attn_implementation = "eager"
    except Exception:
        pass

    model = Fast_dLLM_Qwen3ForCausalLM(cfg)
    model.eval()

    input_ids = torch.randint(4, cfg.vocab_size, (2, 3), dtype=torch.long)

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=5,
            eos_token_id=999999,
            pad_token_id=0,
            block_size=4,
            small_block_size=2,
            threshold=1.0,
            temperature=0.0,
            use_block_cache=False,
        )

    assert out.shape == (2, 8)
    assert torch.equal(out[:, :3], input_ids)
    assert not (out[:, 3:] == cfg.mask_token_id).any().item()

    print("[OK] tiny generate regression check passed.")


def main():
    check_config_generation_tokenizer_loaders()
    check_raw_tokenizer_files()
    check_chat_template_consistency()
    check_no_weight_index_change_required()
    check_automodel_key_alignment()
    check_tiny_generate_still_ok()

    print("[OK] Step 7 tokenizer asset evaluation passed.")


if __name__ == "__main__":
    main()
