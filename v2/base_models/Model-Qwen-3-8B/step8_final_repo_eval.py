import json
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

MODEL_DIR = Path(".").resolve()
sys.path.insert(0, str(MODEL_DIR))

import modeling as modeling_module
from configuration import Fast_dLLM_Qwen3Config
from modeling import Fast_dLLM_Qwen3ForCausalLM


def check_core_files_exist():
    print("[CHECK] core files exist")

    required = [
        "configuration.py",
        "modeling.py",
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "added_tokens.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "model.safetensors.index.json",
        "model-00001-of-00005.safetensors",
        "model-00002-of-00005.safetensors",
        "model-00003-of-00005.safetensors",
        "model-00004-of-00005.safetensors",
        "model-00005-of-00005.safetensors",
    ]

    missing = [x for x in required if not (MODEL_DIR / x).exists()]
    assert not missing, f"Missing files: {missing}"

    print("[OK] core files exist.")


def check_config_tokenizer_generation():
    print("[CHECK] config / tokenizer / generation config")

    cfg = AutoConfig.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        local_files_only=True,
    )
    tok = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        local_files_only=True,
    )
    gen = GenerationConfig.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )

    print("config class:", cfg.__class__.__name__)
    print("model_type:", cfg.model_type)
    print("tokenizer length:", len(tok))
    print("|<MASK>| id:", tok.convert_tokens_to_ids("|<MASK>|"))

    assert cfg.__class__.__name__ == "Fast_dLLM_Qwen3Config"
    assert cfg.model_type == "Fast_dLLM_Qwen3"

    assert cfg.vocab_size == 151936
    assert cfg.hidden_size == 4096
    assert cfg.intermediate_size == 12288
    assert cfg.num_hidden_layers == 36
    assert cfg.num_attention_heads == 32
    assert cfg.num_key_value_heads == 8
    assert cfg.head_dim == 128

    assert cfg.bd_size == 32
    assert cfg.mask_token_id == 151669
    assert cfg.mask_token == "|<MASK>|"
    assert cfg.pad_token_id == 151643
    assert cfg.bos_token_id == 151643
    assert cfg.eos_token_id == 151645

    assert len(tok) == 151670
    assert len(tok) <= cfg.vocab_size
    assert tok.convert_tokens_to_ids("|<MASK>|") == 151669
    assert tok.pad_token_id == 151643
    assert tok.eos_token_id == 151645

    assert gen.bos_token_id == 151643
    assert gen.eos_token_id == [151645, 151643]
    assert gen.pad_token_id == 151643

    assert tok.chat_template is not None
    assert (MODEL_DIR / "chat_template.jinja").exists()

    print("[OK] config/tokenizer/generation checks passed.")


def check_no_bad_mask_id():
    print("[CHECK] no bad mask id")

    text = (MODEL_DIR / "modeling.py").read_text(encoding="utf-8")

    assert "mask_id=151665" not in text
    assert "mask_id: Optional[int] = 151665" not in text
    assert "mask_token_id=151665" not in text

    print("[OK] no hard-coded 151665 mask id found.")


def check_automodel_key_alignment():
    print("[CHECK] AutoModelForCausalLM state_dict key alignment")

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

    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)

    print(f"expected weight keys: {len(expected_keys)}")
    print(f"actual model keys:    {len(actual_keys)}")
    print(f"missing:              {len(missing)}")
    print(f"extra:                {len(extra)}")

    if missing:
        print("[MISSING sample]", missing[:30])
    if extra:
        print("[EXTRA sample]", extra[:30])

    assert not missing
    assert not extra

    attn = model.model.layers[0].self_attn
    assert hasattr(attn, "q_norm")
    assert hasattr(attn, "k_norm")
    assert attn.q_proj.bias is None

    print("[OK] key alignment passed.")


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
        complementary_mask=True,
    )

    try:
        cfg._attn_implementation = "eager"
    except Exception:
        pass

    return cfg


def check_tiny_training_and_generate():
    print("[CHECK] tiny train forward and generate")

    torch.manual_seed(1234)

    cfg = make_tiny_config()
    model = Fast_dLLM_Qwen3ForCausalLM(cfg)

    old_create_block_mask = modeling_module.create_block_mask
    modeling_module.create_block_mask = None

    try:
        model.train()
        input_ids = torch.randint(4, cfg.vocab_size, (2, 8), dtype=torch.long)
        labels = input_ids.clone()
        labels[:, :2] = -100

        out = model(
            input_ids=input_ids,
            labels=labels,
            use_cache=True,
            block_size=4,
        )

        assert out.loss is not None
        assert torch.isfinite(out.loss).item()
        assert out.logits.shape == (4, 8, cfg.vocab_size)
        assert out.past_key_values is None
        assert out.block_past_key_values is None

    finally:
        modeling_module.create_block_mask = old_create_block_mask

    model.eval()
    prompt = torch.randint(4, cfg.vocab_size, (2, 3), dtype=torch.long)

    with torch.no_grad():
        generated = model.generate(
            input_ids=prompt,
            max_new_tokens=5,
            eos_token_id=999999,
            pad_token_id=0,
            block_size=4,
            small_block_size=2,
            threshold=1.0,
            temperature=0.0,
            use_block_cache=False,
        )

    assert generated.shape == (2, 8)
    assert torch.equal(generated[:, :3], prompt)
    assert not (generated[:, 3:] == cfg.mask_token_id).any().item()

    print("[OK] tiny train/generate checks passed.")


def optional_full_load_check():
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
    print("config model_type:", model.config.model_type)

    assert model.__class__.__name__ == "Fast_dLLM_Qwen3ForCausalLM"
    assert model.config.model_type == "Fast_dLLM_Qwen3"
    assert model.config.mask_token_id == 151669

    attn = model.model.layers[0].self_attn
    assert hasattr(attn, "q_norm")
    assert hasattr(attn, "k_norm")
    assert attn.q_proj.bias is None

    print("[OK] full-load check passed.")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--full-load", action="store_true")
    args = parser.parse_args()

    check_core_files_exist()
    check_config_tokenizer_generation()
    check_no_bad_mask_id()
    check_automodel_key_alignment()
    check_tiny_training_and_generate()

    if args.full_load:
        optional_full_load_check()

    print("[OK] Step 8 final repository evaluation passed.")


if __name__ == "__main__":
    main()
