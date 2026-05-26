import json
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = Path(".").resolve()
sys.path.insert(0, str(MODEL_DIR))

import modeling as modeling_module
from configuration import Fast_dLLM_Qwen3Config
from modeling import Fast_dLLM_Qwen3ForCausalLM


def patch_dense_training_mask():
    old_create_block_mask = modeling_module.create_block_mask
    modeling_module.create_block_mask = None
    return old_create_block_mask


def restore_dense_training_mask(old_create_block_mask):
    modeling_module.create_block_mask = old_create_block_mask


def make_tiny_config(complementary_mask=True):
    cfg = Fast_dLLM_Qwen3Config(
        vocab_size=512,
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
        mask_token="|<MASK>|",
        bd_size=4,
        complementary_mask=complementary_mask,
        conplemenrary_mask=complementary_mask,
        attention_bias=False,
        attention_dropout=0.0,
        use_sliding_window=False,
        sliding_window=None,
        use_cache=True,
    )

    try:
        cfg._attn_implementation = "eager"
    except Exception:
        pass

    return cfg


def build_synthetic_batch(vocab_size, batch_size=2, seq_len=8, prompt_len=2):
    assert seq_len % 4 == 0

    input_ids = torch.randint(
        low=4,
        high=vocab_size,
        size=(batch_size, seq_len),
        dtype=torch.long,
    )

    labels = input_ids.clone()
    labels[:, :prompt_len] = -100

    return input_ids, labels


def assert_grad_ok(model, name):
    param = dict(model.named_parameters())[name]
    assert param.grad is not None, f"{name} grad is None"
    assert torch.isfinite(param.grad).all().item(), f"{name} grad contains non-finite values"
    grad_norm = param.grad.detach().float().norm().item()
    assert grad_norm > 0, f"{name} grad norm is zero"
    print(f"  grad[{name}] norm = {grad_norm:.6f}")


def check_direct_tiny_backward(complementary_mask):
    print(f"[CHECK] direct tiny backward, complementary_mask={complementary_mask}")

    torch.manual_seed(1234)

    cfg = make_tiny_config(complementary_mask=complementary_mask)
    model = Fast_dLLM_Qwen3ForCausalLM(cfg)
    model.train()
    model.zero_grad(set_to_none=True)

    input_ids, labels = build_synthetic_batch(
        vocab_size=cfg.vocab_size,
        batch_size=2,
        seq_len=8,
        prompt_len=2,
    )

    old_create_block_mask = patch_dense_training_mask()
    try:
        out = model(
            input_ids=input_ids,
            labels=labels,
            use_cache=True,  # training branch should disable this internally
            block_size=4,
        )
    finally:
        restore_dense_training_mask(old_create_block_mask)

    expected_batch = 4 if complementary_mask else 2

    assert out.loss is not None
    assert torch.isfinite(out.loss).item(), out.loss
    assert out.logits.shape == (expected_batch, 8, cfg.vocab_size), out.logits.shape
    assert out.past_key_values is None
    assert out.block_past_key_values is None

    print("  loss:", float(out.loss.detach().cpu()))

    out.loss.backward()

    assert_grad_ok(model, "lm_head.weight")
    assert_grad_ok(model, "model.embed_tokens.weight")
    assert_grad_ok(model, "model.layers.0.self_attn.q_proj.weight")
    assert_grad_ok(model, "model.layers.0.self_attn.q_norm.weight")
    assert_grad_ok(model, "model.layers.0.mlp.gate_proj.weight")

    print("[OK] direct tiny backward passed.")


def check_auto_config_tiny_backward():
    print("[CHECK] AutoConfig + AutoModelForCausalLM tiny backward")

    torch.manual_seed(5678)

    cfg = AutoConfig.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        local_files_only=True,
    )

    assert cfg.__class__.__name__ == "Fast_dLLM_Qwen3Config"
    assert cfg.model_type == "Fast_dLLM_Qwen3"
    assert cfg.mask_token_id == 151669

    # Shrink real repo config to tiny dimensions while preserving remote-code class.
    cfg.vocab_size = 512
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
    cfg.mask_token = "|<MASK>|"
    cfg.bd_size = 4
    cfg.complementary_mask = True
    cfg.conplemenrary_mask = True
    cfg.attention_bias = False
    cfg.attention_dropout = 0.0
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
    model.train()
    model.zero_grad(set_to_none=True)

    assert model.__class__.__name__ == "Fast_dLLM_Qwen3ForCausalLM"

    input_ids, labels = build_synthetic_batch(
        vocab_size=cfg.vocab_size,
        batch_size=2,
        seq_len=8,
        prompt_len=2,
    )

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

    assert out.loss is not None
    assert torch.isfinite(out.loss).item()
    assert out.logits.shape == (4, 8, cfg.vocab_size)

    print("  loss:", float(out.loss.detach().cpu()))

    out.loss.backward()

    assert_grad_ok(model, "lm_head.weight")
    assert_grad_ok(model, "model.layers.0.self_attn.k_proj.weight")
    assert_grad_ok(model, "model.layers.0.self_attn.k_norm.weight")
    assert_grad_ok(model, "model.layers.1.mlp.down_proj.weight")

    print("[OK] AutoConfig tiny backward passed.")


def check_training_batch_invariants():
    print("[CHECK] training batch preparation invariants")

    torch.manual_seed(42)

    cfg = make_tiny_config(complementary_mask=True)
    model = Fast_dLLM_Qwen3ForCausalLM(cfg)
    model.train()

    input_ids = torch.arange(4, 4 + 2 * 8, dtype=torch.long).reshape(2, 8)
    labels = input_ids.clone()
    labels[:, :2] = -100

    train_input_ids, train_labels = model._prepare_block_diffusion_training_batch(
        input_ids=input_ids,
        labels=labels,
        mask_id=cfg.mask_token_id,
    )

    assert train_input_ids.shape == (4, 16)
    assert train_labels.shape == (4, 8)

    noised_half = train_input_ids[:, :8]
    clean_half = train_input_ids[:, 8:]

    assert torch.equal(clean_half[:2], input_ids)
    assert torch.equal(clean_half[2:], input_ids)
    assert (train_labels[:, :2] == -100).all().item()

    supervised = train_labels != -100
    assert supervised.any().item()
    assert (noised_half[supervised] == cfg.mask_token_id).all().item()

    primary = train_labels[:2, 2:] != -100
    complement = train_labels[2:, 2:] != -100

    # With complementary mask, every trainable token is supervised exactly once
    # across the primary/complementary pair.
    assert torch.equal(primary | complement, torch.ones_like(primary))
    assert not (primary & complement).any().item()

    print("[OK] training batch invariants passed.")


def check_repo_tokenizer_and_config_still_ok():
    print("[CHECK] repo tokenizer/config still OK")

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

    assert cfg.model_type == "Fast_dLLM_Qwen3"
    assert cfg.mask_token_id == 151669
    assert cfg.mask_token == "|<MASK>|"
    assert cfg.pad_token_id == 151643
    assert tok.convert_tokens_to_ids("|<MASK>|") == 151669
    assert len(tok) == 151670
    assert len(tok) <= cfg.vocab_size

    print("[OK] repo tokenizer/config check passed.")


def check_no_hardcoded_bad_mask_id():
    print("[CHECK] no hard-coded bad mask id")

    text = (MODEL_DIR / "modeling.py").read_text(encoding="utf-8")

    assert "mask_id=151665" not in text
    assert "mask_id: Optional[int] = 151665" not in text
    assert "mask_token_id=151665" not in text

    print("[OK] no hard-coded 151665 mask id found.")


def check_state_dict_key_alignment_meta():
    print("[CHECK] meta state_dict key alignment")

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

    print(f"  expected keys: {len(expected_keys)}")
    print(f"  actual keys:   {len(actual_keys)}")
    print(f"  missing:       {len(missing)}")
    print(f"  extra:         {len(extra)}")

    if missing:
        print("[MISSING sample]", missing[:30])
    if extra:
        print("[EXTRA sample]", extra[:30])

    assert not missing
    assert not extra

    print("[OK] meta state_dict key alignment passed.")


def main():
    check_repo_tokenizer_and_config_still_ok()
    check_no_hardcoded_bad_mask_id()
    check_state_dict_key_alignment_meta()
    check_training_batch_invariants()
    check_direct_tiny_backward(complementary_mask=True)
    check_direct_tiny_backward(complementary_mask=False)
    check_auto_config_tiny_backward()

    print("[OK] Step 9A tiny training smoke test passed.")


if __name__ == "__main__":
    main()
