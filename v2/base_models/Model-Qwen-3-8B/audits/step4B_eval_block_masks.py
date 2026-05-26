import json
import sys
from pathlib import Path

import torch

MODEL_DIR = Path(".").resolve()
sys.path.insert(0, str(MODEL_DIR))

from configuration import Fast_dLLM_Qwen3Config
from modeling import (
    BaseModelOutputWithPastAndBlockCache,
    CausalLMOutputWithPastAndBlockCache,
    Fast_dLLM_Qwen3ForCausalLM,
    Fast_dLLM_Qwen3Model,
    block_diff_mask,
    eval_block_diff_mask,
)


def check_dataclasses():
    print("[CHECK] Fast output dataclasses")

    lm_out = CausalLMOutputWithPastAndBlockCache(
        logits=torch.zeros(1, 2, 16),
        past_key_values=None,
        block_past_key_values=None,
    )
    base_out = BaseModelOutputWithPastAndBlockCache(
        last_hidden_state=torch.zeros(1, 2, 8),
        past_key_values=None,
        block_past_key_values=None,
    )

    assert hasattr(lm_out, "block_past_key_values")
    assert hasattr(base_out, "block_past_key_values")
    assert lm_out.logits.shape == (1, 2, 16)
    assert base_out.last_hidden_state.shape == (1, 2, 8)

    print("[OK] Dataclass check passed.")


def check_eval_block_diff_mask():
    print("[CHECK] eval_block_diff_mask")

    block_size = 4
    seq_len = 12

    q_idx = torch.arange(seq_len)[:, None]
    kv_idx = torch.arange(seq_len)[None, :]

    mask = eval_block_diff_mask(q_idx=q_idx, kv_idx=kv_idx, block_size=block_size)

    assert mask.shape == (seq_len, seq_len)
    assert mask.dtype == torch.bool

    # Query block 0 can attend block 0 only.
    assert mask[0, 0:4].all().item()
    assert not mask[0, 4:8].any().item()
    assert not mask[0, 8:12].any().item()

    # Query block 1 can attend blocks 0 and 1, not block 2.
    assert mask[4, 0:8].all().item()
    assert not mask[4, 8:12].any().item()

    # Query block 2 can attend all earlier/current blocks.
    assert mask[8, 0:12].all().item()

    print("[OK] eval_block_diff_mask check passed.")


def check_training_block_diff_mask():
    print("[CHECK] block_diff_mask for [x_t ; x_0] training layout")

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

    # ------------------------------------------------------------------
    # x_t block 1 query: idx 4.
    # It should attend to x_t block 1, not x_t block 0/2.
    # ------------------------------------------------------------------
    q_xt_b1 = 4

    assert mask[q_xt_b1, 4:8].all().item(), "x_t block should attend within same noised block."
    assert not mask[q_xt_b1, 0:4].any().item(), "x_t block should not attend previous noised block."
    assert not mask[q_xt_b1, 8:12].any().item(), "x_t block should not attend future noised block."

    # x_t block 1 should attend previous clean block 0, but not current clean block 1.
    assert mask[q_xt_b1, n + 0:n + 4].all().item(), "x_t block 1 should attend clean block 0."
    assert not mask[q_xt_b1, n + 4:n + 8].any().item(), "x_t block 1 should not attend current clean block."
    assert not mask[q_xt_b1, n + 8:n + 12].any().item(), "x_t block 1 should not attend future clean block."

    # ------------------------------------------------------------------
    # x_t block 2 query: idx 8.
    # It should attend previous clean blocks 0 and 1, not current clean block 2.
    # ------------------------------------------------------------------
    q_xt_b2 = 8

    assert mask[q_xt_b2, 8:12].all().item(), "x_t block 2 should attend within same noised block."
    assert mask[q_xt_b2, n + 0:n + 8].all().item(), "x_t block 2 should attend previous clean blocks."
    assert not mask[q_xt_b2, n + 8:n + 12].any().item(), "x_t block 2 should not attend current clean block."

    # ------------------------------------------------------------------
    # x_0 block 1 query: idx n + 4.
    # It should attend clean blocks 0 and 1, not clean block 2,
    # and should not attend x_t tokens.
    # ------------------------------------------------------------------
    q_x0_b1 = n + 4

    assert not mask[q_x0_b1, 0:n].any().item(), "clean x_0 query should not attend noised x_t tokens."
    assert mask[q_x0_b1, n + 0:n + 8].all().item(), "x_0 block 1 should attend clean blocks 0 and 1."
    assert not mask[q_x0_b1, n + 8:n + 12].any().item(), "x_0 block 1 should not attend future clean block."

    # x_0 block 0 should not attend future clean blocks.
    q_x0_b0 = n
    assert mask[q_x0_b0, n:n + 4].all().item()
    assert not mask[q_x0_b0, n + 4:n + 12].any().item()

    print("[OK] block_diff_mask training-layout check passed.")


def check_4a_structure_still_ok():
    print("[CHECK] 4A structure still intact")

    cfg = Fast_dLLM_Qwen3Config()
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


def check_tiny_forward_still_ok():
    print("[CHECK] Tiny forward still works after 4B")

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

    try:
        cfg._attn_implementation = "eager"
    except Exception:
        pass

    model = Fast_dLLM_Qwen3ForCausalLM(cfg)
    model.eval()

    input_ids = torch.randint(4, cfg.vocab_size, (2, 8), dtype=torch.long)

    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=False)

    assert out.logits.shape == (2, 8, cfg.vocab_size)
    assert torch.isfinite(out.logits).all().item()

    print("[OK] Tiny forward check passed.")


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
    check_dataclasses()
    check_eval_block_diff_mask()
    check_training_block_diff_mask()
    check_4a_structure_still_ok()
    check_tiny_forward_still_ok()
    check_meta_state_dict_still_aligned()

    print("[OK] Milestone 4B evaluation passed.")


if __name__ == "__main__":
    main()
