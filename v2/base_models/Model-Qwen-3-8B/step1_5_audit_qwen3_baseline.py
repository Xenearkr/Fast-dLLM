import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = Path(".").resolve()
OUT_JSON = MODEL_DIR / "step1_5_qwen3_baseline_audit.json"

print(f"[INFO] Auditing model dir: {MODEL_DIR}")

config = AutoConfig.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True,
    local_files_only=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True,
    local_files_only=True,
)

with open(MODEL_DIR / "model.safetensors.index.json", "r", encoding="utf-8") as f:
    index = json.load(f)

weight_map = index["weight_map"]

important_tokens = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<tool_call>",
    "</tool_call>",
    "<|tool_calls_section_begin|>",
    "<|tool_calls_section_end|>",
    "<tool_response>",
    "</tool_response>",
    "<think>",
    "</think>",
    "|<MASK>|",
]

token_ids = {}
for tok in important_tokens:
    try:
        token_ids[tok] = tokenizer.convert_tokens_to_ids(tok)
    except Exception as e:
        token_ids[tok] = f"ERROR: {repr(e)}"

added_vocab = tokenizer.get_added_vocab()

config_summary = {
    "model_type": getattr(config, "model_type", None),
    "architectures": getattr(config, "architectures", None),
    "vocab_size": getattr(config, "vocab_size", None),
    "hidden_size": getattr(config, "hidden_size", None),
    "intermediate_size": getattr(config, "intermediate_size", None),
    "num_hidden_layers": getattr(config, "num_hidden_layers", None),
    "num_attention_heads": getattr(config, "num_attention_heads", None),
    "num_key_value_heads": getattr(config, "num_key_value_heads", None),
    "head_dim": getattr(config, "head_dim", None),
    "attention_bias": getattr(config, "attention_bias", None),
    "rms_norm_eps": getattr(config, "rms_norm_eps", None),
    "rope_theta": getattr(config, "rope_theta", None),
    "max_position_embeddings": getattr(config, "max_position_embeddings", None),
    "bos_token_id": getattr(config, "bos_token_id", None),
    "eos_token_id": getattr(config, "eos_token_id", None),
    "pad_token_id": getattr(config, "pad_token_id", None),
    "tie_word_embeddings": getattr(config, "tie_word_embeddings", None),
    "use_cache": getattr(config, "use_cache", None),
}

tokenizer_summary = {
    "len_tokenizer": len(tokenizer),
    "tokenizer_vocab_size": getattr(tokenizer, "vocab_size", None),
    "bos_token": tokenizer.bos_token,
    "bos_token_id": tokenizer.bos_token_id,
    "eos_token": tokenizer.eos_token,
    "eos_token_id": tokenizer.eos_token_id,
    "pad_token": tokenizer.pad_token,
    "pad_token_id": tokenizer.pad_token_id,
    "unk_token": tokenizer.unk_token,
    "unk_token_id": tokenizer.unk_token_id,
    "additional_special_tokens": tokenizer.additional_special_tokens,
    "special_tokens_map": tokenizer.special_tokens_map,
    "important_token_ids": token_ids,
    "num_added_vocab_entries": len(added_vocab),
    "added_vocab_tail": dict(sorted(added_vocab.items(), key=lambda x: x[1])[-80:]),
    "has_chat_template": tokenizer.chat_template is not None,
    "chat_template_prefix": tokenizer.chat_template[:800] if tokenizer.chat_template else None,
}

# Inspect key names.
keys = list(weight_map.keys())

q_norm_keys = [k for k in keys if k.endswith("self_attn.q_norm.weight")]
k_norm_keys = [k for k in keys if k.endswith("self_attn.k_norm.weight")]
q_proj_bias_keys = [k for k in keys if k.endswith("self_attn.q_proj.bias")]
k_proj_bias_keys = [k for k in keys if k.endswith("self_attn.k_proj.bias")]
v_proj_bias_keys = [k for k in keys if k.endswith("self_attn.v_proj.bias")]
o_proj_bias_keys = [k for k in keys if k.endswith("self_attn.o_proj.bias")]

weight_key_summary = {
    "num_weight_keys": len(keys),
    "num_shards": len(set(weight_map.values())),
    "has_embed_tokens": "model.embed_tokens.weight" in weight_map,
    "has_lm_head": "lm_head.weight" in weight_map,
    "num_q_norm_keys": len(q_norm_keys),
    "num_k_norm_keys": len(k_norm_keys),
    "num_q_proj_bias_keys": len(q_proj_bias_keys),
    "num_k_proj_bias_keys": len(k_proj_bias_keys),
    "num_v_proj_bias_keys": len(v_proj_bias_keys),
    "num_o_proj_bias_keys": len(o_proj_bias_keys),
    "sample_q_norm_keys": q_norm_keys[:3],
    "sample_k_norm_keys": k_norm_keys[:3],
}

def get_tensor_shape(tensor_name: str):
    shard_name = weight_map[tensor_name]
    shard_path = MODEL_DIR / shard_name
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        return list(f.get_tensor(tensor_name).shape), shard_name

shape_names = [
    "model.embed_tokens.weight",
    "lm_head.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.self_attn.q_norm.weight",
    "model.layers.0.self_attn.k_norm.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.up_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
    "model.norm.weight",
]

tensor_shapes = {}
for name in shape_names:
    if name in weight_map:
        shape, shard = get_tensor_shape(name)
        tensor_shapes[name] = {"shape": shape, "shard": shard}
    else:
        tensor_shapes[name] = None

# Lightweight model load test. This only checks class/weights compatibility.
print("[INFO] Loading model for compatibility check...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True,
    local_files_only=True,
    torch_dtype=torch.float16,
    device_map="cpu",
)
model_class = model.__class__.__name__
param_count = sum(p.numel() for p in model.parameters())

audit = {
    "model_dir": str(MODEL_DIR),
    "model_class": model_class,
    "param_count": param_count,
    "config_summary": config_summary,
    "tokenizer_summary": tokenizer_summary,
    "weight_index_summary": weight_key_summary,
    "tensor_shapes": tensor_shapes,
    "decision_hints": {
        "tokenizer_len_le_config_vocab_size": len(tokenizer) <= config.vocab_size,
        "mask_token_already_exists": token_ids["|<MASK>|"] != tokenizer.unk_token_id,
        "qwen3_qk_norm_present_for_all_layers": (
            len(q_norm_keys) == config.num_hidden_layers
            and len(k_norm_keys) == config.num_hidden_layers
        ),
        "attention_bias_keys_absent": (
            len(q_proj_bias_keys)
            + len(k_proj_bias_keys)
            + len(v_proj_bias_keys)
            + len(o_proj_bias_keys)
            == 0
        ),
    },
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(audit, f, indent=2, ensure_ascii=False)

print(f"[OK] Wrote audit report to: {OUT_JSON}")
print(json.dumps(audit["decision_hints"], indent=2, ensure_ascii=False))
print("[INFO] Important token ids:")
print(json.dumps(token_ids, indent=2, ensure_ascii=False))