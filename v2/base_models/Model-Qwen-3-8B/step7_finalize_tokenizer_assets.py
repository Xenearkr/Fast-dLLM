import hashlib
import json
import shutil
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer, GenerationConfig

MODEL_DIR = Path(".").resolve()
AUDIT_DIR = MODEL_DIR / "audits"
OUT_JSON = AUDIT_DIR / "step7_tokenizer_assets_audit.json"

AUDIT_DIR.mkdir(exist_ok=True)

TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "added_tokens.json",
    "special_tokens_map.json",
    "chat_template.jinja",
]


def sha256_file(path: Path):
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


before_hashes = {
    name: sha256_file(MODEL_DIR / name)
    for name in TOKENIZER_FILES + ["config.json", "generation_config.json", "model.safetensors.index.json"]
}

cfg = AutoConfig.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True,
    local_files_only=True,
)

gen_cfg = GenerationConfig.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True,
    local_files_only=True,
)

mask_token = getattr(cfg, "mask_token", "|<MASK>|")
mask_token_id = getattr(cfg, "mask_token_id", 151669)

changed = False
messages = []

# ---------------------------------------------------------------------
# 1. Verify / repair mask token.
# ---------------------------------------------------------------------
current_mask_id = tokenizer.convert_tokens_to_ids(mask_token)

if current_mask_id is None:
    additional = list(tokenizer.additional_special_tokens or [])
    if mask_token not in additional:
        additional.append(mask_token)

    tokenizer.add_special_tokens(
        {"additional_special_tokens": additional}
    )
    changed = True
    messages.append(f"Added missing mask token {mask_token!r}.")
    current_mask_id = tokenizer.convert_tokens_to_ids(mask_token)

if current_mask_id != mask_token_id:
    raise RuntimeError(
        f"Mask token id mismatch: tokenizer has {current_mask_id}, config expects {mask_token_id}."
    )

# Ensure mask token is listed as additional special token.
additional_special_tokens = list(tokenizer.additional_special_tokens or [])
if mask_token not in additional_special_tokens:
    additional_special_tokens.append(mask_token)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": additional_special_tokens}
    )
    changed = True
    messages.append(f"Registered {mask_token!r} as additional_special_token.")

# ---------------------------------------------------------------------
# 2. Verify / repair bos/eos/pad special tokens.
# ---------------------------------------------------------------------
expected_pad_id = int(cfg.pad_token_id)
expected_bos_id = int(cfg.bos_token_id)
expected_eos_id = int(cfg.eos_token_id)

if tokenizer.pad_token_id != expected_pad_id:
    tokenizer.pad_token = tokenizer.convert_ids_to_tokens(expected_pad_id)
    changed = True
    messages.append(f"Set pad_token_id to {expected_pad_id}.")

if tokenizer.bos_token_id != expected_bos_id:
    tokenizer.bos_token = tokenizer.convert_ids_to_tokens(expected_bos_id)
    changed = True
    messages.append(f"Set bos_token_id to {expected_bos_id}.")

if tokenizer.eos_token_id != expected_eos_id:
    tokenizer.eos_token = tokenizer.convert_ids_to_tokens(expected_eos_id)
    changed = True
    messages.append(f"Set eos_token_id to {expected_eos_id}.")

# ---------------------------------------------------------------------
# 3. Save tokenizer only if needed.
# ---------------------------------------------------------------------
if changed:
    tokenizer.save_pretrained(MODEL_DIR)
    print("[WRITE] tokenizer files updated via tokenizer.save_pretrained().")
else:
    print("[INFO] tokenizer files already consistent; no tokenizer save needed.")

# ---------------------------------------------------------------------
# 4. Sync chat_template.jinja from tokenizer.chat_template.
# ---------------------------------------------------------------------
chat_template = getattr(tokenizer, "chat_template", None)

tokenizer_config_path = MODEL_DIR / "tokenizer_config.json"
tokenizer_config = read_json(tokenizer_config_path) or {}

if chat_template is None:
    chat_template = tokenizer_config.get("chat_template")

chat_template_path = MODEL_DIR / "chat_template.jinja"

if chat_template is None and chat_template_path.exists():
    chat_template = chat_template_path.read_text(encoding="utf-8")

if chat_template:
    old_jinja = chat_template_path.read_text(encoding="utf-8") if chat_template_path.exists() else None
    if old_jinja != chat_template:
        chat_template_path.write_text(chat_template, encoding="utf-8")
        messages.append("Synced chat_template.jinja.")
        print("[WRITE] chat_template.jinja synced.")

    if tokenizer_config.get("chat_template") != chat_template:
        tokenizer_config["chat_template"] = chat_template
        with open(tokenizer_config_path, "w", encoding="utf-8") as f:
            json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)
            f.write("\n")
        messages.append("Synced tokenizer_config.json chat_template.")
        print("[WRITE] tokenizer_config.json chat_template synced.")
else:
    messages.append("No chat_template found in tokenizer/tokenizer_config/chat_template.jinja.")

# ---------------------------------------------------------------------
# 5. Post-checks.
# ---------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True,
    local_files_only=True,
)

after_hashes = {
    name: sha256_file(MODEL_DIR / name)
    for name in TOKENIZER_FILES + ["config.json", "generation_config.json", "model.safetensors.index.json"]
}

important_tokens = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
    "<think>",
    "</think>",
    mask_token,
]

important_token_ids = {
    tok: tokenizer.convert_tokens_to_ids(tok)
    for tok in important_tokens
}

assert important_token_ids[mask_token] == mask_token_id
assert len(tokenizer) <= cfg.vocab_size
assert tokenizer.pad_token_id == expected_pad_id
assert tokenizer.bos_token_id == expected_bos_id
assert tokenizer.eos_token_id == expected_eos_id

# model.safetensors.index.json must not change in Step 7.
assert before_hashes["model.safetensors.index.json"] == after_hashes["model.safetensors.index.json"], (
    "model.safetensors.index.json changed unexpectedly during Step 7."
)

audit = {
    "model_dir": str(MODEL_DIR),
    "changed": changed,
    "messages": messages,
    "config": {
        "model_type": cfg.model_type,
        "vocab_size": cfg.vocab_size,
        "mask_token": mask_token,
        "mask_token_id": mask_token_id,
        "pad_token_id": cfg.pad_token_id,
        "bos_token_id": cfg.bos_token_id,
        "eos_token_id": cfg.eos_token_id,
    },
    "generation_config": {
        "bos_token_id": gen_cfg.bos_token_id,
        "eos_token_id": gen_cfg.eos_token_id,
        "pad_token_id": gen_cfg.pad_token_id,
        "temperature": gen_cfg.temperature,
        "top_k": gen_cfg.top_k,
        "top_p": gen_cfg.top_p,
    },
    "tokenizer": {
        "len": len(tokenizer),
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token": tokenizer.bos_token,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "additional_special_tokens_tail": list(tokenizer.additional_special_tokens or [])[-30:],
        "important_token_ids": important_token_ids,
    },
    "file_hash_changes": {
        name: {
            "before": before_hashes[name],
            "after": after_hashes[name],
            "changed": before_hashes[name] != after_hashes[name],
        }
        for name in sorted(before_hashes.keys())
    },
}

OUT_JSON.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

print("[OK] Step 7 tokenizer asset finalization complete.")
print(json.dumps(audit["tokenizer"], indent=2, ensure_ascii=False))
print(f"[OK] Wrote audit report to: {OUT_JSON}")
