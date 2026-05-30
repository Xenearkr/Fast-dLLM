import json
import shutil
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer

MODEL_DIR = Path(".").resolve()
AUDIT_DIR = MODEL_DIR / "audits"
BACKUP_DIR = AUDIT_DIR / "pre_step2_tokenizer_backup"
OUT_JSON = AUDIT_DIR / "step2_tokenizer_mask_audit.json"

MASK_TOKEN = "|<MASK>|"

AUDIT_DIR.mkdir(exist_ok=True)
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

# 1. 备份 tokenizer 相关文件
tokenizer_files = [
    "added_tokens.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "chat_template.jinja",
]

for name in tokenizer_files:
    src = MODEL_DIR / name
    if src.exists():
        dst = BACKUP_DIR / name
        shutil.copy2(src, dst)
        print(f"[BACKUP] {src.name} -> {dst}")

# 2. 加载 config / tokenizer
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

before_len = len(tokenizer)
before_added_vocab = dict(tokenizer.get_added_vocab())
before_additional_special_tokens = list(tokenizer.additional_special_tokens or [])

print(f"[INFO] config.vocab_size = {config.vocab_size}")
print(f"[INFO] len(tokenizer) before = {before_len}")
print(f"[INFO] current MASK id = {tokenizer.convert_tokens_to_ids(MASK_TOKEN)}")

# 3. 新增 |<MASK>|，保留 Qwen3 原 additional_special_tokens
if tokenizer.convert_tokens_to_ids(MASK_TOKEN) is None:
    new_additional_special_tokens = list(before_additional_special_tokens)
    if MASK_TOKEN not in new_additional_special_tokens:
        new_additional_special_tokens.append(MASK_TOKEN)

    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": new_additional_special_tokens}
    )
    print(f"[INFO] num_added = {num_added}")
else:
    print("[INFO] MASK token already exists, skip adding.")

after_len = len(tokenizer)
mask_token_id = tokenizer.convert_tokens_to_ids(MASK_TOKEN)

# 4. 基本安全检查
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
    MASK_TOKEN,
]

important_token_ids = {
    tok: tokenizer.convert_tokens_to_ids(tok)
    for tok in important_tokens
}

assert mask_token_id is not None, "MASK token was not added correctly."
assert mask_token_id < config.vocab_size, (
    f"MASK token id {mask_token_id} >= config.vocab_size {config.vocab_size}; "
    "embedding resize would be required."
)
assert important_token_ids[MASK_TOKEN] not in {
    important_token_ids["<tool_response>"],
    important_token_ids["</tool_response>"],
    important_token_ids["<think>"],
    important_token_ids["</think>"],
}, "MASK token id conflicts with Qwen3 tool/thinking tokens."

requires_resize_embeddings = after_len > config.vocab_size

# 5. 保存 tokenizer 文件
tokenizer.save_pretrained(MODEL_DIR)

# 6. 可选：把 Qwen3 chat_template 单独导出为 chat_template.jinja
#    内容来自 tokenizer_config.json 中的 Qwen3 原生模板，不使用 Fast-Qwen2.5 模板。
if tokenizer.chat_template is not None:
    chat_template_path = MODEL_DIR / "chat_template.jinja"
    chat_template_path.write_text(tokenizer.chat_template, encoding="utf-8")
    print(f"[WRITE] {chat_template_path}")

# 7. 写审计报告
audit = {
    "model_dir": str(MODEL_DIR),
    "mask_token": MASK_TOKEN,
    "mask_token_id": mask_token_id,
    "config_vocab_size": config.vocab_size,
    "len_tokenizer_before": before_len,
    "len_tokenizer_after": after_len,
    "requires_resize_embeddings": requires_resize_embeddings,
    "before_additional_special_tokens": before_additional_special_tokens,
    "after_additional_special_tokens": tokenizer.additional_special_tokens,
    "important_token_ids": important_token_ids,
    "pad_token": tokenizer.pad_token,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token": tokenizer.eos_token,
    "eos_token_id": tokenizer.eos_token_id,
    "special_tokens_map": tokenizer.special_tokens_map,
    "added_vocab_tail": dict(
        sorted(tokenizer.get_added_vocab().items(), key=lambda x: x[1])[-40:]
    ),
    "generated_files_expected": [
        "added_tokens.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "chat_template.jinja",
    ],
}

OUT_JSON.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

print("[OK] Step 2 tokenizer update complete.")
print(f"[OK] Wrote audit report to: {OUT_JSON}")
print(json.dumps({
    "mask_token_id": mask_token_id,
    "len_tokenizer_before": before_len,
    "len_tokenizer_after": after_len,
    "config_vocab_size": config.vocab_size,
    "requires_resize_embeddings": requires_resize_embeddings,
    "important_token_ids": important_token_ids,
}, indent=2, ensure_ascii=False))
