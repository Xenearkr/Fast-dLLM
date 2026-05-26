import json
import shutil
from pathlib import Path

MODEL_DIR = Path(".").resolve()
CONFIG_PATH = MODEL_DIR / "config.json"
AUDIT_DIR = MODEL_DIR / "audits"
BACKUP_DIR = AUDIT_DIR / "pre_step5_config_backup"
OUT_JSON = AUDIT_DIR / "step5_config_json_audit.json"

AUDIT_DIR.mkdir(exist_ok=True)
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

backup_path = BACKUP_DIR / "config_qwen3_original.json"
if not backup_path.exists():
    shutil.copy2(CONFIG_PATH, backup_path)
    print(f"[BACKUP] {CONFIG_PATH} -> {backup_path}")
else:
    print(f"[INFO] Backup already exists: {backup_path}")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = json.load(f)

before = dict(cfg)

# Safety checks: avoid patching the wrong model.
expected_qwen3_shape = {
    "vocab_size": 151936,
    "hidden_size": 4096,
    "intermediate_size": 12288,
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
}

for key, expected in expected_qwen3_shape.items():
    actual = cfg.get(key)
    if actual != expected:
        raise RuntimeError(
            f"Unexpected Qwen3-8B config value for {key}: actual={actual}, expected={expected}. "
            "Refusing to patch config.json."
        )

# Fast-dLLM Qwen3 custom-code entry points.
cfg["architectures"] = ["Fast_dLLM_Qwen3ForCausalLM"]
cfg["model_type"] = "Fast_dLLM_Qwen3"
cfg["auto_map"] = {
    "AutoConfig": "configuration.Fast_dLLM_Qwen3Config",
    "AutoModel": "modeling.Fast_dLLM_Qwen3Model",
    "AutoModelForCausalLM": "modeling.Fast_dLLM_Qwen3ForCausalLM",
}

# Fast-dLLM v2 method fields.
cfg["bd_size"] = 32
cfg["mask_token_id"] = 151669
cfg["mask_token"] = "|<MASK>|"

# Use the corrected spelling, while preserving Fast's historical typo for compatibility.
cfg["complementary_mask"] = True
cfg["conplemenrary_mask"] = True

# Qwen3 tokenizer/padding convention from our Step 1.5/2 audit.
cfg["pad_token_id"] = 151643

# Qwen3-8B has no sliding attention; make this explicit for our custom config.
cfg["layer_types"] = ["full_attention"] * int(cfg["num_hidden_layers"])

# Keep Qwen3 structural fields untouched; only normalize fields that must exist.
cfg["attention_bias"] = False
cfg["attention_dropout"] = float(cfg.get("attention_dropout", 0.0))
cfg["rope_scaling"] = cfg.get("rope_scaling", None)
cfg["rope_theta"] = cfg.get("rope_theta", 1000000)

with open(CONFIG_PATH, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2, ensure_ascii=False)
    f.write("\n")

changed_keys = {
    key: {
        "before": before.get(key, "<MISSING>"),
        "after": cfg.get(key),
    }
    for key in sorted(set(before.keys()) | set(cfg.keys()))
    if before.get(key, "<MISSING>") != cfg.get(key)
}

audit = {
    "model_dir": str(MODEL_DIR),
    "changed_keys": changed_keys,
    "final_core_fields": {
        "architectures": cfg["architectures"],
        "model_type": cfg["model_type"],
        "auto_map": cfg["auto_map"],
        "bd_size": cfg["bd_size"],
        "mask_token_id": cfg["mask_token_id"],
        "mask_token": cfg["mask_token"],
        "complementary_mask": cfg["complementary_mask"],
        "conplemenrary_mask": cfg["conplemenrary_mask"],
        "pad_token_id": cfg["pad_token_id"],
        "layer_types_len": len(cfg["layer_types"]),
        "layer_types_unique": sorted(set(cfg["layer_types"])),
    },
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(audit, f, indent=2, ensure_ascii=False)
    f.write("\n")

print("[OK] Step 5 config.json patch complete.")
print(json.dumps(audit["final_core_fields"], indent=2, ensure_ascii=False))
print(f"[OK] Wrote audit report to: {OUT_JSON}")
