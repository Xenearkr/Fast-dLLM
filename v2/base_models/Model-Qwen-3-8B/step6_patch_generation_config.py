import json
import shutil
from pathlib import Path

MODEL_DIR = Path(".").resolve()
GEN_PATH = MODEL_DIR / "generation_config.json"
CONFIG_PATH = MODEL_DIR / "config.json"
AUDIT_DIR = MODEL_DIR / "audits"
BACKUP_DIR = AUDIT_DIR / "pre_step6_generation_config_backup"
OUT_JSON = AUDIT_DIR / "step6_generation_config_audit.json"

AUDIT_DIR.mkdir(exist_ok=True)
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

backup_path = BACKUP_DIR / "generation_config_pre_step6.json"
if GEN_PATH.exists() and not backup_path.exists():
    shutil.copy2(GEN_PATH, backup_path)
    print(f"[BACKUP] {GEN_PATH} -> {backup_path}")
elif backup_path.exists():
    print(f"[INFO] Backup already exists: {backup_path}")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    model_cfg = json.load(f)

before = {}
if GEN_PATH.exists():
    with open(GEN_PATH, "r", encoding="utf-8") as f:
        before = json.load(f)

# Sanity checks: Step 5 should already be done.
assert model_cfg["model_type"] == "Fast_dLLM_Qwen3"
assert model_cfg["architectures"] == ["Fast_dLLM_Qwen3ForCausalLM"]
assert model_cfg["mask_token_id"] == 151669
assert model_cfg["pad_token_id"] == 151643
assert model_cfg["bos_token_id"] == 151643
assert model_cfg["eos_token_id"] == 151645

# Step 6 policy:
# - Preserve Qwen3 token ids and Qwen3 sampling defaults.
# - Do not add repetition_penalty for now, because our custom Fast generate()
#   does not implement repetition_penalty. Adding ignored fields can mislead
#   downstream users.
gen = {
    "bos_token_id": 151643,
    "do_sample": True,
    "eos_token_id": [151645, 151643],
    "pad_token_id": 151643,
    "temperature": 0.6,
    "top_k": 20,
    "top_p": 0.95,
    "transformers_version": before.get(
        "transformers_version",
        model_cfg.get("transformers_version", "4.51.0"),
    ),
}

with open(GEN_PATH, "w", encoding="utf-8") as f:
    json.dump(gen, f, indent=2, ensure_ascii=False)
    f.write("\n")

changed_keys = {
    key: {
        "before": before.get(key, "<MISSING>"),
        "after": gen.get(key, "<MISSING>"),
    }
    for key in sorted(set(before.keys()) | set(gen.keys()))
    if before.get(key, "<MISSING>") != gen.get(key, "<MISSING>")
}

audit = {
    "model_dir": str(MODEL_DIR),
    "policy": "Keep Qwen3 generation defaults; Fast diffusion decoding knobs live in modeling.py.generate().",
    "changed_keys": changed_keys,
    "final_generation_config": gen,
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(audit, f, indent=2, ensure_ascii=False)
    f.write("\n")

print("[OK] Step 6 generation_config.json patch complete.")
print(json.dumps(gen, indent=2, ensure_ascii=False))
print(f"[OK] Wrote audit report to: {OUT_JSON}")
