import json
import shutil
from pathlib import Path

MODEL_DIR = Path(".").resolve()
TC_PATH = MODEL_DIR / "tokenizer_config.json"
JINJA_PATH = MODEL_DIR / "chat_template.jinja"
AUDIT_DIR = MODEL_DIR / "audits"
BACKUP_DIR = AUDIT_DIR / "pre_step7_chat_template_backup"

AUDIT_DIR.mkdir(exist_ok=True)
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

if not JINJA_PATH.exists():
    raise FileNotFoundError("chat_template.jinja does not exist.")

backup_path = BACKUP_DIR / "tokenizer_config_pre_chat_template_patch.json"
if not backup_path.exists():
    shutil.copy2(TC_PATH, backup_path)
    print(f"[BACKUP] {TC_PATH} -> {backup_path}")

with open(TC_PATH, "r", encoding="utf-8") as f:
    tokenizer_config = json.load(f)

jinja_text = JINJA_PATH.read_text(encoding="utf-8")

before = tokenizer_config.get("chat_template")
tokenizer_config["chat_template"] = jinja_text

with open(TC_PATH, "w", encoding="utf-8") as f:
    json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)
    f.write("\n")

audit = {
    "chat_template_was_present": before is not None,
    "chat_template_length": len(jinja_text),
    "synced_from": str(JINJA_PATH),
    "synced_to": str(TC_PATH),
}

(AUDIT_DIR / "step7_chat_template_patch_audit.json").write_text(
    json.dumps(audit, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("[OK] Synced chat_template.jinja into tokenizer_config.json['chat_template'].")
print(json.dumps(audit, indent=2, ensure_ascii=False))
