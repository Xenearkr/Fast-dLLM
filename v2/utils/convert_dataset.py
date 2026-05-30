"""
使用说明：
- 用于转换Llama-Nemotron数据集格式，使之可直接用于finetune.py

使用方法：
- 对于SFT/code/v1.1，已经搞定了

"""

import json
from pathlib import Path


INPUT_PATH = Path(
    "/home/u-shengbf/Codes/Fast-dLLM/v2/data/"
    "Llama-Nemotron-math-v1.1/SFT/math/math_v1.1.jsonl"
)

OUTPUT_DIR = Path(
    "/home/u-shengbf/Codes/Fast-dLLM/v2/data/"
    "Llama-Nemotron-math-v1.1/train_conversation"
)

SHARD_SIZE = 10000
MAX_EXAMPLES = None  # 调试可改成 1000


def normalize_messages(input_field, output_text, system_prompt=None):
    messages = []
    extracted_system = ""

    if isinstance(input_field, str):
        messages.append({
            "role": "user",
            "content": input_field if input_field else " ",
        })

    elif isinstance(input_field, list):
        for m in input_field:
            if not isinstance(m, dict):
                continue

            role = m.get("role")
            content = m.get("content", "")

            if content is None:
                content = ""
            content = str(content)

            if role == "system":
                if not extracted_system and content:
                    extracted_system = content
                continue

            if role in {"user", "assistant"}:
                messages.append({
                    "role": role,
                    "content": content if content else " ",
                })

    else:
        messages.append({
            "role": "user",
            "content": str(input_field) if input_field is not None else " ",
        })

    output_text = output_text if output_text is not None else " "
    output_text = str(output_text) if output_text else " "

    messages.append({
        "role": "assistant",
        "content": output_text,
    })

    system = system_prompt or extracted_system or ""
    system = str(system) if system is not None else ""

    while messages and messages[0]["role"] != "user":
        messages.pop(0)

    if len(messages) < 2:
        return None, None

    if len(messages) % 2 != 0:
        return None, None

    for i, m in enumerate(messages):
        expected_role = "user" if i % 2 == 0 else "assistant"
        if m["role"] != expected_role:
            return None, None

    return system, messages


def write_shard(instances, shard_idx):
    output_path = OUTPUT_DIR / f"train-{shard_idx:05d}.json"

    obj = {
        "type": "conversation",
        "instances": instances,
    }

    # 不使用 indent，显著减小文件体积和解析开销
    output_path.write_text(
        json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {output_path} with {len(instances)} instances")


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 防止旧 json 混入
    for old_file in OUTPUT_DIR.glob("*.json"):
        old_file.unlink()

    seen = 0
    kept = 0
    skipped = 0
    shard_idx = 0
    shard_instances = []

    with INPUT_PATH.open("r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            seen += 1

            try:
                ex = json.loads(line)
            except json.JSONDecodeError as e:
                skipped += 1
                print(f"[WARN] JSON decode failed at line {line_no}: {e}")
                continue

            system, messages = normalize_messages(
                input_field=ex.get("input", ""),
                output_text=ex.get("output", ""),
                system_prompt=ex.get("system_prompt", ""),
            )

            if messages is None:
                skipped += 1
                continue

            instance = {
                "conversation_id": f"code_v1.1-{line_no}",
                "system": system,
                "messages": messages,
            }

            shard_instances.append(instance)
            kept += 1

            if len(shard_instances) >= SHARD_SIZE:
                write_shard(shard_instances, shard_idx)
                shard_idx += 1
                shard_instances = []

            if kept % 10000 == 0:
                print(f"seen={seen}, kept={kept}, skipped={skipped}")

            if MAX_EXAMPLES is not None and kept >= MAX_EXAMPLES:
                break

    if shard_instances:
        write_shard(shard_instances, shard_idx)

    print("Done.")
    print(f"Input:   {INPUT_PATH}")
    print(f"Output:  {OUTPUT_DIR}")
    print(f"Seen:    {seen}")
    print(f"Kept:    {kept}")
    print(f"Skipped: {skipped}")
    print(f"Shards:  {shard_idx + 1}")


if __name__ == "__main__":
    main()