import json
from pathlib import Path
from datasets import load_dataset


def normalize_messages(input_field, output_text):
    """
    Nemotron SFT:
      input: usually a list of messages, e.g. [{"role": "user", "content": "..."}]
      output: assistant response string

    LMFlow conversation:
      messages: user/assistant alternating list
    """
    if isinstance(input_field, str):
        messages = [{"role": "user", "content": input_field}]
    elif isinstance(input_field, list):
        messages = []
        for m in input_field:
            if not isinstance(m, dict):
                continue

            role = m.get("role")
            content = m.get("content", "")

            if role not in {"system", "user", "assistant"}:
                continue

            # system 建议放到 instance["system"]，不要混进 messages
            if role == "system":
                continue

            messages.append({
                "role": role,
                "content": content if content else " ",
            })
    else:
        messages = [{"role": "user", "content": str(input_field)}]

    # 确保最后追加 assistant output
    output_text = output_text if output_text else " "
    messages.append({
        "role": "assistant",
        "content": output_text,
    })

    # LMFlow 要求从 user 开始，user/assistant 成对
    while messages and messages[0]["role"] != "user":
        messages = messages[1:]

    # 如果出现连续同 role，保守跳过/合并会更好；这里简单过滤异常样本
    if len(messages) < 2:
        return None

    if len(messages) % 2 == 1:
        messages = messages[:-1]

    for i, m in enumerate(messages):
        expected = "user" if i % 2 == 0 else "assistant"
        if m["role"] != expected:
            return None

    return messages


def convert_split(
    split_name,
    output_path,
    max_examples=None,
    streaming=True,
):
    ds = load_dataset(
        "nvidia/Llama-Nemotron-Post-Training-Dataset",
        "SFT",
        split=split_name,
        streaming=streaming,
    )

    instances = []
    n_seen = 0
    n_kept = 0

    for ex in ds:
        n_seen += 1

        # 可选：只保留官方 used_in_training 样本
        # 字段是 string，常见值需你们本地 print 检查
        # if ex.get("used_in_training") not in {"True", "true", "1", True}:
        #     continue

        messages = normalize_messages(
            input_field=ex.get("input", ""),
            output_text=ex.get("output", ""),
        )

        if messages is None:
            continue

        system_prompt = ex.get("system_prompt") or ""

        instance = {
            "conversation_id": f"{split_name}-{n_seen}",
            "system": system_prompt,
            "messages": messages,
        }

        # 可选：保留元信息，LMFlow 不一定使用，但方便追踪
        instance["metadata"] = {
            "category": ex.get("category"),
            "reasoning": ex.get("reasoning"),
            "generator": ex.get("generator"),
            "version": ex.get("version"),
            "license": ex.get("license"),
            "used_in_training": ex.get("used_in_training"),
        }

        instances.append(instance)
        n_kept += 1

        if max_examples is not None and n_kept >= max_examples:
            break

    output = {
        "type": "conversation",
        "instances": instances,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"split={split_name}")
    print(f"seen={n_seen}")
    print(f"kept={n_kept}")
    print(f"wrote={output_path}")


if __name__ == "__main__":
    out_dir = Path("/data/fastdllm_nemotron_sft/train_conversation")

    for split in ["code", "math", "science", "chat", "safety"]:
        convert_split(
            split_name=split,
            output_path=out_dir / f"{split}.json",
            max_examples=None,   # 调试时可设 1000
            streaming=True,
        )