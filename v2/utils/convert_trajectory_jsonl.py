#!/usr/bin/env python3
"""
将 generate_trajectory 输出的 JSONL 转为 LMFlow conversation JSON（含 trajectory 字段）。

每行 JSONL 格式:
  {"messages": [...], "trajectory": [int, ...]}

输出目录下生成 train-*.json:
  {"type": "conversation", "instances": [{..., "trajectory": [...]}, ...]}

用法:
  python utils/convert_trajectory_jsonl.py
  python utils/convert_trajectory_jsonl.py \\
    --input data/Llama-Nemotron-code-v1.1/trajectory/result.jsonl \\
    --output-dir data/Llama-Nemotron-code-v1.1/trajectory/train_conversation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, List, Optional, Tuple

_V2_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_INPUT = _V2_ROOT / "data/Llama-Nemotron-code-v1.1/trajectory/result.jsonl"
_DEFAULT_OUTPUT = _V2_ROOT / "data/Llama-Nemotron-code-v1.1/trajectory/train_conversation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert trajectory JSONL to conversation JSON shards.")
    parser.add_argument("--input", type=Path, default=_DEFAULT_INPUT, help="Input .jsonl path")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="Output directory for train-*.json",
    )
    parser.add_argument("--shard-size", type=int, default=10000, help="Instances per shard file")
    parser.add_argument("--max-examples", type=int, default=None, help="Limit lines (debug)")
    parser.add_argument(
        "--id-prefix",
        type=str,
        default="trajectory",
        help="conversation_id prefix",
    )
    return parser.parse_args()


def validate_messages(messages: Any) -> Optional[str]:
    if not isinstance(messages, list) or len(messages) < 2:
        return "messages must be a list with at least 2 turns"
    if messages[0].get("role") != "user":
        return "first message must be user"
    if len(messages) % 2 != 0:
        return "messages length must be even"
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            return f"message {i} is not a dict"
        role = m.get("role")
        content = m.get("content")
        if role not in ("user", "assistant"):
            return f"invalid role at {i}: {role!r}"
        if content is None:
            return f"empty content at {i}"
        expected = "user" if i % 2 == 0 else "assistant"
        if role != expected:
            return f"expected {expected} at {i}, got {role}"
    return None


def validate_trajectory(trajectory: Any) -> Optional[str]:
    if not isinstance(trajectory, list):
        return "trajectory must be a list"
    if not trajectory:
        return "trajectory is empty"
    for i, v in enumerate(trajectory):
        if not isinstance(v, int):
            return f"trajectory[{i}] is not int: {type(v)}"
    return None


def normalize_instance(
    row: dict,
    line_no: int,
    id_prefix: str,
) -> Tuple[Optional[dict], Optional[str]]:
    messages = row.get("messages")
    err = validate_messages(messages)
    if err:
        return None, err

    trajectory = row.get("trajectory")
    err = validate_trajectory(trajectory)
    if err:
        return None, err

    system = row.get("system") or ""
    if system is None:
        system = ""
    system = str(system)

    instance = {
        "conversation_id": row.get("conversation_id") or f"{id_prefix}-{line_no}",
        "system": system,
        "messages": messages,
        "trajectory": trajectory,
    }
    return instance, None


def write_shard(instances: List[dict], output_dir: Path, shard_idx: int) -> None:
    output_path = output_dir / f"train-{shard_idx:05d}.json"
    obj = {"type": "conversation", "instances": instances}
    output_path.write_text(
        json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output_path} ({len(instances)} instances)")


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.glob("*.json"):
        old.unlink()

    seen = 0
    kept = 0
    skipped = 0
    shard_idx = 0
    n_shards_written = 0
    shard_instances: List[dict] = []

    with input_path.open("r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            seen += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                skipped += 1
                print(f"[WARN] line {line_no}: JSON decode failed: {e}")
                continue

            instance, err = normalize_instance(row, line_no, args.id_prefix)
            if instance is None:
                skipped += 1
                print(f"[WARN] line {line_no}: {err}")
                continue

            shard_instances.append(instance)
            kept += 1

            if len(shard_instances) >= args.shard_size:
                write_shard(shard_instances, output_dir, shard_idx)
                shard_idx += 1
                n_shards_written += 1
                shard_instances = []

            if kept % 5000 == 0 and kept > 0:
                print(f"progress: seen={seen}, kept={kept}, skipped={skipped}")

            if args.max_examples is not None and kept >= args.max_examples:
                break

    if shard_instances:
        write_shard(shard_instances, output_dir, shard_idx)
        n_shards_written += 1

    print("Done.")
    print(f"Input:   {input_path}")
    print(f"Output:  {output_dir}")
    print(f"Seen:    {seen}")
    print(f"Kept:    {kept}")
    print(f"Skipped: {skipped}")
    print(f"Shards:  {n_shards_written}")


if __name__ == "__main__":
    main()
