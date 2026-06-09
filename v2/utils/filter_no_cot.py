"""
筛选 Llama-Nemotron code_v1.jsonl，保留不含 CoT（chain-of-thought）的样本。

与 download_datasets.py 使用相同的数据路径约定。

判定为「有 CoT」的规则（满足任一即排除）：
  - 字段 reasoning 为 on / true / 1 / yes
  - system_prompt 含 "thinking on"（如 detailed thinking on）
  - output 中含非空的 <think>...</think> 块

code_v1.jsonl 前半段多为 reasoning=on，后半段含 reasoning=off 的无 CoT 样本。
code_v1.1.jsonl 为 v1.1 推理增强子集，几乎全部含 CoT，不宜作为无 CoT 来源。

用法：
  python filter_no_cot.py
  python filter_no_cot.py --max-examples 1000   # 调试
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "Llama-Nemotron-code-v1.1"
_DEFAULT_INPUT = _DATA_ROOT / "SFT/code/code_v1.jsonl"
_DEFAULT_OUTPUT = _DATA_ROOT / "SFT/code/code_no_cot.jsonl"

_THINKING_BLOCK_RE = re.compile(
    r"<think>(.*?)</think>",
    re.DOTALL | re.IGNORECASE,
)


def _is_truthy_reasoning_on(value) -> bool:
    if value is True:
        return True
    if value is False or value is None:
        return False
    text = str(value).strip().lower()
    return text in {"on", "true", "1", "yes"}


def _is_truthy_reasoning_off(value) -> bool:
    if value is False:
        return True
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"off", "false", "0", "no"}


def _has_nonempty_thinking_block(output: str) -> bool:
    for match in _THINKING_BLOCK_RE.finditer(output):
        if match.group(1).strip():
            return True
    return False


def has_cot(example: dict) -> bool:
    """样本是否包含 CoT / reasoning trace。"""
    if _is_truthy_reasoning_on(example.get("reasoning")):
        return True
    if _is_truthy_reasoning_off(example.get("reasoning")):
        return False

    system_prompt = str(example.get("system_prompt") or "").strip().lower()
    if "thinking on" in system_prompt:
        return True
    if "thinking off" in system_prompt:
        return False

    output = example.get("output")
    if not isinstance(output, str):
        return False

    if _has_nonempty_thinking_block(output):
        return True

    return False


def filter_no_cot(
    input_path: Path,
    output_path: Path,
    max_examples: int | None = None,
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen = 0
    kept = 0
    skipped_cot = 0
    skipped_bad = 0

    with input_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            seen += 1

            try:
                ex = json.loads(line)
            except json.JSONDecodeError as e:
                skipped_bad += 1
                print(f"[WARN] JSON decode failed at line {line_no}: {e}")
                continue

            if has_cot(ex):
                skipped_cot += 1
            else:
                fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
                kept += 1

            if kept > 0 and kept % 10000 == 0:
                print(f"kept={kept}, seen={seen}, skipped_cot={skipped_cot}")

            if max_examples is not None and kept >= max_examples:
                break

    print("Done.")
    print(f"Input:         {input_path}")
    print(f"Output:        {output_path}")
    print(f"Seen:          {seen}")
    print(f"Kept (no CoT): {kept}")
    print(f"Skipped CoT:   {skipped_cot}")
    print(f"Skipped bad:   {skipped_bad}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter no-CoT samples from code_v1.jsonl")
    parser.add_argument(
        "--input",
        type=Path,
        default=_DEFAULT_INPUT,
        help=f"Input jsonl (default: {_DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"Output jsonl (default: {_DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Stop after keeping this many no-CoT samples (debug)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    filter_no_cot(args.input, args.output, max_examples=args.max_examples)


if __name__ == "__main__":
    main()
