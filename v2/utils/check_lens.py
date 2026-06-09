"""
统计 LMFlow conversation 数据集经 tokenizer + chat template 编码后的序列长度。

与 finetune 一致：使用 apply_chat_template（默认 fast_dllm_v2），
长度 = len(input_ids)（blocking 分块之前）。

用法：
  python check_lens.py
  python check_lens.py --data-dir ../data/Llama-Nemotron-code-v1.1/use
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# lmflow 包在 repo 的 third_party/ 下（与 generate_trajectory_acc.py 一致）
_REPO_ROOT = Path(__file__).resolve().parents[2]
_LMFLOW_PATH = _REPO_ROOT / "third_party"
if str(_LMFLOW_PATH) not in sys.path:
    sys.path.insert(0, str(_LMFLOW_PATH))

from transformers import AutoTokenizer

from lmflow.utils.conversation_template import JINJA_TEMPLATES

_V2_ROOT = Path(__file__).resolve().parents[1]
_DATA_ROOT = _V2_ROOT / "data" / "Llama-Nemotron-code-v1.1"
_DEFAULT_DATA_DIR = _DATA_ROOT / "use"
_DEFAULT_MODEL = "Efficient-Large-Model/Fast_dLLM_v2_7B"
_DEFAULT_TEMPLATE = "fast_dllm_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check tokenized sequence lengths for conversation JSON shards."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_DEFAULT_DATA_DIR,
        help=f"Directory containing train-*.json (default: {_DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=_DEFAULT_MODEL,
        help=f"HuggingFace model/tokenizer path (default: {_DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--conversation-template",
        type=str,
        default=_DEFAULT_TEMPLATE,
        help=f"Chat template name (default: {_DEFAULT_TEMPLATE})",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Pass trust_remote_code=True to from_pretrained",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only load tokenizer from local cache",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Only process first N instances (debug)",
    )
    return parser.parse_args()


def iter_instances(data_dir: Path):
    data_files = sorted(data_dir.glob("*.json"))
    if not data_files:
        raise FileNotFoundError(f"No *.json found under {data_dir}")

    for path in data_files:
        with path.open(encoding="utf-8") as f:
            obj = json.load(f)
        if obj.get("type") != "conversation":
            raise ValueError(f"{path}: expected type=conversation, got {obj.get('type')!r}")
        for inst in obj.get("instances", []):
            yield path.name, inst


def encode_length(tokenizer, chat_template: str, instance: dict) -> int:
    messages = instance.get("messages") or []
    system = instance.get("system")

    conversation = []
    if system:
        conversation.append({"role": "system", "content": system})
    conversation.extend(messages)

    encoded = tokenizer.apply_chat_template(
        conversation=conversation,
        chat_template=chat_template,
        return_assistant_tokens_mask=True,
        return_dict=True,
    )
    return len(encoded["input_ids"])


def main():
    args = parse_args()
    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    if args.conversation_template not in JINJA_TEMPLATES:
        raise ValueError(
            f"Unknown conversation_template: {args.conversation_template}. "
            f"Available: {sorted(JINJA_TEMPLATES)}"
        )
    chat_template = JINJA_TEMPLATES[args.conversation_template]

    print("Loading tokenizer:", args.model_path)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer loaded in {time.time() - t0:.2f}s")

    lengths: list[int] = []
    skipped = 0

    for shard_name, inst in iter_instances(data_dir):
        try:
            length = encode_length(tokenizer, chat_template, inst)
        except Exception as e:
            skipped += 1
            cid = inst.get("conversation_id", "?")
            print(f"[WARN] skip {shard_name} {cid}: {e}")
            continue

        lengths.append(length)

        if len(lengths) % 1000 == 0:
            print(f"processed {len(lengths)} examples...")

        if args.max_examples is not None and len(lengths) >= args.max_examples:
            break

    if not lengths:
        raise RuntimeError("No valid instances processed.")

    max_len = max(lengths)
    min_len = min(lengths)
    avg_len = sum(lengths) / len(lengths)

    print("\n==== Token length stats ====")
    print(f"Data dir:              {data_dir}")
    print(f"Model:                 {args.model_path}")
    print(f"Conversation template: {args.conversation_template}")
    print(f"Instances:             {len(lengths)}")
    print(f"Skipped:               {skipped}")
    print(f"max_len:               {max_len}")
    print(f"min_len:               {min_len}")
    print(f"avg_len:               {avg_len:.2f}")


if __name__ == "__main__":
    main()
