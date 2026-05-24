import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to the Hugging Face model checkpoint directory.",
    )
    parser.add_argument(
        "--max_shard_size",
        default="5GB",
        help="Max shard size for safetensors output.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model_dir = Path(args.model_path).expanduser().resolve()

    if not model_dir.exists():
        raise FileNotFoundError(f"model_path does not exist: {model_dir}")

    if not model_dir.is_dir():
        raise NotADirectoryError(f"model_path is not a directory: {model_dir}")

    print("Loading model from:", model_dir)

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        local_files_only=True,
        low_cpu_mem_usage=True,
    )

    print("Saving safetensors in-place...")

    model.save_pretrained(
        str(model_dir),
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )

    tok = AutoTokenizer.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=True,
    )

    tok.save_pretrained(str(model_dir))

    print("Done.")


if __name__ == "__main__":
    main()