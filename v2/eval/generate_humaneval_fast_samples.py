import sys
import os
import argparse
import json
import time
import types
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def str2bool(value):
    if isinstance(value, bool):
        return value

    value = value.lower()
    if value in ("yes", "true", "t", "y", "1", "on"):
        return True
    if value in ("no", "false", "f", "n", "0", "off"):
        return False

    raise argparse.ArgumentTypeError(
        f"Boolean value expected, got: {value}. "
        "Use true/false, yes/no, 1/0, or on/off."
    )


def import_generation_functions():
    # 获取当前文件所在目录的上一层
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    # 将父目录添加到系统路径
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
        
    try:
        import generation_functions
        return generation_functions
    except ImportError:
        # 兼容本地如果叫 generation_function.py 的情况
        import generation_function
        return generation_function


def load_evalplus_tasks(dataset_name):
    if dataset_name == "mbpp":
        from evalplus.data import get_mbpp_plus
        return get_mbpp_plus()
    if dataset_name == "humaneval":
        from evalplus.data import get_human_eval_plus
        return get_human_eval_plus()

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def shard_items(items, shard_id: int, num_shards: int):
    """Round-robin shard split. Keeps deterministic global task order after merge."""
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")

    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(
            f"shard_id must be in [0, {num_shards}), got {shard_id}"
        )

    return [
        item
        for idx, item in enumerate(items)
        if idx % num_shards == shard_id
    ]


def build_prompt(problem, prompt_mode):
    raw_prompt = problem["prompt"]

    if prompt_mode == "raw":
        return raw_prompt

    if prompt_mode == "chat":
        return (
            "You are an expert Python programmer. "
            "Complete the following programming problem. "
            "Return only valid Python code. "
            "Do not include explanations or Markdown fences.\n\n"
            f"{raw_prompt}"
        )

    raise ValueError(f"Unknown prompt_mode: {prompt_mode}")


def apply_chat_template_if_needed(tokenizer, prompt, prompt_mode):
    if prompt_mode != "chat":
        return prompt

    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def cuda_synchronize_if_needed(device):
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def atomic_write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def update_progress(progress_path, metrics: dict, **updates):
    if progress_path is None:
        return

    metrics.update(updates)
    metrics["last_update_time"] = time.time()
    metrics["wall_elapsed_sec"] = metrics["last_update_time"] - metrics["start_time"]
    atomic_write_json(progress_path, metrics)


def batch_generate(
    model,
    tokenizer,
    prompts,
    device,
    *,
    mask_id,
    bd_size,
    small_block_size,
    max_new_tokens,
    threshold,
    use_block_cache,
):
    input_id_list = []
    seq_lens = []
    max_len = 0
    min_len = 10**18

    for prompt in prompts:
        model_inputs = tokenizer([prompt], return_tensors="pt")
        input_ids = model_inputs["input_ids"].to(device)

        input_id_list.append(input_ids)
        seq_lens.append(input_ids.shape[1])
        max_len = max(max_len, input_ids.shape[1])
        min_len = min(min_len, input_ids.shape[1])

    padded = []
    for input_ids in input_id_list:
        pad_len = max_len - input_ids.shape[1]
        if pad_len > 0:
            pad = torch.full(
                (1, pad_len),
                mask_id,
                dtype=torch.long,
                device=device,
            )
            input_ids = torch.cat([input_ids, pad], dim=1)
        padded.append(input_ids)

    batched_input_ids = torch.cat(padded, dim=0).to(device)
    seq_len_tensor = torch.tensor(seq_lens, dtype=torch.long, device=device)

    with torch.inference_mode():
        generated = model.mdm_sample(
            batched_input_ids,
            tokenizer=tokenizer,
            block_size=bd_size,
            small_block_size=small_block_size,
            max_new_tokens=max_new_tokens,
            mask_id=mask_id,
            min_len=min_len,
            seq_len=seq_len_tensor,
            use_block_cache=use_block_cache,
            threshold=threshold,
        )

    completions = []
    generated_token_counts = []

    for batch_idx, prompt_len in enumerate(seq_lens):
        # generation_functions.batch_sample 返回 dict: {original_batch_idx: full_ids}
        full_ids = generated[batch_idx]
        new_ids = full_ids[prompt_len:]

        # 去掉残留 mask token
        new_ids = new_ids[new_ids != mask_id]

        generated_token_counts.append(int(new_ids.numel()))
        text = tokenizer.decode(new_ids, skip_special_tokens=True)
        completions.append(text)

    return completions, generated_token_counts


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset", choices=["mbpp", "humaneval"], default="humaneval")
    parser.add_argument("--output", required=True)

    parser.add_argument("--prompt_mode", choices=["raw", "chat"], default="raw")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)

    parser.add_argument("--mask_id", type=int, default=151665)
    parser.add_argument("--bd_size", type=int, default=32)
    parser.add_argument("--small_block_size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument(
        "--use_block_cache",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help=(
            "Whether to enable block cache. "
            "Accepts true/false. "
            "Also supports flag-only usage: --use_block_cache means true."
        ),
    )

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")

    parser.add_argument(
        "--shard_id",
        type=int,
        default=0,
        help="Shard id for multi-GPU generation. Default 0.",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Total number of shards for multi-GPU generation. Default 1.",
    )
    parser.add_argument(
        "--progress_file",
        type=str,
        default=None,
        help="Optional JSON file for progress and throughput metrics.",
    )

    args = parser.parse_args()

    if args.max_new_tokens % args.bd_size != 0:
        raise ValueError(
            f"max_new_tokens must be divisible by bd_size. "
            f"Got max_new_tokens={args.max_new_tokens}, bd_size={args.bd_size}"
        )

    problems = load_evalplus_tasks(args.dataset)
    all_items = list(problems.items())

    if args.limit is not None:
        all_items = all_items[: args.limit]

    total_items = len(all_items)
    items = shard_items(
        all_items,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    progress_path = Path(args.progress_file) if args.progress_file else None
    now = time.time()
    metrics = {
        "dataset": args.dataset,
        "status": "initializing",
        "model_path": args.model_path,
        "output": str(output_path),
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "total_items_before_shard": total_items,
        "total_samples": len(items),
        "done_samples": 0,
        "total_batches": (len(items) + args.batch_size - 1) // args.batch_size,
        "done_batches": 0,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "allocated_new_tokens": 0,
        "generated_tokens": 0,
        "generation_time_sec": 0.0,
        "wall_elapsed_sec": 0.0,
        "failed_extract_count": 0,
        "non_compilable_count": None,
        "mask_id": args.mask_id,
        "threshold": args.threshold,
        "use_block_cache": args.use_block_cache,
        "start_time": now,
        "last_update_time": now,
        "error": None,
    }
    update_progress(progress_path, metrics, status="loading_model")

    try:
        generation_functions = import_generation_functions()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if args.dtype == "bf16":
            torch_dtype = torch.bfloat16
        elif args.dtype == "fp16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        print("Loading tokenizer:", args.model_path)
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            local_files_only=args.local_files_only,
        )

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("Loading model:", args.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            local_files_only=args.local_files_only,
        ).to(device)

        model.eval()

        model.mdm_sample = types.MethodType(
            generation_functions.Fast_dLLM_QwenForCausalLM.batch_sample,
            model,
        )

        print(f"Dataset: {args.dataset}")
        print(f"Total problems before shard: {total_items}")
        print(f"Shard id: {args.shard_id}")
        print(f"Num shards: {args.num_shards}")
        print(f"Shard problems: {len(items)}")
        print(f"Output: {output_path}")
        print(f"mask_id: {args.mask_id}")
        print(f"threshold: {args.threshold}")
        print(f"use_block_cache: {args.use_block_cache}")

        update_progress(progress_path, metrics, status="running")

        rows_written = 0
        with output_path.open("w", encoding="utf-8") as f:
            iterator = range(0, len(items), args.batch_size)
            for start in tqdm(
                iterator,
                desc=f"Generating shard {args.shard_id}/{args.num_shards}",
            ):
                batch_items = items[start : start + args.batch_size]

                task_ids = [x[0] for x in batch_items]
                problems_batch = [x[1] for x in batch_items]

                raw_prompts = [
                    build_prompt(problem, args.prompt_mode)
                    for problem in problems_batch
                ]

                prompts = [
                    apply_chat_template_if_needed(tokenizer, p, args.prompt_mode)
                    for p in raw_prompts
                ]

                cuda_synchronize_if_needed(device)
                gen_start = time.perf_counter()
                completions, token_counts = batch_generate(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    device=device,
                    mask_id=args.mask_id,
                    bd_size=args.bd_size,
                    small_block_size=args.small_block_size,
                    max_new_tokens=args.max_new_tokens,
                    threshold=args.threshold,
                    use_block_cache=args.use_block_cache,
                )
                cuda_synchronize_if_needed(device)
                gen_time = time.perf_counter() - gen_start

                for task_id, problem, completion in zip(task_ids, problems_batch, completions):
                    # EvalPlus schema:
                    # solution: 自包含代码，通常包含 prompt。
                    # raw 模式下 problem["prompt"] 是题目给定前缀，completion 是模型续写。
                    if args.prompt_mode == "raw":
                        solution = problem["prompt"] + completion
                    else:
                        # chat 模式下模型可能输出完整代码，让 sanitize 去处理。
                        solution = completion

                    row = {
                        "task_id": task_id,
                        "solution": solution,
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    rows_written += 1

                f.flush()

                metrics["done_batches"] += 1
                metrics["done_samples"] += len(batch_items)
                metrics["allocated_new_tokens"] += len(batch_items) * args.max_new_tokens
                metrics["generated_tokens"] += sum(token_counts)
                metrics["generation_time_sec"] += gen_time
                update_progress(progress_path, metrics, status="running")

        update_progress(progress_path, metrics, status="completed")
        print(f"Done. Wrote {rows_written} samples to {output_path}")
        print(f"Generated tokens: {metrics['generated_tokens']}")
        print(f"Generation time: {metrics['generation_time_sec']:.4f}s")
        if metrics["done_samples"] > 0:
            print(f"Shard TPF: {metrics['generation_time_sec'] / metrics['done_samples']:.4f}s/sample")
        if metrics["generation_time_sec"] > 0:
            print(f"Shard TPS: {metrics['generated_tokens'] / metrics['generation_time_sec']:.4f} tokens/s")

    except Exception as exc:
        update_progress(progress_path, metrics, status="failed", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
