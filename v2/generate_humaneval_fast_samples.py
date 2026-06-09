import argparse
import json
import time
import types
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from generation_speed_metrics import (
    attach_forward_counter,
    build_speed_report,
    count_new_tokens,
    print_speed_report,
    reset_forward_counter,
    restore_forward,
    save_speed_report,
    sync_cuda,
    validate_warmup_new_tokens,
)


def import_generation_functions():
    try:
        import generation_functions
        return generation_functions
    except ImportError:
        # 兼容你本地如果叫 generation_function.py 的情况
        import generation_function
        return generation_function


def load_evalplus_tasks(dataset_name):
    if dataset_name == "mbpp":
        from evalplus.data import get_mbpp_plus
        return get_mbpp_plus()
    elif dataset_name == "humaneval":
        from evalplus.data import get_human_eval_plus
        return get_human_eval_plus()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


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

    with torch.no_grad():
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
    batch_new_tokens = 0

    for batch_idx, prompt_len in enumerate(seq_lens):
        # generation_functions.batch_sample 返回 dict: {original_batch_idx: full_ids}
        full_ids = generated[batch_idx]
        batch_new_tokens += count_new_tokens(full_ids, prompt_len, mask_id)
        new_ids = full_ids[prompt_len:]
        new_ids = new_ids[new_ids != mask_id]

        text = tokenizer.decode(new_ids, skip_special_tokens=True)
        completions.append(text)

    return completions, batch_new_tokens


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset", choices=["mbpp", "humaneval"], default="mbpp")
    parser.add_argument("--output", required=True)

    parser.add_argument("--prompt_mode", choices=["raw", "chat"], default="raw")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)

    parser.add_argument("--mask_id", type=int, default=151665)
    parser.add_argument("--bd_size", type=int, default=32)
    parser.add_argument("--small_block_size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--use_block_cache", action="store_true")

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument(
        "--method",
        default="model",
        help="Label used in speed metrics (e.g. Fast, Trajectory).",
    )
    parser.add_argument(
        "--metrics_output",
        default=None,
        help="Where to write TPS/TPF metrics JSON. Default: <output>.metrics.json",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=3,
        help="Number of warmup generations before timed eval (default: 3). Set 0 to disable.",
    )
    parser.add_argument(
        "--warmup_new_tokens",
        type=int,
        default=32,
        help="max_new_tokens per warmup step; must be divisible by bd_size (default: 32).",
    )

    args = parser.parse_args()

    if args.max_new_tokens % args.bd_size != 0:
        raise ValueError(
            f"max_new_tokens must be divisible by bd_size. "
            f"Got max_new_tokens={args.max_new_tokens}, bd_size={args.bd_size}"
        )
    if args.warmup_steps > 0:
        validate_warmup_new_tokens(args.warmup_new_tokens, args.bd_size)

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

    problems = load_evalplus_tasks(args.dataset)
    items = list(problems.items())

    if args.limit is not None:
        items = items[: args.limit]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_output = (
        Path(args.metrics_output)
        if args.metrics_output is not None
        else output_path.with_name(output_path.stem + ".metrics.json")
    )

    print(f"Dataset: {args.dataset}")
    print(f"Num problems: {len(items)}")
    print(f"Output: {output_path}")
    print(f"Metrics: {metrics_output}")

    rows = []
    total_new_tokens = 0

    forward_counter, original_forward = attach_forward_counter(model)

    if args.warmup_steps > 0 and items:
        warmup_batch = items[: args.batch_size]
        warmup_problems = [x[1] for x in warmup_batch]
        warmup_raw_prompts = [
            build_prompt(problem, args.prompt_mode) for problem in warmup_problems
        ]
        warmup_prompts = [
            apply_chat_template_if_needed(tokenizer, p, args.prompt_mode)
            for p in warmup_raw_prompts
        ]
        print(
            f"Warmup: {args.warmup_steps} step(s), "
            f"{args.warmup_new_tokens} new tokens/step"
        )
        for step in range(args.warmup_steps):
            batch_generate(
                model=model,
                tokenizer=tokenizer,
                prompts=warmup_prompts,
                device=device,
                mask_id=args.mask_id,
                bd_size=args.bd_size,
                small_block_size=args.small_block_size,
                max_new_tokens=args.warmup_new_tokens,
                threshold=args.threshold,
                use_block_cache=args.use_block_cache,
            )
            print(f"  warmup step {step + 1}/{args.warmup_steps} done")
        sync_cuda()
        reset_forward_counter(forward_counter)

    sync_cuda()
    generate_start = time.perf_counter()

    for start in tqdm(range(0, len(items), args.batch_size), desc="Generating"):
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

        completions, batch_new_tokens = batch_generate(
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
        total_new_tokens += batch_new_tokens

        for task_id, problem, completion in zip(task_ids, problems_batch, completions):
            # EvalPlus schema:
            # solution: 自包含代码，通常包含 prompt
            #
            # raw 模式下，problem["prompt"] 是题目给定前缀；
            # completion 是模型续写，因此拼接成完整 solution。
            if args.prompt_mode == "raw":
                solution = problem["prompt"] + completion
            else:
                # chat 模式下模型可能输出完整代码，让 sanitize 去处理。
                solution = completion

            rows.append(
                {
                    "task_id": task_id,
                    "solution": solution,
                }
            )

    sync_cuda()
    generate_time_s = time.perf_counter() - generate_start
    restore_forward(model, original_forward)

    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    speed_report = build_speed_report(
        method=args.method,
        model_path=args.model_path,
        dataset=args.dataset,
        num_samples=len(rows),
        new_tokens=total_new_tokens,
        generate_time_s=generate_time_s,
        forward_calls=forward_counter["count"],
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        bd_size=args.bd_size,
        small_block_size=args.small_block_size,
        threshold=args.threshold,
        use_block_cache=args.use_block_cache,
        warmup_steps=args.warmup_steps,
        warmup_new_tokens=args.warmup_new_tokens if args.warmup_steps > 0 else 0,
    )
    save_speed_report(metrics_output, speed_report)
    print_speed_report(speed_report)

    print(f"Done. Wrote {len(rows)} samples to {output_path}")


if __name__ == "__main__":
    main()