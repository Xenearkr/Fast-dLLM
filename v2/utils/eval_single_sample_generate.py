"""
指标说明：
TPS_new_tokens_per_sec = 生成 token/s
TPF_time_per_forward_sec = 平均每次 forward 耗时
tokens_per_forward = 每次 forward 平均产出 token 数

使用方法：【在v2/目录下运行】

正确性对照：
python tests_local/eval_single_sample_generate.py \
  --model-dir /home/u-shengbf/Codes/Fast-dLLM/v2/base_models/Model-Qwen-3-8B \
  --mode both \
  --threshold 0.0 \
  --max-new-tokens 32 \
  --block-size 32 \
  --small-block-size 8 \
  --print-output

python tests_local/eval_single_sample_generate.py \
  --model-dir /home/u-shengbf/Codes/Fast-dLLM/v2/output_models/finetune_full_20260525_234355 \
  --mode both \
  --threshold 0.9 \
  --max-new-tokens 32 \
  --block-size 32 \
  --small-block-size 8 \
  --print-output

速度比较：
python tests_local/eval_single_sample_generate.py \
  --model-dir /home/u-shengbf/Codes/Fast-dLLM/v2/output_models/finetune_full_20260525_234355 \
  --mode no_cache \
  --threshold 0.9 \
  --max-new-tokens 128 \
  --block-size 32 \
  --small-block-size 8 \
  --print-output

python tests_local/eval_single_sample_generate.py \
  --model-dir /home/u-shengbf/Codes/Fast-dLLM/v2/output_models/finetune_full_20260525_234355 \
  --mode cache \
  --threshold 0.9 \
  --max-new-tokens 128 \
  --block-size 32 \
  --small-block-size 8 \
  --print-output


改用Fast系 尝试：[尚未调试好]
python tests_local/eval_single_sample_generate.py \
  --model-dir /home/u-shengbf/.cache/huggingface/hub/models--Efficient-Large-Model--Fast_dLLM_v2_7B/snapshots/0661abf5f9f0ee338970d091052a26c8efa51974 \
  --mode cache \
  --threshold 0.9 \
  --max-new-tokens 32 \
  --block-size 32 \
  --small-block-size 8 \
  --mask-id 151665 \
  --stop-id 151645 \
  --print-output

python tests_local/eval_single_sample_generate.py \
  --model-dir /home/u-shengbf/Codes/Fast-dLLM/v2/base_models/Model-Qwen-2.5-7B \
  --mode cache \
  --threshold 0.9 \
  --max-new-tokens 32 \
  --block-size 32 \
  --small-block-size 8 \
  --mask-id 151665 \
  --stop-id 151645 \
  --print-output

python tests_local/eval_single_sample_generate.py \
  --model-dir /home/u-shengbf/Codes/Fast-dLLM/v2/output_models/finetune_fast_dLLM_7B_20260521_222723 \
  --mode cache \
  --threshold 0.9 \
  --max-new-tokens 32 \
  --block-size 32 \
  --small-block-size 8 \
  --print-output
"""


import argparse
import json
import time
import random
from typing import Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def sync_cuda():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_token_id(tokenizer, token: str):
    tid = tokenizer.convert_tokens_to_ids(token)
    if tid is None or tid < 0:
        return None
    return int(tid)


def build_input_ids(tokenizer, model, prompt: str, enable_thinking: bool):
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        return_tensors="pt",
    )

    # device_map="auto" 下通常 model.device 可用。
    return input_ids.to(model.device)


def run_once(
    model,
    tokenizer,
    input_ids,
    args,
    use_block_cache: bool,
) -> Dict[str, Any]:
    mask_id = args.mask_id
    if mask_id is None:
        mask_id = resolve_token_id(tokenizer, "|<MASK>|")
    if mask_id is None:
        raise RuntimeError("Cannot resolve |<MASK>| token id. Please pass --mask-id.")

    stop_id = args.stop_id
    if stop_id is None:
        stop_id = resolve_token_id(tokenizer, "<|im_end|>")
    if stop_id is None:
        stop_id = tokenizer.eos_token_id

    forward_records = []
    orig_forward = model.forward

    def wrapped_forward(*f_args, **kwargs):
        input_arg = kwargs.get("input_ids", None)
        if input_arg is None and len(f_args) > 0:
            input_arg = f_args[0]

        rec = {
            "call": len(forward_records),
            "input_len": None if input_arg is None else int(input_arg.shape[1]),
            "use_cache": kwargs.get("use_cache", None),
            "update_past_key_values": kwargs.get("update_past_key_values", None),
            "use_block_cache": kwargs.get("use_block_cache", None),
            "replace_position": kwargs.get("replace_position", None),
            "block_size_arg": kwargs.get("block_size", "<missing>"),
        }
        forward_records.append(rec)
        return orig_forward(*f_args, **kwargs)

    model.forward = wrapped_forward

    sync_cuda()
    t0 = time.perf_counter()

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids.clone(),
            max_new_tokens=args.max_new_tokens,
            block_size=args.block_size,
            small_block_size=args.small_block_size,
            threshold=args.threshold,
            temperature=args.temperature,
            top_p=args.top_p,
            use_block_cache=use_block_cache,
            mask_id=mask_id,
            stop_token=stop_id,
            debug_generate=args.debug_generate,
            debug_tokenizer=tokenizer,
        )

    sync_cuda()
    total_time = time.perf_counter() - t0

    model.forward = orig_forward

    prompt_tokens = int(input_ids.shape[1])
    total_tokens = int(out.shape[1])
    new_tokens = max(0, total_tokens - prompt_tokens)

    forward_calls = len(forward_records)
    num_replace_calls = sum(
        1 for r in forward_records
        if r.get("replace_position", None) is not None
    )
    num_full_block_calls = sum(
        1 for r in forward_records
        if r.get("input_len", None) == args.block_size
        and r.get("use_block_cache", None) is True
    )
    num_small_block_calls = sum(
        1 for r in forward_records
        if r.get("input_len", None) == args.small_block_size
        and r.get("use_block_cache", None) is True
    )

    metrics = {
        "use_block_cache": use_block_cache,
        "prompt_tokens": prompt_tokens,
        "new_tokens": new_tokens,
        "total_tokens": total_tokens,
        "total_time_sec": total_time,
        "TPS_new_tokens_per_sec": new_tokens / total_time if total_time > 0 else None,
        "TPF_time_per_forward_sec": total_time / forward_calls if forward_calls > 0 else None,
        "tokens_per_forward": new_tokens / forward_calls if forward_calls > 0 else None,
        "forward_calls": forward_calls,
        "num_replace_calls": num_replace_calls,
        "num_full_block_calls": num_full_block_calls,
        "num_small_block_calls": num_small_block_calls,
        "threshold": args.threshold,
        "block_size": args.block_size,
        "small_block_size": args.small_block_size,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }

    text = tokenizer.decode(out[0], skip_special_tokens=False)

    return {
        "output_ids": out,
        "text": text,
        "metrics": metrics,
        "forward_records": forward_records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        default="/home/u-shengbf/Codes/Fast-dLLM/v2/base_models/Model-Qwen-3-8B",
    )
    parser.add_argument(
        "--prompt",
        default="Write a Python function to add two numbers.",
    )
    parser.add_argument("--mode", choices=["cache", "no_cache", "both"], default="both")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--small-block-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--mask-id", type=int, default=None)
    parser.add_argument("--stop-id", type=int, default=None)
    parser.add_argument("--print-output", action="store_true")
    parser.add_argument("--print-forward-first-n", type=int, default=20)
    parser.add_argument("--debug-generate", action="store_true")
    args = parser.parse_args()

    assert args.block_size % args.small_block_size == 0, (
        "block_size must be divisible by small_block_size"
    )

    set_seed(args.seed)

    print(f"[INFO] Loading tokenizer from {args.model_dir}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        local_files_only=True,
    )

    print(f"[INFO] Loading model from {args.model_dir}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.eval()

    input_ids = build_input_ids(
        tokenizer=tokenizer,
        model=model,
        prompt=args.prompt,
        enable_thinking=args.enable_thinking,
    )

    print(f"[INFO] input_shape={tuple(input_ids.shape)}", flush=True)
    print(f"[INFO] prompt_tokens={input_ids.shape[1]}", flush=True)

    results = {}

    if args.mode in ["no_cache", "both"]:
        print("\n===== RUN use_block_cache=False =====", flush=True)
        set_seed(args.seed)
        res = run_once(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            args=args,
            use_block_cache=False,
        )
        results["no_cache"] = res
        print(json.dumps(res["metrics"], ensure_ascii=False, indent=2), flush=True)

        if args.print_forward_first_n > 0:
            print("[FORWARD TRACE no_cache]", flush=True)
            for r in res["forward_records"][: args.print_forward_first_n]:
                print(json.dumps(r, ensure_ascii=False), flush=True)

    if args.mode in ["cache", "both"]:
        print("\n===== RUN use_block_cache=True =====", flush=True)
        set_seed(args.seed)
        res = run_once(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            args=args,
            use_block_cache=True,
        )
        results["cache"] = res
        print(json.dumps(res["metrics"], ensure_ascii=False, indent=2), flush=True)

        if args.print_forward_first_n > 0:
            print("[FORWARD TRACE cache]", flush=True)
            for r in res["forward_records"][: args.print_forward_first_n]:
                print(json.dumps(r, ensure_ascii=False), flush=True)

    if args.mode == "both":
        no_cache_ids = results["no_cache"]["output_ids"]
        cache_ids = results["cache"]["output_ids"]
        same = torch.equal(no_cache_ids, cache_ids)
        print(f"\nEQUIVALENCE={same}", flush=True)

        if not same:
            min_len = min(no_cache_ids.shape[1], cache_ids.shape[1])
            neq = (no_cache_ids[:, :min_len] != cache_ids[:, :min_len]).nonzero()
            first_diff = neq[0].tolist() if neq.numel() > 0 else "length_mismatch_only"
            print(f"first_diff={first_diff}", flush=True)

    if args.print_output:
        if "cache" in results:
            print("\n===== OUTPUT use_block_cache=True =====")
            print(results["cache"]["text"])
        elif "no_cache" in results:
            print("\n===== OUTPUT use_block_cache=False =====")
            print(results["no_cache"]["text"])


if __name__ == "__main__":
    main()