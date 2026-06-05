import argparse
import json
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_token_id(tokenizer, token):
    tid = tokenizer.convert_tokens_to_ids(token)
    if tid is None or tid < 0:
        return None
    return int(tid)


def make_prompt_ids(tokenizer, length, device, avoid_ids):
    text = (
        "The quick brown fox jumps over the lazy dog. "
        "你好，今天我们测试 Fast dLLM 的 generate 逻辑。 "
    )
    ids = tokenizer(
        text * max(16, length // 20 + 16),
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids[0].tolist()

    special = set(tokenizer.all_special_ids or [])
    clean = [x for x in ids if x not in special and x not in avoid_ids]

    while len(clean) < length:
        clean.extend(clean)

    return torch.tensor(clean[:length], device=device, dtype=torch.long).unsqueeze(0)


def run_generate(
    model,
    tokenizer,
    input_ids,
    mask_id,
    stop_id,
    args,
    use_block_cache,
):
    forward_calls = []

    orig_forward = model.forward

    def wrapped_forward(*f_args, **kwargs):
        input_arg = kwargs.get("input_ids", None)
        if input_arg is None and len(f_args) > 0:
            input_arg = f_args[0]

        rec = {
            "call": len(forward_calls),
            "input_len": None if input_arg is None else int(input_arg.shape[1]),
            "use_cache": kwargs.get("use_cache", None),
            "update_past_key_values": kwargs.get("update_past_key_values", None),
            "use_block_cache": kwargs.get("use_block_cache", None),
            "replace_position": kwargs.get("replace_position", None),
            "block_size_arg": kwargs.get("block_size", "<missing>"),
        }
        forward_calls.append(rec)

        if len(forward_calls) <= args.print_first_n or len(forward_calls) % args.print_every == 0:
            log(json.dumps(rec, ensure_ascii=False))


        if input_arg is not None:
            vocab_size = int(model.config.vocab_size)
            input_min = int(input_arg.min().item())
            input_max = int(input_arg.max().item())
            invalid = input_min < 0 or input_max >= vocab_size

            rec["input_min"] = input_min
            rec["input_max"] = input_max
            rec["vocab_size"] = vocab_size
            rec["invalid_token_id"] = invalid
            rec["mask_count"] = int((input_arg == mask_id).sum().item())

            if invalid:
                print("INVALID INPUT IDS:", json.dumps(rec, ensure_ascii=False), flush=True)
                raise RuntimeError(
                    f"Invalid token id detected: min={input_min}, max={input_max}, vocab_size={vocab_size}"
                )

        out = orig_forward(*f_args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return out

    model.forward = wrapped_forward

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    t0 = time.time()

    log(f"START generate use_block_cache={use_block_cache}")

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids.clone(),
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            block_size=args.block_size,
            small_block_size=args.small_block_size,
            threshold=args.threshold,
            temperature=0.0,
            top_p=args.top_p,
            mask_id=mask_id,
            stop_token=stop_id,
            use_block_cache=use_block_cache,
        )

    torch.cuda.synchronize()
    dt = time.time() - t0

    model.forward = orig_forward

    log(
        f"DONE generate use_block_cache={use_block_cache}, "
        f"time={dt:.2f}s, forward_calls={len(forward_calls)}, "
        f"output_shape={tuple(output.shape)}"
    )

    num_replace_calls = sum(
        1 for r in forward_calls
        if r.get("replace_position", None) is not None
    )

    num_full_block_calls = sum(
        1 for r in forward_calls
        if r.get("input_len", None) == args.block_size
        and r.get("use_block_cache", None) is True
    )

    num_small_block_calls = sum(
        1 for r in forward_calls
        if r.get("input_len", None) == args.small_block_size
        and r.get("use_block_cache", None) is True
    )

    log(
        f"TRACE SUMMARY: "
        f"num_replace_calls={num_replace_calls}, "
        f"num_full_block_calls={num_full_block_calls}, "
        f"num_small_block_calls={num_small_block_calls}"
    )

    return output, dt, len(forward_calls)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        default="/home/u-shengbf/Codes/Fast-dLLM/v2/base_models/Model-Qwen-3-8B",
    )
    parser.add_argument("--prompt-len", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--small-block-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--mode", choices=["cache", "no_cache", "both"], default="both")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--mask-id", type=int, default=None)
    parser.add_argument("--stop-id", type=int, default=None)
    parser.add_argument("--print-first-n", type=int, default=20)
    parser.add_argument("--print-every", type=int, default=50)
    args = parser.parse_args()

    assert args.max_new_tokens % args.block_size == 0
    assert args.block_size % args.small_block_size == 0

    set_seed(args.seed)

    log("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        use_fast=True,
    )

    log("Loading model")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    model.config._attn_implementation = "eager"
    model.model.config._attn_implementation = "eager"
    
    model.eval()

    device = model.device

    mask_id = args.mask_id
    if mask_id is None:
        mask_id = resolve_token_id(tokenizer, "|<MASK>|")
    if mask_id is None:
        raise RuntimeError("Cannot resolve |<MASK>|. Please pass --mask-id.")

    stop_id = args.stop_id
    if stop_id is None:
        stop_id = resolve_token_id(tokenizer, "<|im_end|>")
    if stop_id is None:
        stop_id = tokenizer.eos_token_id

    log(f"mask_id={mask_id}, mask_token={tokenizer.convert_ids_to_tokens(mask_id)}")
    log(f"stop_id={stop_id}, stop_token={tokenizer.convert_ids_to_tokens(stop_id)}")

    input_ids = make_prompt_ids(
        tokenizer,
        args.prompt_len,
        device,
        avoid_ids={mask_id, stop_id},
    )
    log(f"input_shape={tuple(input_ids.shape)}")

    out_no_cache = None
    out_cache = None

    if args.mode in ["no_cache", "both"]:
        set_seed(args.seed)
        out_no_cache, t_no_cache, n_no_cache = run_generate(
            model,
            tokenizer,
            input_ids,
            mask_id,
            stop_id,
            args,
            use_block_cache=False,
        )

    if args.mode in ["cache", "both"]:
        set_seed(args.seed)
        out_cache, t_cache, n_cache = run_generate(
            model,
            tokenizer,
            input_ids,
            mask_id,
            stop_id,
            args,
            use_block_cache=True,
        )

    if args.mode == "both":
        same = torch.equal(out_no_cache, out_cache)
        log(f"EQUIVALENCE={same}")
        if not same:
            min_len = min(out_no_cache.shape[1], out_cache.shape[1])
            neq = (out_no_cache[:, :min_len] != out_cache[:, :min_len]).nonzero()
            first_diff = neq[0].tolist() if neq.numel() > 0 else "length_mismatch_only"
            log(f"first_diff={first_diff}")
            log("---- no_cache tail ----")
            print(tokenizer.decode(out_no_cache[0][-200:], skip_special_tokens=False))
            log("---- cache tail ----")
            print(tokenizer.decode(out_cache[0][-200:], skip_special_tokens=False))
            raise SystemExit(1)


if __name__ == "__main__":
    main()