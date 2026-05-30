import argparse
import json
import random
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def shift_logits(logits):
    return torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)


def max_abs(a, b):
    return (a.float() - b.float()).abs().max().item()


def resolve_token_id(tokenizer, token):
    tid = tokenizer.convert_tokens_to_ids(token)
    if tid is None or tid < 0:
        return None
    return int(tid)


def build_input_ids(tokenizer, model, prompt, enable_thinking=False):
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        return_tensors="pt",
    )
    return input_ids.to(model.device)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        default="/home/u-shengbf/Codes/Fast-dLLM/v2/output_models/finetune_full_20260525_234355",
    )
    parser.add_argument(
        "--prompt",
        default="Write a Python function to add two numbers.",
    )
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--small-block-size", type=int, default=8)
    parser.add_argument("--mask-id", type=int, default=None)
    parser.add_argument("--stop-id", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--attn-implementation", default=None)
    args = parser.parse_args()

    assert args.block_size % args.small_block_size == 0

    set_seed(args.seed)

    log(f"Loading tokenizer from {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        local_files_only=True,
    )

    log(f"Loading model from {args.model_dir}")
    load_kwargs = dict(
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if args.attn_implementation is not None:
        load_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        **load_kwargs,
    )
    model.eval()

    if args.attn_implementation is not None:
        model.config._attn_implementation = args.attn_implementation
        if hasattr(model, "model"):
            model.model.config._attn_implementation = args.attn_implementation

    mask_id = args.mask_id
    if mask_id is None:
        mask_id = resolve_token_id(tokenizer, "|<MASK>|")
    if mask_id is None:
        raise RuntimeError("Cannot resolve |<MASK>|. Please pass --mask-id.")

    stop_id = args.stop_id
    if stop_id is None:
        stop_id = resolve_token_id(tokenizer, "<|im_end|>")

    input_ids = build_input_ids(
        tokenizer=tokenizer,
        model=model,
        prompt=args.prompt,
        enable_thinking=args.enable_thinking,
    )

    prompt_len = int(input_ids.shape[1])
    block_size = args.block_size
    small = args.small_block_size

    log(f"prompt_len={prompt_len}, block_size={block_size}, small_block_size={small}")
    log(f"mask_id={mask_id}, mask_token={tokenizer.convert_ids_to_tokens(mask_id)}")
    if stop_id is not None:
        log(f"stop_id={stop_id}, stop_token={tokenizer.convert_ids_to_tokens(stop_id)}")
    log(f"attn_impl={getattr(model.config, '_attn_implementation', None)}")

    # 只处理 prompt_len < block_size 或 prompt_len 非整 block 的第一个 partial block。
    # 对你的 case，prompt_len=21，current_block 是：
    # [prompt tokens 0:21] + [mask tokens 21:32]
    if prompt_len >= block_size:
        prefix_len = (prompt_len // block_size) * block_size
        prefix = input_ids[:, :prefix_len]
        rest = input_ids[:, prefix_len:]
    else:
        prefix_len = 0
        prefix = None
        rest = input_ids

    past_key_values = None
    if prefix_len > 0:
        log(f"Prefilling prefix_len={prefix_len}")
        prefill = model.forward(
            input_ids=prefix,
            use_cache=True,
            update_past_key_values=True,
            block_size=block_size,
        )
        past_key_values = prefill.past_key_values

    rest_len = int(rest.shape[1])
    pad_len = block_size - rest_len
    if pad_len <= 0:
        raise RuntimeError(f"This script expects a partial current block, got rest_len={rest_len}")

    masks = torch.full(
        (rest.shape[0], pad_len),
        fill_value=mask_id,
        device=rest.device,
        dtype=rest.dtype,
    )
    current_block = torch.cat([rest, masks], dim=1)
    assert current_block.shape[1] == block_size

    prompt_offset = rest_len
    small_start = (prompt_offset // small) * small
    small_end = small_start + small

    log(
        f"current_block_len={current_block.shape[1]}, "
        f"prompt_offset_in_current_block={prompt_offset}, "
        f"target small block={small_start}:{small_end}"
    )

    current_tokens = tokenizer.convert_ids_to_tokens(current_block[0].tolist())
    log("current_block tokens:")
    print(json.dumps(current_tokens, ensure_ascii=False, indent=2), flush=True)

    # 1. 初始化 block cache：完整 current_block forward。
    init_out = model.forward(
        input_ids=current_block,
        use_cache=True,
        past_key_values=past_key_values,
        update_past_key_values=False,
        use_block_cache=True,
        block_size=block_size,
    )
    block_past_key_values = init_out.block_past_key_values
    assert block_past_key_values is not None

    # 2. 检查 init full-block cache logits 是否等价于 no-cache full recompute。
    ref0 = model.forward(
        input_ids=current_block,
        use_cache=True,
        past_key_values=past_key_values,
        update_past_key_values=False,
        block_size=block_size,
        use_block_cache=False,
    ).logits

    init_shift = shift_logits(init_out.logits)[:, small_start:small_end, :]
    ref0_shift = shift_logits(ref0)[:, small_start:small_end, :]

    mask_slice = current_block[:, small_start:small_end] == mask_id

    log(f"INIT full-cache vs no-cache max_abs all={max_abs(init_shift, ref0_shift):.6g}")
    if mask_slice.any():
        init_mask_logits = init_shift[mask_slice]
        ref0_mask_logits = ref0_shift[mask_slice]
        log(f"INIT full-cache vs no-cache max_abs mask_only={max_abs(init_mask_logits, ref0_mask_logits):.6g}")

    # 3. 模拟 generate 第一轮：用 ref logits 给第一个 mask 位置填一个 argmax token。
    updated = current_block.clone()
    first_mask_abs = prompt_offset
    first_mask_local = first_mask_abs - small_start

    token_for_first_mask = ref0_shift[:, first_mask_local, :].argmax(dim=-1)
    updated[:, first_mask_abs] = token_for_first_mask

    log(
        f"Filled first mask at block_pos={first_mask_abs}, "
        f"local_pos={first_mask_local}, "
        f"token_id={int(token_for_first_mask[0].item())}, "
        f"token={tokenizer.convert_ids_to_tokens(int(token_for_first_mask[0].item()))}"
    )

    # 4. 现在比较：updated current_block 的 full recompute vs replace_position=small_start。
    ref1 = model.forward(
        input_ids=updated,
        use_cache=True,
        past_key_values=past_key_values,
        update_past_key_values=False,
        block_size=block_size,
        use_block_cache=False,
    ).logits
    ref1_shift = shift_logits(ref1)[:, small_start:small_end, :]

    replace_out = model.forward(
        input_ids=updated[:, small_start:small_end],
        use_cache=True,
        past_key_values=past_key_values,
        update_past_key_values=False,
        use_block_cache=True,
        block_past_key_values=block_past_key_values,
        replace_position=small_start,
        block_size=block_size,
    )
    got_shift = shift_logits(replace_out.logits)

    remaining_mask_slice = updated[:, small_start:small_end] == mask_id

    log(f"REPLACE vs full max_abs all={max_abs(got_shift, ref1_shift):.6g}")

    # 对 generate 真正会用的位置，只比较 remaining mask positions。
    if remaining_mask_slice.any():
        got_mask = got_shift[remaining_mask_slice]
        ref_mask = ref1_shift[remaining_mask_slice]
        diff = max_abs(got_mask, ref_mask)
        log(f"REPLACE vs full max_abs remaining_mask_only={diff:.6g}")

        try:
            torch.testing.assert_close(
                got_mask.float(),
                ref_mask.float(),
                atol=args.atol,
                rtol=args.rtol,
            )
            log("[PASS] partial-block replace logits match on remaining mask positions.")
        except AssertionError as e:
            log("[FAIL] partial-block replace logits mismatch on remaining mask positions.")
            print(str(e), flush=True)

            # 额外打印哪个 local mask 位置差异最大。
            abs_diff = (got_shift.float() - ref1_shift.float()).abs()
            local_scores = []
            for local_idx in range(small):
                if bool(remaining_mask_slice[0, local_idx].item()):
                    local_scores.append(
                        (local_idx, float(abs_diff[0, local_idx].max().item()))
                    )
            log(f"remaining mask local max diffs={local_scores}")
            raise
    else:
        log("No remaining mask positions after filling first mask; nothing to compare.")

    sync_cuda()


if __name__ == "__main__":
    main()