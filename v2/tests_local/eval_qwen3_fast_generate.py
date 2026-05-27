"""
运行：【不能四个一起评测，必须单独创建进程，否则会污染评测环境...】
for ITEM in static past block_full block_replace; do
  echo "===== $ITEM ====="
  rm -rf ~/.cache/huggingface/modules/transformers_modules/Model-Qwen-3-8B
  python tests_local/eval_qwen3_fast_generate.py \
    --model-dir /home/u-shengbf/Codes/Fast-dLLM/v2/base_models/Model-Qwen-3-8B \
    --only $ITEM \
    --block-size 32 \
    --small-block-size 8
done
"""

import argparse
import inspect
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def first_device(model):
    try:
        return model.device
    except Exception:
        return next(model.parameters()).device


def set_deterministic(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def resolve_dtype(dtype: str):
    if dtype == "auto":
        return "auto"
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    if dtype == "fp32":
        return torch.float32
    raise ValueError(f"Unknown dtype: {dtype}")


def load_model_and_tokenizer(args):
    log(f"Loading tokenizer from {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        use_fast=True,
    )

    log(f"Loading model from {args.model_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        torch_dtype=resolve_dtype(args.dtype),
        device_map=args.device_map,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


def token_id(tokenizer, token: str) -> Optional[int]:
    try:
        tid = tokenizer.convert_tokens_to_ids(token)
    except Exception:
        return None
    if tid is None:
        return None
    if isinstance(tid, list):
        return tid[0] if tid else None
    if tid == tokenizer.unk_token_id:
        # 对部分 tokenizer，未知 token 会映射到 unk；Qwen 通常没有 unk，但这里保守处理。
        return None
    return int(tid)


def resolve_special_ids(tokenizer, args) -> Tuple[int, int]:
    if args.mask_id is not None:
        mask_id = args.mask_id
    else:
        mask_id = token_id(tokenizer, "|<MASK>|")

    if mask_id is None or mask_id < 0:
        raise RuntimeError(
            "无法解析 |<MASK>| 的 token id。"
            "请确认 Qwen3-Fast tokenizer 已新增 |<MASK>|，或通过 --mask-id 显式传入。"
        )

    if args.stop_id is not None:
        stop_id = args.stop_id
    else:
        im_end = token_id(tokenizer, "<|im_end|>")
        stop_id = im_end if im_end is not None else tokenizer.eos_token_id

    if stop_id is None or stop_id < 0:
        raise RuntimeError("无法解析 stop token。请通过 --stop-id 显式传入。")

    return int(mask_id), int(stop_id)


def get_vocab_size(model, tokenizer) -> int:
    if hasattr(model, "config") and hasattr(model.config, "vocab_size"):
        return int(model.config.vocab_size)
    return len(tokenizer)


def make_normal_ids(
    tokenizer,
    model,
    length: int,
    device,
    avoid_ids: Optional[set] = None,
) -> torch.LongTensor:
    avoid_ids = avoid_ids or set()
    text = (
        "The quick brown fox jumps over the lazy dog. "
        "Large language models can reason about code, math, and natural language. "
        "你好，今天我们测试生成逻辑和缓存逻辑。 "
    )
    ids = tokenizer(
        text * max(8, length // 20 + 8),
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids[0].tolist()

    clean = []
    all_special = set(getattr(tokenizer, "all_special_ids", []) or [])
    for x in ids:
        if x not in avoid_ids and x not in all_special:
            clean.append(x)

    if not clean:
        # 极端 fallback：随机选非 special token。
        vocab = get_vocab_size(model, tokenizer)
        clean = [
            i for i in range(min(vocab, 2000))
            if i not in avoid_ids and i not in all_special
        ]

    while len(clean) < length:
        clean.extend(clean)

    return torch.tensor(clean[:length], dtype=torch.long, device=device).unsqueeze(0)


def shift_logits_like_generate(logits: torch.Tensor) -> torch.Tensor:
    return torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def assert_close_logits(
    name: str,
    got: torch.Tensor,
    ref: torch.Tensor,
    atol: float,
    rtol: float,
):
    if got.shape != ref.shape:
        raise AssertionError(f"{name}: shape mismatch got={got.shape}, ref={ref.shape}")

    diff = max_abs(got, ref)
    ok = torch.allclose(got.float(), ref.float(), atol=atol, rtol=rtol)
    if not ok:
        # 给出更有解释力的 top-k 差异
        pos = (got.float() - ref.float()).abs().argmax().item()
        raise AssertionError(
            f"{name}: logits not close. max_abs={diff:.6g}, "
            f"got_shape={tuple(got.shape)}, ref_shape={tuple(ref.shape)}, flat_argmax={pos}"
        )


def check_static(model, tokenizer, mask_id: int, stop_id: int, args) -> List[CheckResult]:
    results = []
    device = first_device(model)

    results.append(CheckResult(
        "model_eval_mode",
        not model.training,
        f"model.training={model.training}",
    ))

    sig = inspect.signature(model.forward)
    required_forward_args = [
        "input_ids",
        "past_key_values",
        "use_cache",
        "update_past_key_values",
        "block_size",
        "use_block_cache",
        "block_past_key_values",
        "replace_position",
    ]
    missing = [x for x in required_forward_args if x not in sig.parameters]
    results.append(CheckResult(
        "forward_signature",
        len(missing) == 0,
        f"missing={missing}",
    ))

    gen_sig = inspect.signature(model.generate)
    required_generate_args = [
        "input_ids",
        "max_new_tokens",
        "small_block_size",
        "block_size",
        "threshold",
        "temperature",
        "use_block_cache",
    ]
    missing_gen = [x for x in required_generate_args if x not in gen_sig.parameters]
    results.append(CheckResult(
        "generate_signature",
        len(missing_gen) == 0,
        f"missing={missing_gen}",
    ))

    cfg = model.config
    qwen3_hints = {
        "num_hidden_layers": getattr(cfg, "num_hidden_layers", None),
        "num_attention_heads": getattr(cfg, "num_attention_heads", None),
        "num_key_value_heads": getattr(cfg, "num_key_value_heads", None),
        "hidden_size": getattr(cfg, "hidden_size", None),
        "vocab_size": getattr(cfg, "vocab_size", None),
        "rope_theta": getattr(cfg, "rope_theta", None),
    }
    results.append(CheckResult(
        "config_summary",
        True,
        json.dumps(qwen3_hints, ensure_ascii=False),
    ))

    layer0 = None
    try:
        layer0 = model.model.layers[0].self_attn
    except Exception:
        pass

    has_q_norm = hasattr(layer0, "q_norm") if layer0 is not None else False
    has_k_norm = hasattr(layer0, "k_norm") if layer0 is not None else False
    results.append(CheckResult(
        "qwen3_attention_norms",
        has_q_norm and has_k_norm,
        f"has_q_norm={has_q_norm}, has_k_norm={has_k_norm}. "
        "如果这里失败，要确认你们 Qwen3 适配是否把 q_norm/k_norm 接进 attention。"
    ))

    results.append(CheckResult(
        "special_ids",
        mask_id != stop_id and mask_id >= 0 and stop_id >= 0,
        f"mask_id={mask_id}, stop_id={stop_id}, "
        f"mask_token={tokenizer.convert_ids_to_tokens(mask_id)}, "
        f"stop_token={tokenizer.convert_ids_to_tokens(stop_id)}",
    ))

    try:
        src = inspect.getsource(model.generate)
        suspicious = "input_ids.shape[1] > block_size" in src
        results.append(CheckResult(
            "generate_prefill_boundary_scan",
            not suspicious,
            "发现 generate 中可能存在 `input_ids.shape[1] > block_size` 边界逻辑；"
            "请重点测试 prompt_len == block_size。"
            if suspicious else "no suspicious `> block_size` prefill boundary found",
        ))
    except Exception as e:
        results.append(CheckResult(
            "generate_source_scan",
            False,
            f"inspect.getsource failed: {repr(e)}",
        ))

    results.append(CheckResult(
        "device",
        True,
        f"first_device={device}",
    ))

    return results


@torch.no_grad()
def check_forward_past_cache_equivalence(model, tokenizer, mask_id: int, stop_id: int, args):
    """
    验证普通 past_key_values：
    full forward(prefix + tail) 的 tail logits
    应等价于 prefill(prefix) + cached forward(tail) 的 logits。
    """
    device = first_device(model)
    avoid = {mask_id, stop_id}
    block_size = args.block_size

    prefix_len = block_size * 2
    tail_len = block_size
    ids = make_normal_ids(
        tokenizer,
        model,
        prefix_len + tail_len,
        device=device,
        avoid_ids=avoid,
    )
    prefix = ids[:, :prefix_len]
    tail = ids[:, prefix_len:]

    full = model(
        input_ids=ids,
        use_cache=False,
        block_size=block_size,
    ).logits[:, -tail_len:, :]

    prefill = model(
        input_ids=prefix,
        use_cache=True,
        update_past_key_values=True,
        block_size=block_size,
    )

    cached = model(
        input_ids=tail,
        use_cache=True,
        past_key_values=prefill.past_key_values,
        update_past_key_values=False,
        block_size=block_size,
    ).logits

    assert_close_logits(
        "past_cache_equivalence",
        got=cached,
        ref=full,
        atol=args.atol,
        rtol=args.rtol,
    )

    return CheckResult(
        "past_cache_equivalence",
        True,
        f"prefix_len={prefix_len}, tail_len={tail_len}, max_abs={max_abs(cached, full):.6g}",
    )


@torch.no_grad()
def check_block_cache_full_branch(model, tokenizer, mask_id: int, stop_id: int, args):
    """
    验证 use_block_cache=True 第一次 full-block forward 的 logits
    与不用 block cache 的 full-block recompute 一致。
    """
    device = first_device(model)
    avoid = {mask_id, stop_id}
    block_size = args.block_size
    small_block_size = args.small_block_size

    prefix_len = block_size
    full_len = prefix_len + block_size
    ids = make_normal_ids(tokenizer, model, full_len, device, avoid)

    prefix = ids[:, :prefix_len]
    cur_block = ids[:, prefix_len:].clone()

    # 注入一部分 mask，使状态更接近 generate 中 x_t。
    cur_block[:, small_block_size: small_block_size * 2] = mask_id
    cur_block[:, -small_block_size:] = mask_id

    prefill = model(
        input_ids=prefix,
        use_cache=True,
        update_past_key_values=True,
        block_size=block_size,
    )
    past = prefill.past_key_values

    ref_out = model(
        input_ids=cur_block,
        use_cache=True,
        past_key_values=past,
        update_past_key_values=False,
        block_size=block_size,
        use_block_cache=False,
    )
    ref_shifted = shift_logits_like_generate(ref_out.logits)

    got_out = model(
        input_ids=cur_block,
        use_cache=True,
        past_key_values=past,
        update_past_key_values=False,
        block_size=block_size,
        use_block_cache=True,
    )
    got_shifted = shift_logits_like_generate(got_out.logits)

    assert got_out.block_past_key_values is not None, "block_past_key_values is None"

    assert_close_logits(
        "block_cache_full_branch",
        got=got_shifted,
        ref=ref_shifted,
        atol=args.atol,
        rtol=args.rtol,
    )

    return CheckResult(
        "block_cache_full_branch",
        True,
        f"max_abs={max_abs(got_shifted, ref_shifted):.6g}",
    )


@torch.no_grad()
def check_block_cache_replace_branch(model, tokenizer, mask_id: int, stop_id: int, args):
    """
    验证 replace_position 分支：
    1. 先对完整 block 建 block_past_key_values；
    2. 修改某个 small block；
    3. 用 replace_position 局部更新；
    4. 与完整 block recompute 对齐比较。

    注意：Fast generate 对 partial slice 做局部 shift。
    当 small_block_start_idx > 0 时，slice 第 0 个位置的 shifted logit
    与完整 block shifted logit 不严格对齐；但 generate 通常不会用它
    去更新该 small block 的第一个 token。因此这里比较 slice[1:]。
    """
    device = first_device(model)
    avoid = {mask_id, stop_id}
    block_size = args.block_size
    small = args.small_block_size
    assert block_size % small == 0

    prefix_len = block_size
    ids = make_normal_ids(tokenizer, model, prefix_len + block_size, device, avoid)
    prefix = ids[:, :prefix_len]
    cur_block = ids[:, prefix_len:].clone()

    prefill = model(
        input_ids=prefix,
        use_cache=True,
        update_past_key_values=True,
        block_size=block_size,
    )
    past = prefill.past_key_values

    init = model(
        input_ids=cur_block,
        use_cache=True,
        past_key_values=past,
        update_past_key_values=False,
        block_size=block_size,
        use_block_cache=True,
    )
    block_past = init.block_past_key_values
    assert block_past is not None, "initial block_past_key_values is None"

    details = []
    for small_start in range(0, block_size, small):
        small_end = small_start + small

        updated = cur_block.clone()
        replacement = make_normal_ids(tokenizer, model, small, device, avoid)
        updated[:, small_start:small_end] = replacement

        # 在 slice 内注入 mask，模拟局部尚未完全生成。
        if small >= 4:
            updated[:, small_start + 2: small_end] = mask_id

        ref = model(
            input_ids=updated,
            use_cache=True,
            past_key_values=past,
            update_past_key_values=False,
            block_size=block_size,
            use_block_cache=False,
        ).logits
        ref_shifted_slice = shift_logits_like_generate(ref)[:, small_start:small_end, :]

        got = model(
            input_ids=updated[:, small_start:small_end],
            use_cache=True,
            past_key_values=past,
            update_past_key_values=False,
            block_size=block_size,
            use_block_cache=True,
            block_past_key_values=block_past,
            replace_position=small_start,
        ).logits
        got_shifted = shift_logits_like_generate(got)

        # 见函数 docstring：small_start > 0 时跳过 slice 第 0 个位置。
        compare_from = 0 if small_start == 0 else 1
        got_cmp = got_shifted[:, compare_from:, :]
        ref_cmp = ref_shifted_slice[:, compare_from:, :]

        assert_close_logits(
            f"block_cache_replace_branch_start_{small_start}",
            got=got_cmp,
            ref=ref_cmp,
            atol=args.atol,
            rtol=args.rtol,
        )
        details.append(
            f"start={small_start}, compared_positions={got_cmp.shape[1]}, "
            f"max_abs={max_abs(got_cmp, ref_cmp):.6g}"
        )

        # 更新当前 cache，模拟 generate 中连续 replace。
        block_past = model(
            input_ids=updated[:, small_start:small_end],
            use_cache=True,
            past_key_values=past,
            update_past_key_values=False,
            block_size=block_size,
            use_block_cache=True,
            block_past_key_values=block_past,
            replace_position=small_start,
        ).block_past_key_values

    return CheckResult(
        "block_cache_replace_branch",
        True,
        "; ".join(details),
    )


def build_prompt_ids(tokenizer, model, target_len: int, device, mask_id: int, stop_id: int):
    ids = make_normal_ids(
        tokenizer,
        model,
        target_len,
        device=device,
        avoid_ids={mask_id, stop_id},
    )
    assert ids.shape[1] == target_len
    return ids


@torch.no_grad()
def run_generate_once(
    model,
    input_ids,
    tokenizer,
    mask_id: int,
    stop_id: int,
    args,
    use_block_cache: bool,
):
    return model.generate(
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


@torch.no_grad()
def check_generate_equivalence(model, tokenizer, mask_id: int, stop_id: int, args):
    """
    端到端比较：
    temperature=0 下，use_block_cache=False/True 的输出 token ids 应完全一致。
    """
    device = first_device(model)
    block = args.block_size

    prompt_lens = [
        max(1, block // 2),
        block,
        block + 1,
        block * 2 - 1,
        block * 2,
        block * 2 + 1,
    ]

    details = []
    for plen in prompt_lens:
        input_ids = build_prompt_ids(tokenizer, model, plen, device, mask_id, stop_id)

        set_deterministic(args.seed)
        out_no_cache = run_generate_once(
            model, input_ids, tokenizer, mask_id, stop_id, args, use_block_cache=False
        )

        set_deterministic(args.seed)
        out_cache = run_generate_once(
            model, input_ids, tokenizer, mask_id, stop_id, args, use_block_cache=True
        )

        same = torch.equal(out_no_cache, out_cache)
        if not same:
            min_len = min(out_no_cache.shape[1], out_cache.shape[1])
            neq = (out_no_cache[:, :min_len] != out_cache[:, :min_len]).nonzero()
            first_diff = neq[0].tolist() if neq.numel() > 0 else "length_mismatch_only"
            no_cache_text = tokenizer.decode(out_no_cache[0], skip_special_tokens=False)
            cache_text = tokenizer.decode(out_cache[0], skip_special_tokens=False)
            raise AssertionError(
                f"generate_equivalence failed at prompt_len={plen}. "
                f"out_no_cache_shape={tuple(out_no_cache.shape)}, "
                f"out_cache_shape={tuple(out_cache.shape)}, "
                f"first_diff={first_diff}\n"
                f"--- no_cache ---\n{no_cache_text[-1000:]}\n"
                f"--- cache ---\n{cache_text[-1000:]}\n"
            )

        details.append(f"prompt_len={plen}, out_len={out_cache.shape[1]}")

    return CheckResult(
        "generate_equivalence_temperature0",
        True,
        "; ".join(details),
    )


@torch.no_grad()
def check_batch_sampling_shape(model, args):
    """
    检查 sample_with_top_p 在 batch_size > 1 且 temperature > 0 时是否返回正确形状。
    原 Fast-dLLM v2 公开实现里这里有过只 sample p_1t[0] 的问题。
    """
    device = first_device(model)
    vocab = int(getattr(model.config, "vocab_size", 32000))
    logits = torch.randn(
        2,
        args.small_block_size,
        min(vocab, args.sampling_vocab_probe),
        device=device,
        dtype=torch.float32,
    )

    if not hasattr(model, "sample_with_top_p"):
        return CheckResult(
            "batch_sampling_shape",
            False,
            "model has no sample_with_top_p",
        )

    x, p = model.sample_with_top_p(
        logits,
        top_p=args.top_p,
        temperature=0.7,
    )

    ok = tuple(x.shape) == (2, args.small_block_size) and tuple(p.shape[:2]) == (2, args.small_block_size)
    return CheckResult(
        "batch_sampling_shape",
        ok,
        f"x_shape={tuple(x.shape)}, p_shape={tuple(p.shape)}. "
        "如果失败，batch_size>1 且 temperature>0 的随机采样路径不可用；"
        "temperature=0 的 greedy generate 不受这个特定问题影响。",
    )


def print_results(results: List[CheckResult]):
    print("\n========== CHECK RESULTS ==========")
    all_ok = True
    for r in results:
        status = "PASS" if r.ok else "FAIL"
        if not r.ok:
            all_ok = False
        print(f"[{status}] {r.name}: {r.detail}")
    print("===================================\n")
    if not all_ok:
        raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        default="/home/u-shengbf/Codes/Fast-dLLM/v2/base_models/Model-Qwen-3-8B",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="bf16", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--small-block-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--mask-id", type=int, default=None)
    parser.add_argument("--stop-id", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument(
        "--only",
        default="static,past,block_full,block_replace,generate,sampling",
        help="逗号分隔：static,past,block_full,block_replace,generate,sampling",
    )
    parser.add_argument("--sampling-vocab-probe", type=int, default=4096)
    args = parser.parse_args()

    assert args.block_size % args.small_block_size == 0, "block_size must be divisible by small_block_size"
    assert args.max_new_tokens % args.block_size == 0, (
        "建议 max_new_tokens 是 block_size 的整数倍；"
        "Fast-dLLM v2 generate 常见实现使用 max_new_tokens // block_size。"
    )

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    set_deterministic(args.seed)

    model, tokenizer = load_model_and_tokenizer(args)
    mask_id, stop_id = resolve_special_ids(tokenizer, args)

    selected = set(x.strip() for x in args.only.split(",") if x.strip())
    results: List[CheckResult] = []

    if "static" in selected:
        log("Running static checks")
        results.extend(check_static(model, tokenizer, mask_id, stop_id, args))
        # static 里有提示型 FAIL，比如 q_norm/k_norm；先打印但不立即退出。
        print_results(results)

    results = []

    if "past" in selected:
        log("Running past_key_values equivalence")
        results.append(check_forward_past_cache_equivalence(model, tokenizer, mask_id, stop_id, args))

    if "block_full" in selected:
        log("Running block cache full-branch equivalence")
        results.append(check_block_cache_full_branch(model, tokenizer, mask_id, stop_id, args))

    if "block_replace" in selected:
        log("Running block cache replace_position equivalence")
        results.append(check_block_cache_replace_branch(model, tokenizer, mask_id, stop_id, args))

    if "generate" in selected:
        log("Running end-to-end generate equivalence")
        results.append(check_generate_equivalence(model, tokenizer, mask_id, stop_id, args))

    if "sampling" in selected:
        log("Running batch sampling shape check")
        results.append(check_batch_sampling_shape(model, args))

    print_results(results)


if __name__ == "__main__":
    main()