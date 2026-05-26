# generate_mbpp_fast_examples.py
import argparse
import ast
import json
import re
import types
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


COMMON_NON_TARGET_CALLS = {
    "assert", "set", "list", "tuple", "dict", "len", "sum", "min", "max",
    "abs", "all", "any", "sorted", "round", "range", "str", "int", "float",
    "bool", "print", "enumerate", "zip", "map", "filter", "isinstance",
    "type", "reversed", "math", "re",
}


def import_generation_functions():
    try:
        import generation_functions
        return generation_functions
    except ImportError:
        try:
            import generation_function
            return generation_function
        except ImportError:
            return None


def load_mbpp_tasks():
    from evalplus.data import get_mbpp_plus
    return get_mbpp_plus()


def get_entry_point(problem: dict) -> str:
    for key in ("entry_point", "canonical_entry_point"):
        if key in problem and problem[key]:
            return problem[key]

    tests = "\n".join(problem.get("test_list", []))
    calls = re.findall(r"\b([A-Za-z_]\w*)\s*\(", tests)
    calls = [c for c in calls if c not in COMMON_NON_TARGET_CALLS]

    if not calls:
        raise ValueError(f"Cannot infer entry point from tests:\n{tests}")

    return Counter(calls).most_common(1)[0][0]


def build_mbpp_prompt(problem: dict) -> str:
    """
    MBPP 是完整函数生成，不是 HumanEval 式函数补全。
    这里强制模型遵守 EvalPlus 期望的 entry point。
    """
    entry_point = get_entry_point(problem)
    tests = "\n".join(problem.get("test_list", []))
    prompt = problem["prompt"].strip()

    return (
        "You are writing a solution for an EvalPlus MBPP task.\n"
        "Return ONLY executable Python code.\n"
        "Do NOT include Markdown fences.\n"
        "Do NOT include explanations.\n"
        "Do NOT include test cases or assert statements.\n"
        "Do NOT define a differently named top-level function.\n"
        f"The required top-level function name is exactly: {entry_point}\n"
        f"The solution must define `def {entry_point}(...):` or an equivalent callable named `{entry_point}`.\n\n"
        f"Problem:\n{prompt}\n\n"
        f"The function must pass these tests:\n{tests}\n\n"
        "Python code only:\n"
    )


def apply_chat_template(tokenizer, prompt: str, use_chat_template: bool) -> str:
    if not use_chat_template:
        return prompt

    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def is_compilable(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def remove_markdown(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = text.replace("```python", "")
    text = text.replace("```", "")
    return text


def fenced_blocks(text: str):
    return re.findall(r"```(?:python)?\s*\n(.*?)```", text, flags=re.S)


def cut_tail_after_code(text: str) -> str:
    """
    只在已经定位到代码候选后使用。
    不要在原始 MBPP prompt 上直接按 assert 截断。
    """
    text = text.replace("\r\n", "\n")

    stop_patterns = [
        "\n### Explanation",
        "\n## Explanation",
        "\n# Explanation",
        "\n###",
        "\nThe function",
        "\nThis function",
        "\nExplanation",
        "\nIn the given",
        "\nThe `",
        "\nThis code",
        "\nIt works",
        "\nHere",
        "\nNote:",
        "\nassert ",
        "\nassert(",
        "\nprint(",
        "\nif __name__",
        "\n# Test",
        "\n# test",
        "\nTests:",
    ]

    cut = len(text)
    for pat in stop_patterns:
        idx = text.find(pat)
        if idx != -1:
            cut = min(cut, idx)

    return text[:cut].strip() + "\n"


def candidate_from_target_def(text: str, entry_point: str):
    text = remove_markdown(text)
    pattern = rf"def\s+{re.escape(entry_point)}\s*\("
    m = re.search(pattern, text)
    if not m:
        return None

    before = text[:m.start()]
    import_lines = []
    for line in before.splitlines():
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            import_lines.append(line)

    body = text[m.start():]
    if import_lines:
        return "\n".join(import_lines) + "\n\n" + body
    return body


def candidate_from_case_insensitive_target_def(text: str, entry_point: str):
    """
    只用于捕捉 count_substrings vs count_Substrings 这种大小写差异。
    不处理 find_n_largest vs heap_queue_largest 这种语义不同函数名。
    """
    text = remove_markdown(text)
    pattern = r"def\s+([A-Za-z_]\w*)\s*\("
    for m in re.finditer(pattern, text):
        found = m.group(1)
        if found.lower() == entry_point.lower():
            before = text[:m.start()]
            import_lines = []
            for line in before.splitlines():
                stripped = line.strip()
                if stripped.startswith("import ") or stripped.startswith("from "):
                    import_lines.append(line)

            body = text[m.start():]
            if import_lines:
                return "\n".join(import_lines) + "\n\n" + body
            return body

    return None


def candidate_from_first_code_start(text: str):
    text = remove_markdown(text)
    starts = []
    for pat in ("import ", "from ", "def ", "class "):
        idx = text.find(pat)
        if idx != -1:
            starts.append(idx)

    if not starts:
        return text

    return text[min(starts):]


def make_compilable_stub(entry_point: str) -> str:
    """
    兜底：如果模型输出无法抽取为可编译代码，写一个必错但可编译的 stub。
    这样 invalid generation 仍计为 wrong，不会污染为语法错误或触发奇怪 fallback。
    """
    return f"def {entry_point}(*args, **kwargs):\n    return None\n"


class ConservativeRename(ast.NodeTransformer):
    """
    只用于大小写不一致的同名重命名。
    例如 count_substrings -> count_Substrings。
    不用于语义不同的函数名。
    """
    def __init__(self, old_name: str, new_name: str):
        self.old_name = old_name
        self.new_name = new_name

    def visit_FunctionDef(self, node):
        if node.name == self.old_name:
            node.name = self.new_name
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        if node.name == self.old_name:
            node.name = self.new_name
        return self.generic_visit(node)

    def visit_Name(self, node):
        if node.id == self.old_name:
            node.id = self.new_name
        return node


def ensure_entry_point_name_conservative(code: str, entry_point: str) -> str:
    """
    只处理大小写不同但 lower 完全相同的 entry point。
    不会把 find_n_largest 改成 heap_queue_largest。
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    defined_names = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined_names.append(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    defined_names.append(target.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                defined_names.append(node.target.id)

    if entry_point in defined_names:
        return code

    candidates = [name for name in defined_names if name.lower() == entry_point.lower()]
    if not candidates:
        return code

    old_name = candidates[0]
    tree = ConservativeRename(old_name, entry_point).visit(tree)
    ast.fix_missing_locations(tree)

    return ast.unparse(tree) + "\n"


def clean_generated_mbpp_code(raw_text: str, problem: dict):
    """
    生成时即时清理，写入 JSONL 前保证 solution 尽量是 EvalPlus 可执行代码。
    不做激进函数名改写，避免屏蔽模型本身问题。
    """
    entry_point = get_entry_point(problem)
    candidates = []

    for block in fenced_blocks(raw_text):
        candidates.append(("fence", block))

    exact = candidate_from_target_def(raw_text, entry_point)
    if exact is not None:
        candidates.append(("target_def", exact))

    ci = candidate_from_case_insensitive_target_def(raw_text, entry_point)
    if ci is not None:
        candidates.append(("case_insensitive_target_def", ci))

    candidates.append(("first_code", candidate_from_first_code_start(raw_text)))
    candidates.append(("raw_no_md", remove_markdown(raw_text)))

    scored = []
    for name, candidate in candidates:
        code = cut_tail_after_code(candidate)
        code = ensure_entry_point_name_conservative(code, entry_point)

        score = 0
        if f"def {entry_point}" in code:
            score += 100
        if entry_point in code:
            score += 25
        if is_compilable(code):
            score += 50
        if "def " in code or "=" in code:
            score += 10
        if "```" not in code:
            score += 2
        if "Explanation" not in code and "###" not in code:
            score += 2

        scored.append((score, name, code))

    scored.sort(key=lambda x: x[0], reverse=True)

    for _, _, code in scored:
        if entry_point in code and is_compilable(code):
            return code, True

    for _, _, code in scored:
        if is_compilable(code):
            return code, True

    return make_compilable_stub(entry_point), False


def fast_batch_generate(
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
    top_p,
    temperature,
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

    batched_input_ids = torch.cat(padded, dim=0)
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
            top_p=top_p,
            temperature=temperature,
        )

    completions = []
    for batch_idx, prompt_len in enumerate(seq_lens):
        full_ids = generated[batch_idx]
        new_ids = full_ids[prompt_len:]
        new_ids = new_ids[new_ids != mask_id]
        text = tokenizer.decode(new_ids, skip_special_tokens=True)
        completions.append(text)

    return completions


def fast_generate_single_fallback(
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
    top_p,
    temperature,
):
    completions = []

    for prompt in prompts:
        inputs = tokenizer([prompt], return_tensors="pt").to(device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            out = model.generate(
                inputs["input_ids"],
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                block_size=bd_size,
                small_block_size=small_block_size,
                threshold=threshold,
                mask_id=mask_id,
                use_block_cache=use_block_cache,
                top_p=top_p,
                temperature=temperature,
            )

        new_ids = out[0][prompt_len:]
        new_ids = new_ids[new_ids != mask_id]
        text = tokenizer.decode(new_ids, skip_special_tokens=True)
        completions.append(text)

    return completions


def count_non_compilable(path: Path):
    total = 0
    bad = []

    with path.open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            total += 1
            try:
                ast.parse(obj["solution"])
            except SyntaxError as e:
                bad.append((obj["task_id"], str(e), obj["solution"][:500]))

    print(f"Compilability check: {len(bad)}/{total} not compilable")

    for task_id, err, code in bad[:20]:
        print("=" * 80)
        print(task_id)
        print(err)
        print(code)

    return len(bad), total


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)

    parser.add_argument("--mask_id", type=int, default=151665)
    parser.add_argument("--bd_size", type=int, default=32)
    parser.add_argument("--small_block_size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--use_block_cache", action="store_true")

    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.0)

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")

    parser.add_argument(
        "--use_chat_template",
        action="store_true",
        help="Use tokenizer chat template. Default False is usually better for EvalPlus-style MBPP generation.",
    )

    args = parser.parse_args()

    if args.max_new_tokens % args.bd_size != 0:
        raise ValueError(
            f"max_new_tokens must be divisible by bd_size. "
            f"Got max_new_tokens={args.max_new_tokens}, bd_size={args.bd_size}"
        )

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

    generation_functions = import_generation_functions()

    if generation_functions is not None:
        print("Using generation_functions.Fast_dLLM_QwenForCausalLM.batch_sample")
        model.mdm_sample = types.MethodType(
            generation_functions.Fast_dLLM_QwenForCausalLM.batch_sample,
            model,
        )
        generate_fn = fast_batch_generate
    else:
        print("generation_functions.py not found. Falling back to model.generate")
        generate_fn = fast_generate_single_fallback

    problems = load_mbpp_tasks()
    items = list(problems.items())

    if args.limit is not None:
        items = items[: args.limit]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Dataset: mbpp")
    print(f"Num problems: {len(items)}")
    print(f"Output: {output_path}")

    failed_extract = []

    with output_path.open("w", encoding="utf-8") as f:
        for start in tqdm(range(0, len(items), args.batch_size), desc="Generating MBPP"):
            batch_items = items[start : start + args.batch_size]
            task_ids = [x[0] for x in batch_items]
            problems_batch = [x[1] for x in batch_items]

            prompts = [
                apply_chat_template(
                    tokenizer,
                    build_mbpp_prompt(problem),
                    args.use_chat_template,
                )
                for problem in problems_batch
            ]

            completions = generate_fn(
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
                top_p=args.top_p,
                temperature=args.temperature,
            )

            for task_id, problem, completion in zip(task_ids, problems_batch, completions):
                solution, ok = clean_generated_mbpp_code(completion, problem)

                if not ok:
                    failed_extract.append(task_id)

                row = {
                    "task_id": task_id,
                    "solution": solution,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Done. Wrote {len(items)} samples to {output_path}")

    if failed_extract:
        print(f"Warning: {len(failed_extract)} samples used compilable stub fallback.")
        print("First fallback task ids:", failed_extract[:20])

    count_non_compilable(output_path)


if __name__ == "__main__":
    main()