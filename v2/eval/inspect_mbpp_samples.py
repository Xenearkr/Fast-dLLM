"""
使用须知：
用于检查 samples JSONL 是否符合 EvalPlus 期望格式。

用法：

[MBPP]
cd /home/u-shengbf/Codes/Fast-dLLM/v2

python inspect_mbpp_samples.py \
  --dataset mbpp \
  --samples evalplus_results/Qwen2.5_LoRA/mbpp_fast.jsonl \
  --show_n 5 \
  --show_bad_n 10

如果要顺便跑 base tests:
python inspect_evalplus_samples.py \
  --dataset mbpp \
  --samples evalplus_results/Fast/mbpp_fast.jsonl \
  --show_n 5 \
  --show_bad_n 10 \
  --run_base_tests




理想输出：

[MBPP]
not_compilable: 0
missing_entry_point: 0
has_markdown_fence: 0
has_explanation_text: 0
mbpp_prompt_residue: 0
has_assert_tests: 0 最好，但允许少量。

MBPP 的 solution 应该是完整 Python 代码，不应该包含 prompt、Markdown、解释或测试。

"""

import argparse
import ast
import json
import multiprocessing as mp
import re
import traceback
from collections import Counter, defaultdict
from pathlib import Path


COMMON_NON_TARGET_CALLS = {
    "assert",
    "set",
    "list",
    "tuple",
    "dict",
    "len",
    "sum",
    "min",
    "max",
    "abs",
    "all",
    "any",
    "sorted",
    "round",
    "range",
    "str",
    "int",
    "float",
    "bool",
    "print",
    "enumerate",
    "zip",
    "map",
    "filter",
    "isinstance",
    "type",
    "reversed",
    "math",
    "re",
}


EXPLANATION_MARKERS = [
    "The function",
    "This function",
    "Explanation",
    "### Explanation",
    "## Explanation",
    "# Explanation",
    "In the given",
    "This code",
    "It works",
    "Here is",
    "Here’s",
    "Note:",
    "The `",
]


def load_tasks(dataset: str):
    if dataset == "mbpp":
        from evalplus.data import get_mbpp_plus

        return get_mbpp_plus()

    if dataset == "humaneval":
        from evalplus.data import get_human_eval_plus

        return get_human_eval_plus()

    raise ValueError(f"Unsupported dataset: {dataset}")


def infer_entry_point(dataset: str, problem: dict):
    """
    尽量从 EvalPlus problem metadata 中读取 entry point。
    如果没有，则从 prompt / test_list 推断。
    """
    for key in ("entry_point", "canonical_entry_point"):
        if problem.get(key):
            return problem[key]

    if dataset == "humaneval":
        prompt = problem.get("prompt", "")
        m = re.search(r"def\s+([A-Za-z_]\w*)\s*\(", prompt)
        return m.group(1) if m else None

    if dataset == "mbpp":
        tests = "\n".join(problem.get("test_list", []))
        calls = re.findall(r"\b([A-Za-z_]\w*)\s*\(", tests)
        calls = [c for c in calls if c not in COMMON_NON_TARGET_CALLS]
        return Counter(calls).most_common(1)[0][0] if calls else None

    return None


def normalize_solution(dataset: str, problem: dict, row: dict):
    """
    EvalPlus accepts either solution or completion.
    For inspection, normalize both into a full solution-like string.
    """
    if "solution" in row and row["solution"] is not None:
        return row["solution"], "solution"

    if "completion" in row and row["completion"] is not None:
        if dataset == "humaneval":
            return problem.get("prompt", "") + row["completion"], "completion+prompt"
        return row["completion"], "completion"

    return "", "missing"


def parse_ast(code: str):
    try:
        return ast.parse(code)
    except SyntaxError:
        return None


def ast_function_names(code: str):
    """
    仅返回 def/async def 定义的函数名。
    保留这个函数用于打印兼容，但 entry point 判断不再只依赖它。
    """
    tree = parse_ast(code)
    if tree is None:
        return []

    return [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def ast_defined_names(code: str):
    """
    返回顶层定义名，兼容：
    - def foo(...)
    - async def foo(...)
    - class Foo
    - foo = lambda ...
    - foo = ...
    - foo: type = ...

    MBPP 中 next_power_of_2 = lambda ... 这类形式应该算作 entry point 存在。
    """
    tree = parse_ast(code)
    if tree is None:
        return []

    names = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)

        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.append(target.id)

        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                names.append(node.target.id)

    return names


def ast_imports(code: str):
    tree = parse_ast(code)
    if tree is None:
        return []

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")

    return imports


def compile_error(code: str):
    try:
        ast.parse(code)
        return None
    except SyntaxError as e:
        text = e.text.strip() if e.text else ""
        return f"{e.msg} at line {e.lineno}: {text}"


def has_markdown(code: str):
    return "```" in code


def has_explanation_text(code: str):
    return any(marker in code for marker in EXPLANATION_MARKERS)


def has_assert_tests(code: str):
    return re.search(r"(^|\n)\s*assert\s*[\(\s]", code) is not None


def mbpp_has_prompt_residue(problem: dict, code: str):
    prompt = problem.get("prompt", "").strip()
    if not prompt:
        return False

    # MBPP solution 应该是纯代码。若题面原文出现在 solution 中，通常说明 prompt 被写进去了。
    short_prompt = prompt[:60]
    if short_prompt and short_prompt in code:
        return True

    if code.lstrip().startswith('"""') or code.lstrip().startswith("'''"):
        return True

    return False


def humaneval_has_prompt(problem: dict, code: str):
    prompt = problem.get("prompt", "")
    if not prompt:
        return False

    return code.startswith(prompt) or prompt.strip() in code


def run_code_with_tests_worker(code: str, tests: str, queue):
    ns = {}
    try:
        exec(code, ns, ns)
        exec(tests, ns, ns)
        queue.put(("pass", None))
    except BaseException:
        queue.put(("fail", traceback.format_exc(limit=3)))


def run_base_tests(dataset: str, problem: dict, code: str, timeout: float):
    """
    Optional lightweight local execution.
    This is only for debugging.
    Full official scoring should still use evalplus.evaluate.
    """
    entry = infer_entry_point(dataset, problem)

    if dataset == "mbpp":
        tests = "\n".join(problem.get("test_list", []))

    else:
        test = problem.get("test", "")
        if not test or not entry:
            return "skip", "no test/entry_point"
        tests = test + f"\ncheck({entry})\n"

    if not tests.strip():
        return "skip", "no base tests"

    queue = mp.Queue()
    proc = mp.Process(target=run_code_with_tests_worker, args=(code, tests, queue))
    proc.start()
    proc.join(timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        return "timeout", f">{timeout}s"

    if queue.empty():
        return "fail", "process exited without result"

    return queue.get()


def inspect_samples(
    dataset: str,
    samples_path: Path,
    show_n: int,
    show_bad_n: int,
    run_tests: bool,
    test_timeout: float,
):
    problems = load_tasks(dataset)

    rows = []
    with samples_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                rows.append(
                    {
                        "line_no": line_no,
                        "task_id": None,
                        "fatal": f"JSONDecodeError: {e}",
                        "row": line[:200],
                    }
                )
                continue

            rows.append(obj)

    stats = Counter()
    bad_examples = defaultdict(list)
    seen_task_ids = set()

    print("=" * 100)
    print(f"Dataset: {dataset}")
    print(f"Samples: {samples_path}")
    print(f"Rows in file: {len(rows)}")
    print(f"Tasks in EvalPlus: {len(problems)}")
    print("=" * 100)

    if len(rows) != len(problems):
        print(f"[WARN] Row count mismatch: got {len(rows)}, expected {len(problems)}")

    for row in rows:
        if "fatal" in row:
            stats["fatal_json_error"] += 1
            bad_examples["fatal_json_error"].append(row)
            continue

        task_id = row.get("task_id")
        if not task_id:
            stats["missing_task_id"] += 1
            bad_examples["missing_task_id"].append(row)
            continue

        if task_id in seen_task_ids:
            stats["duplicate_task_id"] += 1
            bad_examples["duplicate_task_id"].append(row)

        seen_task_ids.add(task_id)

        if task_id not in problems:
            stats["unknown_task_id"] += 1
            bad_examples["unknown_task_id"].append(row)
            continue

        problem = problems[task_id]
        entry = infer_entry_point(dataset, problem)
        code, schema_type = normalize_solution(dataset, problem, row)

        if schema_type == "missing":
            stats["missing_solution_or_completion"] += 1
            bad_examples["missing_solution_or_completion"].append((task_id, code))
            continue

        stats[f"schema_{schema_type}"] += 1

        err = compile_error(code)
        if err is not None:
            stats["not_compilable"] += 1
            bad_examples["not_compilable"].append((task_id, err, code[:1200]))

        function_names = ast_function_names(code)
        defined_names = ast_defined_names(code)

        if entry and entry not in defined_names:
            stats["missing_entry_point"] += 1
            bad_examples["missing_entry_point"].append(
                (task_id, entry, defined_names, code[:1200])
            )

        if has_markdown(code):
            stats["has_markdown_fence"] += 1
            bad_examples["has_markdown_fence"].append((task_id, code[:1200]))

        if has_explanation_text(code):
            stats["has_explanation_text"] += 1
            bad_examples["has_explanation_text"].append((task_id, code[:1200]))

        if has_assert_tests(code):
            stats["has_assert_tests"] += 1
            bad_examples["has_assert_tests"].append((task_id, code[:1200]))

        if dataset == "mbpp" and mbpp_has_prompt_residue(problem, code):
            stats["mbpp_prompt_residue"] += 1
            bad_examples["mbpp_prompt_residue"].append((task_id, code[:1200]))

        if dataset == "humaneval" and not humaneval_has_prompt(problem, code):
            # 如果 samples 用 completion 字段，EvalPlus 可以接受；
            # 如果 samples 用 solution 字段，则一般应包含完整 prompt。
            stats["humaneval_prompt_not_found"] += 1
            bad_examples["humaneval_prompt_not_found"].append((task_id, code[:1200]))

        if run_tests and err is None and entry and entry in defined_names:
            status, detail = run_base_tests(dataset, problem, code, test_timeout)
            stats[f"base_test_{status}"] += 1

            if status != "pass":
                bad_examples[f"base_test_{status}"].append(
                    (task_id, detail, code[:1200])
                )

    missing_ids = sorted(set(problems.keys()) - seen_task_ids)
    extra_ids = sorted(seen_task_ids - set(problems.keys()))

    if missing_ids:
        stats["missing_task_ids"] = len(missing_ids)

    if extra_ids:
        stats["extra_task_ids"] = len(extra_ids)

    print("\n[Summary]")
    for key, value in stats.most_common():
        print(f"{key}: {value}")

    if missing_ids:
        print("\n[Missing task ids]")
        print(missing_ids[:50], "..." if len(missing_ids) > 50 else "")

    if extra_ids:
        print("\n[Extra/unknown task ids]")
        print(extra_ids[:50], "..." if len(extra_ids) > 50 else "")

    print("\n[First samples]")
    for i, row in enumerate(rows[:show_n]):
        print("=" * 100)
        print(f"Index: {i}")
        print(f"task_id: {row.get('task_id')}")

        if row.get("task_id") in problems:
            problem = problems[row["task_id"]]
            entry = infer_entry_point(dataset, problem)
            code, schema_type = normalize_solution(dataset, problem, row)

            print(f"schema_type: {schema_type}")
            print(f"entry_point: {entry}")
            print(f"function_names: {ast_function_names(code)}")
            print(f"defined_names: {ast_defined_names(code)}")
            print(f"imports: {ast_imports(code)}")
            print(f"compilable: {compile_error(code) is None}")
            print("-" * 100)
            print(code[:1200])
        else:
            print(str(row)[:1200])

    print("\n[Bad examples]")
    for category, examples in bad_examples.items():
        print("\n" + "#" * 100)
        print(f"{category}: {len(examples)}")

        for example in examples[:show_bad_n]:
            print("-" * 100)
            if isinstance(example, tuple):
                for part in example:
                    print(part)
            else:
                print(example)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mbpp", "humaneval"], required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--show_n", type=int, default=5)
    parser.add_argument("--show_bad_n", type=int, default=5)
    parser.add_argument("--run_base_tests", action="store_true")
    parser.add_argument("--test_timeout", type=float, default=2.0)

    args = parser.parse_args()

    inspect_samples(
        dataset=args.dataset,
        samples_path=Path(args.samples),
        show_n=args.show_n,
        show_bad_n=args.show_bad_n,
        run_tests=args.run_base_tests,
        test_timeout=args.test_timeout,
    )


if __name__ == "__main__":
    main()
