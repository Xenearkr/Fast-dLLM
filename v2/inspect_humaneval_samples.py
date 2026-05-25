"""
inspect_humaneval_samples.py

用于检查 HumanEval / HumanEval+ samples JSONL 是否适合直接交给 EvalPlus evaluate。

核心原则：
1. HumanEval 是函数补全任务，但 EvalPlus samples 可以是 solution 或 completion。
2. 如果是 completion 字段，本脚本会用 problem["prompt"] + completion 归一化为完整代码后检查。
3. 如果是 solution 字段，本脚本直接检查 solution。
4. 不检查 humaneval_prompt_not_found：sanitize 后可能会在函数前加 import，或重排代码，这不等价于评测错误。
5. 不用字符串规则检查 Explanation / Here is / assert：HumanEval docstring 中常合法包含这些词。
6. 硬检查项主要是：
   - 行数是否为 164；
   - task_id 是否完整、重复、未知；
   - JSON schema 是否包含 solution 或 completion；
   - 代码是否可编译；
   - 是否定义了目标 entry point；
   - 是否残留 Markdown code fence；
   - 是否存在真正的 AST-level assert 语句。

用法：
cd /home/u-shengbf/Codes/Fast-dLLM/v2
python inspect_humaneval_samples.py \
  --samples evalplus_results/Fast/humaneval_fast.jsonl \
  --show_n 5 \
  --show_bad_n 10

python inspect_humaneval_samples.py \
  --samples evalplus_results/Qwen2.5/humaneval_fast-sanitized.jsonl \
  --show_n 5 \
  --show_bad_n 10

python inspect_humaneval_samples.py \
  --samples evalplus_results/Fast/humaneval_fast-sanitized.jsonl \
  --show_n 5 \
  --show_bad_n 10 \
  --run_base_tests
"""

import argparse
import ast
import json
import multiprocessing as mp
import re
import traceback
from collections import Counter, defaultdict
from pathlib import Path


def load_humaneval_tasks():
    from evalplus.data import get_human_eval_plus
    return get_human_eval_plus()


def infer_entry_point(problem: dict):
    for key in ("entry_point", "canonical_entry_point"):
        if problem.get(key):
            return problem[key]

    prompt = problem.get("prompt", "")
    m = re.search(r"def\s+([A-Za-z_]\w*)\s*\(", prompt)
    return m.group(1) if m else None


def normalize_solution(problem: dict, row: dict):
    """
    EvalPlus supports either:
    - {"task_id": ..., "solution": full_code}
    - {"task_id": ..., "completion": generated_suffix}

    For inspection, normalize both into full executable code.
    """
    if "solution" in row and row["solution"] is not None:
        return row["solution"], "solution"

    if "completion" in row and row["completion"] is not None:
        return problem.get("prompt", "") + row["completion"], "completion+prompt"

    return "", "missing"


def parse_ast(code: str):
    try:
        return ast.parse(code)
    except SyntaxError:
        return None


def compile_error(code: str):
    try:
        ast.parse(code)
        return None
    except SyntaxError as e:
        text = e.text.strip() if e.text else ""
        return f"{e.msg} at line {e.lineno}: {text}"


def ast_function_names(code: str):
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
    HumanEval 通常应该是 def entry_point(...), 但这里也兼容：
    - async def
    - entry_point = lambda ...
    - entry_point = ...
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


def has_markdown_fence(code: str) -> bool:
    return "```" in code


def has_real_assert_nodes(code: str) -> bool:
    """
    只检测 AST 中真正可执行的 assert 语句。
    不会误报 docstring 里的 'assert ...' 示例。
    """
    tree = parse_ast(code)
    if tree is None:
        return False

    return any(isinstance(node, ast.Assert) for node in ast.walk(tree))


def has_top_level_non_docstring_string_expr(code: str) -> bool:
    """
    检测顶层裸字符串表达式。
    模块开头的 docstring 可接受；其他顶层字符串可能是残留自然语言。
    """
    tree = parse_ast(code)
    if tree is None:
        return False

    for idx, node in enumerate(tree.body):
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            if idx == 0:
                continue
            return True

    return False


def run_code_with_tests_worker(code: str, tests: str, queue):
    ns = {}
    try:
        exec(code, ns, ns)
        exec(tests, ns, ns)
        queue.put(("pass", None))
    except BaseException:
        queue.put(("fail", traceback.format_exc(limit=3)))


def run_base_tests(problem: dict, code: str, timeout: float):
    """
    Debug-only base tests. Official scoring should still use evalplus.evaluate.
    """
    entry = infer_entry_point(problem)
    test = problem.get("test", "")

    if not test or not entry:
        return "skip", "no test/entry_point"

    tests = test + f"\ncheck({entry})\n"

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


def inspect_samples(samples_path: Path, show_n: int, show_bad_n: int, run_tests: bool, test_timeout: float):
    problems = load_humaneval_tasks()

    rows = []
    with samples_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                row = json.loads(line)
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

            rows.append(row)

    stats = Counter()
    bad_examples = defaultdict(list)
    seen_task_ids = set()

    print("=" * 100)
    print("Dataset: humaneval")
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
        entry = infer_entry_point(problem)
        code, schema_type = normalize_solution(problem, row)

        if schema_type == "missing":
            stats["missing_solution_or_completion"] += 1
            bad_examples["missing_solution_or_completion"].append((task_id, row))
            continue

        stats[f"schema_{schema_type}"] += 1

        err = compile_error(code)
        if err is not None:
            stats["not_compilable"] += 1
            bad_examples["not_compilable"].append((task_id, err, code[:1200]))

        defined_names = ast_defined_names(code)

        if entry and entry not in defined_names:
            stats["missing_entry_point"] += 1
            bad_examples["missing_entry_point"].append((task_id, entry, defined_names, code[:1200]))

        if has_markdown_fence(code):
            stats["has_markdown_fence"] += 1
            bad_examples["has_markdown_fence"].append((task_id, code[:1200]))

        if has_real_assert_nodes(code):
            stats["has_real_assert_nodes"] += 1
            bad_examples["has_real_assert_nodes"].append((task_id, code[:1200]))

        if has_top_level_non_docstring_string_expr(code):
            stats["has_top_level_string_residue"] += 1
            bad_examples["has_top_level_string_residue"].append((task_id, code[:1200]))

        if run_tests and err is None and entry and entry in defined_names:
            status, detail = run_base_tests(problem, code, test_timeout)
            stats[f"base_test_{status}"] += 1

            if status != "pass":
                bad_examples[f"base_test_{status}"].append((task_id, detail, code[:1200]))

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

        task_id = row.get("task_id")
        if task_id in problems:
            problem = problems[task_id]
            entry = infer_entry_point(problem)
            code, schema_type = normalize_solution(problem, row)

            print(f"schema_type: {schema_type}")
            print(f"entry_point: {entry}")
            print(f"function_names: {ast_function_names(code)}")
            print(f"defined_names: {ast_defined_names(code)}")
            print(f"imports: {ast_imports(code)}")
            print(f"compilable: {compile_error(code) is None}")
            print(f"has_markdown_fence: {has_markdown_fence(code)}")
            print(f"has_real_assert_nodes: {has_real_assert_nodes(code)}")
            print(f"has_top_level_string_residue: {has_top_level_non_docstring_string_expr(code)}")
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
    parser.add_argument("--samples", required=True)
    parser.add_argument("--show_n", type=int, default=5)
    parser.add_argument("--show_bad_n", type=int, default=10)
    parser.add_argument("--run_base_tests", action="store_true")
    parser.add_argument("--test_timeout", type=float, default=2.0)

    args = parser.parse_args()

    inspect_samples(
        samples_path=Path(args.samples),
        show_n=args.show_n,
        show_bad_n=args.show_bad_n,
        run_tests=args.run_base_tests,
        test_timeout=args.test_timeout,
    )


if __name__ == "__main__":
    main()
