# 打补丁：humaneval对比和清洗
import argparse
import ast
import json
from pathlib import Path

from evalplus.data import get_human_eval_plus


def load_jsonl(path: Path):
    data = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            data[obj["task_id"]] = obj
    return data


def code_of(obj: dict, problem: dict) -> str:
    if obj.get("solution") is not None:
        return obj["solution"]
    if obj.get("completion") is not None:
        return problem["prompt"] + obj["completion"]
    return ""


def entry_point(problem: dict) -> str:
    ep = problem.get("entry_point") or problem.get("canonical_entry_point")
    if ep:
        return ep

    import re
    m = re.search(r"def\s+([A-Za-z_]\w*)\s*\(", problem["prompt"])
    if not m:
        raise ValueError(f"Cannot infer entry point from prompt:\n{problem['prompt']}")
    return m.group(1)


def parse_tree(code: str):
    try:
        return ast.parse(code)
    except SyntaxError:
        return None


def is_compilable(code: str) -> bool:
    return parse_tree(code) is not None


def defined_names(code: str):
    tree = parse_tree(code)
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


def has_entry(code: str, ep: str) -> bool:
    return ep in defined_names(code)


def strip_markdown_tokens(code: str) -> str:
    return code.replace("```python", "").replace("```", "")


def salvage_prefix(raw_code: str, ep: str):
    """
    尝试从 raw 中恢复一个可编译且包含 entry point 的前缀。
    适合处理：
      def foo(...):
          ...
      ```
      explanation...
    或尾部自然语言解释污染。
    不会修复函数体内部语法错误。
    """
    candidates = []

    # 1. 原文
    candidates.append(raw_code)

    # 2. 去掉 markdown tokens
    candidates.append(strip_markdown_tokens(raw_code))

    # 3. 按行从后往前截断，找最短污染清除后的可编译前缀
    lines = raw_code.splitlines()
    for end in range(len(lines), 0, -1):
        prefix = "\n".join(lines[:end]).rstrip() + "\n"
        candidates.append(prefix)
        candidates.append(strip_markdown_tokens(prefix))

    for cand in candidates:
        if is_compilable(cand) and has_entry(cand, ep):
            return cand

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--sanitized", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    raw_path = Path(args.raw)
    sanitized_path = Path(args.sanitized)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    problems = get_human_eval_plus()
    raw = load_jsonl(raw_path)
    sanitized = load_jsonl(sanitized_path)

    patched_count = 0
    kept_sanitized_count = 0
    unrecovered_count = 0

    report = []

    with output_path.open("w", encoding="utf-8") as out:
        for task_id, problem in problems.items():
            ep = entry_point(problem)

            raw_obj = raw.get(task_id)
            san_obj = sanitized.get(task_id)

            if raw_obj is None:
                raise KeyError(f"Missing task in raw file: {task_id}")
            if san_obj is None:
                raise KeyError(f"Missing task in sanitized file: {task_id}")

            raw_code = code_of(raw_obj, problem)
            san_code = code_of(san_obj, problem)

            san_ok = is_compilable(san_code) and has_entry(san_code, ep)

            if san_ok:
                final_obj = san_obj
                kept_sanitized_count += 1
            else:
                recovered = salvage_prefix(raw_code, ep)

                if recovered is not None:
                    final_obj = {
                        "task_id": task_id,
                        "solution": recovered,
                    }
                    patched_count += 1
                    report.append(
                        {
                            "task_id": task_id,
                            "entry_point": ep,
                            "reason": "sanitized_missing_or_bad_but_raw_recovered",
                            "sanitized_head": san_code[:300],
                            "recovered_head": recovered[:300],
                        }
                    )
                else:
                    # 保留 sanitized。此时错误更可能是模型没有生成可恢复的目标函数。
                    final_obj = san_obj
                    unrecovered_count += 1
                    report.append(
                        {
                            "task_id": task_id,
                            "entry_point": ep,
                            "reason": "unrecovered_keep_sanitized",
                            "raw_head": raw_code[:300],
                            "sanitized_head": san_code[:300],
                        }
                    )

            out.write(json.dumps(final_obj, ensure_ascii=False) + "\n")

    report_path = output_path.with_suffix(".patch_report.json")
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Wrote patched file: {output_path}")
    print(f"Wrote patch report: {report_path}")
    print(f"kept_sanitized: {kept_sanitized_count}")
    print(f"patched_from_raw: {patched_count}")
    print(f"unrecovered_keep_sanitized: {unrecovered_count}")


if __name__ == "__main__":
    main()