# merge_mbpp_shards.py
import argparse
import json
from pathlib import Path


def load_mbpp_task_order(limit=None):
    from evalplus.data import get_mbpp_plus

    problems = get_mbpp_plus()
    task_ids = list(problems.keys())

    if limit is not None:
        task_ids = task_ids[:limit]

    return task_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_by_task_id = {}
    duplicate_task_ids = []
    total_rows = 0

    for shard_path_str in args.shards:
        shard_path = Path(shard_path_str)

        if not shard_path.exists():
            raise FileNotFoundError(f"Shard file not found: {shard_path}")

        with shard_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                obj = json.loads(line)
                total_rows += 1

                task_id = obj["task_id"]

                if task_id in rows_by_task_id:
                    duplicate_task_ids.append(task_id)

                rows_by_task_id[task_id] = obj

    task_order = load_mbpp_task_order(limit=args.limit)
    task_order_set = set(task_order)

    missing_task_ids = [
        task_id
        for task_id in task_order
        if task_id not in rows_by_task_id
    ]

    extra_task_ids = [
        task_id
        for task_id in rows_by_task_id
        if task_id not in task_order_set
    ]

    if duplicate_task_ids:
        print(f"Warning: duplicate task_ids found: {len(duplicate_task_ids)}")
        print("First duplicate task_ids:", duplicate_task_ids[:20])

    if missing_task_ids:
        print(f"Warning: missing task_ids: {len(missing_task_ids)}")
        print("First missing task_ids:", missing_task_ids[:20])

    if extra_task_ids:
        print(f"Warning: extra task_ids not in MBPP order: {len(extra_task_ids)}")
        print("First extra task_ids:", extra_task_ids[:20])

    if args.strict and (duplicate_task_ids or missing_task_ids or extra_task_ids):
        raise RuntimeError(
            "Shard merge failed strict checks: "
            f"duplicates={len(duplicate_task_ids)}, "
            f"missing={len(missing_task_ids)}, "
            f"extra={len(extra_task_ids)}"
        )

    merged_rows = [
        rows_by_task_id[task_id]
        for task_id in task_order
        if task_id in rows_by_task_id
    ]

    with output_path.open("w", encoding="utf-8") as f:
        for row in merged_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("=" * 80)
    print("Merged MBPP shards")
    print(f"Shard files: {len(args.shards)}")
    print(f"Input rows: {total_rows}")
    print(f"Unique task_ids: {len(rows_by_task_id)}")
    print(f"Output rows: {len(merged_rows)}")
    print(f"Output: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
