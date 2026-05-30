# monitor_humaneval_progress.py
import argparse
import json
import time
from pathlib import Path

from tqdm import tqdm


TERMINAL_STATUSES = {"completed", "failed"}


def read_progress(path: Path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None
    except Exception:
        return None


def aggregate(progress_files):
    rows = []
    for path in progress_files:
        obj = read_progress(path)
        if obj is not None:
            rows.append(obj)

    total_samples = sum(int(x.get("total_samples", 0)) for x in rows)
    done_samples = sum(int(x.get("done_samples", 0)) for x in rows)
    generated_tokens = sum(int(x.get("generated_tokens", 0)) for x in rows)
    allocated_new_tokens = sum(int(x.get("allocated_new_tokens", 0)) for x in rows)
    generation_time_sec = sum(float(x.get("generation_time_sec", 0.0)) for x in rows)

    start_times = [
        float(x.get("start_time"))
        for x in rows
        if x.get("start_time") is not None
    ]
    last_update_times = [
        float(x.get("last_update_time"))
        for x in rows
        if x.get("last_update_time") is not None
    ]

    wall_time_sec = 0.0
    if start_times and last_update_times:
        wall_time_sec = max(last_update_times) - min(start_times)
        wall_time_sec = max(0.0, wall_time_sec)

    statuses = [str(x.get("status", "unknown")) for x in rows]
    all_exist = len(rows) == len(progress_files)
    all_terminal = all_exist and all(s in TERMINAL_STATUSES for s in statuses)

    failed_shards = sum(1 for s in statuses if s == "failed")
    completed_shards = sum(1 for s in statuses if s == "completed")
    loading_shards = sum(1 for s in statuses if s == "loading_model")
    running_shards = sum(1 for s in statuses if s == "running")
    initializing_shards = sum(1 for s in statuses if s == "initializing")

    gen_tpf = generation_time_sec / done_samples if done_samples > 0 else 0.0
    gen_tps = generated_tokens / generation_time_sec if generation_time_sec > 0 else 0.0

    wall_tpf = wall_time_sec / done_samples if done_samples > 0 else 0.0
    wall_tps = generated_tokens / wall_time_sec if wall_time_sec > 0 else 0.0

    return {
        "rows": rows,
        "total_samples": total_samples,
        "done_samples": done_samples,
        "generated_tokens": generated_tokens,
        "allocated_new_tokens": allocated_new_tokens,
        "generation_time_sec": generation_time_sec,
        "wall_time_sec": wall_time_sec,
        "all_exist": all_exist,
        "all_terminal": all_terminal,
        "completed_shards": completed_shards,
        "failed_shards": failed_shards,
        "initializing_shards": initializing_shards,
        "loading_shards": loading_shards,
        "running_shards": running_shards,
        "gen_tpf": gen_tpf,
        "gen_tps": gen_tps,
        "wall_tpf": wall_tpf,
        "wall_tps": wall_tps,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--progress_files", nargs="+", required=True)
    parser.add_argument("--refresh", type=float, default=2.0)
    parser.add_argument("--desc", type=str, default="HumanEval total progress")
    args = parser.parse_args()

    progress_files = [Path(x) for x in args.progress_files]

    bar = tqdm(
        total=0,
        initial=0,
        desc=args.desc,
        unit="sample",
        dynamic_ncols=True,
        smoothing=0.1,
    )

    try:
        while True:
            agg = aggregate(progress_files)

            total = agg["total_samples"]
            done = agg["done_samples"]

            if total > 0 and bar.total != total:
                bar.total = total

            bar.n = min(done, total) if total > 0 else done

            bar.set_postfix(
                {
                    "tokens": agg["generated_tokens"],
                    "TPF": f"{agg['wall_tpf']:.2f}s",
                    "TPS": f"{agg['wall_tps']:.2f}",
                    "gen_TPS": f"{agg['gen_tps']:.2f}",
                    "init": agg["initializing_shards"],
                    "load": agg["loading_shards"],
                    "run": agg["running_shards"],
                    "done": agg["completed_shards"],
                    "fail": agg["failed_shards"],
                },
                refresh=False,
            )
            bar.refresh()

            if agg["all_terminal"]:
                break

            time.sleep(args.refresh)
    finally:
        bar.close()

    agg = aggregate(progress_files)
    print("=" * 80)
    print("Generation progress monitor finished")
    print(f"Samples: {agg['done_samples']}/{agg['total_samples']}")
    print(f"Generated tokens: {agg['generated_tokens']}")
    print(f"Allocated new token slots: {agg['allocated_new_tokens']}")
    print(f"Wall time: {agg['wall_time_sec']:.2f}s")
    print(f"Generation time summed over shards: {agg['generation_time_sec']:.2f}s")
    print(f"Wall TPF: {agg['wall_tpf']:.4f}s/sample")
    print(f"Wall TPS: {agg['wall_tps']:.4f} tokens/s")
    print(f"GPU-summed generation TPF: {agg['gen_tpf']:.4f}s/sample")
    print(f"GPU-summed generation TPS: {agg['gen_tps']:.4f} tokens/s")
    print(f"Completed shards: {agg['completed_shards']}")
    print(f"Failed shards: {agg['failed_shards']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
