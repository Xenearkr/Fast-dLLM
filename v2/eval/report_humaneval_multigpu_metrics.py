# report_humaneval_multigpu_metrics.py
import argparse
import glob
import json
import re
from pathlib import Path


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def aggregate_progress(progress_files):
    rows = []
    for path_str in progress_files:
        obj = read_json(Path(path_str))
        if obj is not None:
            rows.append(obj)

    total_samples = sum(int(x.get("total_samples", 0)) for x in rows)
    done_samples = sum(int(x.get("done_samples", 0)) for x in rows)
    generated_tokens = sum(int(x.get("generated_tokens", 0)) for x in rows)
    allocated_new_tokens = sum(int(x.get("allocated_new_tokens", 0)) for x in rows)
    generation_time_sec = sum(float(x.get("generation_time_sec", 0.0)) for x in rows)
    
    # -------------------------------------------------------------
    # 修改点 1：累加所有 GPU 分片的总 forward 步数
    # -------------------------------------------------------------
    total_forward_steps = sum(int(x.get("total_forward_steps", 0)) for x in rows)

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

    completed_shards = sum(1 for x in rows if x.get("status") == "completed")
    failed_shards = sum(1 for x in rows if x.get("status") == "failed")

    return {
        "num_progress_files": len(progress_files),
        "num_readable_progress_files": len(rows),
        "completed_shards": completed_shards,
        "failed_shards": failed_shards,
        "total_samples": total_samples,
        "done_samples": done_samples,
        "generated_tokens": generated_tokens,
        "total_forward_steps": total_forward_steps, # 导出总步数
        # 计算全局多卡 TPF 指标
        "token_per_forward": generated_tokens / total_forward_steps if total_forward_steps else None,
        "allocated_new_tokens": allocated_new_tokens,
        "generation_time_sec_sum": generation_time_sec,
        "wall_time_sec": wall_time_sec,
        "wall_tpf_sec_per_sample": wall_time_sec / done_samples if done_samples else None,
        "wall_tps_tokens_per_sec": generated_tokens / wall_time_sec if wall_time_sec else None,
        "gpu_summed_tpf_sec_per_sample": generation_time_sec / done_samples if done_samples else None,
        "gpu_summed_tps_tokens_per_sec": generated_tokens / generation_time_sec if generation_time_sec else None,
    }


def parse_pass_metrics_from_text(text: str):
    """
    尽量从 evalplus.evaluate 的 stdout 里抽取 pass@k。
    不强依赖 EvalPlus 的具体版本输出格式。
    """
    metrics = {}
    current_section = None
    observed_pass_values = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        lower = line.lower()

        if not line:
            continue

        if "base + extra" in lower or "base+extra" in lower or "plus" in lower:
            current_section = "plus"
        elif lower == "base" or lower.startswith("base "):
            current_section = "base"

        for m in re.finditer(r"['\"]?(pass@\d+)['\"]?\s*[:=]\s*([0-9]*\.?[0-9]+)", line):
            key = m.group(1)
            value = float(m.group(2))
            observed_pass_values.append((current_section, key, value))

            if current_section:
                metrics[f"{current_section}_{key}"] = value
            else:
                metrics.setdefault(key, value)

        for m in re.finditer(r"\b(pass@\d+)\b\s+([0-9]*\.?[0-9]+)", line):
            key = m.group(1)
            value = float(m.group(2))
            observed_pass_values.append((current_section, key, value))
            if current_section:
                metrics[f"{current_section}_{key}"] = value
            else:
                metrics.setdefault(key, value)

    pass1_values = [v for section, key, v in observed_pass_values if key == "pass@1"]
    if "base_pass@1" not in metrics and pass1_values:
        metrics["base_pass@1"] = pass1_values[0]
    if "plus_pass@1" not in metrics and len(pass1_values) >= 2:
        metrics["plus_pass@1"] = pass1_values[1]

    return metrics


def parse_eval_result_files_from_samples(samples_path: Path):
    """
    尽量从 EvalPlus 结果文件中抽取 base/plus 正确率。
    不同 EvalPlus 版本的文件结构可能不同，所以这里只做保守解析。
    """
    candidates = []
    patterns = [
        str(samples_path.with_suffix("")) + "*eval_results*.json",
        str(samples_path.with_suffix("")) + "*eval_results*.jsonl",
        str(samples_path.parent / (samples_path.stem + "*eval_results*.json")),
        str(samples_path.parent / (samples_path.stem + "*eval_results*.jsonl")),
    ]

    for pat in patterns:
        candidates.extend(glob.glob(pat))

    candidates = sorted(set(candidates))
    parsed = {}

    for path_str in candidates:
        path = Path(path_str)
        try:
            if path.suffix == ".jsonl":
                rows = []
                with path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            rows.append(json.loads(line))

                base_statuses = []
                plus_statuses = []
                for row in rows:
                    if "base_status" in row:
                        base_statuses.append(str(row["base_status"]).lower())
                    if "plus_status" in row:
                        plus_statuses.append(str(row["plus_status"]).lower())
                    if "base" in row and isinstance(row["base"], bool):
                        base_statuses.append("pass" if row["base"] else "fail")
                    if "plus" in row and isinstance(row["plus"], bool):
                        plus_statuses.append("pass" if row["plus"] else "fail")

                if base_statuses:
                    parsed["base_pass@1_from_results"] = sum(x == "pass" for x in base_statuses) / len(base_statuses)
                if plus_statuses:
                    parsed["plus_pass@1_from_results"] = sum(x == "pass" for x in plus_statuses) / len(plus_statuses)

            elif path.suffix == ".json":
                obj = json.loads(path.read_text(encoding="utf-8"))
                text = json.dumps(obj, ensure_ascii=False)
                parsed.update(parse_pass_metrics_from_text(text))
        except Exception:
            continue

    if candidates:
        parsed["eval_result_files"] = candidates

    return parsed


def fmt_float(value, digits=4):
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def maybe_pct(value):
    if value is None:
        return "N/A"
    if 0.0 <= value <= 1.0:
        return f"{100.0 * value:.2f}%"
    return f"{value:.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--progress_files", nargs="+", required=True)
    parser.add_argument("--eval_log", type=str, default=None)
    parser.add_argument("--samples", type=str, default=None)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    progress_metrics = aggregate_progress(args.progress_files)

    eval_metrics = {}
    if args.eval_log is not None and Path(args.eval_log).exists():
        text = Path(args.eval_log).read_text(encoding="utf-8", errors="ignore")
        eval_metrics.update(parse_pass_metrics_from_text(text))

    if args.samples is not None:
        eval_metrics.update(parse_eval_result_files_from_samples(Path(args.samples)))

    report = {
        "progress_metrics": progress_metrics,
        "eval_metrics": eval_metrics,
    }

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 80)
    print("HUMANEVAL MULTI-GPU FINAL REPORT")
    print("=" * 80)
    print("Accuracy")
    print(f"  base pass@1: {maybe_pct(eval_metrics.get('base_pass@1') or eval_metrics.get('base_pass@1_from_results'))}")
    print(f"  plus pass@1: {maybe_pct(eval_metrics.get('plus_pass@1') or eval_metrics.get('plus_pass@1_from_results'))}")

    other_keys = sorted(
        k for k in eval_metrics
        if k not in {
            "base_pass@1",
            "plus_pass@1",
            "base_pass@1_from_results",
            "plus_pass@1_from_results",
            "eval_result_files",
        }
    )
    if other_keys:
        print("  other parsed EvalPlus metrics:")
        for key in other_keys:
            print(f"    {key}: {eval_metrics[key]}")

    print()
    print("Generation throughput")
    print(f"  shards completed/failed: {progress_metrics['completed_shards']}/{progress_metrics['failed_shards']}")
    print(f"  samples: {progress_metrics['done_samples']}/{progress_metrics['total_samples']}")
    print(f"  generated tokens: {progress_metrics['generated_tokens']}")
    
    # -------------------------------------------------------------
    # 修改点 2：在屏幕输出中打印合并后的真实总步数与最终的 token_per_forward
    # -------------------------------------------------------------
    print(f"  total forward steps: {progress_metrics['total_forward_steps']}")
    print(f"  token per forward (TPF): {fmt_float(progress_metrics['token_per_forward'], 4)} tokens/forward")
    
    print(f"  allocated new token slots: {progress_metrics['allocated_new_tokens']}")
    print(f"  wall time: {fmt_float(progress_metrics['wall_time_sec'], 2)} s")
    print(f"  summed GPU generation time: {fmt_float(progress_metrics['generation_time_sec_sum'], 2)} s")
    print(f"  wall Sec/Sample: {fmt_float(progress_metrics['wall_tpf_sec_per_sample'], 4)} s/sample")
    print(f"  wall TPS: {fmt_float(progress_metrics['wall_tps_tokens_per_sec'], 4)} tokens/s")
    print(f"  GPU-summed Sec/Sample: {fmt_float(progress_metrics['gpu_summed_tpf_sec_per_sample'], 4)} s/sample")
    print(f"  GPU-summed TPS: {fmt_float(progress_metrics['gpu_summed_tps_tokens_per_sec'], 4)} tokens/s")

    if "eval_result_files" in eval_metrics:
        print()
        print("EvalPlus result files:")
        for path in eval_metrics["eval_result_files"]:
            print(f"  {path}")

    if args.output_json:
        print()
        print(f"Saved metrics report: {args.output_json}")

    print("=" * 80)


if __name__ == "__main__":
    main()