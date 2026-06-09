"""Generation speed metrics for Fast-dLLM eval (TPS, TPF, forward calls)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def sync_cuda() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def attach_forward_counter(model) -> Tuple[Dict[str, int], Any]:
    counter = {"count": 0}
    original_forward = model.forward

    def counted_forward(*args, **kwargs):
        counter["count"] += 1
        return original_forward(*args, **kwargs)

    model.forward = counted_forward
    return counter, original_forward


def restore_forward(model, original_forward) -> None:
    model.forward = original_forward


def reset_forward_counter(counter: Dict[str, int]) -> None:
    counter["count"] = 0


def validate_warmup_new_tokens(warmup_new_tokens: int, bd_size: int) -> None:
    if warmup_new_tokens <= 0:
        raise ValueError(f"warmup_new_tokens must be positive, got {warmup_new_tokens}")
    if warmup_new_tokens % bd_size != 0:
        raise ValueError(
            f"warmup_new_tokens must be divisible by bd_size. "
            f"Got warmup_new_tokens={warmup_new_tokens}, bd_size={bd_size}"
        )


def count_new_tokens(full_ids, prompt_len: int, mask_id: int) -> int:
    new_ids = full_ids[prompt_len:]
    return int((new_ids != mask_id).sum().item())


def build_speed_report(
    *,
    method: str,
    model_path: str,
    dataset: str,
    num_samples: int,
    new_tokens: int,
    generate_time_s: float,
    forward_calls: int,
    batch_size: int = 1,
    max_new_tokens: int = 512,
    bd_size: int = 32,
    small_block_size: int = 8,
    threshold: float = 1.0,
    use_block_cache: bool = False,
    warmup_steps: int = 0,
    warmup_new_tokens: int = 0,
) -> Dict[str, Any]:
    tps = new_tokens / generate_time_s if generate_time_s > 0 else 0.0
    tpf = new_tokens / forward_calls if forward_calls > 0 else 0.0
    ms_per_token = 1000.0 * generate_time_s / max(new_tokens, 1)
    ms_per_forward = 1000.0 * generate_time_s / max(forward_calls, 1)

    return {
        "method": method,
        "model_path": model_path,
        "dataset": dataset,
        "num_samples": num_samples,
        "new_tokens": new_tokens,
        "generate_time_s": round(generate_time_s, 4),
        "forward_calls": forward_calls,
        "tps": round(tps, 2),
        "tpf": round(tpf, 2),
        "ms_per_token": round(ms_per_token, 2),
        "ms_per_forward": round(ms_per_forward, 2),
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "bd_size": bd_size,
        "small_block_size": small_block_size,
        "threshold": threshold,
        "use_block_cache": use_block_cache,
        "warmup_steps": warmup_steps,
        "warmup_new_tokens": warmup_new_tokens,
    }


def save_speed_report(path: Path, report: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def load_speed_report(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def print_speed_report(report: Dict[str, Any]) -> None:
    print("=" * 72)
    print(f"Generation speed [{report.get('method', '?')}] dataset={report.get('dataset', '?')}")
    print("=" * 72)
    print(f"  samples:        {report.get('num_samples', 0)}")
    print(f"  new_tokens:     {report.get('new_tokens', 0)}")
    print(f"  generate_time:  {report.get('generate_time_s', 0):.4f} s")
    print(f"  forward_calls:  {report.get('forward_calls', 0)}")
    print(f"  TPS:            {report.get('tps', 0):.2f} tokens/s")
    print(f"  TPF:            {report.get('tpf', 0):.2f} tokens/forward")
    print(f"  ms/token:       {report.get('ms_per_token', 0):.2f}")
    print(f"  ms/forward:     {report.get('ms_per_forward', 0):.2f}")
    warmup_steps = report.get("warmup_steps", 0)
    if warmup_steps:
        print(
            f"  warmup:         {warmup_steps} step(s), "
            f"{report.get('warmup_new_tokens', 0)} tokens/step (excluded from metrics)"
        )
    print("=" * 72)


def print_comparison_table(
    reports: List[Dict[str, Any]],
    *,
    title: str = "Generation speed comparison",
) -> None:
    if not reports:
        print("No speed metrics available for comparison.")
        return

    headers = [
        "Method",
        "Samples",
        "NewTokens",
        "Time(s)",
        "TPS",
        "Forwards",
        "TPF",
    ]

    rows: List[List[str]] = []
    for report in reports:
        rows.append(
            [
                str(report.get("method", "?")),
                str(report.get("num_samples", 0)),
                str(report.get("new_tokens", 0)),
                f"{report.get('generate_time_s', 0):.2f}",
                f"{report.get('tps', 0):.2f}",
                str(report.get("forward_calls", 0)),
                f"{report.get('tpf', 0):.2f}",
            ]
        )

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cells: List[str]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    sep = "-+-".join("-" * w for w in widths)

    print()
    print("=" * (len(sep) + 4))
    print(title)
    print("=" * (len(sep) + 4))
    print(fmt_row(headers))
    print(sep)
    for row in rows:
        print(fmt_row(row))
    print("=" * (len(sep) + 4))
    print("TPS = new_tokens / wall_time;  TPF = new_tokens / forward_calls")
    print()
