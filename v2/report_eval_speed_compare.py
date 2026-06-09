#!/usr/bin/env python3
"""Print a side-by-side TPS/TPF comparison table from metrics JSON files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

from generation_speed_metrics import load_speed_report, print_comparison_table


def parse_label_path(raw: str) -> Tuple[str, Path]:
    if "=" not in raw:
        path = Path(raw)
        return path.stem, path
    label, path_str = raw.split("=", 1)
    return label.strip(), Path(path_str.strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare generation speed metrics (TPS/TPF/forward calls)."
    )
    parser.add_argument(
        "--title",
        default="Generation speed comparison",
        help="Table title",
    )
    parser.add_argument(
        "metrics",
        nargs="+",
        help="Metrics files as LABEL=path/to/file.metrics.json",
    )
    args = parser.parse_args()

    reports: List[Dict] = []
    missing: List[str] = []

    for raw in args.metrics:
        label, path = parse_label_path(raw)
        report = load_speed_report(path)
        if report is None:
            missing.append(f"{label} ({path})")
            continue
        report.setdefault("method", label)
        reports.append(report)

    if missing:
        print("Warning: missing metrics files:")
        for item in missing:
            print(f"  - {item}")
        print("Re-run generation with FORCE_REGEN=1 to collect speed metrics.")
        print()

    print_comparison_table(reports, title=args.title)


if __name__ == "__main__":
    main()
