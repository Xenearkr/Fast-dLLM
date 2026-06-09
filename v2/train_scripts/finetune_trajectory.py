#!/usr/bin/env python
# coding=utf-8
"""
Trajectory 监督微调入口：与 finetune.py 相同流程（trajectory + block_mask_num + 轨迹 mask）。

请先运行:
  python utils/convert_trajectory_jsonl.py

再启动本脚本或 finetune_trajectory.sh。
"""

import sys
from pathlib import Path

_V2_ROOT = Path(__file__).resolve().parents[1]
_TRAIN_SCRIPTS = Path(__file__).resolve().parent
for p in (_V2_ROOT, _TRAIN_SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from finetune import main  # noqa: E402


if __name__ == "__main__":
    main()
