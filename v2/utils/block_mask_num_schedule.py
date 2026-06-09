"""
训练过程中将 block_mask_num 从 start 线性增至 end（默认 bd_size）。

由 BlockMaskNumSchedule 维护当前值；BlockMaskNumCallback 在每个 step 更新；
DataCollator 将当前值写入 batch 的 block_mask_num 字段并传入 forward。

与 modeling.forward 中的 trajectory mask 配合：每 block 取块内 max(trajectory) 为 trajectory_end；
>0 时按 trajectory 阈值 mask，== -1 时 mask 块内末尾 block_mask_num 个可训练 token。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Optional

import torch
from transformers import TrainerCallback

logger = logging.getLogger(__name__)

BLOCK_MASK_NUM_FIELD = "block_mask_num"
_MODEL_LIVE_BLOCK_MASK_PATCHED_ATTR = "_fast_dllm_live_block_mask_patched"
_MODEL_BLOCK_MASK_SCHEDULE_ATTR = "_fast_dllm_block_mask_schedule"


@dataclass
class BlockMaskNumSchedule:
    """global_step 从 0 到 max_steps-1 时，current 从 start 均匀增至 end。"""

    start: float = 1.0
    end: float = 32.0
    max_steps: Optional[int] = None
    current: float = 1.0

    def value_at_step(self, global_step: int) -> float:
        if self.max_steps is None or self.max_steps <= 1:
            return float(self.end)
        step = max(0, min(global_step, self.max_steps - 1))
        progress = step / (self.max_steps - 1)
        return self.start + progress * (self.end - self.start)

    def update(self, global_step: int) -> float:
        self.current = self.value_at_step(global_step)
        return self.current


class BlockMaskNumCallback(TrainerCallback):
    def __init__(self, schedule: BlockMaskNumSchedule, log_every: int = 50):
        self.schedule = schedule
        self.log_every = log_every

    def on_train_begin(self, args, state, control, **kwargs):
        max_steps = state.max_steps
        if max_steps is None or max_steps <= 0:
            max_steps = getattr(args, "max_steps", None)
        if max_steps is None or max_steps <= 0:
            train_dataloader = kwargs.get("train_dataloader")
            if train_dataloader is not None:
                steps_per_epoch = len(train_dataloader)
                epochs = int(state.num_train_epochs)
                if steps_per_epoch > 0 and epochs > 0:
                    max_steps = steps_per_epoch * epochs
        if max_steps is not None and max_steps > 0:
            self.schedule.max_steps = int(max_steps)
        self.schedule.update(state.global_step)
        logger.info(
            "block_mask_num schedule: start=%s end=%s max_steps=%s initial=%s",
            self.schedule.start,
            self.schedule.end,
            self.schedule.max_steps,
            self.schedule.current,
        )

    def on_step_begin(self, args, state, control, **kwargs):
        value = self.schedule.update(state.global_step)
        if self.log_every > 0 and state.global_step % self.log_every == 0:
            logger.info(
                "step=%s block_mask_num=%.4f (schedule %s -> %s)",
                state.global_step,
                value,
                self.schedule.start,
                self.schedule.end,
            )


def resolve_block_mask_num(
    block_mask_num,
    bd_size: int,
    *,
    default: Optional[int] = None,
) -> int:
    """将 batch 中的 block_mask_num 解析为 [1, bd_size] 的整数。"""
    if block_mask_num is None:
        if default is None:
            raise ValueError("block_mask_num is required when schedule is disabled.")
        val = float(default)
    elif hasattr(block_mask_num, "detach"):
        val = float(block_mask_num.detach().float().mean().item())
    else:
        val = float(block_mask_num)

    if not math.isfinite(val):
        val = float(default if default is not None else bd_size)
    return max(1, min(int(round(val)), bd_size))


def inject_live_block_mask_num(
    kwargs: dict[str, Any],
    schedule: BlockMaskNumSchedule,
) -> None:
    """用主进程里实时更新的 schedule.current 覆盖 batch 中的 block_mask_num。"""
    input_ids = kwargs.get("input_ids")
    if input_ids is None:
        return
    value = float(schedule.current)
    batch_size = int(input_ids.shape[0])
    kwargs[BLOCK_MASK_NUM_FIELD] = torch.full(
        (batch_size,),
        value,
        dtype=torch.float32,
        device=input_ids.device,
    )


def _resolve_forward_target(model: Any):
    forward = model.forward
    while hasattr(forward, "__wrapped__"):
        forward = forward.__wrapped__
    return forward


def wrap_model_forward_with_live_block_mask_schedule(
    model: Any,
    schedule: Optional[BlockMaskNumSchedule],
) -> Any:
    """
    在 forward 时注入实时的 block_mask_num。

    DataCollator 在 dataloader worker 子进程中运行时会持有 schedule 的 pickle 快照，
    current 不会随训练步更新；因此在主进程 forward 前覆盖该字段。
    """
    setattr(model, _MODEL_BLOCK_MASK_SCHEDULE_ATTR, schedule)
    if schedule is None or getattr(model, _MODEL_LIVE_BLOCK_MASK_PATCHED_ATTR, False):
        return model

    outer_forward = model.forward

    def forward_with_live_block_mask_num(*args, **kwargs):
        if model.training:
            inject_live_block_mask_num(kwargs, schedule)
        return outer_forward(*args, **kwargs)

    forward_with_live_block_mask_num.__name__ = getattr(
        outer_forward, "__name__", "forward"
    )
    forward_with_live_block_mask_num.__doc__ = getattr(outer_forward, "__doc__", None)
    if hasattr(outer_forward, "__signature__"):
        forward_with_live_block_mask_num.__signature__ = outer_forward.__signature__
    forward_with_live_block_mask_num.__wrapped__ = outer_forward

    model.forward = forward_with_live_block_mask_num
    setattr(model, _MODEL_LIVE_BLOCK_MASK_PATCHED_ATTR, True)
    return model
