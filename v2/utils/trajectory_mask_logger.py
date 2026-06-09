"""
训练时记录每一轮 forward 的带 mask 输入（noisy_input_ids），不含 clear / complementary 拼接数据。

输出到 trajectory_now.log；仅 local_rank=0 写入。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, List, Optional

import torch
from transformers import TrainerCallback

from utils.block_mask_num_schedule import (
    BLOCK_MASK_NUM_FIELD,
    _MODEL_BLOCK_MASK_SCHEDULE_ATTR,
    inject_live_block_mask_num,
    resolve_block_mask_num,
)
from utils.trajectory_block_masking import (
    block_trajectory_end,
    build_trajectory_mask_indices,
)
from utils.truncate_tokenized_dataset import (
    TRAJECTORY_FIELD,
    TRAJECTORY_FULL_FIELD,
    _TRAJECTORY_PAD_VALUE,
)

_MODEL_MASK_LOGGING_PATCHED_ATTR = "_fast_dllm_mask_logging_patched"
DEFAULT_TRAJECTORY_NOW_LOG = "trajectory_now.log"


def _is_main_process() -> bool:
    return int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))) == 0


def format_full_trajectory(
    trajectory: Optional[List[int]],
    *,
    bd_size: int,
) -> str:
    """格式化数据集中该样本的完整 trajectory（截断/pad 之前）。"""
    if trajectory is None:
        return "trajectory_full: (missing)"
    if len(trajectory) == 0:
        return "trajectory_full (len=0): (empty)"

    lines = [f"trajectory_full (len={len(trajectory)}):"]
    if bd_size > 0:
        num_blocks = (len(trajectory) + bd_size - 1) // bd_size
        for block_idx in range(num_blocks):
            start = block_idx * bd_size
            end = min(start + bd_size, len(trajectory))
            block_traj = trajectory[start:end]
            traj_end = block_trajectory_end(block_traj)
            lines.append(f"  block {block_idx}: end={traj_end}")
            lines.append(f"    traj={block_traj}")
    else:
        lines.append(f"  values={trajectory}")
    return "\n".join(lines)


def format_truncated_trajectory(
    trajectory: List[int],
    labels: List[int],
    *,
    bd_size: int,
    mask_indices: Optional[List[bool]] = None,
) -> str:
    """格式化截断后与 batch 等长的 trajectory（按 block 展示）。"""
    seq_len = len(trajectory)
    if seq_len != len(labels):
        return (
            f"trajectory_truncated: length mismatch "
            f"traj={len(trajectory)} labels={len(labels)}"
        )

    lines = [f"trajectory_truncated (len={seq_len}):"]
    num_blocks = seq_len // bd_size if bd_size > 0 else 0
    if num_blocks == 0:
        lines.append(f"  values={trajectory}")
        return "\n".join(lines)

    for block_idx in range(num_blocks):
        start = block_idx * bd_size
        end = start + bd_size
        block_traj = trajectory[start:end]
        block_labels = labels[start:end]
        block_mask = (
            mask_indices[start:end]
            if mask_indices is not None and len(mask_indices) == seq_len
            else None
        )
        train_cols = [col for col, lb in enumerate(block_labels) if lb != -100]
        masked_cols = (
            [col for col in train_cols if block_mask[col]]
            if block_mask is not None
            else []
        )
        traj_end = block_trajectory_end(block_traj)
        lines.append(
            f"  block {block_idx}: end={traj_end} "
            f"train_cols={train_cols} masked_cols={masked_cols}"
        )
        lines.append(f"    traj={block_traj}")

    pad_count = sum(1 for v in trajectory if v == _TRAJECTORY_PAD_VALUE)
    if pad_count:
        lines.append(f"  trajectory_pad_count={pad_count} (pad_value={_TRAJECTORY_PAD_VALUE})")
    return "\n".join(lines)


def format_masked_sequence(
    tokenizer,
    token_ids: List[int],
    mask_token_id: int,
) -> str:
    """逐 token 解码；mask 位置显示为 <mask>。"""
    parts: List[str] = []
    for tid in token_ids:
        if int(tid) == mask_token_id:
            parts.append("<mask>")
        else:
            parts.append(tokenizer.decode([int(tid)], skip_special_tokens=False))
    return "".join(parts)


@dataclass
class TrajectoryMaskLoggingState:
    log_path: str = DEFAULT_TRAJECTORY_NOW_LOG
    global_step: int = 0
    forward_count: int = 0
    _file_initialized: bool = field(default=False, repr=False)

    def reset_log(self) -> None:
        if not _is_main_process():
            return
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write(
                "# trajectory masked training data "
                "(noisy_input_ids + trajectory_full + truncated trajectory)\n"
            )
        self._file_initialized = True

    def append_masked_batch(
        self,
        *,
        tokenizer,
        noisy_input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        trajectory: torch.LongTensor,
        trajectory_full: Optional[List[Optional[List[int]]]],
        mask_indices: torch.BoolTensor,
        mask_token_id: int,
        block_mask_num,
        bd_size: int,
    ) -> None:
        if not _is_main_process():
            return
        if not self._file_initialized:
            self.reset_log()

        k = resolve_block_mask_num(block_mask_num, bd_size)
        if hasattr(block_mask_num, "detach"):
            bmn_val = float(block_mask_num.detach().float().mean().item())
        else:
            bmn_val = float(block_mask_num)

        self.forward_count += 1
        lines = [
            "",
            f"========== step={self.global_step} forward={self.forward_count} "
            f"block_mask_num={bmn_val:.4f} (k={k}) ==========",
        ]

        batch_size = noisy_input_ids.shape[0]
        for sample_idx in range(batch_size):
            ids = noisy_input_ids[sample_idx].detach().cpu().tolist()
            sample_labels = labels[sample_idx].detach().cpu()
            mask_count = int((noisy_input_ids[sample_idx] == mask_token_id).sum().item())
            trainable_count = int((sample_labels != -100).sum().item())
            text = format_masked_sequence(tokenizer, ids, mask_token_id)
            traj_row = trajectory[sample_idx].detach().cpu().tolist()
            masked_row = mask_indices[sample_idx].detach().cpu().tolist()
            full_row = None
            if trajectory_full is not None and sample_idx < len(trajectory_full):
                full_row = trajectory_full[sample_idx]
            lines.append(
                f"[sample {sample_idx}] mask_count={mask_count} "
                f"trainable_tokens={trainable_count}"
            )
            lines.append(format_full_trajectory(full_row, bd_size=bd_size))
            lines.append(format_truncated_trajectory(
                traj_row,
                sample_labels.tolist(),
                bd_size=bd_size,
                mask_indices=masked_row,
            ))
            lines.append(text)

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


class TrajectoryMaskLogCallback(TrainerCallback):
    """同步 Trainer global_step 到 TrajectoryMaskLoggingState。"""

    def __init__(self, logging_state: TrajectoryMaskLoggingState):
        self.logging_state = logging_state

    def on_train_begin(self, args, state, control, **kwargs):
        self.logging_state.global_step = state.global_step
        self.logging_state.forward_count = 0
        self.logging_state.reset_log()

    def on_step_begin(self, args, state, control, **kwargs):
        self.logging_state.global_step = state.global_step


def _resolve_forward_target(model: Any):
    forward = model.forward
    while hasattr(forward, "__wrapped__"):
        forward = forward.__wrapped__
    return forward


def wrap_model_forward_with_mask_logging(
    model: Any,
    *,
    tokenizer,
    mask_id: int,
    bd_size: int,
    log_path: str = DEFAULT_TRAJECTORY_NOW_LOG,
) -> TrajectoryMaskLoggingState:
    """
    包装 model.forward：在训练且含 trajectory 时，复现主路径 mask 并写入 log。
    仅记录 noisy_input_ids，不包含 clear / complementary 拼接。
    """
    if getattr(model, _MODEL_MASK_LOGGING_PATCHED_ATTR, False):
        state = getattr(model, "_fast_dllm_mask_logging_state", None)
        if state is None:
            state = TrajectoryMaskLoggingState(log_path=log_path)
            setattr(model, "_fast_dllm_mask_logging_state", state)
        return state

    state = TrajectoryMaskLoggingState(log_path=log_path)
    setattr(model, "_fast_dllm_mask_logging_state", state)

    original_forward = _resolve_forward_target(model)
    outer_forward = model.forward

    def forward_with_mask_logging(*args, **kwargs):
        if model.training:
            trajectory = kwargs.get(TRAJECTORY_FIELD)
            labels = kwargs.get("labels")
            input_ids = kwargs.get("input_ids")
            schedule = getattr(model, _MODEL_BLOCK_MASK_SCHEDULE_ATTR, None)
            if schedule is not None:
                inject_live_block_mask_num(kwargs, schedule)
            block_mask_num = kwargs.get(BLOCK_MASK_NUM_FIELD)
            mask_id_kw = kwargs.get("mask_id", mask_id)

            if (
                trajectory is not None
                and labels is not None
                and input_ids is not None
                and block_mask_num is not None
            ):
                labels_for_mask = labels.clone()
                block_flat = input_ids.reshape(
                    input_ids.shape[0] * input_ids.shape[1] // bd_size,
                    bd_size,
                )
                mask_indices = build_trajectory_mask_indices(
                    labels_for_mask,
                    trajectory,
                    block_mask_num,
                    bd_size,
                )
                x_t = torch.where(mask_indices, mask_id_kw, block_flat).reshape(
                    labels.shape
                )
                noisy_input_ids = input_ids.clone()
                noisy_input_ids[labels != -100] = x_t[labels != -100]

                state.append_masked_batch(
                    tokenizer=tokenizer,
                    noisy_input_ids=noisy_input_ids,
                    labels=labels,
                    trajectory=trajectory,
                    trajectory_full=kwargs.get(TRAJECTORY_FULL_FIELD),
                    mask_indices=mask_indices.reshape(labels.shape),
                    mask_token_id=int(mask_id_kw),
                    block_mask_num=block_mask_num,
                    bd_size=bd_size,
                )

        return outer_forward(*args, **kwargs)

    forward_with_mask_logging.__name__ = getattr(
        outer_forward, "__name__", "forward"
    )
    forward_with_mask_logging.__doc__ = getattr(outer_forward, "__doc__", None)
    if hasattr(outer_forward, "__signature__"):
        forward_with_mask_logging.__signature__ = outer_forward.__signature__
    forward_with_mask_logging.__wrapped__ = outer_forward

    model.forward = forward_with_mask_logging
    setattr(model, _MODEL_MASK_LOGGING_PATCHED_ATTR, True)
    return state
