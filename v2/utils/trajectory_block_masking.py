"""基于 trajectory 的 block mask 索引（与 Fast_dLLM_v2_7B.modeling 中逻辑一致，供单测）。"""

from __future__ import annotations

import torch

from utils.block_mask_num_schedule import resolve_block_mask_num


def block_trajectory_end(block_traj) -> int:
    """块内 trajectory 最大值，用作 trajectory_end。"""
    if hasattr(block_traj, "max"):
        return int(block_traj.max().item())
    return int(max(block_traj))


def build_trajectory_mask_indices(
    labels: torch.LongTensor,
    trajectory: torch.LongTensor,
    block_mask_num,
    bd_size: int,
) -> torch.BoolTensor:
    if labels.shape != trajectory.shape:
        raise ValueError(
            f"labels and trajectory must have the same shape, got "
            f"{tuple(labels.shape)} vs {tuple(trajectory.shape)}"
        )
    if labels.shape[1] % bd_size != 0:
        raise ValueError(
            f"sequence length {labels.shape[1]} must be divisible by bd_size={bd_size}"
        )

    k = resolve_block_mask_num(block_mask_num, bd_size)
    block_labels = labels.reshape(labels.shape[0] * labels.shape[1] // bd_size, bd_size)
    block_traj = trajectory.reshape(block_labels.shape)

    device = labels.device
    trajectory_end = block_traj.max(dim=1).values
    trainable_in_block = block_labels != -100
    has_trainable = trainable_in_block.any(dim=1)

    # 含可训练 token 的 block：块内 max(trajectory) 必须 != 0（-1 走 fallback，>0 走阈值逻辑）
    if has_trainable.any():
        invalid = has_trainable & (trajectory_end == 0)
        if invalid.any():
            bad_ends = trajectory_end[invalid].detach().cpu().tolist()
            bad_idx = invalid.nonzero(as_tuple=True)[0].detach().cpu().tolist()
            raise AssertionError(
                "Each block with trainable tokens (labels != -100) must have "
                f"max trajectory != 0; bad block indices {bad_idx}, "
                f"trajectory_end values {bad_ends}"
            )

    threshold = trajectory_end.unsqueeze(1) - k
    traj_mask = (block_traj > threshold) & trainable_in_block

    col_idx = torch.arange(bd_size, device=device)
    last_k = col_idx >= (bd_size - k)
    fallback_mask = last_k.unsqueeze(0) & trainable_in_block

    use_fallback = (trajectory_end == -1).unsqueeze(1)
    return torch.where(use_fallback, fallback_mask, traj_mask)
