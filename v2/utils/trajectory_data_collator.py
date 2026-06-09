"""
Fast-dLLM finetune：将 trajectory / block_mask_num 与 input_ids / labels 一并 collate 进 batch。

配合 ensure_model_forward_accepts_training_batch_keys，使 HuggingFace Trainer 保留这些列并传入 forward。
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, List, Optional

import torch

from utils.block_mask_num_schedule import BLOCK_MASK_NUM_FIELD, BlockMaskNumSchedule
from utils.truncate_tokenized_dataset import (
    TRAJECTORY_FIELD,
    TRAJECTORY_FULL_FIELD,
    _TRAJECTORY_PAD_VALUE,
)

_MODEL_TRAINING_FORWARD_PATCHED_ATTR = "_fast_dllm_training_forward_patched"

TRAINING_FORWARD_BATCH_KEYS = (
    TRAJECTORY_FIELD,
    TRAJECTORY_FULL_FIELD,
    BLOCK_MASK_NUM_FIELD,
)


@dataclass
class FastDLLMTrajectoryDataCollator:
    """将 tokenized 样本堆叠为 batch，并包含 trajectory（int64，形状 [B, L]）。"""

    trajectory_pad_value: int = _TRAJECTORY_PAD_VALUE
    block_mask_schedule: Optional[BlockMaskNumSchedule] = None

    def __call__(self, features: List[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if not features:
            return {}

        batch: dict[str, torch.Tensor] = {}

        for key in ("input_ids", "attention_mask", "labels"):
            if key in features[0]:
                batch[key] = torch.tensor([f[key] for f in features], dtype=torch.long)

        seq_len = batch["input_ids"].shape[1] if "input_ids" in batch else None
        trajectories = []
        for f in features:
            traj = f.get(TRAJECTORY_FIELD)
            if traj is None:
                if seq_len is None:
                    raise ValueError(
                        "Feature missing trajectory and input_ids; cannot infer sequence length."
                    )
                trajectories.append([self.trajectory_pad_value] * seq_len)
            else:
                if seq_len is not None and len(traj) != seq_len:
                    raise ValueError(
                        f"trajectory length {len(traj)} != input_ids length {seq_len}"
                    )
                trajectories.append(traj)

        batch[TRAJECTORY_FIELD] = torch.tensor(trajectories, dtype=torch.long)

        trajectory_fulls = []
        for f in features:
            full = f.get(TRAJECTORY_FULL_FIELD)
            if full is None:
                trajectory_fulls.append(None)
            else:
                trajectory_fulls.append([int(x) for x in full])
        batch[TRAJECTORY_FULL_FIELD] = trajectory_fulls

        # 占位字段；实际训练值由 wrap_model_forward_with_live_block_mask_schedule
        # 在 forward 前按 schedule.current 覆盖（worker 子进程中的 schedule 快照不会更新）。
        if self.block_mask_schedule is not None:
            batch_size = batch["input_ids"].shape[0]
            value = float(self.block_mask_schedule.start)
            batch[BLOCK_MASK_NUM_FIELD] = torch.full(
                (batch_size,),
                value,
                dtype=torch.float32,
            )
        return batch


def ensure_model_forward_accepts_trajectory(model: Any) -> Any:
    """兼容旧调用；等价于 ensure_model_forward_accepts_training_batch_keys。"""
    return ensure_model_forward_accepts_training_batch_keys(model)


def ensure_model_forward_accepts_training_batch_keys(model: Any) -> Any:
    """
    保证 forward 签名包含 trajectory、block_mask_num，Trainer 不会丢弃这些列。
    """
    if getattr(model, _MODEL_TRAINING_FORWARD_PATCHED_ATTR, False):
        return model

    forward = model.forward
    sig = inspect.signature(forward)
    missing = [k for k in TRAINING_FORWARD_BATCH_KEYS if k not in sig.parameters]
    if not missing:
        setattr(model, _MODEL_TRAINING_FORWARD_PATCHED_ATTR, True)
        return model

    params = list(sig.parameters.values())
    annotations = {
        TRAJECTORY_FIELD: Optional[torch.LongTensor],
        TRAJECTORY_FULL_FIELD: Optional[list],
        BLOCK_MASK_NUM_FIELD: Optional[torch.Tensor],
    }
    new_params = [
        inspect.Parameter(
            key,
            kind=inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation=annotations[key],
        )
        for key in missing
    ]
    # 新参数必须插在 **kwargs 之前，否则 inspect 会报参数顺序错误。
    varkw_idx = next(
        (i for i, p in enumerate(params) if p.kind == inspect.Parameter.VAR_KEYWORD),
        None,
    )
    if varkw_idx is not None:
        params = params[:varkw_idx] + new_params + params[varkw_idx:]
    else:
        params.extend(new_params)
    new_signature = sig.replace(parameters=params)

    # 与其它 forward wrapper 一致：无 self，调用时捕获的 bound forward。
    forward_fn = forward

    def forward_with_training_batch_keys(*args, **kwargs):
        return forward_fn(*args, **kwargs)

    forward_with_training_batch_keys.__name__ = getattr(forward, "__name__", "forward")
    forward_with_training_batch_keys.__doc__ = getattr(forward, "__doc__", None)
    forward_with_training_batch_keys.__signature__ = new_signature
    forward_with_training_batch_keys.__wrapped__ = forward

    model.forward = forward_with_training_batch_keys
    setattr(model, _MODEL_TRAINING_FORWARD_PATCHED_ATTR, True)
    return model
