"""
在 model.tokenize 之后、group_text 之前，将每条样本规整到 max_length。

- input_ids / attention_mask / labels：> max_length 截断；< max_length 右侧 pad 到 max_length
  （与 finetuner group_text 中 pad 逻辑一致：input_ids 用 pad_token_id，mask=0，labels=-100）
- trajectory：固定规整到 max_length；过短末尾填 -1

不修改 lmflow 内部函数；通过 wrap_model_tokenize_with_truncate 注入到 finetune 流程。
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, List, Literal, Optional

logger = logging.getLogger(__name__)

TRAJECTORY_FIELD = "trajectory"
TRAJECTORY_FULL_FIELD = "trajectory_full"
_TRAJECTORY_PAD_VALUE = -1
_TRUNCATE_KEYS = ("input_ids", "attention_mask", "labels")


def normalize_trajectory_length(
    trajectory: Any,
    max_length: int,
    truncation_side: Literal["right", "left"] = "right",
) -> List[int]:
    """将 trajectory 规整为长度 max_length：过长截断，过短末尾填 -1。"""
    if max_length <= 0:
        raise ValueError(f"max_length must be positive, got {max_length}")

    if trajectory is None:
        return [_TRAJECTORY_PAD_VALUE] * max_length

    if not isinstance(trajectory, list):
        raise TypeError(f"trajectory must be a list, got {type(trajectory)}")

    traj = [int(x) for x in trajectory]
    n = len(traj)

    if n > max_length:
        if truncation_side == "right":
            traj = traj[:max_length]
        else:
            traj = traj[n - max_length :]
    elif n < max_length:
        traj = traj + [_TRAJECTORY_PAD_VALUE] * (max_length - n)

    return traj


def pad_tokenized_example_to_max_length(
    example: dict,
    max_length: int,
    pad_token_id: int,
    padding_side: Literal["right", "left"] = "right",
) -> dict:
    """将 input_ids 等序列 pad 到 max_length（参考 finetuner group_text pad 逻辑）。"""
    out = example
    seq_len = len(out["input_ids"])
    if seq_len >= max_length:
        return out

    pad_length = max_length - seq_len
    if padding_side == "right":
        out["input_ids"] = out["input_ids"] + [pad_token_id] * pad_length
        if "attention_mask" in out:
            out["attention_mask"] = out["attention_mask"] + [0] * pad_length
        if "labels" in out:
            out["labels"] = out["labels"] + [-100] * pad_length
    elif padding_side == "left":
        out["input_ids"] = [pad_token_id] * pad_length + out["input_ids"]
        if "attention_mask" in out:
            out["attention_mask"] = [0] * pad_length + out["attention_mask"]
        if "labels" in out:
            out["labels"] = [-100] * pad_length + out["labels"]
    else:
        raise ValueError(f"padding_side must be 'right' or 'left', got {padding_side!r}")

    return out


def truncate_tokenized_example(
    example: dict,
    max_length: int,
    pad_token_id: int,
    truncation_side: Literal["right", "left"] = "right",
    padding_side: Literal["right", "left"] = "right",
) -> dict:
    """单条 tokenized 样本：截断/pad token 列到 max_length，并规整 trajectory。"""
    out = dict(example)

    seq_len = len(out["input_ids"])
    if seq_len > max_length:
        if truncation_side == "right":
            sl = slice(0, max_length)
        else:
            sl = slice(seq_len - max_length, seq_len)

        for key in _TRUNCATE_KEYS:
            if key in out:
                out[key] = out[key][sl]

    pad_tokenized_example_to_max_length(
        out,
        max_length=max_length,
        pad_token_id=pad_token_id,
        padding_side=padding_side,
    )

    raw_traj = out.get(TRAJECTORY_FIELD)
    if raw_traj is None:
        out[TRAJECTORY_FULL_FIELD] = None
    else:
        out[TRAJECTORY_FULL_FIELD] = [int(x) for x in raw_traj]

    out[TRAJECTORY_FIELD] = normalize_trajectory_length(
        raw_traj,
        max_length=max_length,
        truncation_side=truncation_side,
    )

    return out


def truncate_tokenized_lm_dataset(
    lm_dataset,
    max_length: int = 512,
    pad_token_id: int = 0,
    truncation_side: Literal["right", "left"] = "right",
    padding_side: Literal["right", "left"] = "right",
):
    """
    对 lmflow Dataset（tokenize 输出）做 map，将 input_ids 规整到 max_length。
    """
    if max_length is None or max_length <= 0:
        raise ValueError(f"max_length must be positive, got {max_length}")

    data_args = lm_dataset.get_data_args()
    hf_dataset = lm_dataset.get_backend_dataset()

    before_lengths = [len(row["input_ids"]) for row in hf_dataset]
    over_before = sum(1 for n in before_lengths if n > max_length)
    under_before = sum(1 for n in before_lengths if n < max_length)

    map_kwargs = {
        "desc": f"Truncating/padding tokenized sequences to max_length={max_length}",
    }
    if not getattr(data_args, "streaming", False):
        truncate_fingerprint = hashlib.md5(
            (
                hf_dataset._fingerprint
                + f"###truncate_max_length={max_length}"
                + f"###{TRAJECTORY_FULL_FIELD}=1"
            ).encode("utf-8")
        ).hexdigest()
        map_kwargs.update(
            {
                "num_proc": getattr(data_args, "preprocessing_num_workers", None),
                "load_from_cache_file": not getattr(data_args, "overwrite_cache", False),
                "new_fingerprint": truncate_fingerprint,
            }
        )

    def _map_fn(ex):
        return truncate_tokenized_example(
            ex,
            max_length=max_length,
            pad_token_id=pad_token_id,
            truncation_side=truncation_side,
            padding_side=padding_side,
        )

    truncated_hf = hf_dataset.map(
        _map_fn,
        **{k: v for k, v in map_kwargs.items() if v is not None},
    )

    lm_dataset.backend_dataset = truncated_hf

    after_lengths = [len(row["input_ids"]) for row in truncated_hf]
    over_after = sum(1 for n in after_lengths if n > max_length)
    under_after = sum(1 for n in after_lengths if n < max_length)
    max_after = max(after_lengths) if after_lengths else 0
    min_after = min(after_lengths) if after_lengths else 0

    traj_lengths = [len(row[TRAJECTORY_FIELD]) for row in truncated_hf]

    logger.info(
        "Post-tokenize truncate/pad: max_length=%s, pad_token_id=%s, samples=%s, "
        "input_ids over before=%s under before=%s, after all len in [%s,%s], "
        "trajectory_len=%s",
        max_length,
        pad_token_id,
        len(truncated_hf),
        over_before,
        under_before,
        min_after,
        max_after,
        traj_lengths[0] if traj_lengths else None,
    )
    print(
        f"[truncate_tokenized_dataset] max_length={max_length}, "
        f"input_ids truncated {over_before}, padded {under_before} / {len(truncated_hf)} samples "
        f"(len after: {min_after}-{max_after}), "
        f"trajectory len {traj_lengths[0] if traj_lengths else max_length} "
        f"(pad={_TRAJECTORY_PAD_VALUE})"
    )

    return lm_dataset


def wrap_model_tokenize_with_truncate(
    model,
    max_length: Optional[int] = 512,
    pad_token_id: Optional[int] = None,
    truncation_side: Literal["right", "left"] = "right",
    padding_side: Optional[Literal["right", "left"]] = None,
):
    """
    包装 HFDecoderModel.tokenize：先 tokenize，再将 input_ids 规整到 max_length。
    返回 model 以便链式调用。
    """
    if max_length is None:
        max_length = 512

    tokenizer = model.tokenizer
    if pad_token_id is None:
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        pad_token_id = tokenizer.pad_token_id

    if padding_side is None:
        padding_side = getattr(tokenizer, "padding_side", "right") or "right"

    if getattr(model, "_tokenize_truncate_wrapped", False):
        return model

    original_tokenize = model.tokenize

    def tokenize_with_truncate(dataset, *args, **kwargs):
        tokenized = original_tokenize(dataset, *args, **kwargs)
        return truncate_tokenized_lm_dataset(
            tokenized,
            max_length=max_length,
            pad_token_id=pad_token_id,
            truncation_side=truncation_side,
            padding_side=padding_side,
        )

    model.tokenize = tokenize_with_truncate
    model._tokenize_truncate_wrapped = True
    model._tokenize_truncate_max_length = max_length
    return model
