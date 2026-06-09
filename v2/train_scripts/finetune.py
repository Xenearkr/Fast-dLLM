#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Statistics and Machine Learning Research Group at HKUST. All rights reserved.
"""A one-line summary of the module or program, terminated by a period.

Leave one blank line.  The rest of this docstring should contain an
overall description of the module or program.  Optionally, it may also
contain a brief description of exported classes and functions and/or usage
examples.

Typical usage example:

  foo = ClassFoo()
  bar = foo.FunctionBar()
"""

import sys
import os
from pathlib import Path

# sys.path.remove(os.path.abspath(os.path.dirname(sys.argv[0])))
_V2_ROOT = Path(__file__).resolve().parents[1]
if str(_V2_ROOT) not in sys.path:
    sys.path.insert(0, str(_V2_ROOT))

from transformers import HfArgumentParser

from utils.conversation_tokenize_with_trajectory import wrap_model_tokenize_with_trajectory
from utils.truncate_tokenized_dataset import wrap_model_tokenize_with_truncate
from utils.block_mask_num_schedule import (
    BlockMaskNumCallback,
    BlockMaskNumSchedule,
    wrap_model_forward_with_live_block_mask_schedule,
)
from utils.trajectory_data_collator import (
    FastDLLMTrajectoryDataCollator,
    ensure_model_forward_accepts_training_batch_keys,
)
from utils.trajectory_mask_logger import (
    TrajectoryMaskLogCallback,
    wrap_model_forward_with_mask_logging,
)

from lmflow.args import (
    ModelArguments,
    DatasetArguments,
    AutoArguments,
)

from lmflow.datasets.dataset import Dataset
from lmflow.models.auto_model import AutoModel
from lmflow.pipeline.auto_pipeline import AutoPipeline


def main():
	# Parses arguments
    pipeline_name = "finetuner"
    PipelineArguments = AutoArguments.get_pipeline_args_class(pipeline_name)

    parser = HfArgumentParser((ModelArguments, DatasetArguments, PipelineArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, pipeline_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, pipeline_args = parser.parse_args_into_dataclasses()

    # Initialization
    finetuner = AutoPipeline.get_pipeline(
        pipeline_name=pipeline_name,
        model_args=model_args,
        data_args=data_args,
        pipeline_args=pipeline_args,
    )
    dataset = Dataset(data_args)
    model = AutoModel.get_model(model_args)
    
    data_args.bd_size = model.backend_model.config.bd_size
    data_args.mask_id = model.tokenizer.encode("|<MASK>|")[0]

    # tokenize 时保留并对齐 trajectory；之后再截断到 block_size
    wrap_model_tokenize_with_trajectory(model)
    truncate_max_len = data_args.block_size if data_args.block_size else 512
    wrap_model_tokenize_with_truncate(model, max_length=truncate_max_len)

    bd_size = int(data_args.bd_size)
    block_mask_schedule = None
    extra_callbacks = None
    if pipeline_args.block_mask_num_schedule:
        mask_end = pipeline_args.block_mask_num_end
        if mask_end is None:
            mask_end = bd_size
        block_mask_schedule = BlockMaskNumSchedule(
            start=float(pipeline_args.block_mask_num_start),
            end=float(mask_end),
        )
        extra_callbacks = [BlockMaskNumCallback(block_mask_schedule)]

    ensure_model_forward_accepts_training_batch_keys(model.get_backend_model())
    wrap_model_forward_with_live_block_mask_schedule(
        model.get_backend_model(),
        block_mask_schedule,
    )

    mask_log_state = wrap_model_forward_with_mask_logging(
        model.get_backend_model(),
        tokenizer=model.tokenizer,
        mask_id=data_args.mask_id,
        bd_size=bd_size,
    )
    mask_log_callback = TrajectoryMaskLogCallback(mask_log_state)
    if extra_callbacks is None:
        extra_callbacks = [mask_log_callback]
    else:
        extra_callbacks = list(extra_callbacks) + [mask_log_callback]

    trajectory_collator = FastDLLMTrajectoryDataCollator(
        block_mask_schedule=block_mask_schedule,
    )

    # Finetuning
    tuned_model = finetuner.tune(
        model=model,
        dataset=dataset,
        data_collator=trajectory_collator,
        extra_trainer_callbacks=extra_callbacks,
    )


if __name__ == '__main__':
    main()
