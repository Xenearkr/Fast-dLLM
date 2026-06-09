"""
Conversation tokenize 扩展：tokenize 后原样保留 instance 中的 trajectory 字段。

lmflow 默认 tokenize 会 remove_columns 删掉 trajectory；本模块在
conversation_tokenize_function 之后把原始 trajectory 写回 token_dict。

用法：finetune.py 中调用 wrap_model_tokenize_with_trajectory(model)
"""

from __future__ import annotations

from typing import Union

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from lmflow.args import DatasetArguments
from lmflow.tokenization.hf_decoder_model import conversation_tokenize_function
from lmflow.utils.conversation_template import ConversationTemplate

TRAJECTORY_FIELD = "trajectory"


def _normalize_system_value(system):
    """与 generate_trajectory.build_generation_prompt 一致：空 system 视为无 system。"""
    return system or None


def _normalize_examples_system_for_tokenize(examples):
    """将 batch 中 system=='' 规范为 None，避免与轨迹生成时的 prompt 构造不一致。"""
    if "system" not in examples:
        return examples

    normalized = dict(examples)
    normalized["system"] = [_normalize_system_value(s) for s in examples["system"]]
    return normalized


def conversation_tokenize_function_with_trajectory(
    examples,
    data_args: DatasetArguments,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    column_names,
    conversation_template: Union[ConversationTemplate, str],
) -> dict:
    """与 lmflow conversation_tokenize_function 相同，并保留原始 trajectory。"""
    examples = _normalize_examples_system_for_tokenize(examples)
    token_dict = conversation_tokenize_function(
        examples,
        data_args,
        tokenizer,
        column_names,
        conversation_template,
    )

    num_example = len(examples[column_names[0]])
    if TRAJECTORY_FIELD in examples:
        token_dict[TRAJECTORY_FIELD] = list(examples[TRAJECTORY_FIELD])
    else:
        token_dict[TRAJECTORY_FIELD] = [None] * num_example

    return token_dict


_TRAJECTORY_TOKENIZE_FINGERPRINT_SUFFIX = (
    "###preserve_trajectory=1###trajectory_full=1###normalize_empty_system=1"
)


def _snapshot_conversation_tokenize_functions():
    import lmflow.models.hf_decoder_model as hf_model_module
    import lmflow.tokenization.hf_decoder_model as hf_tok_module

    return (
        hf_tok_module.conversation_tokenize_function,
        hf_model_module.conversation_tokenize_function,
    )


def _apply_conversation_tokenize_with_trajectory():
    import lmflow.models.hf_decoder_model as hf_model_module
    import lmflow.tokenization.hf_decoder_model as hf_tok_module

    hf_tok_module.conversation_tokenize_function = (
        conversation_tokenize_function_with_trajectory
    )
    hf_model_module.conversation_tokenize_function = (
        conversation_tokenize_function_with_trajectory
    )


def _restore_conversation_tokenize_functions(old_tok_fn, old_model_fn):
    import lmflow.models.hf_decoder_model as hf_model_module
    import lmflow.tokenization.hf_decoder_model as hf_tok_module

    hf_tok_module.conversation_tokenize_function = old_tok_fn
    hf_model_module.conversation_tokenize_function = old_model_fn


def wrap_model_tokenize_with_trajectory(model) -> object:
    """
    包装 model.tokenize：在调用期间用 conversation_tokenize_function_with_trajectory
    替换 lmflow 默认 conversation_tokenize_function。

    须同时 patch tokenization 与 models.hf_decoder_model 两处引用；
    后者在 import 时已绑定旧函数，仅改 tokenization 模块无效。
    """
    if getattr(model, "_tokenize_trajectory_wrapped", False):
        return model

    original_tokenize = model.tokenize

    def tokenize_with_trajectory(dataset, *args, **kwargs):
        old_tok_fn, old_model_fn = _snapshot_conversation_tokenize_functions()
        _apply_conversation_tokenize_with_trajectory()

        old_get_fingerprint = dataset.get_fingerprint

        def get_fingerprint_with_trajectory():
            return old_get_fingerprint() + _TRAJECTORY_TOKENIZE_FINGERPRINT_SUFFIX

        dataset.get_fingerprint = get_fingerprint_with_trajectory
        try:
            return original_tokenize(dataset, *args, **kwargs)
        finally:
            dataset.get_fingerprint = old_get_fingerprint
            _restore_conversation_tokenize_functions(old_tok_fn, old_model_fn)

    model.tokenize = tokenize_with_trajectory
    model._tokenize_trajectory_wrapped = True
    return model
