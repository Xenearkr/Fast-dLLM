import re
import shutil
from pathlib import Path

MODEL_DIR = Path(".").resolve()
MODELING_PATH = MODEL_DIR / "modeling.py"
AUDIT_DIR = MODEL_DIR / "audits"
BACKUP_DIR = AUDIT_DIR / "pre_step4A_modeling_backup"

AUDIT_DIR.mkdir(exist_ok=True)
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

backup_path = BACKUP_DIR / "modeling_qwen3_original.py"
if not backup_path.exists():
    shutil.copy2(MODELING_PATH, backup_path)
    print(f"[BACKUP] {MODELING_PATH} -> {backup_path}")
else:
    print(f"[INFO] Backup already exists: {backup_path}")

src = MODELING_PATH.read_text(encoding="utf-8")

# ---------------------------------------------------------------------
# 1. Convert Transformers source-tree relative imports to model-repo imports.
# ---------------------------------------------------------------------
import_replacements = {
    "from ...activations import ACT2FN": "from transformers.activations import ACT2FN",
    "from ...cache_utils import Cache, DynamicCache": "from transformers.cache_utils import Cache, DynamicCache",
    "from ...generation import GenerationMixin": "from transformers.generation import GenerationMixin",
    "from ...integrations import use_kernel_forward_from_hub, use_kernel_func_from_hub, use_kernelized_func": (
        "from transformers.integrations import use_kernel_forward_from_hub, use_kernel_func_from_hub, use_kernelized_func"
    ),
    "from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask": (
        "from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask"
    ),
    "from ...modeling_flash_attention_utils import FlashAttentionKwargs": (
        "from transformers.modeling_flash_attention_utils import FlashAttentionKwargs"
    ),
    "from ...modeling_layers import (": "from transformers.modeling_layers import (",
    "from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast": (
        "from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast"
    ),
    "from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update": (
        "from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update"
    ),
    "from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel": (
        "from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel"
    ),
    "from ...processing_utils import Unpack": "from transformers.processing_utils import Unpack",
    "from ...utils import TransformersKwargs, auto_docstring, can_return_tuple": (
        "from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple"
    ),
    "from ...utils.generic import maybe_autocast, merge_with_config_defaults": (
        "from transformers.utils.generic import maybe_autocast, merge_with_config_defaults"
    ),
    "from ...utils.output_capturing import capture_outputs": (
        "from transformers.utils.output_capturing import capture_outputs"
    ),
}

for old, new in import_replacements.items():
    src = src.replace(old, new)

# Replace config import. Do this before and after name replacement defensively.
src = re.sub(
    r"from\s+\.configuration_qwen3\s+import\s+Qwen3Config",
    "try:\n    from .configuration import Fast_dLLM_Qwen3Config\nexcept ImportError:\n    from configuration import Fast_dLLM_Qwen3Config",
    src,
)

# ---------------------------------------------------------------------
# 2. Rename Qwen3 classes into Fast_dLLM_Qwen3 classes.
#    Keep internal attribute names unchanged, so weight keys remain compatible.
# ---------------------------------------------------------------------
name_replacements = {
    "Qwen3ForSequenceClassification": "Fast_dLLM_Qwen3ForSequenceClassification",
    "Qwen3ForTokenClassification": "Fast_dLLM_Qwen3ForTokenClassification",
    "Qwen3ForQuestionAnswering": "Fast_dLLM_Qwen3ForQuestionAnswering",
    "Qwen3ForCausalLM": "Fast_dLLM_Qwen3ForCausalLM",
    "Qwen3PreTrainedModel": "Fast_dLLM_Qwen3PreTrainedModel",
    "Qwen3DecoderLayer": "Fast_dLLM_Qwen3DecoderLayer",
    "Qwen3Attention": "Fast_dLLM_Qwen3Attention",
    "Qwen3RotaryEmbedding": "Fast_dLLM_Qwen3RotaryEmbedding",
    "Qwen3RMSNorm": "Fast_dLLM_Qwen3RMSNorm",
    "Qwen3MLP": "Fast_dLLM_Qwen3MLP",
    "Qwen3Model": "Fast_dLLM_Qwen3Model",
    "Qwen3Config": "Fast_dLLM_Qwen3Config",
}

for old, new in sorted(name_replacements.items(), key=lambda kv: len(kv[0]), reverse=True):
    src = src.replace(old, new)

# The config import may have been transformed by the Qwen3Config replacement above.
src = re.sub(
    r"from\s+\.configuration_qwen3\s+import\s+Fast_dLLM_Qwen3Config",
    "try:\n    from .configuration import Fast_dLLM_Qwen3Config\nexcept ImportError:\n    from configuration import Fast_dLLM_Qwen3Config",
    src,
)

# ---------------------------------------------------------------------
# 3. Patch RotaryEmbedding for our standalone Fast_dLLM_Qwen3Config.
#    Qwen3 upstream may use config.rope_parameters; our config uses rope_theta/rope_scaling.
# ---------------------------------------------------------------------
rotary_class = r'''
class Fast_dLLM_Qwen3RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config: Fast_dLLM_Qwen3Config, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config

        rope_parameters = getattr(config, "rope_parameters", None)
        rope_scaling = getattr(config, "rope_scaling", None)

        if isinstance(rope_parameters, dict):
            self.rope_type = rope_parameters.get("rope_type", rope_parameters.get("type", "default"))
        elif rope_parameters is not None:
            self.rope_type = getattr(rope_parameters, "rope_type", "default")
        elif isinstance(rope_scaling, dict):
            self.rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))
        else:
            self.rope_type = "default"

        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_default_rope_parameters(
        config: Fast_dLLM_Qwen3Config | None = None,
        device: Optional["torch.device"] = None,
        seq_len: int | None = None,
    ) -> tuple["torch.Tensor", float]:
        if config is None:
            raise ValueError("config must be provided for RoPE parameter initialization.")

        rope_parameters = getattr(config, "rope_parameters", None)
        if isinstance(rope_parameters, dict) and "rope_theta" in rope_parameters:
            base = rope_parameters["rope_theta"]
        else:
            base = getattr(config, "rope_theta", 1000000.0)

        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        attention_factor = 1.0

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


'''

src = re.sub(
    r"class Fast_dLLM_Qwen3RotaryEmbedding\(nn\.Module\):.*?\n\ndef rotate_half",
    rotary_class + "def rotate_half",
    src,
    flags=re.DOTALL,
)

# ---------------------------------------------------------------------
# 4. Add compatibility aliases at the end.
#    These aliases are useful during migration but are not the primary exported classes.
# ---------------------------------------------------------------------
alias_block = r'''

# ---------------------------------------------------------------------
# Backward-compatible aliases during migration.
# ---------------------------------------------------------------------
Qwen3Config = Fast_dLLM_Qwen3Config
Qwen3RMSNorm = Fast_dLLM_Qwen3RMSNorm
Qwen3MLP = Fast_dLLM_Qwen3MLP
Qwen3RotaryEmbedding = Fast_dLLM_Qwen3RotaryEmbedding
Qwen3Attention = Fast_dLLM_Qwen3Attention
Qwen3DecoderLayer = Fast_dLLM_Qwen3DecoderLayer
Qwen3PreTrainedModel = Fast_dLLM_Qwen3PreTrainedModel
Qwen3Model = Fast_dLLM_Qwen3Model
Qwen3ForCausalLM = Fast_dLLM_Qwen3ForCausalLM
'''

# Replace __all__ with Fast names and append aliases.
src = re.sub(
    r"__all__\s*=\s*\[[\s\S]*?\]\s*$",
    '''__all__ = [
    "Fast_dLLM_Qwen3ForCausalLM",
    "Fast_dLLM_Qwen3PreTrainedModel",
    "Fast_dLLM_Qwen3Model",
    "Fast_dLLM_Qwen3ForSequenceClassification",
    "Fast_dLLM_Qwen3ForTokenClassification",
    "Fast_dLLM_Qwen3ForQuestionAnswering",
]
''' + alias_block,
    src,
)

MODELING_PATH.write_text(src, encoding="utf-8")
print(f"[OK] Wrote Milestone 4A skeleton to: {MODELING_PATH}")

# Basic static checks.
required = [
    "class Fast_dLLM_Qwen3Attention",
    "self.q_norm",
    "self.k_norm",
    "bias=config.attention_bias",
    "class Fast_dLLM_Qwen3ForCausalLM",
    "class Fast_dLLM_Qwen3Model",
    "Fast_dLLM_Qwen3Config",
]

missing = [x for x in required if x not in src]
if missing:
    raise RuntimeError(f"Missing required markers after transformation: {missing}")

for forbidden in [
    "from ...",
    "from .configuration_qwen3 import",
]:
    if forbidden in src:
        raise RuntimeError(f"Forbidden marker remains in modeling.py: {forbidden}")

print("[OK] Static marker check passed.")
