from pathlib import Path
import shutil

MODEL_DIR = Path(".").resolve()
MODELING_PATH = MODEL_DIR / "modeling.py"
AUDIT_DIR = MODEL_DIR / "audits"
BACKUP_DIR = AUDIT_DIR / "pre_step4B_modeling_backup"

AUDIT_DIR.mkdir(exist_ok=True)
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

backup_path = BACKUP_DIR / "modeling_step4A_passed.py"
if not backup_path.exists():
    shutil.copy2(MODELING_PATH, backup_path)
    print(f"[BACKUP] {MODELING_PATH} -> {backup_path}")
else:
    print(f"[INFO] Backup already exists: {backup_path}")

text = MODELING_PATH.read_text(encoding="utf-8")

# ---------------------------------------------------------------------
# 1. Add dataclass import.
# ---------------------------------------------------------------------
if "from dataclasses import dataclass" not in text:
    text = text.replace(
        "from collections.abc import Callable\n",
        "from collections.abc import Callable\nfrom dataclasses import dataclass\n",
        1,
    )
    print("[PATCH] Added dataclass import.")
else:
    print("[INFO] dataclass import already exists.")

# ---------------------------------------------------------------------
# 2. Add flex_attention / create_block_mask import with fallback.
# ---------------------------------------------------------------------
flex_import_block = '''try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
except Exception:
    flex_attention = None
    create_block_mask = None
'''

if "from torch.nn.attention.flex_attention import flex_attention, create_block_mask" not in text and "flex_attention = None" not in text:
    text = text.replace(
        "from torch import nn\n",
        "from torch import nn\n" + flex_import_block,
        1,
    )
    print("[PATCH] Added flex_attention/create_block_mask import fallback.")
else:
    print("[INFO] flex_attention/create_block_mask import already exists.")

# ---------------------------------------------------------------------
# 3. Add Fast-dLLM output dataclasses and block-diffusion masks.
#    This block intentionally does not modify forward logic yet.
# ---------------------------------------------------------------------
block_4b = r'''
# ---------------------------------------------------------------------
# Fast-dLLM v2 block-diffusion utilities.
# Added in Milestone 4B. These utilities are not connected to forward()
# until later milestones.
# ---------------------------------------------------------------------

@dataclass
class CausalLMOutputWithPastAndBlockCache(CausalLMOutputWithPast):
    block_past_key_values: Optional[Cache] = None


@dataclass
class BaseModelOutputWithPastAndBlockCache(BaseModelOutputWithPast):
    block_past_key_values: Optional[Cache] = None


def fused_flex_attention(q, k, v, mask=None):
    """
    Thin wrapper around torch flex_attention used by Fast-dLLM training.

    The actual forward path is not connected in Milestone 4B. If the current
    PyTorch build does not provide flex_attention, this function raises a clear
    error when called.
    """
    if flex_attention is None:
        raise RuntimeError(
            "torch.nn.attention.flex_attention is unavailable in this environment. "
            "Install a PyTorch version with flex_attention support before enabling "
            "Fast-dLLM training attention."
        )
    return flex_attention(q, k, v, block_mask=mask, enable_gqa=True)


def block_diff_mask(b, h, q_idx, kv_idx, block_size=None, n=None):
    """
    Construct the Fast-dLLM block-diffusion attention mask for training.

    The concatenated training sequence is [x_t ; x_0], where each half has
    length n. The mask is composed of:

    1. Block Diagonal Mask:
       tokens attend bidirectionally within the same block of the same half.

    2. Offset Block-Causal Mask:
       noised tokens x_t^b attend to previous clean blocks x_0^{<b}.

    3. Block-Causal Mask:
       clean tokens x_0^b attend to clean blocks x_0^{<=b}.

    Args:
        b, h:
            Batch and head indices. They are ignored by the logical mask but
            kept for create_block_mask compatibility.
        q_idx:
            Query indices.
        kv_idx:
            Key/value indices.
        block_size:
            Block size D.
        n:
            Length of each half sequence. Total concatenated length is 2n.

    Returns:
        Boolean tensor indicating allowed attention positions.
    """
    if block_size is None:
        raise ValueError("block_size must be provided.")
    if n is None:
        raise ValueError("n must be provided.")

    x0_flag_q = q_idx >= n
    x0_flag_kv = kv_idx >= n

    block_q = torch.where(
        x0_flag_q,
        (q_idx - n) // block_size,
        q_idx // block_size,
    )
    block_kv = torch.where(
        x0_flag_kv,
        (kv_idx - n) // block_size,
        kv_idx // block_size,
    )

    # M_BD: within-block bidirectional attention inside the same half.
    block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)

    # M_OBC: noised x_t block attends to previous clean x_0 blocks.
    offset_block_causal = (
        (block_q > block_kv)
        & (x0_flag_kv == 1)
        & (x0_flag_q == 0)
    )

    # M_BC: clean x_0 block attends to current and previous clean blocks.
    block_causal = (
        (block_q >= block_kv)
        & (x0_flag_kv == 1)
        & (x0_flag_q == 1)
    )

    return block_diagonal | offset_block_causal | block_causal


def eval_block_diff_mask(q_idx, kv_idx, block_size=None):
    """
    Construct the block-causal mask used by Fast-dLLM inference.

    This mask is block-level causal: tokens in a query block may attend to
    key/value tokens from the same block and all earlier blocks.

    Args:
        q_idx:
            Query indices.
        kv_idx:
            Key/value indices.
        block_size:
            Block size D.

    Returns:
        Boolean tensor indicating allowed attention positions.
    """
    if block_size is None:
        raise ValueError("block_size must be provided.")

    block_q = q_idx // block_size
    block_kv = kv_idx // block_size
    return block_q >= block_kv


'''

if "class CausalLMOutputWithPastAndBlockCache" not in text:
    marker = "\n\n@use_kernel_forward_from_hub(\"RMSNorm\")"
    if marker not in text:
        raise RuntimeError("Could not find insertion marker before Fast_dLLM_Qwen3RMSNorm.")
    text = text.replace(marker, "\n" + block_4b + marker, 1)
    print("[PATCH] Inserted 4B dataclasses and block mask utilities.")
else:
    print("[INFO] 4B dataclasses/mask utilities already exist.")

MODELING_PATH.write_text(text, encoding="utf-8")

required_markers = [
    "class CausalLMOutputWithPastAndBlockCache",
    "class BaseModelOutputWithPastAndBlockCache",
    "def fused_flex_attention",
    "def block_diff_mask",
    "def eval_block_diff_mask",
]

missing = [m for m in required_markers if m not in text]
if missing:
    raise RuntimeError(f"Missing required 4B markers: {missing}")

print("[OK] Milestone 4B patch applied.")
