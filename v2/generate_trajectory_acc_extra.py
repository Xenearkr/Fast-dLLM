"""
基于 generate_trajectory_acc.py，按 block 生成轨迹，并在每个 block 解码完成后用
ground truth 替换模型输出，作为下一 block 的上下文（block 级 teacher forcing）。

与 generate_trajectory_acc.py 的区别：
  - 每解码完一个 block，记录该 block 内各 token 的 trajectory
  - 进入下一 block 前，将刚解码的 block 替换为数据集中 assistant 的 ground truth token
  - 刷新 KV cache，确保后续 block 在 GT 前缀上继续解码
  - 最终输出的 trajectory 仍是模型在各 block 上的真实解码顺序；token 序列以 GT 为准

数据源 / 输出格式 / 多卡加速与 generate_trajectory_acc.py 相同。

示例：
  accelerate launch generate_trajectory_acc_extra.py \\
    --model_path base_models/Fast_dLLM_v2_7B_full \\
    --dataset_path data/Llama-Nemotron-code-v1.1/train_conversation_no_cot \\
    --dataset_files train-00000.json \\
    --output data/Llama-Nemotron-code-v1.1/trajectory/no_cot_subset_gt/train-00001.json \\
    --output_format conversation \\
    --batch_size 4 --max_new_tokens 512 --local_files_only
"""
import json
import sys
import types
from pathlib import Path
import argparse

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import accelerate

DEFAULT_DATASET_PATH = (
    "/home/u-chenx/Fast-dLLM/v2/data/Llama-Nemotron-code-v1.1/train_conversation_no_cot"
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LMFLOW_ROOT = _REPO_ROOT / "third_party"
if str(_LMFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(_LMFLOW_ROOT))

from lmflow.utils.conversation_template import JINJA_TEMPLATES


def _refresh_past_key_values(self, input_ids, block_size):
    """用 GT 前缀重建 KV cache。"""
    aligned_len = input_ids.shape[1] // block_size * block_size
    if aligned_len <= block_size:
        return None
    output = self.forward(
        input_ids=input_ids[:, :aligned_len],
        use_cache=True,
        update_past_key_values=True,
        block_size=block_size,
    )
    return output.past_key_values


def _apply_ground_truth_block(x_t, ground_truth_ids, block_idx, block_size):
    """仅将当前已解码 block（及可能的 block 边界 token）替换为 GT。"""
    block_start = block_idx * block_size
    block_end = min((block_idx + 1) * block_size, ground_truth_ids.shape[1], x_t.shape[1])
    if block_end > block_start:
        x_t[:, block_start:block_end] = ground_truth_ids[:, block_start:block_end]

    boundary_pos = block_end
    if boundary_pos < ground_truth_ids.shape[1] and boundary_pos < x_t.shape[1]:
        x_t[:, boundary_pos] = ground_truth_ids[:, boundary_pos]
    return x_t


@torch.no_grad()
def batch_sample_gt_context(
    self,
    input_ids,
    tokenizer,
    block_size,
    max_new_tokens,
    small_block_size,
    min_len,
    seq_len,
    ground_truth_ids,
    mask_id=151665,
    threshold=0.95,
    stop_token=151645,
    use_block_cache=False,
    top_p=0.95,
    temperature=0.0,
    prompt_padded_len=None,
):
    """
    与 batch_sample 相同，但每个 block 解码后：
      1. 保留 trajectory 记录
      2. 用 ground_truth_ids 覆盖该 block
      3. 重建 past_key_values
    """
    num_blocks = max_new_tokens // block_size + seq_len.max().item() // block_size
    batch_size = input_ids.shape[0]

    if prompt_padded_len is None:
        prompt_padded_len = int(seq_len.max().item())

    if min_len > block_size:
        output = self.forward(
            input_ids=input_ids[:, :(min_len // block_size * block_size)],
            use_cache=True,
            update_past_key_values=True,
            block_size=block_size,
        )
        logits, past_key_values = output.logits, output.past_key_values
        if min_len % block_size == 0:
            predict_sample_idx = (seq_len == min_len)
            predict_logits = logits[predict_sample_idx, -1:, :]
            next_token = predict_logits.argmax(dim=-1)
            if input_ids.shape[1] <= min_len:
                input_ids = torch.cat([input_ids, next_token], dim=1)
            else:
                input_ids[predict_sample_idx, min_len] = next_token.squeeze(dim=-1)
    else:
        past_key_values = None

    seq_block_idx = seq_len // block_size
    finished_flag = torch.zeros((batch_size), device=self.device, dtype=torch.bool)

    start_block_idx = min_len // block_size
    num_small_blocks = block_size // small_block_size

    sample_indices = torch.arange(batch_size, device=self.device)
    finished_samples = {}
    finished_trajectories = {}

    traj_lens = torch.full(
        (batch_size,),
        prompt_padded_len + max_new_tokens,
        device=self.device,
        dtype=torch.long,
    )
    trajectory = torch.zeros(
        batch_size,
        traj_lens.max().item(),
        device=self.device,
        dtype=torch.long,
    )
    decode_count = torch.zeros(batch_size, device=self.device, dtype=torch.long)

    if min_len > block_size and min_len % block_size == 0:
        predict_sample_idx = (seq_len == min_len)
        decode_count += predict_sample_idx.long()
        trajectory[predict_sample_idx, min_len] = decode_count[predict_sample_idx]

    def align_trajectory_to_sample(traj_row, sample_len, current_count):
        if traj_row.shape[0] >= sample_len:
            return traj_row[:sample_len].clone()
        pad_len = sample_len - traj_row.shape[0]
        pad = torch.full(
            (pad_len,),
            fill_value=current_count + 1,
            device=traj_row.device,
            dtype=traj_row.dtype,
        )
        return torch.cat([traj_row, pad])

    def record_unmasks(traj, counts, cur_seq_len, cur_traj_lens, unmask_idx, abs_start):
        if not unmask_idx.any():
            return
        batch_idx, local_idx = unmask_idx.nonzero(as_tuple=True)
        abs_pos = abs_start + local_idx
        valid = (abs_pos >= cur_seq_len[batch_idx]) & (abs_pos < cur_traj_lens[batch_idx])
        batch_idx = batch_idx[valid]
        abs_pos = abs_pos[valid]
        if batch_idx.numel() == 0:
            return
        counts[batch_idx] += 1
        traj[batch_idx, abs_pos] = counts[batch_idx]

    def record_append_decode(traj, counts, cur_seq_len, cur_traj_lens, abs_pos):
        valid = (abs_pos >= cur_seq_len) & (abs_pos < cur_traj_lens)
        batch_idx = torch.arange(traj.shape[0], device=traj.device)[valid]
        if batch_idx.numel() == 0:
            return
        counts[batch_idx] += 1
        traj[batch_idx, abs_pos] = counts[batch_idx]

    for block_idx in range(start_block_idx, num_blocks):
        if (seq_block_idx == block_idx).all():
            x_init = mask_id * torch.ones(
                (input_ids.shape[0], block_size - input_ids.shape[1] % block_size),
                device=self.device,
                dtype=torch.long,
            )
            x_init = torch.cat([input_ids, x_init], dim=1)
            input_ids = x_init
        else:
            x_init = input_ids[:, :(block_idx + 1) * block_size]

        x_init[finished_flag, -block_size:] = tokenizer.pad_token_id
        x_t = x_init.clone()
        block_past_key_values = None

        while True:
            mask_idx = (x_t[:, -block_size:] == mask_id)
            if mask_idx.sum() == 0:
                for sample_idx in range(x_t.shape[0]):
                    if finished_flag[sample_idx] and seq_len[sample_idx] < (block_idx + 1) * block_size:
                        stop_positions = (x_t[sample_idx, seq_len[sample_idx]:] == stop_token).nonzero()
                        if stop_positions.numel() > 0:
                            stop_token_idx = stop_positions[0][0]
                            x_t[sample_idx, seq_len[sample_idx] + stop_token_idx + 1 :] = tokenizer.pad_token_id
                output = self.forward(
                    input_ids=x_t[:, -block_size:],
                    use_cache=True,
                    past_key_values=past_key_values,
                    update_past_key_values=True,
                    block_size=block_size,
                )
                logits, past_key_values = output.logits, output.past_key_values
                next_token = logits[:, -1:, :].argmax(dim=-1)
                next_token[finished_flag] = tokenizer.pad_token_id
                x_t = torch.cat([x_t, next_token], dim=1)
                record_append_decode(trajectory, decode_count, seq_len, traj_lens, x_t.shape[1] - 1)
                break

            for small_block_idx in range(num_small_blocks):
                small_block_start_idx = small_block_idx * small_block_size
                small_block_end_idx = small_block_start_idx + small_block_size

                start = -block_size + small_block_start_idx
                end = None if block_size == small_block_end_idx else -block_size + small_block_end_idx
                while True:
                    mask_idx = (x_t[:, -block_size:] == mask_id)
                    if mask_idx[:, start:end].sum() == 0:
                        break

                    if use_block_cache:
                        if block_past_key_values is None or (x_t[:, -block_size + small_block_start_idx] == mask_id).any():
                            output = self.forward(
                                input_ids=x_t[:, -block_size:],
                                use_cache=True,
                                past_key_values=past_key_values,
                                update_past_key_values=False,
                                use_block_cache=True,
                            )
                            logits, block_past_key_values = output.logits, output.block_past_key_values
                            logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                            logits = logits[:, start:end]
                        else:
                            logits = self.forward(
                                input_ids=x_t[:, start:end],
                                use_cache=True,
                                past_key_values=past_key_values,
                                update_past_key_values=False,
                                use_block_cache=True,
                                block_past_key_values=block_past_key_values,
                                replace_position=small_block_start_idx,
                            ).logits
                            logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                    else:
                        logits = self.forward(
                            input_ids=x_t[:, -block_size:],
                            use_cache=True,
                            past_key_values=past_key_values,
                            update_past_key_values=False,
                        ).logits
                        logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                        logits = logits[:, start:end]

                    x_1, p_1t = self.sample_with_top_p(logits, top_p=top_p, temperature=temperature)
                    x1_p = torch.squeeze(torch.gather(p_1t, dim=-1, index=torch.unsqueeze(x_1, -1)), -1)
                    x1_p = torch.where(mask_idx[:, start:end], x1_p, -torch.inf)

                    unmask_idx = (x1_p > threshold)
                    max_prob_idx = x1_p.argmax(dim=-1)
                    unmask_idx[torch.arange(x_1.shape[0]), max_prob_idx] = True
                    unmask_idx = unmask_idx & mask_idx[:, start:end]

                    abs_start = x_t.shape[1] + start
                    record_unmasks(trajectory, decode_count, seq_len, traj_lens, unmask_idx, abs_start)
                    x_t[:, start:end][unmask_idx] = x_1[unmask_idx]

                    finished_row_flags = ((x_1 == stop_token) & unmask_idx).any(dim=1)
                    finished_flag = finished_flag | finished_row_flags

        # block 解码完成：用 GT 替换当前 block，再进入下一 block
        x_t = _apply_ground_truth_block(x_t, ground_truth_ids, block_idx, block_size)

        if input_ids.shape[1] == x_t.shape[1]:
            input_ids = x_t
        else:
            input_ids[:, :(block_idx + 1) * block_size] = x_t[:, :-1]
            if (seq_block_idx == block_idx).all():
                input_ids = torch.cat([input_ids, x_t[:, -1:]], dim=1)
            else:
                if input_ids.shape[1] <= (block_idx + 1) * block_size:
                    input_ids = x_t
                else:
                    input_ids[seq_block_idx == block_idx, (block_idx + 1) * block_size] = x_t[
                        seq_block_idx == block_idx, (block_idx + 1) * block_size
                    ]

        past_key_values = _refresh_past_key_values(self, input_ids, block_size)
        seq_block_idx[seq_block_idx == block_idx] = block_idx + 1

    for sample_idx in range(batch_size):
        sample_len = int(traj_lens[sample_idx].item())
        finished_samples[sample_idx] = ground_truth_ids[sample_idx, :sample_len]
        finished_trajectories[sample_idx] = align_trajectory_to_sample(
            trajectory[sample_idx],
            sample_len,
            decode_count[sample_idx].item(),
        )

    assert len(finished_samples) == batch_size
    assert len(finished_trajectories) == batch_size
    return finished_samples, finished_trajectories


def resolve_input_json_files(
    dataset_path: str,
    dataset_files: list[str] | None,
    dataset_glob: str | None,
) -> list[Path]:
    base = Path(dataset_path) if dataset_path else Path(".")

    if dataset_files:
        resolved: list[Path] = []
        for raw in dataset_files:
            candidate = Path(raw)
            if not candidate.is_absolute():
                candidate = base / candidate
            candidate = candidate.resolve()
            if not candidate.is_file():
                raise FileNotFoundError(f"Dataset file not found: {candidate}")
            resolved.append(candidate)
        return sorted(set(resolved))

    if not base.exists():
        raise FileNotFoundError(f"Dataset not found: {base}")

    if base.is_file():
        return [base.resolve()]

    pattern = dataset_glob or "*.json"
    json_files = sorted(base.glob(pattern))
    if not json_files:
        raise FileNotFoundError(f"No files matched {pattern!r} under {base}")
    return [p.resolve() for p in json_files]


def load_conversation_dataset(json_files: list[Path], limit=None):
    items = []
    for json_file in json_files:
        with json_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if data.get("type") != "conversation":
            raise ValueError(
                f"Expected conversation dataset in {json_file}, got type={data.get('type')!r}"
            )

        for i, inst in enumerate(data["instances"]):
            task_id = inst.get("conversation_id") or f"{json_file.stem}-{i}"
            if not inst.get("messages"):
                continue
            if inst["messages"][-1]["role"] != "assistant":
                raise ValueError(
                    f"Instance {task_id} in {json_file} has no assistant message for ground truth"
                )
            items.append((task_id, inst, str(json_file.name)))
            if limit is not None and len(items) >= limit:
                return items

    return items


def resolve_output_path(output: str, output_format: str) -> tuple[Path, str]:
    path = Path(output)
    if output_format == "auto":
        if path.suffix == ".json":
            output_format = "conversation"
        elif path.is_dir():
            output_format = "conversation"
        else:
            output_format = "jsonl"

    if output_format == "conversation":
        if path.suffix == ".json":
            return path, output_format
        if path.suffix == ".jsonl":
            raise ValueError(
                "output ends with .jsonl but output_format=conversation; "
                "use a .json path or a directory."
            )
        return path / "train-00000.json", output_format

    if path.is_dir():
        return path / "trajectory.jsonl", output_format
    return path, output_format


def write_output_rows(rows: list[dict], output_file: Path, output_format: str) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "conversation":
        payload = {"type": "conversation", "instances": rows}
        output_file.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        return

    with output_file.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_generation_prompt(instance, tokenizer, conversation_template):
    messages = list(instance["messages"])
    system = instance.get("system") or None

    if not messages:
        raise ValueError(f"Empty messages in instance {instance.get('conversation_id')}")

    if messages[-1]["role"] == "assistant":
        messages = messages[:-1]

    conversation = []
    if system:
        conversation.append({"role": "system", "content": system})
    conversation.extend(messages)

    return tokenizer.apply_chat_template(
        conversation=conversation,
        chat_template=conversation_template,
        tokenize=False,
        add_generation_prompt=True,
    )


def build_ground_truth_token_ids(
    instance,
    tokenizer,
    conversation_template,
    *,
    prompt_padded_len: int,
    max_new_tokens: int,
    mask_id: int,
):
    """构造与 batch padding 对齐的 GT token 序列：prompt + mask pad + assistant。"""
    messages = list(instance["messages"])
    system = instance.get("system") or None

    conversation = []
    if system:
        conversation.append({"role": "system", "content": system})
    conversation.extend(messages)

    full_text = tokenizer.apply_chat_template(
        conversation=conversation,
        chat_template=conversation_template,
        tokenize=False,
        add_generation_prompt=False,
    )
    full_ids = tokenizer([full_text], return_tensors="pt")["input_ids"][0]

    prompt_text = build_generation_prompt(instance, tokenizer, conversation_template)
    prompt_ids = tokenizer([prompt_text], return_tensors="pt")["input_ids"][0]
    prompt_len = int(prompt_ids.shape[0])

    gen_gt = full_ids[prompt_len:]
    total_len = prompt_padded_len + max_new_tokens
    gt = torch.full((total_len,), tokenizer.pad_token_id, dtype=torch.long)

    gt[:prompt_len] = prompt_ids
    if prompt_padded_len > prompt_len:
        gt[prompt_len:prompt_padded_len] = mask_id

    gen_len = min(int(gen_gt.shape[0]), max_new_tokens)
    if gen_len > 0:
        gt[prompt_padded_len : prompt_padded_len + gen_len] = gen_gt[:gen_len]

    return gt


def batch_generate_gt_context(
    model,
    tokenizer,
    prompts,
    instances,
    conversation_template,
    device,
    *,
    mask_id,
    bd_size,
    small_block_size,
    max_new_tokens,
    threshold,
    use_block_cache,
):
    input_id_list = []
    gt_list = []
    seq_lens = []
    max_len = 0
    min_len = 10**18

    for prompt, instance in zip(prompts, instances):
        model_inputs = tokenizer([prompt], return_tensors="pt")
        input_ids = model_inputs["input_ids"].to(device)

        input_id_list.append(input_ids)
        seq_lens.append(input_ids.shape[1])
        max_len = max(max_len, input_ids.shape[1])
        min_len = min(min_len, input_ids.shape[1])

    padded = []
    for input_ids in input_id_list:
        pad_len = max_len - input_ids.shape[1]
        if pad_len > 0:
            pad = torch.full(
                (1, pad_len),
                mask_id,
                dtype=torch.long,
                device=device,
            )
            input_ids = torch.cat([input_ids, pad], dim=1)
        padded.append(input_ids)

    for instance in instances:
        gt = build_ground_truth_token_ids(
            instance,
            tokenizer,
            conversation_template,
            prompt_padded_len=max_len,
            max_new_tokens=max_new_tokens,
            mask_id=mask_id,
        )
        gt_list.append(gt)

    batched_input_ids = torch.cat(padded, dim=0).to(device)
    ground_truth_ids = torch.stack(gt_list, dim=0).to(device)
    seq_len_tensor = torch.tensor(seq_lens, dtype=torch.long, device=device)

    with torch.no_grad():
        finished_samples, finished_trajectories = model.mdm_sample(
            batched_input_ids,
            tokenizer=tokenizer,
            block_size=bd_size,
            small_block_size=small_block_size,
            max_new_tokens=max_new_tokens,
            mask_id=mask_id,
            min_len=min_len,
            seq_len=seq_len_tensor,
            ground_truth_ids=ground_truth_ids,
            use_block_cache=use_block_cache,
            threshold=threshold,
            prompt_padded_len=max_len,
        )

    trajectories = []
    for batch_idx, prompt_len in enumerate(seq_lens):
        full_ids = finished_samples[batch_idx]
        traj = finished_trajectories[batch_idx]

        if full_ids.shape[0] != traj.shape[0]:
            raise RuntimeError(
                f"trajectory length {traj.shape[0]} != sample length {full_ids.shape[0]} "
                f"for batch index {batch_idx}"
            )

        trajectories.append(traj.detach().cpu().tolist())

    return trajectories


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", required=True)
    parser.add_argument(
        "--dataset_path",
        default=DEFAULT_DATASET_PATH,
        help="Conversation JSON file or directory (base dir for --dataset_files)",
    )
    parser.add_argument(
        "--dataset_files",
        nargs="+",
        default=None,
        help="One or more train-*.json shards (relative to --dataset_path unless absolute)",
    )
    parser.add_argument(
        "--dataset_glob",
        default=None,
        help="Glob under --dataset_path when --dataset_files is not set (default: *.json)",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--output_format",
        choices=["auto", "jsonl", "conversation"],
        default="auto",
        help="conversation: single train-*.json for finetune_trajectory; jsonl: intermediate",
    )
    parser.add_argument(
        "--conversation_template",
        default="fast_dllm_v2",
        help="Chat template name, same as finetune_a2d_v0.sh",
    )

    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=512)

    parser.add_argument("--mask_id", type=int, default=151665)
    parser.add_argument("--bd_size", type=int, default=32)
    parser.add_argument("--small_block_size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=torch.inf)
    parser.add_argument("--use_block_cache", action="store_true")

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")

    args = parser.parse_args()

    if args.max_new_tokens % args.bd_size != 0:
        raise ValueError(
            f"max_new_tokens must be divisible by bd_size. "
            f"Got max_new_tokens={args.max_new_tokens}, bd_size={args.bd_size}"
        )

    accelerator = accelerate.Accelerator()
    device = accelerator.device

    if args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    if accelerator.is_main_process:
        print("Loading tokenizer:", args.model_path)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.conversation_template not in JINJA_TEMPLATES:
        raise ValueError(
            f"Unknown conversation_template: {args.conversation_template}. "
            f"Available: {sorted(JINJA_TEMPLATES)}"
        )
    conversation_template = JINJA_TEMPLATES[args.conversation_template]

    if accelerator.is_main_process:
        print("Loading model:", args.model_path)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        local_files_only=args.local_files_only,
        low_cpu_mem_usage=True,
    )
    model = model.to(device)
    model.mdm_sample = types.MethodType(batch_sample_gt_context, model)
    model.eval()

    generation_model = model

    json_files = resolve_input_json_files(
        args.dataset_path,
        args.dataset_files,
        args.dataset_glob,
    )
    all_items = load_conversation_dataset(json_files, limit=args.limit)

    with accelerator.split_between_processes(all_items) as local_items:
        worker_items = list(local_items)

    output_path, output_format = resolve_output_path(args.output, args.output_format)
    if accelerator.is_main_process:
        print("Input JSON files:")
        for p in json_files:
            print(f"  - {p}")
        print(f"Total problems across all GPUs: {len(all_items)}")
        print(f"Problems assigned per GPU (approx): {len(worker_items)}")
        print(f"Output: {output_path} (format={output_format})")
        print("Mode: block-level GT context (teacher forcing between blocks)")

    local_rows = []

    disable_tqdm = not accelerator.is_main_process
    for start in tqdm(range(0, len(worker_items), args.batch_size), desc="Generating", disable=disable_tqdm):
        batch_items = worker_items[start : start + args.batch_size]
        if not batch_items:
            continue

        task_ids = [x[0] for x in batch_items]
        instances_batch = [x[1] for x in batch_items]
        source_files = [x[2] for x in batch_items]

        prompts = [
            build_generation_prompt(instance, tokenizer, conversation_template)
            for instance in instances_batch
        ]

        trajectories = batch_generate_gt_context(
            model=generation_model,
            tokenizer=tokenizer,
            prompts=prompts,
            instances=instances_batch,
            conversation_template=conversation_template,
            device=device,
            mask_id=args.mask_id,
            bd_size=args.bd_size,
            small_block_size=args.small_block_size,
            max_new_tokens=args.max_new_tokens,
            threshold=args.threshold,
            use_block_cache=args.use_block_cache,
        )

        for task_id, instance, source_file, trajectory in zip(
            task_ids, instances_batch, source_files, trajectories
        ):
            system = instance.get("system")
            if system is None:
                system = ""
            row = {
                "conversation_id": task_id,
                "system": system,
                "messages": instance["messages"],
                "trajectory": trajectory,
            }
            if output_format == "jsonl":
                row["source_file"] = source_file
            local_rows.append(row)

    all_gathered_rows = accelerate.utils.gather_object(local_rows)

    if accelerator.is_main_process:
        write_output_rows(all_gathered_rows, output_path, output_format)
        print(f"Successfully saved {len(all_gathered_rows)} trajectories to {output_path}")
        if output_format == "conversation":
            print(
                "Train with: DATASET_PATH="
                f"{output_path.parent} bash train_scripts/finetune_trajectory.sh"
            )


if __name__ == "__main__":
    main()
