"""
用与 finetune_a2d_v0.sh 相同的方式加载 / 分词 conversation 数据，再调用 mdm_sample 生成轨迹。

输出 JSONL 每行字段：
  - conversation_id
  - completion: 生成文本（从真实 prompt 长度之后 decode，去掉残留 mask）
  - trajectory: 与完整 token 序列等长的 decode 步数（prompt 为 0，生成区为 1,2,...）
  - reference: 数据集中原始 assistant 标注
"""
import json
import sys
import types
from pathlib import Path
import argparse

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_DATASET_PATH = (
    "/home/u-chenx/Fast-dLLM/v2/data/Llama-Nemotron-code-v1.1/use/train-00000.json"
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LMFLOW_ROOT = _REPO_ROOT / "third_party"
if str(_LMFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(_LMFLOW_ROOT))

from lmflow.utils.conversation_template import JINJA_TEMPLATES


@torch.no_grad()
def batch_sample(
    self,
    input_ids,
    tokenizer,
    block_size,
    max_new_tokens,
    small_block_size,
    min_len,
    seq_len,
    mask_id=151665,
    threshold=0.95,
    stop_token=151645,
    use_block_cache=False,
    top_p=0.95,
    temperature=0.0,
    prompt_padded_len=None,
):
    """与 generation_functions.Fast_dLLM_QwenForCausalLM.batch_sample 相同，并记录解码顺序 trajectory。"""
    num_blocks = max_new_tokens // block_size + seq_len.max().item() // block_size
    batch_size = input_ids.shape[0]

    if prompt_padded_len is None:
        prompt_padded_len = int(seq_len.max().item())

    if min_len > block_size:
        output = self.forward(input_ids=input_ids[:, :(min_len // block_size * block_size)], use_cache=True, update_past_key_values=True, block_size=block_size)
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
    # 方案 A：记录 [seq_len, prompt_padded_len) 的 padding unmask + [prompt_padded_len, ...) 的新 block
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

    def align_trajectory_to_sample(traj_row, sample_row, current_count):
        """
        将轨迹张量裁剪或填充至与样本等长。
        如果轨迹长度不足，在尾部填充 current_count + 1
        """
        sample_len = sample_row.shape[0]
        
        # 情况 1：如果轨迹长度已经大于等于样本长度，进行截断
        if traj_row.shape[0] >= sample_len:
            return traj_row[:sample_len].clone()
        
        # 情况 2：如果轨迹长度不足，计算需要填充的长度
        pad_len = sample_len - traj_row.shape[0]
        
        # 使用 torch.full 创建一个填充张量，其值全部为 current_count + 1
        pad = torch.full(
            (pad_len,),
            fill_value=current_count + 1,
            device=traj_row.device,
            dtype=traj_row.dtype,
        )
        
        # 拼接并返回
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
        if finished_flag.all():
            break
        if (seq_block_idx == block_idx).all():
            x_init = mask_id * torch.ones((input_ids.shape[0], block_size-input_ids.shape[1]%block_size), device=self.device, dtype=torch.long)
            x_init = torch.cat([input_ids, x_init], dim=1)
            input_ids = x_init
        else:
            x_init = input_ids[:, :(block_idx + 1)*block_size]

        x_init[finished_flag, -block_size:] = tokenizer.pad_token_id
        x_t = x_init.clone()
        step = 0
        block_past_key_values = None
        while True:
            mask_idx = (x_t[:, -block_size:] == mask_id)
            if mask_idx.sum() == 0:
                for sample_idx in range(x_t.shape[0]):
                    if finished_flag[sample_idx] and seq_len[sample_idx] < (block_idx + 1) * block_size:
                        stop_token_idx = (x_t[sample_idx, seq_len[sample_idx]:] == stop_token).nonzero()[0][0]
                        x_t[sample_idx, seq_len[sample_idx]+stop_token_idx+1:] = tokenizer.pad_token_id
                if finished_flag.all():
                    break
                output = self.forward(input_ids=x_t[:, -block_size:], use_cache=True, past_key_values=past_key_values, update_past_key_values=True, block_size=block_size)
                logits, past_key_values = output.logits, output.past_key_values
                next_token = logits[:, -1:, :].argmax(dim=-1)
                next_token[finished_flag] = tokenizer.pad_token_id
                x_t = torch.cat([x_t, next_token], dim=1)
                record_append_decode(trajectory, decode_count, seq_len, traj_lens, x_t.shape[1] - 1)
                step += 1

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
                        if block_past_key_values is None or (x_t[:, -block_size+small_block_start_idx] == mask_id).any():
                            output = self.forward(input_ids=x_t[:, -block_size:], use_cache=True, past_key_values=past_key_values, update_past_key_values=False, use_block_cache=True)
                            logits, block_past_key_values = output.logits, output.block_past_key_values
                            logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                            logits = logits[:, start:end]
                        else:
                            logits = self.forward(input_ids=x_t[:,start:end], use_cache=True, past_key_values=past_key_values, update_past_key_values=False, use_block_cache=True, block_past_key_values=block_past_key_values, replace_position=small_block_start_idx).logits
                            logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                    else:
                        logits = self.forward(input_ids=x_t[:, -block_size:], use_cache=True, past_key_values=past_key_values, update_past_key_values=False).logits
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

                    step += 1

        if input_ids.shape[1] ==  x_t.shape[1]:
            input_ids = x_t
        else:
            input_ids[:, :(block_idx + 1)*block_size] = x_t[:, :-1]
            if (seq_block_idx == block_idx).all():
                input_ids = torch.cat([input_ids, x_t[:, -1:]], dim=1)
            else:
                if input_ids.shape[1] <= (block_idx + 1)*block_size:
                    input_ids = x_t
                else:
                    input_ids[seq_block_idx == block_idx, (block_idx + 1)*block_size] = x_t[seq_block_idx == block_idx, (block_idx + 1)*block_size]
        seq_block_idx[seq_block_idx == block_idx] = block_idx + 1
        if finished_flag.any():
            for sample_idx in range(x_t.shape[0]):
                if finished_flag[sample_idx]:
                    original_idx = sample_indices[sample_idx].item()
                    finished_samples[original_idx] = x_t[sample_idx:sample_idx+1].clone().squeeze(dim=0)
                    finished_trajectories[original_idx] = align_trajectory_to_sample(
                        trajectory[sample_idx], x_t[sample_idx], decode_count[sample_idx].item()
                    )
            sample_indices = sample_indices[~finished_flag]
            input_ids = input_ids[~finished_flag]
            seq_block_idx = seq_block_idx[~finished_flag]
            seq_len = seq_len[~finished_flag]
            traj_lens = traj_lens[~finished_flag]
            x_t = x_t[~finished_flag]
            trajectory = trajectory[~finished_flag]
            decode_count = decode_count[~finished_flag]

            for layer_id in range(len(past_key_values)):
                past_key_values.key_cache[layer_id] = past_key_values.key_cache[layer_id][~finished_flag]
                past_key_values.value_cache[layer_id] = past_key_values.value_cache[layer_id][~finished_flag]

            finished_flag = finished_flag[~finished_flag]

    if len(finished_samples) < batch_size:
        for sample_idx in range(x_t.shape[0]):
            original_idx = sample_indices[sample_idx].item()
            finished_samples[original_idx] = x_t[sample_idx:sample_idx+1].clone().squeeze(dim=0)
            finished_trajectories[original_idx] = align_trajectory_to_sample(
                trajectory[sample_idx], x_t[sample_idx], decode_count[sample_idx].item()
            )

    assert len(finished_samples) == batch_size
    assert len(finished_trajectories) == batch_size
    return finished_samples, finished_trajectories


def load_conversation_dataset(data_path, limit=None):
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.is_dir():
        json_files = sorted(path.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No *.json files found in {path}")
    else:
        json_files = [path]

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
            items.append((task_id, inst))
            if limit is not None and len(items) >= limit:
                return items

    return items


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


def batch_generate(
    model,
    tokenizer,
    prompts,
    device,
    *,
    mask_id,
    bd_size,
    small_block_size,
    max_new_tokens,
    threshold,
    use_block_cache,
):
    """返回 (completions, trajectories)，trajectories[i] 与 finished_samples[i] 等长的 decode 步数列表。"""
    input_id_list = []
    seq_lens = []
    max_len = 0
    min_len = 10**18

    for prompt in prompts:
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

    batched_input_ids = torch.cat(padded, dim=0).to(device)
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
            use_block_cache=use_block_cache,
            threshold=threshold,
            prompt_padded_len=max_len,
        )

    completions = []
    trajectories = []

    for batch_idx, prompt_len in enumerate(seq_lens):
        full_ids = finished_samples[batch_idx]
        traj = finished_trajectories[batch_idx]

        if full_ids.shape[0] != traj.shape[0]:
            raise RuntimeError(
                f"trajectory length {traj.shape[0]} != sample length {full_ids.shape[0]} "
                f"for batch index {batch_idx}"
            )

        new_ids = full_ids[prompt_len:]
        new_ids = new_ids[new_ids != mask_id]
        text = tokenizer.decode(new_ids, skip_special_tokens=True)
        completions.append(text)
        trajectories.append(traj.detach().cpu().tolist())

    return completions, trajectories




def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", required=True)
    parser.add_argument(
        "--dataset_path",
        default=DEFAULT_DATASET_PATH,
        help="Conversation JSON file or directory (same format as finetune dataset_path)",
    )
    parser.add_argument("--output", required=True)
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
    parser.add_argument("--threshold", type=float, default=torch.inf) #这里强制每一步解码一个位置
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

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

    print("Loading model:", args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        local_files_only=args.local_files_only,
    ).to(device)

    model.eval()

    model.mdm_sample = types.MethodType(batch_sample, model)

    items = load_conversation_dataset(args.dataset_path, limit=args.limit)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {args.dataset_path}")
    print(f"Num problems: {len(items)}")
    print(f"Output: {output_path}")

    rows = []

    for start in tqdm(range(0, len(items), args.batch_size), desc="Generating"):
        batch_items = items[start : start + args.batch_size]

        task_ids = [x[0] for x in batch_items]
        instances_batch = [x[1] for x in batch_items]

        prompts = [
            build_generation_prompt(instance, tokenizer, conversation_template)
            for instance in instances_batch
        ]

        completions, trajectories = batch_generate(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            device=device,
            mask_id=args.mask_id,
            bd_size=args.bd_size,
            small_block_size=args.small_block_size,
            max_new_tokens=args.max_new_tokens,
            threshold=args.threshold,
            use_block_cache=args.use_block_cache,
        )

        for task_id, instance, completion, trajectory in zip(
            task_ids, instances_batch, completions, trajectories
        ):
            rows.append(
                {
                    "conversation_id": task_id,
                    "completion": completion,
                    "trajectory": trajectory,
                    "reference": instance["messages"][-1]["content"]
                    if instance["messages"] and instance["messages"][-1]["role"] == "assistant"
                    else "",
                }
            )

    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Done. Wrote {len(rows)} samples to {output_path}")


if __name__ == "__main__":
    main()