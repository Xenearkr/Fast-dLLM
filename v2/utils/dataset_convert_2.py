import json
from pathlib import Path
from typing import Optional, Tuple

def extract_last_python_block(content: str) -> Optional[str]:
    """
    提取最后一个 ```python 代码块，从该标记开始直到内容末尾。
    不再查找闭合的 ```，也不再补充结尾的 ```。
    """
    last_idx = content.rfind("```python")
    if last_idx == -1:
        return None
    # 直接返回从 ```python 到末尾的全部内容
    return content[last_idx:]

'''
def extract_last_python_block(content: str) -> Optional[str]:
    """提取最后一个 ```python 代码块，并确保以 ``` 结尾。"""
    last_idx = content.rfind("```python")
    if last_idx == -1:
        return None
    sub = content[last_idx:]
    next_triple = sub.find("```", 9)
    if next_triple != -1:
        # 包含结束的 ```
        return sub[:next_triple + 3]
    else:
        return sub + "\n```"
'''

def preprocess_instance(instance: dict) -> Optional[dict]:
    """处理单个样本，保留非 assistant 消息，过滤/替换 assistant 代码块。"""
    original_messages = instance.get("messages", [])
    new_messages = []
    for msg in original_messages:
        role = msg.get("role")
        if role != "assistant":
            new_messages.append(msg.copy())
            continue
        content = msg.get("content", "")
        code_block = extract_last_python_block(content)
        if code_block is not None:
            new_msg = msg.copy()
            new_msg["content"] = code_block
            new_messages.append(new_msg)
    if not any(m.get("role") == "assistant" for m in new_messages):
        return None
    instance["messages"] = new_messages
    return instance

def preprocess_file(input_path: Path, output_path: Path) -> Tuple[int, int]:
    """预处理单个 JSON 文件，保存结果。返回 (原始样本数, 保留样本数)。"""
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    instances = data.get("instances", [])
    original_cnt = len(instances)
    new_instances = []
    for inst in instances:
        new_inst = preprocess_instance(inst)
        if new_inst is not None:
            new_instances.append(new_inst)
    data["instances"] = new_instances
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return original_cnt, len(new_instances)

def preprocess_all_files(input_dir: Path, output_dir: Path, file_pattern: str = "train-*.json", suffix: str = "-processed") -> None:
    """批量预处理所有匹配文件，输出到 output_dir，文件名添加 suffix。"""
    json_files = sorted(input_dir.glob(file_pattern))
    if not json_files:
        print(f"[错误] 在 {input_dir} 下未找到匹配 {file_pattern} 的文件")
        return
    print(f"开始预处理，共找到 {len(json_files)} 个文件...")
    total_orig = 0
    total_kept = 0
    for idx, in_path in enumerate(json_files, 1):
        stem = in_path.stem
        out_name = f"{stem}{suffix}.json"
        out_path = output_dir / out_name
        orig, kept = preprocess_file(in_path, out_path)
        discarded = orig - kept
        total_orig += orig
        total_kept += kept
        print(f"[{idx}/{len(json_files)}] {in_path.name} -> {out_name} | 原始: {orig} | 保留: {kept} | 丢弃: {discarded}")
    print(f"\n===== 汇总统计 =====")
    print(f"总原始样本数: {total_orig}")
    print(f"总保留样本数: {total_kept}")
    print(f"总丢弃样本数: {total_orig - total_kept}")
    print("所有文件预处理完成。")

def compute_average_code_length(processed_dir: Path, num_samples: int = 100, suffix: str = "-processed") -> float:
    """计算预处理后前 num_samples 个样本的代码块平均长度。"""
    processed_files = sorted(processed_dir.glob(f"*{suffix}.json"))
    if not processed_files:
        print(f"[错误] 在 {processed_dir} 下未找到任何以 {suffix}.json 结尾的文件")
        return 0.0
    lengths = []
    remaining = num_samples
    for file_path in processed_files:
        if remaining <= 0:
            break
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        instances = data.get("instances", [])[:remaining]
        for inst in instances:
            last_assistant_content = None
            for msg in reversed(inst.get("messages", [])):
                if msg.get("role") == "assistant":
                    last_assistant_content = msg.get("content", "")
                    break
            if last_assistant_content is not None:
                lengths.append(len(last_assistant_content))
        remaining -= len(instances)
    if not lengths:
        print("[警告] 未找到任何含有 assistant 回复的样本")
        return 0.0
    return sum(lengths) / len(lengths)

def output_first_n_responses(processed_dir: Path, n: int, output_txt: Optional[Path] = None, suffix: str = "-processed") -> None:
    """输出前 n 个样本的回复内容（代码块）。"""
    processed_files = sorted(processed_dir.glob(f"*{suffix}.json"))
    if not processed_files:
        print(f"[错误] 在 {processed_dir} 下未找到任何以 {suffix}.json 结尾的文件")
        return
    collected = []
    remaining = n
    for file_path in processed_files:
        if remaining <= 0:
            break
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        instances = data.get("instances", [])[:remaining]
        for idx, inst in enumerate(instances):
            last_assistant_content = None
            for msg in reversed(inst.get("messages", [])):
                if msg.get("role") == "assistant":
                    last_assistant_content = msg.get("content", "")
                    break
            if last_assistant_content is not None:
                collected.append({
                    "file": file_path.name,
                    "index": idx,
                    "content": last_assistant_content
                })
        remaining -= len(instances)

    print(f"\n📝 前 {len(collected)} 个样本的回复内容（代码块，以 ``` 结尾）如下：\n")
    for i, item in enumerate(collected, 1):
        print(f"===== 样本 {i} (文件: {item['file']}, 文件内序号: {item['index']}) =====")
        print(item["content"])
        print("\n" + "-" * 60 + "\n")

    if output_txt:
        with output_txt.open("w", encoding="utf-8") as f:
            f.write(f"前 {len(collected)} 个样本的回复内容\n\n")
            for i, item in enumerate(collected, 1):
                f.write(f"===== 样本 {i} (文件: {item['file']}, 文件内序号: {item['index']}) =====\n")
                f.write(item["content"])
                f.write("\n\n" + "-" * 60 + "\n\n")
        print(f"已同时将结果保存至: {output_txt}")

if __name__ == "__main__":
    # 原始数据路径（输入）
    INPUT_DIR = Path("/home/u-shengbf/Codes/Fast-dLLM/v2/data/Llama-Nemotron-code-v1.1/train_conversation")
    # 处理后输出路径
    OUTPUT_DIR = Path("/home/u-shengbf/Codes/Fast-dLLM/v2/data/Llama-Nemotron-code-v1.1/clear")

    # 1. 预处理所有文件（处理后的回复均以 ``` 结尾）
    preprocess_all_files(INPUT_DIR, OUTPUT_DIR, file_pattern="train-*.json", suffix="-processed")

    # 2. 统计前 100 个样本的平均代码块长度
    NUM_SAMPLES = 10000
    avg_len = compute_average_code_length(OUTPUT_DIR, num_samples=NUM_SAMPLES, suffix="-processed")
    print(f"\n前 {NUM_SAMPLES} 个样本中，截取后的代码块平均字符数为: {avg_len:.2f}")

    # 3. 输出前 5 个回复（可修改 y 值）
    # OUTPUT_Y = 5
    # output_first_n_responses(OUTPUT_DIR, n=OUTPUT_Y, output_txt=OUTPUT_DIR / "sample_responses.txt")