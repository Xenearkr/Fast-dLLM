import json
import time
from pathlib import Path
from transformers import AutoTokenizer

# ================= 配置参数 =================
# 预处理后的数据目录（use 文件夹）
DATA_DIR = Path("/home/u-shengbf/Codes/Fast-dLLM/v2/data/Llama-Nemotron-code-v1.1/clear")
# Tokenizer 路径
TOKENIZER_PATH = "/home/u-shengbf/Codes/Fast-dLLM/v2/base_models/Model-Qwen-3-8B"

# 采样数量（用于估算）
MAX_EXAMPLES = 20000
# 批处理大小
BATCH_SIZE = 32
# 是否对单个回复截断（防止过长拖慢速度），None 表示不截断
MAX_CHARS_PER_RESPONSE = None
# ===========================================

def get_all_processed_files(data_dir: Path, suffix: str = "-processed.json"):
    """返回所有预处理后的 JSON 文件列表（按文件名排序）"""
    return sorted(data_dir.glob(f"*{suffix}"))

def count_total_samples(data_dir: Path, suffix: str = "-processed.json") -> int:
    """统计所有预处理文件中的样本总数（每个文件中的 instances 数量之和）"""
    total = 0
    for file_path in get_all_processed_files(data_dir, suffix):
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        total += len(data.get("instances", []))
    return total

def extract_assistant_response(instance: dict) -> str:
    """
    从单个样本中提取最后一条 assistant 消息的 content。
    预处理后的样本中，assistant 消息已经是截取好的 ```python 代码块（且以 ``` 结尾）。
    """
    messages = instance.get("messages", [])
    # 倒序查找最后一条 assistant 消息
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if MAX_CHARS_PER_RESPONSE is not None:
                content = content[:MAX_CHARS_PER_RESPONSE]
            return content
    return ""  # 理论上每个保留的样本都有 assistant，但防御性返回空字符串

def tokenize_batch(tokenizer, texts):
    """批量 tokenize，返回每个文本的 token 长度列表"""
    encoded = tokenizer(
        texts,
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )
    return [len(ids) for ids in encoded["input_ids"]]

def main():
    print(f"数据目录: {DATA_DIR}")
    print(f"Tokenizer 路径: {TOKENIZER_PATH}")

    # 1. 统计总样本数
    print("正在统计总样本数...")
    total_samples = count_total_samples(DATA_DIR)
    print(f"总样本数: {total_samples}")

    if total_samples == 0:
        print("错误：没有找到任何样本，请检查预处理是否完成。")
        return

    # 2. 加载 tokenizer
    print("加载 tokenizer...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=True,
    )
    print(f"Tokenizer 加载完成，耗时 {time.time()-t0:.2f}s")
    print(f"Tokenizer 类型: {type(tokenizer)}")

    # 3. 逐文件读取，收集样本回复并批量 tokenize
    processed_files = get_all_processed_files(DATA_DIR)
    print(f"找到 {len(processed_files)} 个预处理文件，开始采样（目标 {MAX_EXAMPLES} 个样本）...\n")

    total_tokens = 0
    total_chars = 0
    count = 0
    batch_texts = []
    start_time = time.time()
    last_report_time = start_time

    for file_path in processed_files:
        if count >= MAX_EXAMPLES:
            break
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        instances = data.get("instances", [])
        for inst in instances:
            if count >= MAX_EXAMPLES:
                break
            response = extract_assistant_response(inst)
            if not response:
                continue
            batch_texts.append(response)
            total_chars += len(response)
            count += 1
            # 达到 batch size 或 达到最大样本数时进行 tokenize
            if len(batch_texts) >= BATCH_SIZE or count >= MAX_EXAMPLES:
                lens = tokenize_batch(tokenizer, batch_texts)
                total_tokens += sum(lens)
                batch_texts = []
                # 进度报告
                now = time.time()
                if now - last_report_time >= 5 or count >= MAX_EXAMPLES:
                    avg_tok = total_tokens / count if count > 0 else 0
                    avg_char = total_chars / count if count > 0 else 0
                    est_total_tokens = avg_tok * total_samples
                    print(f"已处理 {count}/{MAX_EXAMPLES} 个样本 | "
                          f"平均 token/样本: {avg_tok:.2f} | "
                          f"平均字符/样本: {avg_char:.2f} | "
                          f"预估总 token 数: {est_total_tokens/1e9:.3f}B | "
                          f"耗时: {now-start_time:.1f}s")
                    last_report_time = now

    # 处理剩余的不足 batch 的数据
    if batch_texts:
        lens = tokenize_batch(tokenizer, batch_texts)
        total_tokens += sum(lens)

    if count == 0:
        print("错误：未提取到任何有效的 assistant 回复。")
        return

    # 4. 最终统计
    avg_tokens = total_tokens / count
    avg_chars = total_chars / count
    estimated_total_tokens = avg_tokens * total_samples
    ratio_to_1b = estimated_total_tokens / 1_000_000_000
    estimated_blocks_512 = estimated_total_tokens / 512

    print("\n" + "=" * 60)
    print("📊 【统计报告：预处理后回复的 Token 数量】")
    print("=" * 60)
    print(f"采样样本数        : {count}")
    print(f"总样本数（全量）  : {total_samples}")
    print(f"平均字符/样本     : {avg_chars:.2f}")
    print(f"平均 token/样本   : {avg_tokens:.2f}")
    print(f"预估总 token 数   : {estimated_total_tokens:,.0f} ({estimated_total_tokens/1e9:.4f}B)")
    print(f"相当于 1B 的倍数  : {ratio_to_1b:.4f}")
    print(f"预估 512-token 块 : {estimated_blocks_512:,.0f}")
    print(f"达到 1B 所需的平均 token/样本: {1_000_000_000 / total_samples:.2f}")

if __name__ == "__main__":
    main()