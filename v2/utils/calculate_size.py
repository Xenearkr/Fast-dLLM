import json
import time
from pathlib import Path
from transformers import AutoTokenizer
import numpy as np  # 引入 numpy 方便计算百分位数

# ================= 配置参数 =================
# 预处理后的数据目录（use 文件夹）
DATA_DIR = Path("/home/u-shengbf/Codes/Fast-dLLM/v2/data/Llama-Nemotron-code-v1.1/clear")
# Tokenizer 路径
TOKENIZER_PATH = "/home/u-shengbf/Codes/Fast-dLLM/v2/base_models/Model-Qwen-3-8B"

# 采样数量（用于估算长度分布）
MAX_EXAMPLES = 10000
# 批处理大小
BATCH_SIZE = 64
# ===========================================

def get_all_processed_files(data_dir: Path, suffix: str = ".json"):
    """返回所有预处理后的 JSON 文件列表（按文件名排序）"""
    return sorted(data_dir.glob(f"*{suffix}"))

def count_total_samples(data_dir: Path, suffix: str = ".json") -> int:
    """统计所有预处理文件中的样本总数"""
    total = 0
    for file_path in get_all_processed_files(data_dir, suffix):
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        total += len(data.get("instances", []))
    return total

def extract_full_conversation_text(instance: dict) -> str:
    """
    还原 LMFlow 训练时的真实状态，将包含 system, user, assistant 的
    整条对话按照标准的 ChatML / 模板格式拼接成一个完整的字符串。
    """
    messages = instance.get("messages", [])
    full_text = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        # 模拟 Qwen / ChatML 风格的特殊 Token 拼接
        full_text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    return full_text

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

    # 3. 逐文件读取，收集全量样本并批量 tokenize
    processed_files = get_all_processed_files(DATA_DIR)
    print(f"找到 {len(processed_files)} 个预处理文件，开始采样（目标 {MAX_EXAMPLES} 个样本）...\n")

    all_lengths = []  # 用于保存所有样本的 Token 长度，方便后续计算分布
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
                
            # 核心修改：提取拼接了 User 代码和 System 模板后的全量文本
            full_conversation = extract_full_conversation_text(inst)
            if not full_conversation.strip():
                continue
                
            batch_texts.append(full_conversation)
            total_chars += len(full_conversation)
            count += 1
            
            # 达到 batch size 或 达到最大样本数时进行 tokenize
            if len(batch_texts) >= BATCH_SIZE or count >= MAX_EXAMPLES:
                lens = tokenize_batch(tokenizer, batch_texts)
                all_lengths.extend(lens)  # 记录长度
                batch_texts = []
                
                # 进度报告
                now = time.time()
                if now - last_report_time >= 5 or count >= MAX_EXAMPLES:
                    avg_tok = sum(all_lengths) / len(all_lengths) if all_lengths else 0
                    avg_char = total_chars / count if count > 0 else 0
                    est_total_tokens = avg_tok * total_samples
                    print(f"已处理 {count}/{MAX_EXAMPLES} 个样本 | "
                          f"当前平均 token: {avg_tok:.2f} | "
                          f"预估总 token 数: {est_total_tokens/1e9:.3f}B | "
                          f"耗时: {now-start_time:.1f}s")
                    last_report_time = now

    # 处理剩余的不足 batch 的数据
    if batch_texts:
        lens = tokenize_batch(tokenizer, batch_texts)
        all_lengths.extend(lens)

    if count == 0:
        print("错误：未提取到任何有效的文本。")
        return

    # 4. 计算高级分布指标
    all_lengths = np.array(all_lengths)
    total_tokens = np.sum(all_lengths)
    avg_tokens = np.mean(all_lengths)
    avg_chars = total_chars / count
    estimated_total_tokens = avg_tokens * total_samples
    
    # 长度区间比例统计
    samples_gt_384 = np.sum(all_lengths > 384)
    samples_gt_512 = np.sum(all_lengths > 512)
    samples_gt_1024 = np.sum(all_lengths > 1024)

    print("\n" + "=" * 60)
    print("📊 【训练样本真实 Token 长度分布报告】")
    print("=" * 60)
    print(f"采样样本数         : {count}")
    print(f"总样本数（全量）   : {total_samples}")
    print(f"平均字符/样本      : {avg_chars:.2f}")
    print(f"平均 Token/样本    : {avg_tokens:.2f}")
    print(f"预估总 Token 数    : {estimated_total_tokens:,.0f} ({estimated_total_tokens/1e9:.4f}B)")
    print("-" * 60)
    print("📈 【详细长度分位数 (Percentiles)】")
    print(f"  50% 的样本长度在  {np.percentile(all_lengths, 50):.0f}  Token 以内 (中位数)")
    print(f"  90% 的样本长度在  {np.percentile(all_lengths, 90):.0f}  Token 以内")
    print(f"  95% 的样本长度在  {np.percentile(all_lengths, 95):.0f}  Token 以内")
    print(f"  99% 的样本长度在  {np.percentile(all_lengths, 99):.0f}  Token 以内")
    print(f"  最大样本 Token 长度: {np.max(all_lengths)}")
    print("-" * 60)
    print("🚨 【超长截断/瓶颈预警风险】")
    print(f"  长度 > 384 的样本数: {samples_gt_384} (占比 {samples_gt_384 / count * 100:.2f}%)")
    print(f"  长度 > 512 的样本数: {samples_gt_512} (占比 {samples_gt_512 / count * 100:.2f}%)")
    print(f"  长度 > 1024 的样本数: {samples_gt_1024} (占比 {samples_gt_1024 / count * 100:.2f}%)")
    print("=" * 60)

if __name__ == "__main__":
    main()