import json
import re
from pathlib import Path

# 配置你的数据路径
INPUT_DIR = Path("/home/u-shengbf/Codes/Fast-dLLM/v2/data/Llama-Nemotron-code-v1/use")

def analyze_length_distribution():
    # 正则表达式：用于匹配并剔除思维链
    think_pattern = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
    
    # 用于存储长度的列表
    original_lengths = []
    cleaned_lengths = []
    
    total_files = 0
    total_samples = 0
    with_think_count = 0
    
    print(f"正在扫描目录并计算长度分布: {INPUT_DIR}")
    
    json_files = list(INPUT_DIR.glob("*.json"))
    
    for file_path in json_files:
        total_files += 1
        try:
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                instances = data.get("instances", []) if isinstance(data, dict) else data
                
                for inst in instances:
                    total_samples += 1
                    
                    # 遍历查找 assistant 的回复
                    messages = inst.get("messages", [])
                    for msg in messages:
                        if msg.get("role") == "assistant":
                            content = msg.get("content", "")
                            if not content or not isinstance(content, str):
                                continue
                            
                            # 1. 记录原始长度
                            orig_len = len(content)
                            original_lengths.append(orig_len)
                            
                            # 2. 记录去除 <think> 后的长度
                            # sub() 替换为空字符串，即剔除
                            cleaned_content = think_pattern.sub("", content)
                            clean_len = len(cleaned_content)
                            cleaned_lengths.append(clean_len)
                            
                            # 统计是否含有 think 标签
                            if orig_len != clean_len:
                                with_think_count += 1
                            break # 处理完该样本的 assistant 回复即可
                            
        except Exception as e:
            print(f"[错误] 读取文件 {file_path.name} 失败: {e}")
            continue

    # 输出统计报告
    if total_samples > 0 and original_lengths:
        avg_orig = sum(original_lengths) / len(original_lengths)
        avg_clean = sum(cleaned_lengths) / len(cleaned_lengths)
        total_orig = sum(original_lengths)
        total_clean = sum(cleaned_lengths)
        reduction_rate = (1 - (total_clean / total_orig)) * 100

        print("\n" + "="*40)
        print("📊 文本长度分布对比报告:")
        print(f"扫描样本总数: {total_samples}")
        print(f"含有思维链的样本: {with_think_count} ({with_think_count/total_samples:.2%})")
        print("-" * 40)
        print(f"平均原始长度: {avg_orig:.2f} 字符")
        print(f"平均清洗后长度: {avg_clean:.2f} 字符")
        print("-" * 40)
        print(f"总体字符缩减量: {total_orig - total_clean} 字符")
        print(f"总体压缩比例: {reduction_rate:.2f}%")
        print("="*40 + "\n")
    else:
        print("未发现有效样本数据。")

if __name__ == "__main__":
    analyze_length_distribution()