import json
from pathlib import Path

# 数据集所在目录
DIR_PATH = Path("/home/u-shengbf/Codes/Fast-dLLM/v2/data/Llama-Nemotron-code-v1.1/use")

# 获取前 10 个 json 文件
json_files = sorted(DIR_PATH.glob("train-*.json"))[:10]

if not json_files:
    print(f"[错误] 未在路径下找到 train-*.json 文件，请检查路径：\n{DIR_PATH}")
    exit()

print(f"正在深度分析前 {len(json_files)} 个文件的回答（assistant）长度...\n")

# 用于全局统计
global_lengths = []

# 打印表格表头
print(f"{'文件名':<18} | {'总样本数':<8} | {'平均长度(字)':<10} | {'最大长度':<8} | {'最小长度':<8}")
print("-" * 65)

for file_path in json_files:
    try:
        # 核心：整包读取单行大 JSON，效率最高
        with file_path.open("r", encoding="utf-8") as f:
            data = json.loads(f.read())
        
        instances = data.get("instances", [])
        file_lengths = []
        
        # 提取当前文件中所有 assistant 的回答长度
        for inst in instances:
            for msg in inst.get("messages", []):
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    file_lengths.append(len(content))
        
        if file_lengths:
            avg_len = sum(file_lengths) / len(file_lengths)
            max_len = max(file_lengths)
            min_len = min(file_lengths)
            global_lengths.extend(file_lengths)
            
            print(f"{file_path.name:<18} | {len(file_lengths):<8} | {int(avg_len):<12} | {max_len:<8} | {min_len:<8}")
        else:
            print(f"{file_path.name:<18} | {len(instances):<8} | 0            | 0        | 0")
            
    except Exception as e:
        print(f"读取文件 {file_path.name} 失败，错误原因: {e}")

# 输出全局区间分布统计
if global_lengths:
    total_count = len(global_lengths)
    print("\n" + "=" * 60)
    print("📊 【前 10 个文件全量长度分布报告】")
    print("=" * 60)
    print(f" 🔹 总计分析样本数 : {total_count} 条")
    print(f" 🔹 全局平均长度   : {int(sum(global_lengths) / total_count)} 字符")
    print(f" 🔹 全局最大长度   : {max(global_lengths)} 字符")
    print(f" 🔹 全局最小长度   : {min(global_lengths)} 字符")
    print("-" * 60)
    
    # 区间递进统计
    bin_1k = sum(1 for l in global_lengths if l <= 1000)
    bin_5k = sum(1 for l in global_lengths if 1000 < l <= 5000)
    bin_10k = sum(1 for l in global_lengths if 5000 < l <= 10000)
    bin_huge = sum(1 for l in global_lengths if l > 10000)
    
    print(" 💡 长度区间详细占比：")
    print(f"   - 短文本 (<= 1k 字符)     : {bin_1k:<6} 条 ({bin_1k / total_count * 100:.2f}%)")
    print(f"   - 中等文本 (1k ~ 5k 字符)  : {bin_5k:<6} 条 ({bin_5k / total_count * 100:.2f}%)")
    print(f"   - 长文本 (5k ~ 10k 字符)   : {bin_10k:<6} 条 ({bin_10k / total_count * 100:.2f}%)")
    print(f"   - 超长文本 (> 10k 字符)    : {bin_huge:<6} 条 ({bin_huge / total_count * 100:.2f}%)")
    print("=" * 60)