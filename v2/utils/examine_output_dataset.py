# 用于检验convert转换结果正确性
import json
from pathlib import Path

path = Path("/home/u-shengbf/Codes/Fast-dLLM/v2/data/Llama-Nemotron-code-v1.1/train_conversation")

max_preview = 3
total_attempted = 0  # 尝试总数，用来强制退出，防止死循环

print("正在精准流式解析前几条数据...")

inside_instance = False
instance_lines = []
brace_depth = 0  # 大括号嵌套深度计数器

with path.open("r", encoding="utf-8") as f:
    for line_num, line in enumerate(f, 1):
        # 提取外层的 type
        if '"type":' in line and not inside_instance:
            try:
                print("type:", line.split('"type":')[1].strip().strip('",'))
            except Exception:
                pass

        # 发现新实例的特征行
        if '"conversation_id":' in line and not inside_instance:
            inside_instance = True
            instance_lines = ["{ \n", line]  # 手动补上左大括号，并加入当前行
            # 初始深度为 1，并加上当前行可能包含的括号
            brace_depth = 1 + line.count("{") - line.count("}")
            continue

        if inside_instance:
            instance_lines.append(line)
            # 核心：根据当前行的大括号数量动态增减深度
            brace_depth += line.count("{")
            brace_depth -= line.count("}")

            # 当深度降回 0 时，说明这才是真正属于该 instance 自身关闭的大括号
            if brace_depth <= 0:
                inside_instance = False
                total_attempted += 1
                
                # 组合成完整的 json 字符串
                full_json_str = "".join(instance_lines).strip()
                if full_json_str.endswith(","):
                    full_json_str = full_json_str[:-1]
                
                try:
                    ex = json.loads(full_json_str)
                    print("=" * 80)
                    print(f"【成功流式解析第 {total_attempted} 个样本】")
                    print("conversation_id:", ex.get("conversation_id"))
                    print("system:", ex.get("system", "")[:100])
                    # 只打印前 2 轮对话防止刷屏
                    for m in ex.get("messages", [])[:2]: 
                        print(f"  [{m['role']}]: {m['content'][:150]}...")
                except json.JSONDecodeError:
                    print(f"\n[错误] 第 {line_num} 行附近截取的块解析失败。")
                    print("截取的文本前 200 字符：\n", full_json_str[:200])
                    print("截取的文本后 100 字符：\n", full_json_str[-100:])
                
                # 重置状态
                instance_lines = []
                brace_depth = 0
                
                # 保险栓：管它成功还是失败，达到指定次数必须断开，绝不连读
                if total_attempted >= max_preview:
                    break

print("\n预览结束。")