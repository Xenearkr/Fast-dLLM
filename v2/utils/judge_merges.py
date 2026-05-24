# 用于细致的检查tokenizer的区别
import json
from pathlib import Path
from collections import Counter

qwen_path = Path("/home/u-shengbf/Codes/Model-Qwen-2.5-7B/tokenizer.json")
fast_path = Path("/home/u-shengbf/Codes/Model-Fast-v2/tokenizer.json")

qwen_tok = json.loads(qwen_path.read_text(encoding="utf-8"))
fast_tok = json.loads(fast_path.read_text(encoding="utf-8"))

qwen_merges = qwen_tok["model"]["merges"]
fast_merges = fast_tok["model"]["merges"]


def normalize_merge(x):
    """
    tokenizer.json 里的 merges 可能是:
      "Ġ t"
    或:
      ["Ġ", "t"]

    这里统一成 tuple，方便比较。
    """
    if isinstance(x, str):
        parts = x.split()
        if len(parts) != 2:
            return (x,)
        return tuple(parts)
    if isinstance(x, list):
        return tuple(x)
    return tuple(x)


qwen_norm = [normalize_merge(x) for x in qwen_merges]
fast_norm = [normalize_merge(x) for x in fast_merges]

print("Qwen merges length:", len(qwen_norm))
print("Fast merges length:", len(fast_norm))
print("Same length:", len(qwen_norm) == len(fast_norm))

# 1. 找出相同 index 上不同的 merge rule
first_diffs = []

min_len = min(len(qwen_norm), len(fast_norm))

for i in range(min_len):
    if qwen_norm[i] != fast_norm[i]:
        first_diffs.append((i, qwen_norm[i], fast_norm[i]))
        if len(first_diffs) >= 50:
            break

print("\nFirst index-wise differences:")
if not first_diffs:
    print("No index-wise differences in shared range.")
else:
    for i, q_merge, f_merge in first_diffs:
        print(f"[{i}]")
        print("  Qwen:", q_merge)
        print("  Fast:", f_merge)

# 2. 检查尾部多出来的部分
if len(qwen_norm) > len(fast_norm):
    print("\nQwen has extra trailing merges:")
    for i, merge in enumerate(qwen_norm[len(fast_norm):len(fast_norm) + 50], start=len(fast_norm)):
        print(f"[{i}] Qwen:", merge)

elif len(fast_norm) > len(qwen_norm):
    print("\nFast has extra trailing merges:")
    for i, merge in enumerate(fast_norm[len(qwen_norm):len(qwen_norm) + 50], start=len(qwen_norm)):
        print(f"[{i}] Fast:", merge)

# 3. set 级别差异：只存在于一方的 merge rule
qwen_set = set(qwen_norm)
fast_set = set(fast_norm)

only_in_qwen = sorted(qwen_set - fast_set)
only_in_fast = sorted(fast_set - qwen_set)

print("\nOnly in Qwen count:", len(only_in_qwen))
for x in only_in_qwen[:50]:
    print("  ", x)

print("\nOnly in Fast count:", len(only_in_fast))
for x in only_in_fast[:50]:
    print("  ", x)

# 4. Counter 级别差异：检测重复 merge rule 的数量差异
qwen_counter = Counter(qwen_norm)
fast_counter = Counter(fast_norm)

counter_diff = []

all_keys = set(qwen_counter) | set(fast_counter)
for k in all_keys:
    if qwen_counter[k] != fast_counter[k]:
        counter_diff.append((k, qwen_counter[k], fast_counter[k]))

print("\nMerge count differences:", len(counter_diff))
for merge, q_count, f_count in counter_diff[:50]:
    print(f"  {merge}: Qwen={q_count}, Fast={f_count}")

# 5. 如果 set 完全一样，但 index-wise 不一样，说明只是顺序不同
same_multiset = qwen_counter == fast_counter
same_order = qwen_norm == fast_norm

print("\nSummary:")
print("Same ordered merges:", same_order)
print("Same merge multiset:", same_multiset)

if same_multiset and not same_order:
    print("Conclusion: same merge rules, but order/rank differs.")
elif not same_multiset:
    print("Conclusion: merge rule contents differ.")
else:
    print("Conclusion: merges are identical.")