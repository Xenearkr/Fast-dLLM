import json
from pathlib import Path

qwen_tok = json.loads(Path("/home/u-shengbf/Codes/Model-Qwen-3-8B/tokenizer.json").read_text())
fast_tok = json.loads(Path("/home/u-shengbf/Codes/Model-Fast-v2/tokenizer.json").read_text())

# 1. 底层 BPE 必须一致
assert qwen_tok["model"]["type"] == fast_tok["model"]["type"]
assert qwen_tok["model"]["vocab"] == fast_tok["model"]["vocab"]
# print(qwen_tok["model"]["merges"])
assert qwen_tok["model"]["merges"] == fast_tok["model"]["merges"]

# 2. pipeline 关键组件建议一致
for key in ["normalizer", "pre_tokenizer", "decoder", "post_processor"]:
    print(key, qwen_tok.get(key) == fast_tok.get(key))

# 3. added tokens 看差异
qwen_added = {x["content"]: x for x in qwen_tok.get("added_tokens", [])}
fast_added = {x["content"]: x for x in fast_tok.get("added_tokens", [])}

print("Only in Fast:", sorted(set(fast_added) - set(qwen_added)))
print("Only in Qwen:", sorted(set(qwen_added) - set(fast_added)))

assert fast_added["|<MASK>|"]["id"] == 151665