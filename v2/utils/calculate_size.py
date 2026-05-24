# 取2000估算
import json
import time
from pathlib import Path
from transformers import AutoTokenizer


INPUT_PATH = Path(
    "/home/u-shengbf/Codes/Fast-dLLM/v2/data/"
    "Llama-Nemotron-code-v1.1/SFT/code/code_v1.1.jsonl"
)

# 改成你本地的 Qwen / Fast tokenizer 目录
TOKENIZER_PATH = "/home/u-shengbf/Codes/Fast-dLLM/v2/base_models/Model-Qwen-2.5-7B"

TOTAL_EXAMPLES = 492_606

# 先不要取太大，1000/2000 足够估一个量级
MAX_EXAMPLES = 2000

# 批量 tokenize，越大越快但越吃内存
BATCH_SIZE = 32

# 防止极端超长样本拖死估算。
# 设为 None 表示不截断文本字符。
# 如果还是慢，可以设成 20000 或 50000。
MAX_CHARS_PER_EXAMPLE = None


def extract_text(ex):
    parts = []

    system_prompt = ex.get("system_prompt")
    if system_prompt:
        parts.append(str(system_prompt))

    input_field = ex.get("input", "")

    if isinstance(input_field, str):
        parts.append(input_field)

    elif isinstance(input_field, list):
        for m in input_field:
            if isinstance(m, dict):
                content = m.get("content", "")
                if content:
                    parts.append(str(content))

    else:
        parts.append(str(input_field))

    output = ex.get("output", "")
    if output:
        parts.append(str(output))

    text = "\n".join(parts)

    if MAX_CHARS_PER_EXAMPLE is not None:
        text = text[:MAX_CHARS_PER_EXAMPLE]

    return text


def tokenize_batch(tok, texts):
    encoded = tok(
        texts,
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )
    return [len(ids) for ids in encoded["input_ids"]]


def main():
    assert INPUT_PATH.exists(), f"Input file not found: {INPUT_PATH}"

    print("Loading tokenizer...")
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=True,
    )

    print(f"Tokenizer loaded in {time.time() - t0:.2f}s")
    print("Tokenizer class:", type(tok))
    print("Input:", INPUT_PATH)
    print("MAX_EXAMPLES:", MAX_EXAMPLES)
    print("BATCH_SIZE:", BATCH_SIZE)

    total_tokens = 0
    total_chars = 0
    count = 0
    skipped = 0

    batch = []

    t_start = time.time()
    t_last = time.time()

    with INPUT_PATH.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if count >= MAX_EXAMPLES:
                break

            line = line.strip()
            if not line:
                continue

            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            text = extract_text(ex)
            if not text:
                skipped += 1
                continue

            batch.append(text)
            total_chars += len(text)

            if len(batch) >= BATCH_SIZE:
                lens = tokenize_batch(tok, batch)
                total_tokens += sum(lens)
                count += len(batch)
                batch = []

                now = time.time()
                if now - t_last >= 5:
                    avg_tok = total_tokens / max(count, 1)
                    avg_chars = total_chars / max(count, 1)
                    est_total = avg_tok * TOTAL_EXAMPLES

                    print(
                        f"processed={count}, "
                        f"avg_tokens={avg_tok:.1f}, "
                        f"avg_chars={avg_chars:.1f}, "
                        f"est_total_tokens={est_total/1e9:.3f}B, "
                        f"elapsed={now - t_start:.1f}s"
                    )
                    t_last = now

    if batch:
        lens = tokenize_batch(tok, batch)
        total_tokens += sum(lens)
        count += len(batch)

    avg_tokens = total_tokens / count
    avg_chars = total_chars / count
    estimated_total = avg_tokens * TOTAL_EXAMPLES
    ratio_to_1b = estimated_total / 1_000_000_000
    estimated_blocks_512 = estimated_total / 512

    print("\n==== Result ====")
    print("sampled examples:", count)
    print("skipped:", skipped)
    print("avg chars/example:", round(avg_chars, 2))
    print("avg tokens/example:", round(avg_tokens, 2))
    print("estimated total tokens:", round(estimated_total))
    print("estimated total tokens B:", round(estimated_total / 1e9, 4))
    print("ratio to 1B:", round(ratio_to_1b, 4))
    print("estimated 512-token blocks:", round(estimated_blocks_512))

    required_avg_for_1b = 1_000_000_000 / TOTAL_EXAMPLES
    print("required avg tokens/example for 1B:", round(required_avg_for_1b, 2))


if __name__ == "__main__":
    main()