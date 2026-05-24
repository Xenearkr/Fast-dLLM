mkdir -p evalplus_results

python generate_evalplus_samples.py \
  --model_path "Efficient-Large-Model/Fast_dLLM_v2_7B" \
  --dataset mbpp \
  --output evalplus_results/mbpp_fast.jsonl \
  --prompt_mode raw \
  --batch_size 1 \
  --max_new_tokens 512 \
  --mask_id 151665 \
  --bd_size 32 \
  --small_block_size 8 \
  --threshold 1.0 \
  --local_files_only
  # --limit 5 \