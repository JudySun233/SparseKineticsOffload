# an example config
python scripts/eval.py \
  --input data/aime24.jsonl \
  --output results/aime24/dense/aime24_qwen3-8b_dense.jsonl \
  --model Qwen/Qwen3-8B \
  --max-model-len 4096 \
  --enable-lmcache \
  --cpu-budget-gb 4.0 \
  --gpu-memory-utilization 0.8 \
  --limit 4
