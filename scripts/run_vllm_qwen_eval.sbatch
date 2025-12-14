#!/bin/bash
#SBATCH --job-name=gen-qwen14b
#SBATCH --output=logs/gen_qwen14b-%j.out
#SBATCH --error=logs/gen_qwen14b-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

## Activate virtualenv: prefer central venv, fallback to repository .venv
if [ -f "$HOME/Kinetics/.venv/bin/activate" ]; then
  source "$HOME/Kinetics/.venv/bin/activate"
elif [ -f ".venv/bin/activate" ]; then
  source ".venv/bin/activate"
fi

# Ensure CUDA tools/libraries are on PATH/LD_LIBRARY_PATH so nvidia-smi and CUDA are available
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

nvidia-smi

python scripts/eval.py \
  --input data/aime24_3.jsonl \
  --output results/aime24/dense/aime24_qwen3-4b_dense_3.jsonl \
  --model Qwen/Qwen3-14B \
  --max-model-len 32768 \
  --temperature 0.2 \
  --top-p 1.0 \
  --top-k 0 \
  --repetition-penalty 1.05 \
  --limit 5
# python scripts/eval_I_cpu.py