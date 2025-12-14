#!/bin/bash
#SBATCH --job-name=gen-qwen14b
#SBATCH --output=logs/gen_qwen14b-%A_%a.out
#SBATCH --error=logs/gen_qwen14b-%A_%a.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --array=0-5

if [ -f "$HOME/Kinetics/.venv/bin/activate" ]; then
  source "$HOME/Kinetics/.venv/bin/activate"
elif [ -f ".venv/bin/activate" ]; then
  source ".venv/bin/activate"
fi
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

INPUT=data/aime24_${SLURM_ARRAY_TASK_ID}.jsonl
OUT=results/aime24/dense/aime24_qwen3-14b_dense_${SLURM_ARRAY_TASK_ID}.jsonl

python scripts/eval.py \
  --input "$INPUT" \
  --output "$OUT" \
  --model Qwen/Qwen3-14B \
  --max-model-len 32768 \
  --temperature 0.2 \
  --top-p 1.0 \
  --top-k 0 \
  --repetition-penalty 1.05 \
  --limit 0
