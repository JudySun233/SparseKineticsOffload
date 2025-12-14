"""
Generate predictions on the AIME24 dataset using vLLM for fast inference.

Dependencies:
- vllm, transformers, torch

Install:
  pip install vllm transformers torch

Usage:
  python scripts/eval.py \
    --input aime24/dense/aime24_qwen3-8b_dense.jsonl \
    --output results/aime24_outputs.jsonl \
    --model Qwen/Qwen3-8B \
    --max-model-len 16384
"""
import argparse
import json
import logging
import os
import time
import re
from typing import Dict
from vllm import LLM, SamplingParams

# copied from kinetics_cost_models/utils.py
MATH_QUERY_TEMPLATE = """
Solve the following math problem efficiently and clearly.  The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.

{Question}{Response}
""".strip()

def apply_chat_template(example, tokenizer, template):
    chat = [
        {"role": "user", "content": template.format(Question=example["query"], Response=example["prediction"])}
    ]
    example["token_length"] = len(tokenizer.apply_chat_template(
        chat, tokenize=True, add_generation_prompt=True
    ))
    prefix = [
        {"role": "user", "content": template.format(Question=example["query"], Response="")}
    ]
    example["prefix_length"] = len(tokenizer.apply_chat_template(
        prefix, tokenize=True, add_generation_prompt=True
    ))
    return example

def extract_prompt(obj: Dict) -> str:
    key = "problem"
    if key in obj and isinstance(obj[key], str) and obj[key].strip():
        return obj[key].strip()
    return ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate AIME24 predictions with Qwen-8B (vLLM).")
    p.add_argument("--input", type=str, required=True, help="Path to input JSONL with AIME24 examples.")
    p.add_argument("--output", type=str, required=True, help="Path to write output JSONL results.")
    p.add_argument("--model", type=str, default="Qwen/Qwen3-8B", help="Model name/path for vLLM (default Qwen/Qwen3-8B).")
    # Remove user-facing max-tokens; we derive dynamic budget from max_model_len - prefix_length
    p.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature (0 for greedy).")
    p.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling.")
    p.add_argument("--top-k", type=int, default=0, help="Top-k sampling (0 = disabled).")
    p.add_argument("--repetition-penalty", type=float, default=1.05, help="Repetition penalty (>1 discourages repeats).")
    p.add_argument("--greedy", action="store_true", help="Force greedy decoding (temperature=0, top_k=1).")
    p.add_argument("--max-model-len", type=int, default=16384, help="Model context window for vLLM.")
    p.add_argument("--limit", type=int, default=0, help="If >0, only process this many examples (for testing).")
    p.add_argument("--batch-size", type=int, default=1, help="Number of examples to process per batch (default: 1, i.e., no batching).")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return p.parse_args()

def open_llm(model: str, max_model_len: int) -> LLM:
    return LLM(model=model, max_model_len=max_model_len)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


    args = parse_args()

    # Set random seeds for reproducibility
    import random
    import numpy as np
    import torch
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    # If vLLM supports a seed argument, add it to LLM() as well (not all versions do)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    logging.info("Loading model %s", args.model)

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        enable_prefix_caching=True,
    )

    tokenizer = llm.get_tokenizer()

    base_kwargs = {}
    if args.greedy:
        base_kwargs = {"temperature": 0.0, "top_k": 1}
        if args.repetition_penalty and args.repetition_penalty > 1.0:
            base_kwargs["repetition_penalty"] = args.repetition_penalty
    else:
        base_kwargs = {"temperature": args.temperature, "top_p": args.top_p}
        if args.top_k and args.top_k > 0:
            base_kwargs["top_k"] = args.top_k
        if args.repetition_penalty and args.repetition_penalty > 1.0:
            base_kwargs["repetition_penalty"] = args.repetition_penalty

    overall_start = time.perf_counter()

    with open(args.input, "r") as fin, open(args.output, "w") as fout:
        processed_count = 0
        idx = 0
        batch = []
        obj_list = []
        total = 0
        while True:
            if args.limit and processed_count >= args.limit:
                break
            line = fin.readline()
            if not line:
                break
            obj = json.loads(line)
            prompt_body = extract_prompt(obj)
            query = MATH_QUERY_TEMPLATE.replace("{Question}", prompt_body).replace("{Response}", "")
            batch.append(query)
            obj_list.append((obj, idx))
            idx += 1
            if len(batch) == args.batch_size or (args.limit and processed_count + len(batch) >= args.limit) or not line:
                # Compute prefix_length and dynamic_max_tokens for each item in batch
                prefix_lengths = []
                dynamic_max_tokens_list = []
                for q in batch:
                    prefix_chat = [{"role": "user", "content": q}]
                    prefix_length = len(tokenizer.apply_chat_template(prefix_chat, tokenize=True, add_generation_prompt=True))
                    prefix_lengths.append(prefix_length)
                    dynamic_max_tokens_list.append(args.max_model_len - prefix_length)
                # Use min dynamic_max_tokens in batch for all (to avoid overflow)
                min_max_tokens = min(dynamic_max_tokens_list)
                sampling = SamplingParams(max_tokens=min_max_tokens, **base_kwargs)
                t0 = time.perf_counter()
                outputs = llm.generate(batch, sampling)
                t1 = time.perf_counter()
                for i, out in enumerate(outputs):
                    obj, real_idx = obj_list[i]
                    text = out.outputs[0].text
                    try:
                        gen_token_len = len(out.outputs[0].token_ids)
                    except Exception:
                        gen_token_len = None
                    choices = []
                    if "solution" in obj:
                        cleaned_sol = re.sub(r"\\boxed|\\begin\{.*?\}|\\end\{.*?\}", "", obj["solution"]).replace("$", "").strip().strip("{} ")
                        if cleaned_sol:
                            choices = [cleaned_sol]
                    id_val = obj.get("id") if obj.get("id") is not None else real_idx
                    rec = {
                        "model": args.model,
                        "query": batch[i],
                        "prediction": [text],
                        "choices": choices,
                        "prefix_length": prefix_lengths[i],
                        "token_length": gen_token_len,
                        "id": id_val,
                        "latency_s": (t1 - t0) / len(batch),
                        "dynamic_max_tokens": min_max_tokens,
                        "max_model_len": args.max_model_len,
                    }
                    fout.write(json.dumps(rec) + "\n")
                    total += 1
                    processed_count += 1
                batch = []
                obj_list = []
        # Handle any remaining items in batch
        if batch:
            prefix_lengths = []
            dynamic_max_tokens_list = []
            for q in batch:
                prefix_chat = [{"role": "user", "content": q}]
                prefix_length = len(tokenizer.apply_chat_template(prefix_chat, tokenize=True, add_generation_prompt=True))
                prefix_lengths.append(prefix_length)
                dynamic_max_tokens_list.append(args.max_model_len - prefix_length)
            min_max_tokens = min(dynamic_max_tokens_list)
            sampling = SamplingParams(max_tokens=min_max_tokens, **base_kwargs)
            t0 = time.perf_counter()
            outputs = llm.generate(batch, sampling)
            t1 = time.perf_counter()
            for i, out in enumerate(outputs):
                obj, real_idx = obj_list[i]
                text = out.outputs[0].text
                try:
                    gen_token_len = len(out.outputs[0].token_ids)
                except Exception:
                    gen_token_len = None
                choices = []
                if "solution" in obj:
                    cleaned_sol = re.sub(r"\\boxed|\\begin\{.*?\}|\\end\{.*?\}", "", obj["solution"]).replace("$", "").strip().strip("{} ")
                    if cleaned_sol:
                        choices = [cleaned_sol]
                id_val = obj.get("id") if obj.get("id") is not None else real_idx
                rec = {
                    "model": args.model,
                    "query": batch[i],
                    "prediction": [text],
                    "choices": choices,
                    "prefix_length": prefix_lengths[i],
                    "token_length": gen_token_len,
                    "id": id_val,
                    "latency_s": (t1 - t0) / len(batch),
                    "dynamic_max_tokens": min_max_tokens,
                    "max_model_len": args.max_model_len,
                }
                fout.write(json.dumps(rec) + "\n")
                total += 1
                processed_count += 1

        dur = time.perf_counter() - overall_start
        logging.info("Done. Wrote results to %s. Examples processed: %d. Time: %.2fs", args.output, total, dur)


if __name__ == "__main__":
    main()
