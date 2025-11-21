"""
Fit GPU-side cost dominance by sweeping (L_in, L_out) without LMCache.

For each (prompt length, max_tokens) combo:
- Disable LMCache and run vLLM generate once.
- Record latency, actual tokenized L_in / L_out, and GPU memory peak.
- Compute C_comp and C_mem,total from Kinetics Eq.(1),(2) using model structure (P, r, D).
- Write all points to JSON for later regression to see which term dominates latency.
"""
import argparse
import json
import logging
import multiprocessing as mp
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import torch
from transformers import AutoConfig
from vllm import LLM, SamplingParams


@dataclass
class ModelStruct:
    params: float  # total active parameters (count, not billions)
    gqa_ratio: float  # num_heads / num_kv_heads
    kv_bytes_per_token: float  # per-token KV size across all layers in bytes
    num_layers: int
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    dtype: str


def infer_model_struct(model_name: str, dtype: str, params_billions: float, gqa_ratio_cli: Optional[float]) -> ModelStruct:
    """Infer model structural constants from HF config; fall back to CLI defaults."""
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    hidden_size = getattr(cfg, "hidden_size", getattr(cfg, "n_embd", None))
    num_layers = getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", None))
    num_heads = getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", None))
    num_kv_heads = getattr(cfg, "num_key_value_heads", getattr(cfg, "n_kv_head", num_heads))

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(dtype.lower(), torch.bfloat16)
    bytes_per_elem = torch.tensor([], dtype=torch_dtype).element_size()

    if None in (hidden_size, num_layers, num_heads, num_kv_heads):
        raise ValueError("Could not infer hidden_size / layers / heads from config; please override via CLI.")
    
    # D = K + V, all layers ≈ 2 × num_layers × hidden_size × bytes_per_elem
    kv_bytes_per_token = 2 * num_layers * hidden_size * bytes_per_elem  
    gqa_ratio = gqa_ratio_cli if gqa_ratio_cli is not None else float(num_heads) / float(num_kv_heads)
    params = params_billions * 1e9  # Eq uses absolute parameter count

    return ModelStruct(
        params=params,
        gqa_ratio=gqa_ratio,
        kv_bytes_per_token=kv_bytes_per_token,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        dtype=str(torch_dtype).replace("torch.", ""),
    )


def compute_C_comp_and_C_mem(P: float, r: float, D: float, N: int, Lin: int, Lout: int) -> Dict[str, float]:
    """Kinetics Eq.(1),(2) using provided structural constants."""
    C_comp = (
        2 * N * P * Lout
        + 2 * r * N * Lin * D * Lout
        + r * N * D * (Lout**2)
    )
    C_mem = 2 * Lin * D * Lout + N * D * (Lout**2)
    return {"C_comp": C_comp, "C_mem_total": C_mem}


def make_prompts(num_prompts: int, approx_tokens: int, seed: int) -> List[str]:
    """Use semi-structured, reasoning-heavy prompts to elicit longer outputs."""
    rng = random.Random(seed)
    seed_sentences = [
        "Break the task into numbered subgoals and explain trade offs.",
        "Use equations and intermediate variables when helpful for precision.",
        "Enumerate edge cases and discuss why they do or do not change the plan.",
        "Keep a running scratchpad of partial results before finalizing.",
        "Cross-check the reasoning with at least two alternative hypotheses.",
    ]
    hard_question = (
        "Given a robot swarm exploring an unknown maze with energy and bandwidth constraints, "
        "derive a control policy to minimize time-to-map under adversarial failures. "
        "Provide a step-by-step plan, math formulation, and final policy sketch."
    )

    prompts: List[str] = []
    for i in range(num_prompts):
        words: List[str] = []
        while len(words) < approx_tokens:
            sentence = rng.choice(seed_sentences)
            words.extend(sentence.split())
        body = " ".join(words[:approx_tokens])
        prompts.append(
            f"[Prompt {i}] {body}\nQuestion: {hard_question}\nAnswer with detailed reasoning then a concise final policy."
        )
    return prompts


def ensure_within_context(prompt_token_counts: List[int], max_tokens: int, max_model_len: int) -> bool:
    if not prompt_token_counts:
        return False
    return max(prompt_token_counts) + max_tokens <= max_model_len


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit GPU-dominant cost terms by sweeping (Lin, Lout) without LMCache.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B", help="Model name or path.")
    parser.add_argument("--max-model-len", type=int, default=16384, help="Context window to cap Lin+Lout.")
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[512, 1024, 2048, 3072, 4096, 6144, 8192, 9216], help="Approx token counts for prompts (Lin targets).")
    parser.add_argument("--max-tokens", type=int, nargs="+", default=[128, 256, 512, 1024, 2048], help="Generation cap (Lout upper bound).")
    parser.add_argument("--num-prompts", type=int, default=1, help="Batch size; keep 1 for Long-CoT scenario.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Low temp to reduce variance while allowing length variety.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Keep sampling broad; can lower for determinism.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="vLLM GPU mem cap fraction.")
    parser.add_argument("--output", type=str, default="scripts/fit_igpu_results.json", help="Path to JSON results.")
    parser.add_argument("--seed", type=int, default=13, help="Seed for prompt construction.")
    # Structural params
    parser.add_argument("--params-billions", type=float, default=8.0, help="Active parameter count in billions (P).")
    parser.add_argument("--gqa-ratio", type=float, default=None, help="Override GQA ratio r = num_heads/num_kv_heads.")
    parser.add_argument("--kv-bytes-per-token", type=float, default=None, help="Override KV size per token (bytes, all layers).")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="KV dtype used by vLLM; affects inferred bytes.")
    return parser.parse_args()


def disable_lmcache_env() -> None:
    """Force LMCache off so kv_transfer_config=None is honored."""
    os.environ["LMCACHE_LOCAL_CPU"] = "0"
    os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "0"
    os.environ["LMCACHE_CHUNK_SIZE"] = "0"
    os.environ["LMCACHE_DISABLE"] = "1"


def main() -> None:
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    disable_lmcache_env()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    model_struct = infer_model_struct(
        model_name=args.model,
        dtype=args.dtype,
        params_billions=args.params_billions,
        gqa_ratio_cli=args.gqa_ratio,
    )
    logging.info("Model struct: %s", asdict(model_struct))

    llm = LLM(
        model=args.model,
        kv_transfer_config=None,  # LMCache off
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
    )
    tokenizer = llm.get_tokenizer()

    results: List[Dict] = []
    for approx_tokens in args.prompt_lengths:
        prompts = make_prompts(num_prompts=args.num_prompts, approx_tokens=approx_tokens, seed=args.seed + approx_tokens)
        prompt_token_counts = [len(tokenizer.encode(p)) for p in prompts]

        for max_tokens in args.max_tokens:
            if not ensure_within_context(prompt_token_counts, max_tokens=max_tokens, max_model_len=args.max_model_len):
                skip_res = {
                    "skipped": True,
                    "skip_reason": "prompt_len + max_tokens exceeds max_model_len",
                    "approx_tokens": approx_tokens,
                    "prompt_token_counts": prompt_token_counts,
                    "max_tokens": max_tokens,
                }
                results.append(skip_res)
                logging.info("SKIP: %s", skip_res)
                continue

            sampling_params = SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=max_tokens,
            )

            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            outputs = llm.generate(prompts, sampling_params)
            torch.cuda.synchronize()
            end = time.perf_counter()

            Lout_list = [len(o.outputs[0].token_ids) for o in outputs]
            avg_Lout = sum(Lout_list) / len(Lout_list)
            max_gpu_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)

            costs = compute_C_comp_and_C_mem(
                P=model_struct.params,
                r=model_struct.gqa_ratio,
                D=args.kv_bytes_per_token or model_struct.kv_bytes_per_token,
                N=args.num_prompts,
                Lin=max(prompt_token_counts),
                Lout=int(avg_Lout),
            )

            res = {
                "model": args.model,
                "max_model_len": args.max_model_len,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "num_prompts": args.num_prompts,
                "approx_prompt_tokens": approx_tokens,
                "prompt_token_counts": prompt_token_counts,
                "max_tokens": max_tokens,
                "latency_s": end - start,
                "Lout_tokens": Lout_list,
                "avg_Lout": avg_Lout,
                "max_gpu_mem_gb": max_gpu_mem_gb,
                "P_params": model_struct.params,
                "gqa_ratio": model_struct.gqa_ratio,
                "kv_bytes_per_token": args.kv_bytes_per_token or model_struct.kv_bytes_per_token,
                "dtype": model_struct.dtype,
                **costs,
            }
            results.append(res)
            logging.info("CONFIG RESULT: %s", res)

            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)

    logging.info("Saved %d configs to %s", len(results), args.output)


if __name__ == "__main__":
    main()
