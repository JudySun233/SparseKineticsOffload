"""
Fit I_cpu by sweeping prompt length / max_tokens / CPU KV budget.

The script runs a single-model, LMCache-enabled generation loop and collects
(bytes_cpu<->gpu, cost_ms) pairs from LMCache logs so we can regress CPU
offload unit price. Default: Long-CoT (N=1); A5000 24GB.
"""
import argparse
import json
import logging
import multiprocessing as mp
import os
import random
import re
import time
from dataclasses import dataclass, asdict
from typing import Dict, List

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import torch
from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.v1.cache_engine import LMCacheEngineBuilder
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig


@dataclass
class LMCacheEvent:
    size_gb: float
    cost_ms: float


@dataclass
class LMCacheAggregates:
    total_size_gb: float = 0.0
    total_cost_ms: float = 0.0
    num_events: int = 0

    def add(self, size_gb: float, cost_ms: float) -> None:
        """Accumulate one LMCache copy event."""
        self.total_size_gb += size_gb
        self.total_cost_ms += cost_ms
        self.num_events += 1

    @property
    def ms_per_gb(self) -> float:
        """Return marginal CPU offload latency (ms / GB)."""
        if self.total_size_gb == 0:
            return 0.0
        return self.total_cost_ms / self.total_size_gb


lmcache_stats = LMCacheAggregates()
lmcache_events: List[LMCacheEvent] = []


class LMCacheLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        """Parse LMCache INFO logs to pull size / cost."""
        global lmcache_stats, lmcache_events
        msg = record.getMessage().lower()
        match = re.search(r"size:\s*([0-9.]+)\s*gb.*cost\s*([0-9.]+)\s*ms", msg)
        if not match:
            return
        try:
            size_gb = float(match.group(1))
            cost_ms = float(match.group(2))
            lmcache_stats.add(size_gb=size_gb, cost_ms=cost_ms)
            lmcache_events.append(LMCacheEvent(size_gb=size_gb, cost_ms=cost_ms))
        except ValueError:
            pass


def attach_lmcache_logger() -> None:
    """Attach LMCacheLogHandler to *all* lmcache loggers."""
    handler = LMCacheLogHandler()

    def _ensure(logger: logging.Logger) -> None:
        if not any(isinstance(h, LMCacheLogHandler) for h in logger.handlers):
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    _ensure(logging.getLogger("lmcache"))
    for name, logger in logging.Logger.manager.loggerDict.items():
        if not isinstance(logger, logging.Logger):
            continue
        if name == "lmcache" or name.startswith("lmcache."):
            _ensure(logger)



def reset_lmcache_counters() -> None:
    """Clear aggregated and per-event stats between configs."""
    global lmcache_stats, lmcache_events
    lmcache_stats = LMCacheAggregates()
    lmcache_events = []


def setup_lmcache_env(cpu_gb: float, chunk_size: int) -> None:
    """Push LMCache env vars so the next engine build uses them."""
    os.environ["LMCACHE_CHUNK_SIZE"] = str(chunk_size)  # tune copy chunk size; CLI also uses 256
    os.environ["LMCACHE_LOCAL_CPU"] = "True"  # force CPU backend so offload is enabled
    os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = str(cpu_gb)  # CPU KV budget (GB) we sweep over


def make_prompts(num_prompts: int, approx_tokens: int, seed: int) -> List[str]:
    """Create tougher, semi-structured prompts to elicit longer generations."""
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
        # Repeat informative seed sentences instead of random tokens to keep vocab reasonable.
        while len(words) < approx_tokens:
            sentence = rng.choice(seed_sentences)
            words.extend(sentence.split())
        body = " ".join(words[:approx_tokens])
        prompts.append(
            f"[Prompt {i}] {body}\nQuestion: {hard_question}\nAnswer with detailed reasoning then a concise final policy."
        )
    return prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit CPU offload unit price with LMCache sweeps.")
    # Model to benchmark; Qwen3-8B matches the serve command shown above.
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    # Context window for this run; 16384 lets 9k+2k settings fit to induce budget pressure.
    parser.add_argument("--max-model-len", type=int, default=16384)
    # Target prompt lengths (Lin) to expose different KV scales; includes midpoints for smoother fit.
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[1024, 2048, 3072, 4096, 6144, 8192, 9216, 12288])
    # Max new tokens (Lout upper bound) per request; more midpoints to vary KV and runtime.
    parser.add_argument("--max-tokens", type=int, nargs="+", default=[128, 256, 512, 1024, 2048])
    # CPU KV budgets in GB (LMCACHE_MAX_LOCAL_CPU_SIZE); include small caps to trigger pressure.
    # parser.add_argument("--cpu-budgets-gb", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0, 16.0])
    parser.add_argument("--cpu-budget-gb", type=float, default=1.0)
    # Batch size; keep 1 to stay in the Long-CoT setting without best-of-N.
    parser.add_argument("--num-prompts", type=int, default=1)
    # Generation temperature; 0.0 for deterministic outputs so timing noise is reduced.
    parser.add_argument("--temperature", type=float, default=0.1)
    # top_p = 1.0 keeps nucleus sampling disabled to avoid extra randomness.
    parser.add_argument("--top-p", type=float, default=1.0)
    # LMCache chunk size that controls CPU<->GPU copy granularity; 256 matches your serve cmd.
    parser.add_argument("--kv-chunk-size", type=int, default=256)
    # GPU memory utilization cap; 0.9 keeps a buffer on A5000 24GB to avoid OOM.
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    # Where to write JSON results; default stays under scripts/ for convenience.
    parser.add_argument("--output", type=str, default="scripts/fit_icpu_results.json")
    # Random seed so prompt bodies are reproducible across runs.
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def ensure_within_context(prompt_token_counts: List[int], max_tokens: int, max_model_len: int) -> bool:
    """Guard against Lin + Lout exceeding the configured context window."""
    if not prompt_token_counts:
        return False
    longest = max(prompt_token_counts)
    return longest + max_tokens <= max_model_len


def run_one_config(
    llm: LLM,
    tokenizer,
    num_prompts: int,
    approx_tokens: int,
    max_tokens: int,
    cpu_gb: float,
    args: argparse.Namespace,
) -> Dict:
    reset_lmcache_counters()
    setup_lmcache_env(cpu_gb=cpu_gb, chunk_size=args.kv_chunk_size)

    prompts = make_prompts(num_prompts=num_prompts, approx_tokens=approx_tokens, seed=args.seed + approx_tokens)
    prompt_token_counts = [len(tokenizer.encode(p)) for p in prompts]
    if not ensure_within_context(prompt_token_counts, max_tokens=max_tokens, max_model_len=args.max_model_len):
        return {
            "skipped": True,
            "skip_reason": "prompt_len + max_tokens exceeds max_model_len",
            "approx_tokens": approx_tokens,
            "max_tokens": max_tokens,
            "cpu_budget_gb": cpu_gb,
            "prompt_token_counts": prompt_token_counts,
        }

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

    total_out_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    avg_out = total_out_tokens / len(outputs)
    max_gpu_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)

    result = {
        "model": args.model,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "kv_chunk_size": args.kv_chunk_size,
        "cpu_budget_gb": cpu_gb,
        "num_prompts": num_prompts,
        "approx_prompt_tokens": approx_tokens,
        "prompt_token_counts": prompt_token_counts,
        "max_tokens": max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "latency_s": end - start,
        "avg_Lout": avg_out,
        "total_out_tokens": total_out_tokens,
        "max_gpu_mem_gb": max_gpu_mem_gb,
        "lmcache_total_size_gb": lmcache_stats.total_size_gb,
        "lmcache_total_size_bytes": lmcache_stats.total_size_gb * (1024 ** 3),
        "lmcache_total_cost_ms": lmcache_stats.total_cost_ms,
        "lmcache_events": lmcache_stats.num_events,
        "lmcache_ms_per_gb": lmcache_stats.ms_per_gb,
        "lmcache_event_trace": [asdict(ev) for ev in lmcache_events],
    }

    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    attach_lmcache_logger()
    args = parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    kv_cfg = KVTransferConfig(
        kv_connector="LMCacheConnectorV1",
        kv_role="kv_both",
    )

    cpu_gb = args.cpu_budget_gb
    setup_lmcache_env(cpu_gb=cpu_gb, chunk_size=args.kv_chunk_size)

    llm = LLM(
        model=args.model,
        kv_transfer_config=kv_cfg,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
    )
    tokenizer = llm.get_tokenizer()

    results: List[Dict] = []

    for approx_tokens in args.prompt_lengths:
        for max_tokens in args.max_tokens:
            res = run_one_config(
                llm=llm,
                tokenizer=tokenizer,
                num_prompts=args.num_prompts,
                approx_tokens=approx_tokens,
                max_tokens=max_tokens,
                cpu_gb=cpu_gb,
                args=args,
            )
            logging.info("CONFIG RESULT: %s", res)
            results.append(res)

            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)

        # LMCacheEngineBuilder.destroy(ENGINE_NAME)

    logging.info("Saved %d configs to %s", len(results), args.output)
    LMCacheEngineBuilder.destroy(ENGINE_NAME)


if __name__ == "__main__":
    main()
