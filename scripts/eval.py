"""
Generate predictions on the AIME24 dataset with a Qwen-8B model and save results as JSONL.

Dependencies:
- vllm, transformers, torch

Install (example):
  pip install vllm transformers torch

Usage (example):
  python scripts/generate_aime24_qwen8b.py \
    --input aime24/dense/aime24_qwen3-8b_dense.jsonl \
    --output results/aime24_qwen8b_outputs.jsonl \
    --model Qwen/Qwen3-8B \
    --max-tokens 256

The script will try to extract a prompt from common keys in each JSON object
("prompt", "question", "input", "instruction", "context"). If none are
found the example is skipped.
"""
import argparse
import json
import logging
import os
import time
import re
from dataclasses import dataclass, asdict
from typing import Dict, List
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

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


# LMCache logging capture (copied/adapted from scripts/fit_I_cpu.py)
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
        self.total_size_gb += size_gb
        self.total_cost_ms += cost_ms
        self.num_events += 1

    @property
    def ms_per_gb(self) -> float:
        if self.total_size_gb == 0:
            return 0.0
        return self.total_cost_ms / self.total_size_gb


lmcache_stats = LMCacheAggregates()
lmcache_events: list[LMCacheEvent] = []
lmcache_stats_batch = LMCacheAggregates()
lmcache_events_batch: list[LMCacheEvent] = []


class LMCacheLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        global lmcache_stats, lmcache_events
        raw = record.getMessage()
        cleaned = re.sub(r"\x1b\[[0-9;]*m", "", raw).lower()
        match = re.search(r"size:\s*([0-9]+(?:\.[0-9]+)?)\s*gb[,\s]+cost[:\s]+([0-9]+(?:\.[0-9]+)?)\s*ms", cleaned, re.IGNORECASE)
        if not match:
            return
        try:
            size_gb = float(match.group(1))
            cost_ms = float(match.group(2))
            lmcache_stats.add(size_gb=size_gb, cost_ms=cost_ms)
            lmcache_events.append(LMCacheEvent(size_gb=size_gb, cost_ms=cost_ms))
            lmcache_stats_batch.add(size_gb=size_gb, cost_ms=cost_ms)
            lmcache_events_batch.append(LMCacheEvent(size_gb=size_gb, cost_ms=cost_ms))
        except ValueError:
            pass


def attach_lmcache_logger() -> None:
    handler = LMCacheLogHandler()
    handler.setLevel(logging.DEBUG)

    def _ensure(logger: logging.Logger) -> None:
        if not any(isinstance(h, LMCacheLogHandler) for h in logger.handlers):
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = True

    _ensure(logging.getLogger("lmcache"))
    for name, logger in logging.Logger.manager.loggerDict.items():
        if not isinstance(logger, logging.Logger):
            continue
        if name == "lmcache" or name.startswith("lmcache."):
            _ensure(logger)


def reset_lmcache_counters_global() -> None:
    global lmcache_stats, lmcache_events
    lmcache_stats = LMCacheAggregates()
    lmcache_events = []

def reset_lmcache_counters_batch() -> None:
    global lmcache_stats_batch, lmcache_events_batch
    lmcache_stats_batch = LMCacheAggregates()
    lmcache_events_batch = []


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
    p.add_argument("--max-model-len", type=int, default=16384, help="Model context window for vLLM.")
    p.add_argument("--greedy", action="store_true", help="Force greedy decoding (temperature=0, top_k=1).")
    p.add_argument("--limit", type=int, default=0, help="If >0, only process this many examples (for testing).")
    p.add_argument("--enable-lmcache", action="store_true", help="Enable LMCache CPU offload via vLLM kv_transfer_config")
    p.add_argument("--cpu-budget-gb", type=float, default=1.0, help="LMCache CPU budget in GB (LMCACHE_MAX_LOCAL_CPU_SIZE)")
    p.add_argument("--kv-chunk-size", type=int, default=256, help="LMCache chunk size (LMCACHE_CHUNK_SIZE)")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90, help="Desired GPU memory utilization (0-1) passed to vLLM")
    return p.parse_args()


def open_llm(model: str, max_model_len: int) -> LLM:
    return LLM(model=model, max_model_len=max_model_len)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    logging.info("Loading model %s", args.model)

    kv_cfg = None
    if getattr(args, "enable_lmcache", False):
        logging.info("Enabling LMCache offload: cpu_budget_gb=%s, kv_chunk_size=%s", args.cpu_budget_gb, args.kv_chunk_size)
        # Let vLLM runs single-process so our logging handler
        # can capture LMCache logs within the same Python process.
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        os.environ["LMCACHE_LOCAL_CPU"] = "True"
        os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = str(args.cpu_budget_gb)
        os.environ["LMCACHE_CHUNK_SIZE"] = str(args.kv_chunk_size)
        attach_lmcache_logger()
        kv_cfg = KVTransferConfig(kv_connector="LMCacheConnectorV1", kv_role="kv_both")

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        kv_transfer_config=kv_cfg,
        gpu_memory_utilization=getattr(args, "gpu_memory_utilization", None),
        enable_prefix_caching=True,
    )
    if getattr(args, "enable_lmcache", False):
        attach_lmcache_logger()
    
    tokenizer = llm.get_tokenizer()

    base_kwargs = {}
    if args.greedy:
        base_kwargs = {
            "temperature": 0.0,
            "top_k": 1,
        }
        if args.repetition_penalty and args.repetition_penalty > 1.0:
            base_kwargs["repetition_penalty"] = args.repetition_penalty
    else:
        base_kwargs = {
            "temperature": args.temperature,
            "top_p": args.top_p,
        }
        if args.top_k and args.top_k > 0:
            base_kwargs["top_k"] = args.top_k
        if args.repetition_penalty and args.repetition_penalty > 1.0:
            base_kwargs["repetition_penalty"] = args.repetition_penalty

    total = 0
    start_all = time.perf_counter()

    def extract_choice(sol: str) -> List[str]:
        s = sol.strip()
        s = re.sub(r"\\boxed|\\begin\{.*?\}|\\end\{.*?\}", "", s)
        s = s.replace("$", "").strip()
        s = s.strip("{} ")
        return [s] if s else []

    with open(args.input, "r") as fin, open(args.output, "w") as fout:

        fout.write("{}\n")
        buffer_prompts: List[str] = []
        buffer_orig: List[Dict] = []
        processed_count = 0

        def flush_buffer():
            nonlocal total, processed_count
            if not buffer_prompts:
                return
            t0 = time.perf_counter()
            if getattr(args, "enable_lmcache", False):
                reset_lmcache_counters_batch()

            prefix_lengths: List[int] = []
            for q in buffer_prompts:
                prefix_chat = [
                    {"role": "user", "content": MATH_QUERY_TEMPLATE.replace("{Question}", q).replace("{Response}", "")}
                ]
                pl = len(tokenizer.apply_chat_template(prefix_chat, tokenize=True, add_generation_prompt=True))
                prefix_lengths.append(pl)

            remaining_caps = [args.max_model_len - pl for pl in prefix_lengths]
            if any(r <= 0 for r in remaining_caps):
                for i, r in enumerate(remaining_caps):
                    print(f"Example {i} has non-positive remaining capacity {r}, skipping.")
                buffer_prompts.clear()
                buffer_orig.clear()
                return

            # For batch >1 we must use a single max_tokens = min(all remaining caps)
            dynamic_max_tokens = min(remaining_caps)
            dynamic_sampling = SamplingParams(max_tokens=dynamic_max_tokens, **base_kwargs)

            outputs = llm.generate(buffer_prompts, dynamic_sampling)

            t1 = time.perf_counter()

            if getattr(args, "enable_lmcache", False):
                stable_rounds = 0
                last_batch_events = lmcache_stats_batch.num_events
                for _ in range(8):
                    time.sleep(0.01)
                    if lmcache_stats_batch.num_events == last_batch_events:
                        stable_rounds += 1
                    else:
                        stable_rounds = 0
                        last_batch_events = lmcache_stats_batch.num_events
                    if stable_rounds >= 2:
                        break

            for i, out in enumerate(outputs):
                orig = buffer_orig[i]

                text = out.outputs[0].text

                prefix_length = prefix_lengths[i]
                gen_token_len = None
                try:
                    gen_token_len = len(out.outputs[0].token_ids)
                except Exception:
                    print("Warning: could not get generated token length from vLLM output.")

                choices = extract_choice(orig.get("solution"))

                id_val = orig.get("id") if orig.get("id") is not None else orig.get("_index", processed_count)

                out_rec = {
                    "model": args.model,
                    "query": orig.get("query", orig.get("problem") or orig.get("prompt") or ""),
                    "prediction": [text],
                    "choices": choices,
                    "prefix_length": prefix_length,
                    "token_length": gen_token_len,
                    "id": id_val,
                    "batch_latency_s": t1 - t0,
                }

                if getattr(args, "enable_lmcache", False):
                    batch_gb = lmcache_stats_batch.total_size_gb
                    batch_cost = lmcache_stats_batch.total_cost_ms
                    batch_events_list = [asdict(e) for e in lmcache_events_batch]
                    cumulative_gb = lmcache_stats.total_size_gb
                    cumulative_cost = lmcache_stats.total_cost_ms
                    out_rec["lmcache_offload_batch_gb"] = batch_gb
                    out_rec["lmcache_offload_batch_cost_ms"] = batch_cost
                    out_rec["lmcache_offload_batch_ms_per_gb"] = (batch_cost / batch_gb) if batch_gb > 0 else 0.0
                    out_rec["lmcache_offload_batch_events"] = batch_events_list
                    out_rec["lmcache_offload_cumulative_gb"] = cumulative_gb
                    out_rec["lmcache_offload_cumulative_cost_ms"] = cumulative_cost
                    out_rec["lmcache_offload_cumulative_ms_per_gb"] = (cumulative_cost / cumulative_gb) if cumulative_gb > 0 else 0.0
                    out_rec["logged_offload_gb"] = cumulative_gb
                    out_rec["lmcache_source"] = "logged"
                else:
                    out_rec["lmcache_offload_batch_gb"] = None
                    out_rec["lmcache_offload_batch_cost_ms"] = None
                    out_rec["lmcache_offload_batch_ms_per_gb"] = None
                    out_rec["lmcache_offload_batch_events"] = None
                    out_rec["lmcache_offload_cumulative_gb"] = None
                    out_rec["lmcache_offload_cumulative_cost_ms"] = None
                    out_rec["lmcache_offload_cumulative_ms_per_gb"] = None
                    out_rec["logged_offload_gb"] = None
                    out_rec["lmcache_source"] = "none"
                fout.write(json.dumps(out_rec) + "\n")
                total += 1
                processed_count += 1

            buffer_prompts.clear()
            buffer_orig.clear()

        for idx, line in enumerate(fin):
            if args.limit and processed_count >= args.limit:
                break
            
            obj = json.loads(line)

            prompt_body = extract_prompt(obj)

            query = MATH_QUERY_TEMPLATE.replace("{Question}", prompt_body).replace("{Response}", "")
            obj_copy = dict(obj)
            obj_copy["_index"] = idx
            obj_copy["query"] = query

            buffer_prompts.append(query)
            buffer_orig.append(obj_copy)

            if len(buffer_prompts) >= 1:
                flush_buffer()

        # final flush
        flush_buffer()

    dur = time.perf_counter() - start_all
    logging.info("Done. Wrote results to %s. Examples processed: %d. Time: %.2fs", args.output, total, dur)


if __name__ == "__main__":
    main()
