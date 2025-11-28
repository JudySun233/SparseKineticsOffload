#!/usr/bin/env python
"""
Evaluate Qwen3-8B-Q6_K on AIME24 via llama.cpp llama-server with
Long-CoT style prompting, 4 samples per question, boxed answers, and
majority voting.

Usage example (after starting llama-server on the same node):

    ./build/bin/llama-server -m models/models/Qwen3-8B-Q6_K/Qwen3-8B-Q6_K.gguf -ngl 22 -c 44000 --port 8000 --host 0.0.0.0

    python scripts/qwen3_8b_eval.py --server-url http://127.0.0.1:8000 --model-id qwen3_8b --max-context 44000 --max-new-tokens 42000 --gpu-layers 22 --offload-desc "qwen3_8b_q6k_cpu_offload_10" --max-examples 30
"""

import argparse
import datetime
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
from collections import Counter

import requests
from datasets import load_dataset

# Optional: for robust math parsing
try:
    from sympy.parsing.latex import parse_latex
    from sympy.parsing.sympy_parser import parse_expr
    from latex2sympy2 import latex2sympy
    HAS_SYMPY = True
except Exception:
    HAS_SYMPY = False
    def parse_latex(s):  # type: ignore
        raise RuntimeError("parse_latex not available")
    def parse_expr(s):  # type: ignore
        raise RuntimeError("parse_expr not available")
    def latex2sympy(s):  # type: ignore
        raise RuntimeError("latex2sympy not available")


# -----------------------------
# Configuration dataclasses
# -----------------------------

@dataclass
class EvalConfig:
    server_url: str
    model_id: str

    # Dataset config
    dataset_name: str = "HuggingFaceH4/aime_2024"
    dataset_split: str = "train"
    max_examples: Optional[int] = None  # only evaluate first N examples if set

    # Qwen3-8B generation config (from your config)
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    max_context: int = 34000     # metadata only; real value is set in llama-server
    max_new_tokens: int = 8192  # up to 32k tokens per completion

    # Offloading / system config (metadata only)
    gpu_layers: Optional[int] = None
    offload_desc: str = "all_gpu_no_offload"

    # HTTP config
    request_timeout: float = 3400.0  # seconds

    # Evaluation config
    num_samples_per_question: int = 2  # 4 traces per question

    # Output
    output_dir: str = "outputs"
    script_invocation: Optional[str] = None


# -----------------------------
# Prompt construction (Qwen template)
# -----------------------------

def build_qwen_messages(problem: str) -> List[Dict[str, str]]:
    """
    Build chat messages for Qwen3 according to your template:

        return [
            {"role": "user",
             "content": prompt + "\\nPlease reason step by step, but keep your reasoning concise (no more than about 20 steps), "
                                  "and put your final answer within \\\\boxed{}."}
        ]
    """
    prompt = problem.strip()
    content = (
        prompt
        + "\nPlease reason step by step, , but keep your reasoning concise (no more than about 20 steps), and put your final answer within \\\\boxed{}."
    )
    return [
        {"role": "user", "content": content}
    ]


# -----------------------------
# Answer extraction and parsing
# -----------------------------

def extract_answer(text: str) -> Optional[str]:
    """
    Extract the boxed answer from model output.

    This is adapted from your Qwen code:

        If 'boxed' appears, we look at the substring after it.
        If the next character is '{', we parse matching braces.
        Otherwise, we take the substring up to the next '$'.

    Returns a stripped string (e.g., '204') or None if not found.
    """
    if "boxed" not in text:
        return None

    ans = text.split("boxed", 1)[-1]
    if len(ans) == 0:
        return None

    ans = ans.lstrip()  # remove leading spaces/newlines
    if ans and ans[0] == "{":
        # parse {...} with possible nested braces
        stack = 1
        a = ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
        return a.strip()
    else:
        # no starting '{', cut at next '$' if any
        a = ans.split("$")[0].strip()
        return a.strip() if a else None


def parse_func(s: str):
    """
    Try multiple parsers to convert string to a symbolic expression.
    Mirrors your original idea: try parse_latex, parse_expr, latex2sympy.
    """
    if not HAS_SYMPY:
        return s
    for f in (parse_latex, parse_expr, latex2sympy):
        try:
            return f(s.replace("\\\\", "\\"))
        except Exception:
            try:
                return f(s)
            except Exception:
                pass
    return s


def math_equal(pred_str: str, gold_str: str) -> bool:
    """
    Compare predicted answer string and ground truth string.

    For AIME24 both should be integers 0-999, but we allow some robustness
    via parse_func. If parsing fails, fall back to stripped string equality.
    """
    pred_str_clean = pred_str.strip().rstrip(".")
    gold_str_clean = gold_str.strip().rstrip(".")

    # Try to parse as integers via sympy
    try:
        pred_expr = parse_func(pred_str_clean)
        gold_expr = parse_func(gold_str_clean)
        pred_int = int(str(pred_expr))
        gold_int = int(str(gold_expr))
        return pred_int == gold_int
    except Exception:
        pass

    # Fallback: direct integer conversion
    try:
        return int(pred_str_clean) == int(gold_str_clean)
    except Exception:
        return pred_str_clean == gold_str_clean


# -----------------------------
# llama.cpp HTTP helpers
# -----------------------------

def call_llama_server_chat(cfg: EvalConfig, messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Call llama-server /v1/chat/completions with given messages.

    We use Qwen3-8B hyperparameters from EvalConfig, and request a single
    completion (n=1). For each question we call this function multiple times
    sequentially to obtain num_samples_per_question traces.
    """
    url = cfg.server_url.rstrip("/") + "/v1/chat/completions"

    payload = {
        "model": cfg.model_id,
        "messages": messages,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "top_k": cfg.top_k,
        "max_tokens": cfg.max_new_tokens,
        "n": 1,          # single completion per call (no parallel traces)
        "stream": False,
    }

    t0 = time.time()
    resp = requests.post(url, json=payload, timeout=cfg.request_timeout)
    t1 = time.time()

    resp.raise_for_status()
    data = resp.json()
    data["_latency_ms"] = (t1 - t0) * 1000.0
    return data


def extract_text_and_usage_chat(resp: Dict[str, Any]) -> (str, Optional[Dict[str, int]]):
    """
    Extract completion text and usage from /v1/chat/completions response.
    """
    choices = resp.get("choices", [])
    if not choices:
        raise ValueError("No 'choices' field in response")

    message = choices[0].get("message", {}) or {}
    text = message.get("content", "")

    usage = resp.get("usage")
    return text, usage


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


# -----------------------------
# Evaluation per question
# -----------------------------

def evaluate_single_question(
    cfg: EvalConfig,
    ex_id: int,
    problem: str,
    gold_answer_str: str,
) -> Dict[str, Any]:
    """
    For one AIME24 problem:
      - Generate num_samples_per_question completions sequentially.
      - Extract boxed answers for each trace.
      - Do majority vote over extracted answers.
      - Determine correctness and aggregate statistics.

    Returns a dict containing:
      {
        "id": ex_id,
        "problem": ...,
        "gold_answer_str": ...,
        "traces": [...],
        "majority_answer": str or None,
        "is_correct_majority": bool,
        "num_samples": int,
        "total_completion_tokens": int,
        "total_prompt_tokens": int,
        "total_latency_ms": float,
      }
    """
    messages = build_qwen_messages(problem)
    traces: List[Dict[str, Any]] = []

    total_completion_tokens = 0
    total_prompt_tokens = 0
    total_latency_ms = 0.0

    # Ground truth as string (AIME24 answers are integers)
    ground_truth = gold_answer_str.strip()

    for sample_idx in range(cfg.num_samples_per_question):
        try:
            resp = call_llama_server_chat(cfg, messages)
        except Exception as e:
            trace_data = {
                "sample_id": sample_idx,
                "error": str(e),
                "text": None,
                "extracted_answer": None,
                "parsed_answer": None,
                "is_correct": False,
                "latency_ms": None,
                "usage": None,
            }
            traces.append(trace_data)
            continue

        text, usage = extract_text_and_usage_chat(resp)
        latency_ms = resp.get("_latency_ms", None)

        if latency_ms is not None:
            total_latency_ms += latency_ms

        if usage is not None:
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
        else:
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0

        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens

        extracted_answer = extract_answer(text) if text is not None else None
        parsed_answer = parse_func(extracted_answer) if (extracted_answer and HAS_SYMPY) else extracted_answer

        is_correct = False
        if extracted_answer is not None and ground_truth is not None:
            try:
                is_correct = math_equal(extracted_answer, ground_truth)
            except Exception:
                is_correct = False

        trace_data = {
            "sample_id": sample_idx,
            "text": text,
            "extracted_answer": extracted_answer,
            "parsed_answer": str(parsed_answer) if parsed_answer is not None else None,
            "is_correct": is_correct,
            "latency_ms": latency_ms,
            "usage": usage,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
        traces.append(trace_data)

    # Majority vote over extracted answers (raw strings)
    answers = [t["extracted_answer"] for t in traces if t.get("extracted_answer")]
    if answers:
        counts = Counter(answers)
        majority_answer, _ = counts.most_common(1)[0]
        try:
            is_correct_majority = math_equal(majority_answer, ground_truth)
        except Exception:
            is_correct_majority = False
    else:
        majority_answer = None
        is_correct_majority = False

    result = {
        "id": ex_id,
        "problem": problem,
        "gold_answer_str": gold_answer_str,
        "gold_answer_int": int(gold_answer_str) if gold_answer_str.isdigit() else None,
        "traces": traces,
        "majority_answer": majority_answer,
        "is_correct_majority": is_correct_majority,
        "num_samples": cfg.num_samples_per_question,
        "total_completion_tokens": total_completion_tokens,
        "total_prompt_tokens": total_prompt_tokens,
        "total_latency_ms": total_latency_ms,
    }
    return result


# -----------------------------
# Main evaluation loop
# -----------------------------

def run_eval(cfg: EvalConfig) -> None:
    """Run AIME24 evaluation under the given configuration."""
    ensure_dir(cfg.output_dir)

    # Load dataset
    ds = load_dataset(cfg.dataset_name, split=cfg.dataset_split)
    if cfg.max_examples is not None:
        ds = ds.select(range(cfg.max_examples))

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        cfg.output_dir,
        f"aime24_qwen3_8b_q6k_{timestamp}.json",
    )

    results: List[Dict[str, Any]] = []
    num_questions = 0
    num_correct_questions = 0

    total_prompt_tokens_all = 0
    total_completion_tokens_all = 0
    total_latency_ms_all = 0.0

    for example in ds:
        ex_id = int(example["id"])
        problem = example["problem"]
        gold_answer_str = example["answer"]

        question_result = evaluate_single_question(cfg, ex_id, problem, gold_answer_str)
        results.append(question_result)

        num_questions += 1
        if question_result["is_correct_majority"]:
            num_correct_questions += 1

        total_prompt_tokens_all += question_result["total_prompt_tokens"]
        total_completion_tokens_all += question_result["total_completion_tokens"]
        total_latency_ms_all += question_result["total_latency_ms"]

        acc_so_far = num_correct_questions / num_questions if num_questions > 0 else 0.0
        print(
            f"[ID {ex_id}] majority_answer={question_result['majority_answer']}, "
            f"gold={gold_answer_str}, "
            f"correct={question_result['is_correct_majority']}, "
            f"acc_so_far={acc_so_far:.3f}"
        )

        # Stream write partial JSON so you can inspect midway
        _write_partial_json(
            output_path,
            cfg,
            results,
            num_questions,
            num_correct_questions,
            total_prompt_tokens_all,
            total_completion_tokens_all,
            total_latency_ms_all,
        )

    print(f"\nSaved evaluation results to: {output_path}")
    if num_questions > 0:
        final_acc = num_correct_questions / num_questions
        print(f"Final accuracy over {num_questions} questions (majority vote): {final_acc:.3f}")


def _write_partial_json(
    output_path: str,
    cfg: EvalConfig,
    results: List[Dict[str, Any]],
    num_questions: int,
    num_correct_questions: int,
    total_prompt_tokens: int,
    total_completion_tokens: int,
    total_latency_ms: float,
) -> None:
    """Write current run_config + summary + question-level results to JSON."""
    accuracy = (
        num_correct_questions / num_questions if num_questions > 0 else 0.0
    )
    avg_latency_ms = (
        total_latency_ms / num_questions if num_questions > 0 else None
    )

    total_tokens = total_prompt_tokens + total_completion_tokens
    total_time_sec = total_latency_ms / 1000.0
    if total_time_sec > 0:
        tokens_per_second = total_tokens / total_time_sec
    else:
        tokens_per_second = None

    run_summary = {
        "num_questions": num_questions,
        "num_correct_questions": num_correct_questions,
        "accuracy_majority": accuracy,
        "avg_latency_ms_per_question": avg_latency_ms,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "tokens_per_second_overall": tokens_per_second,
    }

    output = {
        "run_config": asdict(cfg),
        "run_summary": run_summary,
        "results": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


# -----------------------------
# Argument parsing
# -----------------------------

def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3-8B-Q6_K on AIME24 via llama.cpp server."
    )
    parser.add_argument(
        "--server-url",
        type=str,
        required=True,
        help="Base URL of llama-server, e.g. http://127.0.0.1:8000",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="qwen3_8b",
        help="Model id string to send in the OpenAI-style request.",
    )
    parser.add_argument(
        "--max-context",
        type=int,
        default=34000,
        help="Max context length (tokens) configured for llama-server (metadata only).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8192,
        help="Max new tokens per completion.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p nucleus sampling parameter.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-k sampling parameter.",
    )
    parser.add_argument(
        "--gpu-layers",
        type=int,
        default=32,
        help="Number of layers placed on GPU when starting llama-server (metadata only).",
    )
    parser.add_argument(
        "--offload-desc",
        type=str,
        default="all_gpu_no_offload",
        help="Human-readable description of weight/KV offloading config.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=3400.0,
        help="Timeout in seconds for each HTTP request.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to store evaluation JSON results.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="If set, only evaluate on the first N examples (for debugging).",
    )
    parser.add_argument(
        "--num-samples-per-question",
        type=int,
        default=4,
        help="Number of completions (traces) per question for majority vote.",
    )

    args = parser.parse_args()

    import sys
    script_invocation = "python " + " ".join(sys.argv)

    cfg = EvalConfig(
        server_url=args.server_url,
        model_id=args.model_id,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_context=args.max_context,
        max_new_tokens=args.max_new_tokens,
        gpu_layers=args.gpu_layers,
        offload_desc=args.offload_desc,
        request_timeout=args.request_timeout,
        output_dir=args.output_dir,
        max_examples=args.max_examples,
        num_samples_per_question=args.num_samples_per_question,
        script_invocation=script_invocation,
    )
    return cfg


if __name__ == "__main__":
    config = parse_args()
    run_eval(config)