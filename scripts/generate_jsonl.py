#!/usr/bin/env python3
"""
Generate predictions for `math-ai/aime24` using Qwen-14B and write JSONL
formatted for the cost model scripts.

Produces a JSONL file whose first line is metadata (JSON) and subsequent
lines are records with fields: `query`, `prediction` (list), `choices` (list),
`prefix_length`, `token_length`, `question_id`.

Usage example:
  python generate_qwen14b_jsonl.py --out-dir ../aime24/dense --split test --max-examples 20

Be careful: loading Qwen-14B requires large GPU memory or offloading; adjust
`device_map` / `torch_dtype` as needed. The script supports a `--dry-run`
mode that only prepares prompts and writes empty predictions for testing.
"""
import os
import sys
import argparse
import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B", help="HF model id to load for generation")
    p.add_argument("--hf-name", default="math-ai/aime24", help="Hugging Face dataset id")
    p.add_argument("--split", default="test", help="dataset split to use")
    p.add_argument("--out-dir", default="aime24/dense", help="output directory")
    p.add_argument("--out-file", default=None, help="output filename (optional)")
    p.add_argument("--sparse", choices=["dense","topk","blocktopk"], default="dense")
    p.add_argument("--max-examples", type=int, default=None, help="limit for quick tests")
    p.add_argument("--batch-size", type=int, default=1, help="number of prompts per generation batch")
    p.add_argument("--num-return-sequences", type=int, default=4, help="best-of-N samples per prompt")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--device-map", default="auto", help="device_map for model loading (e.g., 'auto' or 'cpu')")
    p.add_argument("--dtype", default="bfloat16", choices=["float16","bfloat16","float32"], help="torch dtype for model")
    p.add_argument("--greedy", action="store_true", help="use greedy decoding (do_sample=False)")
    p.add_argument("--dry-run", action="store_true", help="do not load model; write placeholders only")
    return p.parse_args()


def dtype_from_str(s):
    if s == "float16":
        return torch.float16
    if s == "bfloat16":
        return torch.bfloat16
    return torch.float32


def main():
    args = build_args()
    ds = load_dataset(args.hf_name, split=args.split)
    print(f"Loaded dataset {args.hf_name} split={args.split} with {len(ds)} examples.")
    os.makedirs(args.out_dir, exist_ok=True)

    model_name = args.model.split('/')[-1].lower().replace('.', '-')
    out_file = args.out_file or os.path.join(args.out_dir, f"{args.hf_name.split('/')[-1]}_{model_name}_{args.sparse}.jsonl")
    print(f"Output file: {out_file}")

    if args.sparse == "dense":
        metadata = {}
    elif args.sparse == "topk":
        metadata = {"attn_local": 128, "attn_topk": 64}
    else:
        metadata = {"attn_local": 128, "attn_block_size": 16, "attn_block_topk": 8}

    print(f"Using metadata: {metadata}")
    # If dry-run, we won't load the heavy model; just use a tokenizer for length calc
    tokenizer = None
    model = None
    if args.dry_run:
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        except Exception:
            tokenizer = None
    else:
        print(f"Loading tokenizer and model {args.model} (this may take time and memory)...")
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        print("Tokenizer loaded.")
        torch_dtype = dtype_from_str(args.dtype)
        try:
            print(f"Loading model {args.model} with device_map={args.device_map} and torch_dtype={torch_dtype}...")
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                device_map=args.device_map,
                torch_dtype=torch_dtype,
            )
            print("Model loaded.")
            try:
                print("Model loaded on devices:", model.hf_device_map)
            except Exception:
                print("Model loaded (device mapping not available)")
            # determine a sensible device for inputs/generation
            try:
                first_param = next(model.parameters())
                model_primary_device = first_param.device
                print(f"Primary model parameter device: {model_primary_device}")
            except Exception:
                model_primary_device = None
                print("Could not determine primary model device; will default to CPU for inputs")
            model.eval()
        except Exception as e:
            print("Model load failed:", e)
            print("You can try --dry-run or change --device-map / --dtype.")
            return
    print("Model and tokenizer loaded.")

    def get_prompt_and_choices(ex):
        # for math-ai/aime24 observed fields: id, problem, solution, url
        if "problem" in ex:
            q = ex["problem"]
        elif "question" in ex:
            q = ex["question"]
        elif "text" in ex:
            q = ex["text"]
        else:
            # fallback pick first string field
            q = None
            for k, v in ex.items():
                if isinstance(v, str):
                    q = v
                    break
            if q is None:
                q = ""

        if "solution" in ex:
            choices = [ex["solution"]]
        elif "answer" in ex:
            choices = [ex["answer"]]
        else:
            choices = [""]
        return q, choices

    write_n = 0
    with open(out_file, "w", encoding="utf8") as fo:
        fo.write(json.dumps(metadata, ensure_ascii=False) + "\n")

        total = len(ds) if args.max_examples is None else min(args.max_examples, len(ds))
        print(f"Processing {total} examples from {args.hf_name}:{args.split}")

        i = 0
        while i < total:
            batch = []
            indices = list(range(i, min(i + args.batch_size, total)))
            for idx in indices:
                ex = ds[idx]
                q, choices = get_prompt_and_choices(ex)
                batch.append((idx, q, choices, ex.get("id", idx)))
            print(f"Generating batch of {len(batch)} examples (examples {i} to {i + len(batch) - 1})...")
            # compute prefix lengths
            prefix_lens = []
            for (_, q, _, _) in batch:
                if tokenizer is not None and q is not None:
                    try:
                        prefix_lens.append(len(tokenizer.encode(q, add_special_tokens=False)))
                    except Exception:
                        prefix_lens.append(0)
                else:
                    prefix_lens.append(0)
            print(f"Prefix lengths for batch: {prefix_lens}")
            # generate predictions
            preds_for_batch = []
            if args.dry_run or model is None:
                # placeholder empty predictions lists
                for _ in batch:
                    preds_for_batch.append([""] * args.num_return_sequences)
            else:
                prompts = [q for (_, q, _, _) in batch]
                # tokenization and generation per-prompt
                for prompt in prompts:
                    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
                    # move input_ids to the same device as the model's primary parameters
                    if 'model_primary_device' in locals() and model_primary_device is not None:
                        try:
                            input_ids = input_ids.to(model_primary_device)
                        except Exception:
                            print("Warning: failed to move input_ids to model_primary_device; using CPU tensors")
                    else:
                        if hasattr(model, 'device'):
                            try:
                                input_ids = input_ids.to(model.device)
                            except Exception:
                                pass

                    # Use inference_mode and (if float16) autocast to speed generation
                    do_sample = not args.greedy
                    with torch.inference_mode():
                        # if using float16 and CUDA is available, use autocast for faster kernels
                        use_autocast = (args.dtype == "float16") and torch.cuda.is_available()
                        if use_autocast:
                            autocast_ctx = torch.cuda.amp.autocast(dtype=torch.float16)
                        else:
                            class _DummyCtx:
                                def __enter__(self_):
                                    return None
                                def __exit__(self_, exc_type, exc, tb):
                                    return False
                            autocast_ctx = _DummyCtx()

                        with autocast_ctx:
                            print(f"Generating for prompt (truncated): {prompt[:50]}...")
                            out = model.generate(
                                input_ids,
                                do_sample=do_sample,
                                num_return_sequences=args.num_return_sequences,
                                max_new_tokens=args.max_new_tokens,
                                temperature=args.temperature,
                                top_p=args.top_p,
                                pad_token_id=tokenizer.eos_token_id,
                            )
                            print(f"Generation complete for this prompt.")

                    # out shape: (num_return_sequences, seq_len) for single-input
                    gen_texts = []
                    # handle outputs when num_return_sequences may be 1
                    nrs = args.num_return_sequences if args.num_return_sequences >= 1 else 1
                    for r in range(nrs):
                        out_seq = out[r] if nrs > 1 else out[0]
                        gen_tokens = out_seq[input_ids.shape[-1]:]
                        gen_part = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
                        gen_texts.append(gen_part)
                    preds_for_batch.append(gen_texts)
            print(f"Generated predictions for batch.")
            # write records
            for (idx, q, choices, qid), pref_len, preds in zip(batch, prefix_lens, preds_for_batch):
                # compute token_length as prefix + max generated token length among preds
                gen_token_lens = []
                for p in preds:
                    if tokenizer is not None:
                        try:
                            gen_token_lens.append(len(tokenizer.encode(p, add_special_tokens=False)))
                        except Exception:
                            gen_token_lens.append(0)
                    else:
                        gen_token_lens.append(0)
                max_gen = max(gen_token_lens) if gen_token_lens else 0
                token_length = pref_len + max_gen

                rec = {
                    "query": q,
                    "prediction": preds,
                    "choices": choices,
                    "prefix_length": pref_len,
                    "token_length": token_length,
                    "question_id": qid
                }
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                write_n += 1

                # periodic progress update so logs show activity
                if write_n % 5 == 0:
                    print(f"Wrote {write_n}/{total} examples to {out_file}")
                    fo.flush()
                    sys.stdout.flush()

            i += len(batch)

    print(f"Wrote {write_n} examples to {out_file} (metadata={metadata})")


if __name__ == "__main__":
    main()
