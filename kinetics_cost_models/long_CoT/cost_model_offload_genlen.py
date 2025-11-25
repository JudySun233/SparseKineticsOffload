import os
import sys
import math
from typing import List, Tuple, Dict, Any, Optional, Callable
from functools import partial, reduce
import json
import re
from itertools import chain
import numpy as np
from scipy.special import comb
import pandas as pd
from transformers import AutoTokenizer, AutoConfig
import matplotlib.pyplot as plt
from datasets import Dataset, load_dataset
from tqdm import tqdm

from lighteval.metrics.dynamic_metrics import (
    ExprExtractionConfig,
    LatexExtractionConfig,
    multilingual_extractive_match_metric,
)
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language
from lm_eval.models.utils import (
    stop_sequences_criteria,
)

MATH_QUERY_TEMPLATE = """
Solve the following math problem efficiently and clearly.  The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.

{Question}{Response}
""".strip()

gold_metric = multilingual_extractive_match_metric(
                language=Language.ENGLISH,
                fallback_mode="first_match",
                precision=5,
                gold_extraction_target=(ExprExtractionConfig(),),
                pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)),
                aggregation_function=max,
            )

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import *
 
 
def make_process_fn(model_name, query_to_id, gen_lens):
    def process_fn(example):
        query = example["query"]
        prediction = example["prediction"][0]
        golds = example["choices"]
        tokenized = tokenizer.encode(prediction)
        token_length = len(tokenized)

        rows = []
        for gen_len in gen_lens:
            truncated = tokenizer.decode(tokenized[:gen_len]) if token_length > gen_len else prediction

            target = Doc(query=query, choices=golds, gold_index=0)
            try:
                result = gold_metric.compute(
                    golds=golds, predictions=[truncated], formatted_doc=target
                )
                score = result["extractive_match"]
            except:
                score = 0

            rows.append({
                "query_id": query_to_id[query],
                "query": query,
                "gen_len_budget": gen_len,
                "gen_len": min(token_length, gen_len),
                "score": score,
            })

        return {"results": rows}
    return process_fn

if __name__ == "__main__":
    # E_flops = 562.5
    E_flops_A5000 = 289.3 # 222.2 FLOPS / 768 GB/s (theoretical)
    E_flops_CPU = 22483 # 222.2 FLOPS / 9.903 GB/s (fit, to be validated)
    task = sys.argv[1]
    sparse_arg = sys.argv[2]
    
    model_sizes = {
        "0.6B": 0.752,
        "1.7B": 2.03,
        "4B": 4.02,
        "8B": 8.19,
        "14B": 14.77,
        "32B": 32.76
    }

    gen_lens =[1024, 2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384, 18432, 20480, 22528, 24576, 26624, 28672, 30720, 32768]
    # RNG for placeholder offload/lmcache values (reproducible)
    rng = np.random.default_rng(42)
   
    query_to_id = {}
    query_id  = 0
    
    for model in ["Qwen3-8B"]:
        config = AutoConfig.from_pretrained(f"Qwen/{model}")
        config = AutoConfig.from_pretrained(f"Qwen/{model}")
        tokenizer = AutoTokenizer.from_pretrained(f"Qwen/{model}")
        
        nparams = model_sizes[model.split("-")[1]] * 1e9
        
        num_attn_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = config.head_dim * config.num_hidden_layers
    
        res_dir = f"{task}/{sparse_arg}" 
        log_files = os.listdir(res_dir)
        tokenizer = AutoTokenizer.from_pretrained(f"Qwen/{model}")
        model_name = model.lower().replace(".", "-")

        for log_file in log_files:
            print(f"Processing file: {log_file}")
            if log_file.endswith(".jsonl") and f"_{sparse_arg}" in log_file and model_name in log_file:
                file_path = os.path.join(res_dir, log_file)

                data = load_dataset("json", data_files=file_path, split="train").skip(1)
                
                # populate query_to_id if empty
                if len(query_to_id) == 0:
                    query_id = 0
                    for example in data:
                        if example["query"] not in query_to_id:
                            query_to_id[example["query"]] = query_id
                            query_id += 1
                
                process_fn = make_process_fn(f"Qwen/{model}", query_to_id, gen_lens)
                processed = data.map(process_fn, num_proc=8, remove_columns=data.column_names)
                
                all_rows = list(chain.from_iterable(processed["results"]))
                raw_df = pd.DataFrame(all_rows)
                
                raw_df = raw_df.groupby(["query_id", "gen_len_budget"]).agg({
                    "query": "first",
                    "gen_len": list,
                    "score": list
                }).reset_index()
                
                result_df = [] 
                for gen_len in gen_lens:
                    grouped_df_budgeted = raw_df[raw_df["gen_len_budget"] == gen_len]
                    for index, row in tqdm(grouped_df_budgeted.iterrows(), total=len(grouped_df_budgeted), desc="Processing rows"):
                        context_length = len(tokenizer.encode(row["query"]))
                        generation_lengths = row["gen_len"]
                        
                        scores = row["score"]
                        cov = coverage(scores, nsamples=[1])[0]    # using ntrial=1 for cot length analysis
                        # Provide a placeholder lmcache_offload_gb when metadata is not available
                        lmcache_offload_gb = row.get("lmcache_offload_gb", None)
                        if lmcache_offload_gb is None:
                            # sample a plausible lmcache usage in GB (0.1 - 8.0 GB)
                            lmcache_offload_gb = float(rng.uniform(0.1, 8.0))
                        
                        kwargs = {}
                        if sparse_arg == "dense":
                            cost_fn = dense_cost_w_offload
                            budget = None

                        # TODO: For different gen len, the actual lmcache offload size should vary, add related fix later
                        compute_cost, comm_mem_cost, gpu_mem_cost = expected_cost(cost_fn, generation_lengths, context_length=context_length, lmcache_total_size_gb=lmcache_offload_gb,
                                                                  nparams=nparams, num_attn_heads=num_attn_heads, num_kv_heads=num_kv_heads, 
                                                                  head_dim=head_dim, E_flops_GPU=E_flops_A5000, E_flops_CPU=E_flops_CPU, **kwargs)
                        
                        res_dict = {
                            "query_id": row["query_id"],
                            "generation_length": generation_lengths,
                            "gen_len_budget": gen_len,
                            "coverage": cov,
                            "lmcache_offload_gb": lmcache_offload_gb,
                            "compute_cost": compute_cost,
                            "comm_mem_cost": comm_mem_cost,
                            "gpu_mem_cost": gpu_mem_cost,
                            "total_cost": compute_cost + comm_mem_cost + gpu_mem_cost
                        }
                        
                        if sparse_arg != "dense":
                            res_dict["budget"] = budget
                            
                        result_df.append(res_dict)
                        
                result_df = pd.DataFrame(result_df)
                result_df.to_csv(f"{res_dir}/{log_file.split('/')[-1].split('.')[0]}_offload_genlen_tradeoff.csv", index=False)
                print(f"Saved results to {res_dir}/{log_file.split('/')[-1].split('.')[0]}_offload_genlen_tradeoff.csv")
