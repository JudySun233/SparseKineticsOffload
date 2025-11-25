import argparse
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.vllm.vllm_model import VLLMModel, VLLMModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
import pprint

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--benchmarks", default="aime24")
    p.add_argument("--max-new-tokens", type=int, default=32768)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--max-model-length", type=int, default=33280)
    p.add_argument("--swap-space", type=int, default=16, help="GB of host swap space for KV cache offload")
    p.add_argument("--max-num-seqs", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature; 0.0 means greedy")
    p.add_argument("--top-k", type=int, default=50, help="top-k sampling")
    p.add_argument("--top-p", type=float, default=0.95, help="top-p (nucleus) sampling")
    p.add_argument("--max-samples", type=int, default=1)
    p.add_argument("--output-dir", default="./aime24/dense", help="Directory to write evaluation results")
    return p.parse_args()


args = parse_args()

MODEL_NAME = args.model
BENCHMARKS = args.benchmarks

TASKS = f"lighteval|{BENCHMARKS}|0|0"

evaluation_tracker = EvaluationTracker(output_dir=args.output_dir)
pipeline_params = PipelineParameters(
    launcher_type=ParallelismManager.NONE,
    max_samples=args.max_samples,
)

generation_parameters = {
    "max_new_tokens": args.max_new_tokens,
    "temperature": float(args.temperature),
    "top_k": int(args.top_k),
    "top_p": float(args.top_p),
}


config = VLLMModelConfig(
    model_name=MODEL_NAME,
    generation_parameters=generation_parameters,
    use_chat_template=True,
    gpu_memory_utilization=args.gpu_memory_utilization,
    max_model_length=args.max_model_length,
    swap_space=args.swap_space,
    max_num_seqs=args.max_num_seqs,
)

model = None
inst_err = None
model = VLLMModel(config)

pipeline = Pipeline(
    model=model,
    pipeline_parameters=pipeline_params,
    evaluation_tracker=evaluation_tracker,
    tasks=TASKS,
)

results = pipeline.evaluate()
pipeline.show_results()
results = pipeline.get_results()
pprint.pprint(type(results))

import os, json

def to_dense_record(sample_id, resp):
    query = resp.get("query", "") if isinstance(resp, dict) else ""
    predictions = resp.get("predictions", []) if isinstance(resp, dict) else []
    pred_text = predictions[0] if predictions else resp.get("text", "")
    record = {
        "id": sample_id,
        "query": query,
        "prediction": [pred_text],
        # fill 'choices' if you have gold options, else empty list
        "choices": resp.get("choices", []),
        # optional fields you can compute or leave
        "prefix_length": len(query.split()),
        "token_length": len(pred_text.split()),
    }
    return record

os.makedirs(args.output_dir, exist_ok=True)
out_dense = os.path.join(args.output_dir, "aime24_dense.jsonl")
with open(out_dense, "w", encoding="utf-8") as fw:
    if isinstance(results, dict):
        iterator = results.items()
    else:
        iterator = enumerate(results)
    for sid, resp in iterator:
        record = to_dense_record(sid, resp)
        json.dump(record, fw, ensure_ascii=False)
        fw.write("\n")
print("Wrote dense JSONL:", out_dense)
