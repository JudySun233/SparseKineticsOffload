# Kinetics++

**Kinetics++** extends test-time scaling laws to **single-GPU, memory-constrained inference** by explicitly modeling **CPU–GPU offloading costs**.  
It builds on *Kinetics* and introduces a practical cost model for real-world deployments where GPU memory is limited.

---

## Motivation

Existing test-time scaling laws assume:
- Large batch sizes  
- All model weights fit in GPU memory  
- Negligible CPU–GPU communication cost  

These assumptions fail in single-GPU settings. When models exceed VRAM, **CPU offloading over PCIe becomes the dominant cost**, fundamentally changing optimal scaling behavior.

---

## Key Idea

We introduce **rFLOPs (resource-aware FLOPs)**, a unified cost metric that accounts for:
- GPU computation
- GPU memory access
- CPU–GPU communication due to offloading

This enables principled **accuracy–cost analysis** under realistic hardware constraints.

---

## Main Findings

- Larger models are **more token-efficient**, reaching accuracy plateaus with fewer tokens.
- Once **CPU offloading is required**, larger models become **strictly Pareto-dominated** by smaller GPU-resident models.
- CPU–GPU communication rapidly dominates inference cost.
- rFLOPs accurately predicts real latency, showing linear scaling with generation length.

---

## Experimental Setup

- **Hardware**: Single NVIDIA A5000 (24GB VRAM)
- **Models**: Qwen3-1.7B / 4B / 8B / 14B
- **Benchmark**: AIME24
- **Inference Engines**:
  - `vLLM` for decoding
  - `llama.cpp` for layer-wise CPU offloading
- **Scaling Axes**:
  - Generation length
  - Offloading depth (number of layers offloaded)

---

## Repository Contents

### Directory Structure

- **`kinetics_cost_models/`** - Cost modeling and analysis
  - `long_CoT/` - Chain-of-thought scaling (generation length)
  - `best_of_N/` - Best-of-N sampling analysis

- **`scripts/`** - Experimental evaluation scripts
  - `cpu_weights_offload/` - CPU offloading experiments
  - `fit_hardware_coefficients/` - Hardware profiling and regression

- **`benchmarks/`** - Dense and sparse inference implementations

- **`results/`** - Experimental outputs and measurements
  - AIME24 results, cost analysis, GPU/CPU profiling data

- **`data/`** - Input datasets

---

## Key Takeaway

**Under single-GPU memory constraints, optimal test-time scaling favors smaller models that fully fit on the GPU.**  
CPU offloading introduces a communication-dominated regime that reshapes the Pareto frontier.

