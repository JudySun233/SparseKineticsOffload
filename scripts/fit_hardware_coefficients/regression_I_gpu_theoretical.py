"""
Fit GPU-dominant cost terms by sweeping (C_comp, C_mem_total) without LMCache.

We use a *single-parameter* time-domain model:

    T_i ≈ γ_gpu * C_dense,i

where

    C_dense,i = C_comp,i + I_gpu * C_mem_elems,i

Here:
    - C_comp_i      : compute cost term from Kinetics dense model (FLOPs-like units).
    - C_mem_elems_i : KV memory access term in *elements* (we convert from bytes
                      using dtype's bytes-per-element to match Kinetics' notion
                      of D as "KV size per token" in number of elements).
    - I_gpu         : fixed theoretical weight on KV memory, derived from
                      B200's I and A5000's FLOPs/BW ratio (we use 5.78e2).
    - γ_gpu         : fitted time-per-'dense unit' scaling factor.

We read C_comp and C_mem_total (bytes) from JSON, convert C_mem_total to elements,
compute C_dense, fit γ_gpu with least squares (no intercept), and plot T_pred vs T_true.

Result:
    - Prints I_gpu (theoretical), γ_gpu, R^2.
    - Saves a scatter plot of T_pred vs T_true with a y = x reference line.
"""

import json
import os
import argparse
from typing import List, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Argument parsing
# -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit GPU cost terms T ≈ γ_gpu * (C_comp + I_gpu * C_mem_elems) and plot results."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="/home/haojias/SparseKineticsOffload/results/gpu_without_offloading_results/fit_I_gpu_results.json",
        help="Path to fit_I_gpu_results.json",
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default="../results/gpu_without_offloading_results/fit_I_gpu_dense_theoretical.png",
        help="Path to save regression plot (PNG).",
    )
    parser.add_argument(
        "--igpu",
        type=float,
        default=5.78e2,
        help="Theoretical I_gpu (weight on C_mem_elems). Default 5.78e2.",
    )
    return parser.parse_args()


# -----------------------------
# Helpers
# -----------------------------
def load_results(path: str) -> List[Dict]:
    """Load list of result dicts from JSON file."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Input JSON file not found: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list of configs, got type {type(data)}")
    return data


def dtype_to_bytes_per_element(dtype: str) -> float:
    """
    Map dtype string to bytes per element.
    Extend this mapping if more dtypes appear.
    """
    d = dtype.lower()
    if "bfloat16" in d or d in ("bf16",):
        return 2.0
    if d in ("float16", "fp16", "half"):
        return 2.0
    if d in ("float32", "fp32"):
        return 4.0
    if d in ("float64", "fp64", "double"):
        return 8.0
    if "int8" in d or "uint8" in d:
        return 1.0
    # Fallback if unknown: assume 2 bytes/element
    return 2.0


def extract_arrays(configs: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract arrays of:
        T          : latency in seconds
        C_comp     : compute cost (as stored in JSON)
        C_mem_elem : KV memory access in *elements* (converted from bytes)
    """
    T_list = []
    C_comp_list = []
    C_mem_bytes_list = []

    bytes_per_elem = None

    for cfg in configs:
        if "latency_s" not in cfg or "C_comp" not in cfg or "C_mem_total" not in cfg:
            continue

        T = float(cfg["latency_s"])
        C_comp = float(cfg["C_comp"])
        C_mem_bytes = float(cfg["C_mem_total"])

        if T <= 0 or C_comp <= 0 or C_mem_bytes <= 0:
            continue

        if bytes_per_elem is None:
            # Infer bytes_per_element from dtype (default to bf16 if missing)
            dtype = cfg.get("dtype", "bfloat16")
            bytes_per_elem = dtype_to_bytes_per_element(dtype)
            print(f"[Info] Inferred bytes_per_element from dtype='{dtype}': {bytes_per_elem} bytes/elem")

        T_list.append(T)
        C_comp_list.append(C_comp)
        C_mem_bytes_list.append(C_mem_bytes)

    if len(T_list) == 0:
        raise ValueError("No valid samples found with latency_s, C_comp, C_mem_total > 0")
    if bytes_per_elem is None:
        bytes_per_elem = 2.0
        print("[Warning] Could not infer dtype; defaulting bytes_per_element = 2.0")

    T_arr = np.array(T_list, dtype=np.float64)
    C_comp_arr = np.array(C_comp_list, dtype=np.float64)
    C_mem_bytes_arr = np.array(C_mem_bytes_list, dtype=np.float64)

    # Convert bytes to approximate "KV elements"
    C_mem_elems_arr = C_mem_bytes_arr / bytes_per_elem

    return T_arr, C_comp_arr, C_mem_elems_arr


def fit_gamma_given_Igpu(
    T: np.ndarray,
    C_comp: np.ndarray,
    C_mem_elems: np.ndarray,
    I_gpu: float,
) -> Tuple[float, np.ndarray, float]:
    """
    For a fixed I_gpu, fit γ_gpu in:

        C_dense = C_comp + I_gpu * C_mem_elems
        T      ≈ γ_gpu * C_dense

    Least squares with zero intercept:

        γ_gpu = (C_dense^T T) / (C_dense^T C_dense)
    """
    C_dense = C_comp + I_gpu * C_mem_elems

    denom = float(np.dot(C_dense, C_dense))
    if denom <= 0:
        raise RuntimeError("Non-positive denominator in gamma fit; check C_dense.")

    gamma_gpu = float(np.dot(C_dense, T) / denom)
    T_pred = gamma_gpu * C_dense

    ss_res = float(np.sum((T - T_pred) ** 2))
    ss_tot = float(np.sum((T - T.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return gamma_gpu, T_pred, r2


def plot_regression(T_true: np.ndarray, T_pred: np.ndarray, out_path: str):
    """
    Scatter plot: T_pred vs T_true, with y=x reference line.
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(T_pred, T_true, alpha=0.6, edgecolor="none")
    min_val = float(min(T_true.min(), T_pred.min()))
    max_val = float(max(T_true.max(), T_pred.max()))
    plt.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1.0, label="y = x")

    plt.xlabel("Predicted latency T_pred (s)")
    plt.ylabel("Measured latency T_true (s)")
    plt.title("Fit GPU cost: T ≈ γ_gpu * (C_comp + I_gpu * C_mem_elems)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()



def main():
    args = parse_args()

    # Load JSON configs
    configs = load_results(args.input)

    # Extract T, C_comp, C_mem_elems
    T_arr, C_comp_arr, C_mem_elems_arr = extract_arrays(configs)
    print(f"Loaded {len(T_arr)} valid samples from {args.input}")

    # Fixed theoretical I_gpu (weight on C_mem_elems)
    I_gpu = float(args.igpu)
    print(f"[Info] Using theoretical I_gpu = {I_gpu:.6e} (per KV element)")

    # Fit gamma_gpu for this fixed I_gpu
    gamma_gpu, T_pred, R2 = fit_gamma_given_Igpu(
        T_arr,
        C_comp_arr,
        C_mem_elems_arr,
        I_gpu,
    )

    # Print formulas and fitted parameters
    print("=== Single-parameter fit: T_i ≈ γ_gpu * C_dense,i ===")
    print("Model definition:")
    print("    C_mem_elems,i = C_mem_bytes,i / bytes_per_element")
    print("    C_dense,i     = C_comp,i + I_gpu * C_mem_elems,i")
    print("    T_i           ≈ γ_gpu * C_dense,i")
    print()
    print(f"I_gpu (theoretical, per KV elem) = {I_gpu:.6e}")
    print(f"γ_gpu (seconds per 'dense unit') = {gamma_gpu:.6e}")
    print(f"R^2 (goodness of fit)            = {R2:.6f}")

    # Plot regression quality
    plot_regression(T_arr, T_pred, args.output_plot)
    print(f"Saved regression plot to: {args.output_plot}")


if __name__ == "__main__":
    main()
