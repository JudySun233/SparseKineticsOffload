"""
Fit GPU-dominant cost terms by sweeping (C_comp, C_mem_total) without LMCache.

Time-domain model (with intercept):

    T_i ≈ a * C_comp_i + b * C_mem_i + c

where

    T_i        : measured latency in seconds
    C_comp_i   : compute cost term from Kinetics formula (FLOPs-like units)
    C_mem_i    : KV memory access term, converted to "elements" instead of bytes
                 to better match the original Kinetics definition of D as
                 "KV size per token" in number of elements.
    c          : constant per-request overhead (scheduler, kernel launch, etc.)

We read C_mem_total_bytes from JSON (which is in bytes), then convert it to
an approximate "KV elements count" via:

    C_mem_elems = C_mem_bytes / bytes_per_element(dtype)

For bfloat16 / float16, bytes_per_element = 2.

Result:
    - Prints a, b, c, R^2
    - Saves a scatter plot of T_pred vs T_true with a y = x reference line.
"""

import json
import os
import argparse
from typing import List, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit GPU cost terms T ≈ a*C_comp + b*C_mem + c (C_mem in elements) and plot results."
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
        default="scripts/fit_I_gpu_regression_with_c.png",
        help="Path to save regression plot (PNG).",
    )
    return parser.parse_args()


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
    This is a simple heuristic; extend if more dtypes appear.
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
    # Fallback: assume 2 bytes per element
    return 2.0


def extract_arrays(configs: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract arrays of:
        - T: latency in seconds
        - C_comp: compute cost (unchanged)
        - C_mem_elems: KV memory access in *elements* (converted from bytes)
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

        # Infer bytes_per_element from dtype (once)
        if bytes_per_elem is None:
            dtype = cfg.get("dtype", "bfloat16")
            bytes_per_elem = dtype_to_bytes_per_element(dtype)
            print(f"[Info] Inferred bytes_per_element from dtype='{dtype}': {bytes_per_elem} bytes/elem")
        else:
            # Optional: sanity-check that dtype is consistent
            pass

        T_list.append(T)
        C_comp_list.append(C_comp)
        C_mem_bytes_list.append(C_mem_bytes)

    if len(T_list) == 0:
        raise ValueError("No valid samples found with latency_s, C_comp, C_mem_total > 0")

    if bytes_per_elem is None:
        # Fallback if no dtype was found at all
        bytes_per_elem = 2.0
        print("[Warning] Could not infer dtype; defaulting bytes_per_element = 2.0")

    T_arr = np.array(T_list, dtype=np.float64)
    C_comp_arr = np.array(C_comp_list, dtype=np.float64)
    C_mem_bytes_arr = np.array(C_mem_bytes_list, dtype=np.float64)

    # Convert bytes to approximate "KV elements"
    C_mem_elems_arr = C_mem_bytes_arr / bytes_per_elem

    return T_arr, C_comp_arr, C_mem_elems_arr


def fit_linear_with_intercept(
    T: np.ndarray,
    C_comp: np.ndarray,
    C_mem: np.ndarray,
) -> Tuple[float, float, float, np.ndarray, float]:
    """
    Fit T ≈ a*C_comp + b*C_mem + c (with intercept).

    We solve least squares for [a, b, c] in:

        T ≈ X @ [a, b, c]^T

    where X = [C_comp, C_mem, 1].
    """
    # Design matrix: [C_comp, C_mem, 1]
    ones = np.ones_like(C_comp)
    X = np.stack([C_comp, C_mem, ones], axis=1)

    # Solve min ||X w - T||_2
    coef, residuals, rank, s_vals = np.linalg.lstsq(X, T, rcond=None)
    a, b, c = coef

    T_pred = X @ coef
    ss_res = float(np.sum((T - T_pred) ** 2))
    ss_tot = float(np.sum((T - T.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return float(a), float(b), float(c), T_pred, r2


def plot_regression(T_true: np.ndarray, T_pred: np.ndarray, out_path: str):
    """
    Make a scatter plot of T_pred vs T_true, with a y=x reference line.
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(T_pred, T_true, alpha=0.6, edgecolor="none")
    min_val = float(min(T_true.min(), T_pred.min()))
    max_val = float(max(T_true.max(), T_pred.max()))
    plt.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1.0, label="y = x")

    plt.xlabel("Predicted latency T_pred (s)")
    plt.ylabel("Measured latency T_true (s)")
    plt.title("Fit GPU cost: T ≈ a*C_comp + b*C_mem_elems + c")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    args = parse_args()

    # 1. Load JSON
    configs = load_results(args.input)

    # 2. Extract T, C_comp, C_mem (in elements)
    T_arr, C_comp_arr, C_mem_arr = extract_arrays(configs)
    print(f"Loaded {len(T_arr)} valid samples from {args.input}")

    # 3. Fit T ≈ a*C_comp + b*C_mem + c
    a, b, c, T_pred, r2 = fit_linear_with_intercept(T_arr, C_comp_arr, C_mem_arr)

    # 4. Print fitted parameters
    print("=== Linear fit: T ≈ a*C_comp + b*C_mem_elems + c ===")
    print("Units (approximate):")
    print("  a : seconds per 'compute unit'         (s / C_comp)")
    print("  b : seconds per KV element             (s / C_mem_elems)")
    print("  c : constant per-request overhead      (s)")
    print()
    print(f"a = {a:.6e}")
    print(f"b = {b:.6e}")
    print(f"c = {c:.6e}")
    print(f"R^2 (goodness of fit) = {r2:.6f}")

    # 5. Plot T_pred vs T_true
    plot_regression(T_arr, T_pred, args.output_plot)
    print(f"Saved regression plot to: {args.output_plot}")


if __name__ == "__main__":
    main()
