"""
Fit GPU-dominant cost terms by sweeping (C_comp, C_mem_total) without LMCache.

Model:
    T_i ≈ a * C_comp_i + b * C_mem_total_i

Where:
    T_i           = latency_s (measured wall-clock time)
    C_comp_i      = compute cost from Kinetics formula
    C_mem_total_i = total KV memory cost from Kinetics formula

Result:
    Prints a, b, R^2 and saves a scatter plot:
        x-axis: T_pred = a*C_comp + b*C_mem_total
        y-axis: T_true = latency_s
"""

import json
import os
import argparse
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit GPU cost terms T ≈ a*C_comp + b*C_mem_total and plot results."
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
        default="../results/gpu_without_offloading_results/fit_I_gpu_regression.png",
        help="Path to save regression plot (PNG).",
    )
    return parser.parse_args()


def load_results(path: str) -> List[Dict]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Input JSON file not found: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list of configs, got type {type(data)}")
    return data


def extract_arrays(configs: List[Dict]):
    T_list = []
    C_comp_list = []
    C_mem_list = []

    for cfg in configs:
        if "latency_s" not in cfg or "C_comp" not in cfg or "C_mem_total" not in cfg:
            continue

        T = float(cfg["latency_s"])
        C_comp = float(cfg["C_comp"])
        C_mem = float(cfg["C_mem_total"])

        if T <= 0 or C_comp <= 0 or C_mem < 0:
            continue

        T_list.append(T)
        C_comp_list.append(C_comp)
        C_mem_list.append(C_mem)

    if len(T_list) == 0:
        raise ValueError("No valid samples found with latency_s, C_comp, C_mem_total > 0")

    T_arr = np.array(T_list, dtype=np.float64)              # shape (n,)
    C_comp_arr = np.array(C_comp_list, dtype=np.float64)    # shape (n,)
    C_mem_arr = np.array(C_mem_list, dtype=np.float64)      # shape (n,)

    return T_arr, C_comp_arr, C_mem_arr


def fit_linear_no_intercept(T: np.ndarray, C_comp: np.ndarray, C_mem: np.ndarray):
    """
    min ||X @ [a,b]^T - T||_2
    """
    # X shape: (n_samples, 2)
    X = np.stack([C_comp, C_mem], axis=1)  # [C_comp, C_mem_total]

    # lstsq returns (coef, residuals, rank, s)
    coef, residuals, rank, s_vals = np.linalg.lstsq(X, T, rcond=None)
    a, b = coef[0], coef[1]

    T_pred = X @ coef
    ss_res = np.sum((T - T_pred) ** 2)
    ss_tot = np.sum((T - T.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return a, b, T_pred, r2


def plot_regression(T_true: np.ndarray, T_pred: np.ndarray, out_path: str):
    """
    Plot T_pred vs T_true scattered graph
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(T_pred, T_true, alpha=0.6, edgecolor="none")
    min_val = min(T_true.min(), T_pred.min())
    max_val = max(T_true.max(), T_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1.0, label="y = x")

    plt.xlabel("Predicted latency T_pred (s)")
    plt.ylabel("Measured latency T_true (s)")
    plt.title("Fit GPU cost: T ≈ a*C_comp + b*C_mem_total")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    args = parse_args()
    configs = load_results(args.input)

    # get T, C_comp, C_mem_total
    T_arr, C_comp_arr, C_mem_arr = extract_arrays(configs)
    print(f"Loaded {len(T_arr)} valid samples from {args.input}")

    # T ≈ a*C_comp + b*C_mem
    a, b, T_pred, r2 = fit_linear_no_intercept(T_arr, C_comp_arr, C_mem_arr)

    print("=== Linear fit: T ≈ a*C_comp + b*C_mem_total ===")
    print(f"a (s per compute unit)      = {a:.6e}")
    print(f"b (s per HBM byte)          = {b:.6e}")
    print(f"R^2 (goodness of fit)       = {r2:.6f}")

    plot_regression(T_arr, T_pred, args.output_plot)
    print(f"Saved regression plot to: {args.output_plot}")


if __name__ == "__main__":
    main()
