"""
regression_I_cpu.py

Fit CPU↔GPU KV offloading bandwidth using LMCache logs.

We assume a simple linear time model for offloading:

    time_i ≈ alpha_cpu * bytes_i

where:
    bytes_i : total KV bytes transferred via CPU↔GPU offload for sample i
    time_i  : total offload time (seconds) for sample i
    alpha_cpu : seconds per byte (1 / bandwidth)

Instead of doing a full least-squares, we use the effective bandwidth:

    BW_cpu_gpu_fit = sum_i bytes_i / sum_i time_i    [bytes / second]

which is equivalent to a bytes-weighted average slope.

Then:

    alpha_cpu = 1 / BW_cpu_gpu_fit

We compute R^2 using:
    time_true_i  = measured offload time
    time_pred_i  = bytes_i / BW_cpu_gpu_fit

and also save a plot of:
    x: bytes (GB)
    y: time (ms)
    with a fitted line.

We do NOT use total end-to-end latency_s here, only the LMCache offload time.
"""

import json
import os
import argparse
from typing import List, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit CPU-GPU offload bandwidth from LMCache logs and plot regression."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "/home/haojias/SparseKineticsOffload/results/cpu_offloading_results/fit_I_cpu_results_cpu1.0gb.json",
            "/home/haojias/SparseKineticsOffload/results/cpu_offloading_results/fit_I_cpu_results_cpu2.0gb.json",
            "/home/haojias/SparseKineticsOffload/results/cpu_offloading_results/fit_I_cpu_results_cpu4.0gb.json",
            "/home/haojias/SparseKineticsOffload/results/cpu_offloading_results/fit_I_cpu_results_cpu8.0gb.json",
        ],
        help="List of JSON files produced by fit_I_cpu experiments.",
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default="../results/cpu_offloading_results/fit_I_cpu_regression.png",
        help="Path to save regression plot (PNG).",
    )
    parser.add_argument(
        "--min_size_gb",
        type=float,
        default=0.0,
        help="Optional: filter out samples with lmcache_total_size_gb < min_size_gb "
             "to reduce noise from very small transfers.",
    )
    return parser.parse_args()


def load_records(paths: List[str]) -> List[Dict]:
    """Load and concatenate all records from multiple JSON files."""
    all_records: List[Dict] = []
    for p in paths:
        if not os.path.isfile(p):
            print(f"[Warning] Input file not found, skip: {p}")
            continue
        with open(p, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            all_records.extend(data)
        else:
            print(f"[Warning] File {p} does not contain a list; skip.")
    if not all_records:
        raise RuntimeError("No valid records loaded from given input files.")
    return all_records


def extract_offload_samples(
    records: List[Dict],
    min_size_gb: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract offload bytes and times from LMCache statistics.

    Returns:
        bytes_arr: shape (n,), float64, total offloaded bytes per sample
        time_arr : shape (n,), float64, total offload time per sample [seconds]
    """
    bytes_list: List[float] = []
    time_list: List[float] = []

    for rec in records:
        size_gb = rec.get("lmcache_total_size_gb", None)
        size_bytes = rec.get("lmcache_total_size_bytes", None)
        cost_ms = rec.get("lmcache_total_cost_ms", None)

        if size_bytes is None or cost_ms is None or size_gb is None:
            continue

        # Filter out invalid / zero entries
        if size_bytes <= 0.0 or cost_ms <= 0.0:
            continue

        if size_gb < min_size_gb:
            # Skip very small transfers to reduce noise
            continue

        bytes_list.append(float(size_bytes))
        time_list.append(float(cost_ms) / 1000.0)  # convert ms -> seconds

    if not bytes_list:
        raise RuntimeError("No valid LMCache offload samples after filtering.")

    bytes_arr = np.asarray(bytes_list, dtype=np.float64)
    time_arr = np.asarray(time_list, dtype=np.float64)

    return bytes_arr, time_arr


def fit_bandwidth(bytes_arr: np.ndarray, time_arr: np.ndarray) -> Tuple[float, float, np.ndarray, float]:
    """
    Fit effective CPU-GPU bandwidth using:

        BW_fit = sum(bytes_i) / sum(time_i)   [bytes / second]

    Then compute:
        time_pred_i = bytes_i / BW_fit

    Also compute:
        R^2 between time_true and time_pred.

    Returns:
        bw_bytes_per_s, alpha_cpu, time_pred, r2
    """
    total_bytes = float(np.sum(bytes_arr))
    total_time = float(np.sum(time_arr))

    bw_bytes_per_s = total_bytes / total_time
    alpha_cpu = 1.0 / bw_bytes_per_s  # seconds per byte

    time_pred = bytes_arr / bw_bytes_per_s

    ss_res = float(np.sum((time_arr - time_pred) ** 2))
    ss_tot = float(np.sum((time_arr - time_arr.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return bw_bytes_per_s, alpha_cpu, time_pred, r2


def plot_regression(
    bytes_arr: np.ndarray,
    time_arr: np.ndarray,
    time_pred: np.ndarray,
    out_path: str,
):
    """
    Plot offload time vs offload bytes with fitted line.

    x-axis : offloaded bytes (GB)
    y-axis : offload time (ms)
    """
    bytes_gb = bytes_arr / 1e9
    time_ms = time_arr * 1e3
    time_pred_ms = time_pred * 1e3

    plt.figure(figsize=(6, 6))
    plt.scatter(bytes_gb, time_ms, alpha=0.6, label="Measured", edgecolor="none")
    # Fit line in GB-ms space
    # Because BW_fit is constant, time_pred is a straight line through origin
    # We can just plot a line from 0 to max(bytes).
    x_line = np.linspace(0.0, bytes_gb.max() * 1.05, 100)
    # slope in ms/GB:
    # time = bytes / BW  =>  t(ms) = (bytes_GB * 1e9 / BW_bytes_per_s) * 1e3
    #                      = bytes_GB * (1e12 / BW_bytes_per_s)
    if bytes_gb.max() > 0:
        slope_ms_per_gb = (time_pred_ms.max() / bytes_gb.max())
    else:
        slope_ms_per_gb = 0.0
    y_line = slope_ms_per_gb * x_line
    plt.plot(x_line, y_line, "k--", label="Fitted line")

    plt.xlabel("Offloaded KV size (GB)")
    plt.ylabel("Offload time (ms)")
    plt.title("CPU-GPU KV offload: time vs size")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    args = parse_args()

    # Load records from all input JSON files
    records = load_records(args.inputs)
    print(f"Loaded {len(records)} records from {len(args.inputs)} files.")

    # Extract offload bytes & times
    bytes_arr, time_arr = extract_offload_samples(records, min_size_gb=args.min_size_gb)
    print(f"Using {len(bytes_arr)} samples after filtering (min_size_gb={args.min_size_gb}).")

    # Fit effective bandwidth
    bw_bytes_per_s, alpha_cpu, time_pred, r2 = fit_bandwidth(bytes_arr, time_arr)
    bw_gb_per_s = bw_bytes_per_s / 1e9

    print("=== CPU-GPU offload bandwidth fit (through origin) ===")
    print(f"BW_cpu_gpu_fit (bytes/s)  = {bw_bytes_per_s:.6e}")
    print(f"BW_cpu_gpu_fit (GB/s)     = {bw_gb_per_s:.3f} GB/s")
    print(f"alpha_cpu (s/byte)        = {alpha_cpu:.6e}")
    print(f"R^2 (time vs bytes model) = {r2:.6f}")

    # Plot regression
    plot_regression(bytes_arr, time_arr, time_pred, args.output_plot)
    print(f"Saved regression plot to: {args.output_plot}")


if __name__ == "__main__":
    main()
