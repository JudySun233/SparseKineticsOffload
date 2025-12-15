import numpy as np
import matplotlib.pyplot as plt

def ctts_cost(
    *,
    P,                # total params
    L,                # num layers
    D,                # KV dimension
    r,                # GQA ratio
    Lin,              # prompt length
    Lout,             # generation length
    N,                # num trials (1)
    Igpu,             # GPU memory intensity
    Icomm,            # CPU-GPU comm intensity
    K,                # num offloaded layers
    bytes_per_param=2 # FP16
):
    """
    Kinetics++ CTTS cost model
    Returns cost in FLOPs (can normalize later).
    """

    # Calculate parameters per layer
    P_layer = P / L
    bP_layer = bytes_per_param * P_layer

    # Linear layer compute cost
    linear_compute = 2 * N * P * Lout

    # Attention mechanism compute cost
    attention_compute = (
        2 * r * N * Lin * D * Lout +
        r * N * D * (Lout ** 2)
    )

    # GPU memory access cost
    memory_cost = Igpu * (
        2 * P * Lout +
        2 * Lin * D * Lout +
        N * D * (Lout ** 2)
    )

    # Weight offloading communication cost
    offload_cost = Icomm * (
        bP_layer * K * (Lin + N * Lout)
    )

    return linear_compute + attention_compute + memory_cost + offload_cost

results = {
    "Qwen-3-1.7B": {
        "P": 1.7e9,
        "L": 24,
        "acc": {
            2048: 0.10, 4096: 0.27, 6144: 0.27, 8192: 0.30,
            10240: 0.30, 12288: 0.30, 14336: 0.30,
            16384: 0.33, 18432: 0.33, 20480: 0.33,
            22528: 0.33, 24576: 0.33,
            28768: 0.33, 30768: 0.33, 32768: 0.33,
        }
    },
    "Qwen-3-4B": {
        "P": 4e9,
        "L": 32,
        "acc": {
            2048: 0.033, 4096: 0.200, 6144: 0.333, 8192: 0.367,
            10240: 0.433, 12288: 0.500, 14336: 0.533,
            16384: 0.533, 18432: 0.567, 20480: 0.567,
            22528: 0.600, 24576: 0.600,
            28296: 0.600, 32296: 0.600,
        }
    },
    "Qwen-3-8B": {
        "P": 8e9,
        "L": 36,
        "acc": {
            2048: 0.1333, 4096: 0.2000, 6144: 0.3333,
            8192: 0.4000, 10240: 0.4000, 12288: 0.4667,
            14336: 0.5000, 16384: 0.5667,
            18432: 0.5667, 20480: 0.5667,
            22528: 0.5667, 24576: 0.6000,
            28768: 0.6333, 30768: 0.6333, 32768: 0.6333,
        }
    },
    "Qwen-3-14B": {
        "P": 14e9,
        "L": 40,
        "acc": {
            2048: 0.0667, 4096: 0.1667, 6144: 0.3333,
            8192: 0.5000, 10240: 0.6000, 12288: 0.6000,
            14336: 0.6333, 16384: 0.6333,
            18432: 0.6333, 20480: 0.6333,
            22528: 0.6333, 24576: 0.6333,
            28768: 0.6333, 30768: 0.6333, 32768: 0.6333,
        }
    },
}

# Cost–Generation Length plot (Kinetics-style)
# Fixed experiment constants
Lin = 1024
N = 1
r = 1.0

# KV dimension (example – adjust if needed)
D = 128 * 8   # head_dim × num_kv_heads (proxy)

# Hardware constants (from your code)
Igpu = 36.16     # E_flops_A5000
Icomm = 881.6    # E_flops_CPU

# Offloading sweep
K_fracs = [0.0, 0.25, 0.5, 0.75, 1.0]

plt.figure(figsize=(8, 6))

for model_name, m in results.items():
    P, L = m["P"], m["L"]
    Louts = sorted(m["acc"].keys())

    # Baseline case with no offloading
    costs = [
        ctts_cost(
            P=P, L=L, D=D, r=r,
            Lin=Lin, Lout=Lout,
            N=N, Igpu=Igpu, Icomm=Icomm,
            K=0
        ) / 1e12
        for Lout in Louts
    ]

    plt.plot(costs, Louts, marker="o", label=model_name)

plt.xlabel("Cost (Tera-eFLOPs)")
plt.ylabel("Generation Length")
plt.title("Generation Length vs Cost (Kinetics++)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Cost–Accuracy plot (Figure 4 style, with offloading sweeps)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for ax, (model_name, m) in zip(axes, results.items()):
    P, L = m["P"], m["L"]

    for frac in K_fracs:
        K = int(frac * L)
        costs, accs = [], []

        for Lout, acc in sorted(m["acc"].items()):
            cost = ctts_cost(
                P=P, L=L, D=D, r=r,
                Lin=Lin, Lout=Lout,
                N=N, Igpu=Igpu, Icomm=Icomm,
                K=K
            )
            costs.append(cost / 1e12)
            accs.append(acc)

        ax.plot(costs, accs, marker="o", label=f"K/L={frac:.2f}")

    ax.set_title(model_name)
    ax.set_xlabel("Cost (Tera-eFLOPs)")
    ax.set_ylabel("Accuracy")
    ax.grid(True)
    ax.legend(fontsize=8)

plt.suptitle("Accuracy vs Cost under Layer Offloading (Kinetics++)")
plt.tight_layout()

# Show all plots at once
plt.show()
