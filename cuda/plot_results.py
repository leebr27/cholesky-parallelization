"""
Plot Part 4 (CUDA) results: GFLOP/s and speedup vs matrix size N,
plus a Nsight Systems kernel time breakdown pie chart.
Data from cholesky_cuda_108357.out and nsys profile as reported in README.md.

Usage:
    python plot_results.py
Outputs:
    cuda_performance.png
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Timing data ───────────────────────────────────────────────────────────────
# Serial baselines from Part 1
serial = {256: 0.003062, 512: 0.058234, 1024: 0.776923, 2048: 12.502958}

N          = [512,     1024,    2048,    3072    ]
best_time  = [0.008844, 0.021121, 0.072146, 0.240840]
gflops     = [5.059,   16.946,  39.688,  40.125  ]
speedup    = [serial.get(n, None) for n in N]
# N=3072 serial not measured; extrapolate via O(N³) from N=2048 baseline
serial_3072_est = serial[2048] * (3072 / 2048) ** 3
speedup = [
    serial[512]  / best_time[0],
    serial[1024] / best_time[1],
    serial[2048] / best_time[2],
    serial_3072_est / best_time[3],
]

# ── Nsight kernel breakdown (N=2048, GPU time) ────────────────────────────────
kernel_labels = ["rank1_update_kernel\n(254.0 ms, 95.1%)",
                 "scale_column_kernel\n(13.2 ms, 4.9%)"]
kernel_times  = [254.0, 13.2]
kernel_colors = ["steelblue", "lightskyblue"]

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
fig.suptitle("Part 4 — CUDA Cholesky Performance (NVIDIA L4 / A10G)",
             fontweight="bold")

# ── Panel 1: GFLOP/s vs N ────────────────────────────────────────────────────
ax = axes[0]
ax.plot(N, gflops, "o-", color="steelblue", linewidth=2, markersize=7)
for x, g in zip(N, gflops):
    ax.annotate(f"{g:.1f}", (x, g), textcoords="offset points",
                xytext=(5, 4), fontsize=9)
ax.set_xlabel("Matrix size N")
ax.set_ylabel("GFLOP/s")
ax.set_title("GFLOP/s vs N (GPU, L4)")
ax.set_xticks(N)
ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
ax.set_ylim(0, max(gflops) * 1.25)
ax.grid(True, linestyle=":", alpha=0.6)

# ── Panel 2: Speedup vs serial ───────────────────────────────────────────────
ax = axes[1]
bars = ax.bar([str(n) for n in N], speedup,
              color=["lightcoral" if s < 1 else "steelblue" for s in speedup],
              edgecolor="white", linewidth=0.8)
ax.axhline(1.0, color="k", linestyle="--", linewidth=1.0, label="Break-even (1×)")
for bar, s in zip(bars, speedup):
    ax.text(bar.get_x() + bar.get_width() / 2,
            s + (max(speedup) * 0.01),
            f"{s:.1f}×", ha="center", va="bottom", fontsize=9)
ax.set_xlabel("Matrix size N")
ax.set_ylabel("Speedup vs serial CPU")
ax.set_title("GPU Speedup vs Serial Baseline")
ax.set_ylim(0, max(speedup) * 1.2)
ax.legend(fontsize=9)
ax.grid(True, axis="y", linestyle=":", alpha=0.6)
# Annotate N=512 as sub-1×
ax.annotate("Kernel-launch\noverhead dominates", xy=(0, speedup[0]),
            xytext=(0.6, speedup[0] * 3),
            fontsize=8, color="tomato",
            arrowprops=dict(arrowstyle="->", color="tomato"))

# ── Panel 3: Nsight kernel time pie (N=2048) ─────────────────────────────────
ax = axes[2]
wedges, texts, autotexts = ax.pie(
    kernel_times, labels=kernel_labels, colors=kernel_colors,
    autopct="%1.1f%%", startangle=90,
    textprops={"fontsize": 8.5},
    wedgeprops={"linewidth": 0.8, "edgecolor": "white"},
)
for at in autotexts:
    at.set_fontsize(9)
    at.set_fontweight("bold")
ax.set_title("GPU Kernel Time Breakdown\n(Nsight, N=2048, A10G)")

plt.tight_layout()
plt.savefig("cuda_performance.png", dpi=150, bbox_inches="tight")
print("Saved cuda_performance.png")
