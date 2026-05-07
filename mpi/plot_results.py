"""
Plot Part 3 (MPI) results: strong scaling speedup and GFLOP/s.
Data from cholesky_mpi_108317.out as reported in README.md.

Usage:
    python plot_results.py
Outputs:
    mpi_performance.png
"""

import matplotlib.pyplot as plt
import numpy as np

# ── Strong scaling data (N = 1024) ────────────────────────────────────────────
serial_time = 0.776923   # Part 1 baseline (N=1024)

ranks_strong  = [2,        4,        8,        16       ]
best_time_s   = [0.093178, 0.074562, 0.046557, 0.036596 ]
gflops_strong = [3.841,    4.800,    7.688,    9.780    ]
speedup       = [serial_time / t for t in best_time_s]

# ── Large problem data (N = 2048, 16 ranks) ───────────────────────────────────
# Serial baseline for N=2048 from Part 1
serial_time_2048 = 12.502958

large_ranks  = [16      ]
large_time   = [0.149886]
large_gflops = [19.103  ]
large_speedup = serial_time_2048 / large_time[0]

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
fig.suptitle("Part 3 — MPI Cholesky Scaling", fontweight="bold")

# ── Panel 1: Strong scaling speedup (N=1024) ─────────────────────────────────
ax = axes[0]
ax.plot(ranks_strong, speedup, "o-", color="steelblue", linewidth=2,
        markersize=7, label="Measured speedup")
ax.plot(ranks_strong, ranks_strong, "k--", linewidth=1.2, label="Ideal linear")
for x, s in zip(ranks_strong, speedup):
    ax.annotate(f"{s:.1f}×", (x, s), textcoords="offset points",
                xytext=(5, 4), fontsize=9)
ax.set_xlabel("MPI Ranks")
ax.set_ylabel(f"Speedup vs serial (t_serial = {serial_time:.3f} s)")
ax.set_title("Strong Scaling Speedup (N=1024)")
ax.set_xticks(ranks_strong)
ax.legend(fontsize=9)
ax.grid(True, linestyle=":", alpha=0.6)

# ── Panel 2: GFLOP/s — strong scaling + large problem bar ────────────────────
ax = axes[1]
x_pos = np.arange(len(ranks_strong) + 1)
labels = [str(r) for r in ranks_strong] + ["16\n(N=2048)"]
gflops_all = gflops_strong + large_gflops
colors = ["steelblue"] * len(ranks_strong) + ["darkorange"]

bars = ax.bar(x_pos, gflops_all, color=colors, edgecolor="white", linewidth=0.8)
for bar, g in zip(bars, gflops_all):
    ax.text(bar.get_x() + bar.get_width() / 2, g + 0.15,
            f"{g:.2f}", ha="center", va="bottom", fontsize=9)

ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.set_xlabel("MPI Ranks  (N=1024 unless noted)")
ax.set_ylabel("GFLOP/s")
ax.set_title("Achieved GFLOP/s")
ax.set_ylim(0, max(gflops_all) * 1.2)

# Legend patches
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="steelblue", label="N=1024"),
                   Patch(color="darkorange", label="N=2048")],
          fontsize=9)
ax.grid(True, axis="y", linestyle=":", alpha=0.6)

plt.tight_layout()
plt.savefig("mpi_performance.png", dpi=150, bbox_inches="tight")
print("Saved mpi_performance.png")
