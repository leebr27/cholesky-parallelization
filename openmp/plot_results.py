"""
Plot Part 2 (OpenMP) results: strong scaling and weak scaling.
Data from cholesky_openmp_108269.out as reported in README.md.

Usage:
    python plot_results.py
Outputs:
    openmp_performance.png
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Strong scaling data (N = 1024) ────────────────────────────────────────────
threads_strong = [1,  2,  4,  8,  16  ]
best_time_s    = [0.854345, 0.350302, 0.172024, 0.089407, 0.083800]
gflops_strong  = [0.419, 1.022, 2.081, 4.003, 4.271]
speedup        = [0.854345 / t for t in best_time_s]   # relative to 1-thread OpenMP

# ── Weak scaling data (N³/threads = const, base N=512 @ 1 thread) ────────────
threads_weak   = [1,   2,   4,   8,    16   ]
N_weak         = [512, 645, 812, 1024, 1290 ]
best_time_w    = [0.061153, 0.024440, 0.025532, 0.091712, 0.279336]
gflops_weak    = [0.732, 3.660, 6.990, 3.903, 2.562]
efficiency_w   = [1.00, 2.50, 2.39, 0.67, 0.22]   # from README (normalized to 1-thread)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
fig.suptitle("Part 2 — OpenMP Cholesky Scaling (N = 1024 strong, N³/P=const weak)",
             fontweight="bold")

# ── Panel 1: Strong scaling speedup ──────────────────────────────────────────
ax = axes[0]
ax.plot(threads_strong, speedup, "o-", color="steelblue", linewidth=2,
        markersize=7, label="Measured speedup")
ax.plot(threads_strong, threads_strong, "k--", linewidth=1.2, label="Ideal linear")
for x, y in zip(threads_strong, speedup):
    ax.annotate(f"{y:.2f}×", (x, y), textcoords="offset points",
                xytext=(5, 4), fontsize=8)
ax.set_xlabel("Threads")
ax.set_ylabel("Speedup (vs 1-thread)")
ax.set_title("Strong Scaling Speedup (N=1024)")
ax.set_xticks(threads_strong)
ax.legend(fontsize=9)
ax.grid(True, linestyle=":", alpha=0.6)

# ── Panel 2: Strong scaling GFLOP/s ──────────────────────────────────────────
ax = axes[1]
ax.plot(threads_strong, gflops_strong, "s-", color="darkorange", linewidth=2,
        markersize=7)
for x, y in zip(threads_strong, gflops_strong):
    ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                xytext=(5, 4), fontsize=8)
ax.set_xlabel("Threads")
ax.set_ylabel("GFLOP/s")
ax.set_title("Strong Scaling GFLOP/s (N=1024)")
ax.set_xticks(threads_strong)
ax.grid(True, linestyle=":", alpha=0.6)

# ── Panel 3: Weak scaling efficiency ─────────────────────────────────────────
ax = axes[2]
ax.plot(threads_weak, efficiency_w, "^-", color="seagreen", linewidth=2,
        markersize=7, label="Efficiency (normalised)")
ax.axhline(1.0, color="k", linestyle="--", linewidth=1.2, label="Ideal (1.0)")
# Annotate with N labels
for x, eff, n in zip(threads_weak, efficiency_w, N_weak):
    ax.annotate(f"N={n}\n{eff:.2f}×", (x, eff), textcoords="offset points",
                xytext=(4, 5), fontsize=7.5)
ax.set_xlabel("Threads")
ax.set_ylabel("Efficiency (best-time₁ × threads / best-timeₜ)")
ax.set_title("Weak Scaling Efficiency (N³/P = const)")
ax.set_xticks(threads_weak)
ax.legend(fontsize=9)
ax.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.savefig("openmp_performance.png", dpi=150, bbox_inches="tight")
print("Saved openmp_performance.png")
