"""
Plot Part 5 (mpi4py) results: mpi4py vs C++ MPI comparison —
strong scaling times, GFLOP/s, and overhead factor.
Data from cholesky_mpi4py_108385.out as reported in README.md.

Usage:
    python plot_results.py
Outputs:
    mpi4py_performance.png
"""

import matplotlib.pyplot as plt
import numpy as np

# ── Strong scaling data (N = 1024) ────────────────────────────────────────────
ranks = [2,        4,        8,        16       ]

mpi4py_time   = [0.914531, 1.103367, 0.569766, 0.288484]
mpi4py_gflops = [0.391,    0.324,    0.628,    1.241   ]

cpp_time      = [0.093178, 0.074562, 0.046557, 0.036596]
cpp_gflops    = [3.841,    4.800,    7.688,    9.780   ]

overhead      = [m / c for m, c in zip(mpi4py_time, cpp_time)]

# ── Large problem data (N = 2048, 16 ranks) ───────────────────────────────────
large_labels    = ["mpi4py\nN=2048", "C++ MPI\nN=2048"]
large_times     = [1.185425,  0.149886]
large_gflops_v  = [2.415,     19.103  ]

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
fig.suptitle("Part 5 — mpi4py vs C++ MPI Cholesky Comparison", fontweight="bold")

x = np.arange(len(ranks))
width = 0.35

# ── Panel 1: Runtime comparison (N=1024) ─────────────────────────────────────
ax = axes[0]
b1 = ax.bar(x - width / 2, mpi4py_time, width, label="mpi4py",
            color="steelblue", edgecolor="white")
b2 = ax.bar(x + width / 2, cpp_time, width, label="C++ MPI",
            color="darkorange", edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels([str(r) for r in ranks])
ax.set_xlabel("MPI Ranks")
ax.set_ylabel("Best time (s)")
ax.set_title("Runtime (N=1024)")
ax.legend(fontsize=9)
ax.grid(True, axis="y", linestyle=":", alpha=0.6)

# ── Panel 2: GFLOP/s comparison (N=1024) + large problem ─────────────────────
ax = axes[1]
ax.plot(ranks, mpi4py_gflops, "o-", color="steelblue", linewidth=2,
        markersize=7, label="mpi4py (N=1024)")
ax.plot(ranks, cpp_gflops, "s-", color="darkorange", linewidth=2,
        markersize=7, label="C++ MPI (N=1024)")

# Mark the N=2048 large-problem points at rank=16
ax.scatter([16], [large_gflops_v[0]], marker="*", s=150, color="steelblue",
           zorder=5, label="mpi4py (N=2048)")
ax.scatter([16], [large_gflops_v[1]], marker="*", s=150, color="darkorange",
           zorder=5, label="C++ MPI (N=2048)")
ax.annotate(f"{large_gflops_v[0]:.2f}", (16, large_gflops_v[0]),
            textcoords="offset points", xytext=(-28, 6), fontsize=8,
            color="steelblue")
ax.annotate(f"{large_gflops_v[1]:.2f}", (16, large_gflops_v[1]),
            textcoords="offset points", xytext=(4, 6), fontsize=8,
            color="darkorange")

ax.set_xlabel("MPI Ranks")
ax.set_ylabel("GFLOP/s")
ax.set_title("GFLOP/s (N=1024; ★ = N=2048 @ 16 ranks)")
ax.set_xticks(ranks)
ax.legend(fontsize=8)
ax.grid(True, linestyle=":", alpha=0.6)

# ── Panel 3: Overhead factor (mpi4py / C++ MPI time) ─────────────────────────
ax = axes[2]
bar_color = ["#4c72b0", "#c44e52", "#55a868", "#8172b2"]
bars = ax.bar([str(r) for r in ranks], overhead,
              color=bar_color, edgecolor="white", linewidth=0.8)
for bar, ov in zip(bars, overhead):
    ax.text(bar.get_x() + bar.get_width() / 2, ov + 0.1,
            f"{ov:.1f}×", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.axhline(1.0, color="k", linestyle="--", linewidth=1.1, label="No overhead (1×)")
ax.set_xlabel("MPI Ranks")
ax.set_ylabel("Overhead factor (mpi4py time / C++ time)")
ax.set_title("Python Overhead vs C++ MPI (N=1024)")
ax.set_ylim(0, max(overhead) * 1.2)
ax.legend(fontsize=9)
ax.grid(True, axis="y", linestyle=":", alpha=0.6)

plt.tight_layout()
plt.savefig("mpi4py_performance.png", dpi=150, bbox_inches="tight")
print("Saved mpi4py_performance.png")
