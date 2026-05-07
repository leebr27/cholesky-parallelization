"""
Plot Part 1 (serial) results: runtime and GFLOP/s vs matrix size N.
Data from cholesky_serial_108259.out as reported in README.md.

Usage:
    python plot_results.py
Outputs:
    serial_performance.png
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Data from README ──────────────────────────────────────────────────────────
N          = [256,    512,    1024,   2048   ]
best_time  = [0.003062, 0.058234, 0.776923, 12.502958]
avg_time   = [0.003079, 0.058305, 0.781225, 12.802376]
gflops     = [1.827,  0.768,  0.461,  0.229  ]

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
fig.suptitle("Part 1 — Serial Cholesky Performance", fontweight="bold")

# Left: runtime vs N (log-log)
ax = axes[0]
ax.loglog(N, best_time, "o-", color="steelblue", linewidth=2,
          markersize=7, label="Best time")
ax.loglog(N, avg_time,  "s--", color="gray", linewidth=1.5,
          markersize=6, label="Avg time")

# Reference O(N³) line anchored to the N=256 point
n_ref = np.array(N, dtype=float)
cubic = best_time[0] * (n_ref / N[0]) ** 3
ax.loglog(n_ref, cubic, ":", color="tomato", linewidth=1.5, label=r"$O(N^3)$ ref")

ax.set_xlabel("Matrix size N")
ax.set_ylabel("Wall time (s)")
ax.set_title("Runtime vs N")
ax.set_xticks(N)
ax.set_xticklabels([str(n) for n in N])
ax.xaxis.set_minor_locator(ticker.NullLocator())
ax.legend(fontsize=9)
ax.grid(True, which="both", linestyle=":", alpha=0.6)

# Right: GFLOP/s vs N
ax = axes[1]
bars = ax.bar([str(n) for n in N], gflops, color="steelblue", edgecolor="white",
              linewidth=0.8)
for bar, g in zip(bars, gflops):
    ax.text(bar.get_x() + bar.get_width() / 2, g + 0.03,
            f"{g:.3f}", ha="center", va="bottom", fontsize=9)
ax.set_xlabel("Matrix size N")
ax.set_ylabel("GFLOP/s")
ax.set_title("Achieved GFLOP/s vs N")
ax.set_ylim(0, max(gflops) * 1.25)
ax.grid(True, axis="y", linestyle=":", alpha=0.6)

plt.tight_layout()
plt.savefig("serial_performance.png", dpi=150, bbox_inches="tight")
print("Saved serial_performance.png")
