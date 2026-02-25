import numpy as np
import matplotlib.pyplot as plt

sites = [2, 4, 6]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
t = np.arange(21)  # t = 0, 1, ..., 20

fig, ax = plt.subplots(figsize=(8, 5))

for k, color in zip(sites, colors):
    data = np.loadtxt(f"data/{k}-site/QFI.txt", comments='#')  # shape (20, 21)
    best_run = np.argmax(data[:, -1])  # run with highest QFI at final time
    best_qfi = data[best_run]
    ax.semilogy(t, np.where(best_qfi > 0, best_qfi, np.nan),
                color=color, linewidth=2, label=f"k={k} (run {best_run})")

N = 20
ax.axhline(N*2**2, color='gray', linestyle='--', linewidth=1.5, label="SQL2")
ax.axhline(N*4**2, color='gray', linestyle='--', linewidth=1.5, label="SQL4")
ax.axhline(N*6**2, color='gray', linestyle='--', linewidth=1.5, label="SQL6")

ax.set_xlabel("Time", fontsize=13)
ax.set_ylabel("QFI (log scale)", fontsize=13)
ax.set_title("Best QFI vs Time (N=20, highest final QFI of 20 runs)", fontsize=14)
ax.set_xlim(0, 20)
ax.legend(fontsize=12)
ax.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.savefig("figures/qfi_comparison.png", dpi=150)
plt.show()
print("Saved to figures/qfi_comparison.png")
