import numpy as np
import matplotlib.pyplot as plt

sites = [2, 4, 6, 8]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
t = np.linspace(0, 20, 21)  # t = 0, T/n, ..., T  (n=100 intervals)

fig, ax = plt.subplots(figsize=(5, 6))

N = 20
for k, color in zip(sites, colors):
    data = np.loadtxt(f"data/{k}-site/QFI.txt", comments='#')
    best_run = np.argmax(np.max(data, axis=1))  # run with highest QFI at any time
    best_qfi = data[best_run]
    ax.semilogy(t, np.where(best_qfi > 0, best_qfi, np.nan),
                color=color, linewidth=2, label=f"k={k} (run {best_run})")
    ax.axhline(N*(k-1)**2, color=color, linestyle='--', linewidth=1.5, label=f"SQL k={k}")

ax.set_xlabel("Time", fontsize=13)
ax.set_ylabel("QFI", fontsize=13)
ax.set_title("Best QFI vs Time (N=20, SQL=N(k-1)^2)", fontsize=14)
ax.set_xlim(0, 20)
ax.legend(fontsize=12)
ax.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.savefig("figures/qfi_comparison.png", dpi=150)
plt.show()
print("Saved to figures/qfi_comparison.png")
