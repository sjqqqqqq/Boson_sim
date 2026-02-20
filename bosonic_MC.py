# -*- coding: utf-8 -*-
"""
Piecewise-constant pulse simulation of boson dynamics.

T = 10, divided into N_INTERVALS = 10 equal intervals of duration 1.
Within each interval, J, U, Delta are drawn once at random:
  J     ~ Uniform[0, 1]
  U     ~ Uniform[-1, 1]
  Delta ~ Uniform[-1, 1]

Time evolution uses STEPS_PER_INTERVAL Trotter steps per interval
(dt = T_INTERVAL / STEPS_PER_INTERVAL).  Total time steps >> N_INTERVALS.

Results plotted: J(t), U(t), Delta(t), QFI(t)/SQL vs time.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.special import comb
import time as tm
import general_functions

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
N                  = 20
k                  = 3
N_INTERVALS        = 10   # number of piecewise-constant intervals
T_INTERVAL         = 1.0  # duration of each interval  (total T = 10)
STEPS_PER_INTERVAL = 100   # Trotter steps per interval  → dt = 0.01, 1000 steps total
SEED               = 42

T_TOTAL  = N_INTERVALS * T_INTERVAL          # 10
N_STEPS  = N_INTERVALS * STEPS_PER_INTERVAL  # 200
dt       = T_INTERVAL / STEPS_PER_INTERVAL   # 0.05


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(SEED)

    d    = int(comb(N + k - 1, k - 1))
    psi0 = np.zeros(d, dtype=complex)
    psi0[0] = 1.0

    matrices        = general_functions.common_matrices(N, k)
    Sz              = matrices["Sz"]
    Szd             = np.diag(Sz)
    Hopping         = matrices["Hopping"]
    Interaction_sum = np.sum(matrices["Interaction"], axis=0)

    # SQL reference
    states_dict = general_functions.common_states(N, k)
    sql_state   = states_dict["SQL"] / np.linalg.norm(states_dict["SQL"])
    SQL         = general_functions.fisher_info_pure(sql_state, Sz)

    # Draw random parameters — one value per interval
    J_vals     = rng.uniform( 0.0, 1.0, N_INTERVALS)
    U_vals     = rng.uniform(-1.0, 1.0, N_INTERVALS)
    Delta_vals = rng.uniform(-1.0, 1.0, N_INTERVALS)

    print(f"Piecewise-constant pulse simulation")
    print(f"  N={N}, k={k}, d={d}, SQL={SQL:.2f}")
    print(f"  {N_INTERVALS} intervals x {STEPS_PER_INTERVAL} steps, dt={dt}, T={T_TOTAL}")

    # QFI array: one entry per time point including t=0
    QFI    = np.empty(N_STEPS + 1)
    QFI[0] = general_functions.fisher_info_pure(psi0, Sz)

    psi  = psi0.copy()
    step = 1
    t0   = tm.time()

    for iv in range(N_INTERVALS):
        # Precompute propagators for this interval (constant within interval)
        interaction_diag = np.exp(-1j * dt * U_vals[iv]     * Interaction_sum)
        detuning_diag    = np.exp(-1j * dt * Delta_vals[iv] * Szd)
        hop_mat          = expm(-1j * dt * J_vals[iv] * Hopping)

        for _ in range(STEPS_PER_INTERVAL):
            psi       = interaction_diag * psi
            psi       = detuning_diag    * psi
            psi       = hop_mat @ psi
            QFI[step] = general_functions.fisher_info_pure(psi, Sz)
            step     += 1

    print(f"Done in {tm.time()-t0:.2f}s")

    # -----------------------------------------------------------------------
    # Build plotting arrays
    # -----------------------------------------------------------------------
    # QFI: N_STEPS+1 evenly-spaced time points  [0, dt, 2dt, ..., T_TOTAL]
    t_qfi = np.linspace(0.0, T_TOTAL, N_STEPS + 1)

    # Sample J, U, Delta at every QFI time point (piecewise constant)
    iv_idx  = np.minimum(np.arange(N_STEPS + 1) // STEPS_PER_INTERVAL,
                         N_INTERVALS - 1)
    J_sampled     = J_vals[iv_idx]
    U_sampled     = U_vals[iv_idx]
    Delta_sampled = Delta_vals[iv_idx]

    # Save text files: two columns (time, value)
    np.savetxt("J.txt",     J_sampled,     header="J",     comments="")
    np.savetxt("U.txt",     U_sampled,     header="U",     comments="")
    np.savetxt("Delta.txt", Delta_sampled, header="Delta", comments="")
    np.savetxt("QFI.txt",   QFI,           header="QFI",   comments="")
    print("Saved J.txt, U.txt, Delta.txt, QFI.txt")

    # Parameters: staircase — value held constant inside each interval
    # Use N_INTERVALS+1 boundary points so plt.step covers [0, T_TOTAL]
    t_step     = np.arange(N_INTERVALS + 1, dtype=float) * T_INTERVAL
    J_step     = np.append(J_vals,     J_vals[-1])
    U_step     = np.append(U_vals,     U_vals[-1])
    Delta_step = np.append(Delta_vals, Delta_vals[-1])

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    fig.suptitle(
        f"Piecewise-constant pulse  "
        f"(N={N}, k={k},  {N_INTERVALS} intervals × {STEPS_PER_INTERVAL} steps,  dt={dt})"
    )

    step_kw = dict(where="post")

    axes[0].step(t_step, J_step,     color="tab:blue",   **step_kw)
    axes[0].set_ylabel("J")
    axes[0].set_ylim(-0.05, 1.1)

    axes[1].step(t_step, U_step,     color="tab:orange", **step_kw)
    axes[1].set_ylabel("U")
    axes[1].axhline(0, color="k", lw=0.5, ls="--")
    axes[1].set_ylim(-1.15, 1.15)

    axes[2].step(t_step, Delta_step, color="tab:green",  **step_kw)
    axes[2].set_ylabel("Delta")
    axes[2].axhline(0, color="k", lw=0.5, ls="--")
    axes[2].set_ylim(-1.15, 1.15)

    axes[3].plot(t_qfi, QFI, color="tab:red")
    axes[3].axhline(SQL, color="k", lw=0.8, ls="--", label=f"SQL = {SQL:.1f}")
    axes[3].set_ylabel("QFI")
    axes[3].set_xlabel("Time")
    axes[3].legend()

    # Vertical lines at interval boundaries
    for ax in axes:
        for tb in np.arange(1, N_INTERVALS) * T_INTERVAL:
            ax.axvline(tb, color="gray", lw=0.5, ls=":")

    axes[0].set_xlim(0, T_TOTAL)

    plt.tight_layout()
    plt.savefig("pulse_results.png", dpi=150)
    plt.show()
    print("Plot saved to pulse_results.png")


if __name__ == "__main__":
    main()
