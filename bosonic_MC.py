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
STEPS_PER_INTERVAL = 100  # Trotter steps per interval  → dt = 0.01, 1000 steps total
SEED               = 42

T_TOTAL = N_INTERVALS * T_INTERVAL          # 10
N_STEPS = N_INTERVALS * STEPS_PER_INTERVAL  # 1000
dt      = T_INTERVAL / STEPS_PER_INTERVAL   # 0.01


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def draw_parameters(rng):
    J_vals     = rng.uniform( 0.0, 1.0, N_INTERVALS)
    U_vals     = rng.uniform(-1.0, 1.0, N_INTERVALS)
    Delta_vals = rng.uniform(-1.0, 1.0, N_INTERVALS)
    return J_vals, U_vals, Delta_vals


def run_simulation(psi0, Sz, Hopping, Interaction_sum, J_vals, U_vals, Delta_vals):
    QFI    = np.empty(N_STEPS + 1)
    QFI[0] = general_functions.fisher_info_pure(psi0, Sz)

    psi  = psi0.copy()
    step = 1

    for iv in range(N_INTERVALS):
        interaction_diag = np.exp(-1j * dt * U_vals[iv]     * Interaction_sum)
        detuning_diag    = np.exp(-1j * dt * Delta_vals[iv] * np.diag(Sz))
        hop_mat          = expm(-1j * dt * J_vals[iv] * Hopping)

        for _ in range(STEPS_PER_INTERVAL):
            psi       = interaction_diag * psi
            psi       = detuning_diag    * psi
            psi       = hop_mat @ psi
            QFI[step] = general_functions.fisher_info_pure(psi, Sz)
            step     += 1

    return QFI


def sample_parameters(J_vals, U_vals, Delta_vals):
    """Expand interval-constant values to one entry per time step."""
    iv_idx        = np.minimum(np.arange(N_STEPS + 1) // STEPS_PER_INTERVAL,
                               N_INTERVALS - 1)
    J_sampled     = J_vals[iv_idx]
    U_sampled     = U_vals[iv_idx]
    Delta_sampled = Delta_vals[iv_idx]
    return J_sampled, U_sampled, Delta_sampled


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def save_results(J_sampled, U_sampled, Delta_sampled, QFI):
    np.savetxt("J.txt",     J_sampled)
    np.savetxt("U.txt",     U_sampled)
    np.savetxt("Delta.txt", Delta_sampled)
    np.savetxt("QFI.txt",   QFI)
    print("Saved J.txt, U.txt, Delta.txt, QFI.txt")


def print_table(J_sampled, U_sampled, Delta_sampled, QFI):
    print(f"\n{'t':>4}  {'J':>8}  {'U':>8}  {'Delta':>8}  {'QFI':>10}")
    print("-" * 46)
    for t_int in range(N_INTERVALS + 1):
        idx = t_int * STEPS_PER_INTERVAL
        print(f"{t_int:>4}  {J_sampled[idx]:>8.4f}  {U_sampled[idx]:>8.4f}"
              f"  {Delta_sampled[idx]:>8.4f}  {QFI[idx]:>10.4f}")


def plot_results(t_qfi, J_sampled, U_sampled, Delta_sampled, QFI,
                 J_vals, U_vals, Delta_vals, SQL):
    # Staircase arrays for step plots
    t_step     = np.arange(N_INTERVALS + 1, dtype=float) * T_INTERVAL
    J_step     = np.append(J_vals,     J_vals[-1])
    U_step     = np.append(U_vals,     U_vals[-1])
    Delta_step = np.append(Delta_vals, Delta_vals[-1])

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

    for ax in axes:
        for tb in np.arange(1, N_INTERVALS) * T_INTERVAL:
            ax.axvline(tb, color="gray", lw=0.5, ls=":")

    axes[0].set_xlim(0, T_TOTAL)

    plt.tight_layout()
    plt.savefig("pulse_results.png", dpi=150)
    plt.show()
    print("Plot saved to pulse_results.png")


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
    Hopping         = matrices["Hopping"]
    Interaction_sum = np.sum(matrices["Interaction"], axis=0)

    states_dict = general_functions.common_states(N, k)
    sql_state   = states_dict["SQL"] / np.linalg.norm(states_dict["SQL"])
    SQL         = general_functions.fisher_info_pure(sql_state, Sz)

    J_vals, U_vals, Delta_vals = draw_parameters(rng)

    print(f"Piecewise-constant pulse simulation")
    print(f"  N={N}, k={k}, d={d}, SQL={SQL:.2f}")
    print(f"  {N_INTERVALS} intervals x {STEPS_PER_INTERVAL} steps, dt={dt}, T={T_TOTAL}")

    t0  = tm.time()
    QFI = run_simulation(psi0, Sz, Hopping, Interaction_sum, J_vals, U_vals, Delta_vals)
    print(f"Done in {tm.time()-t0:.2f}s")

    t_qfi = np.linspace(0.0, T_TOTAL, N_STEPS + 1)
    J_sampled, U_sampled, Delta_sampled = sample_parameters(J_vals, U_vals, Delta_vals)

    # save_results(J_sampled, U_sampled, Delta_sampled, QFI)
    print_table(J_sampled, U_sampled, Delta_sampled, QFI)
    # plot_results(t_qfi, J_sampled, U_sampled, Delta_sampled, QFI,
    #              J_vals, U_vals, Delta_vals, SQL)


if __name__ == "__main__":
    main()
