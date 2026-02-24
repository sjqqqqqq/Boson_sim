# -*- coding: utf-8 -*-
"""
bosonic_cluster.py — bosonic_fast.py adapted for Linux HPC clusters.

THREE CHANGES vs bosonic_fast.py
──────────────────────────────────
1. Thread pinning:  OpenBLAS/MKL/OMP are set to 1 thread per process BEFORE
   numpy/scipy are imported.  Without this, each worker spawns O(cpu_count)
   BLAS threads, causing severe oversubscription on many-core nodes.

2. SLURM-aware worker count:  reads SLURM_CPUS_PER_TASK (or SLURM_NTASKS) so
   the pool uses only the cores actually allocated to the job, not every core
   on the shared node.  Falls back to os.cpu_count() when not under SLURM.

3. Workers capped at N_RUNS:  spawning more processes than tasks wastes time
   on operator initialisation (build_operators) in idle workers.
"""

import os

# ── Must happen BEFORE numpy/scipy are imported ────────────────────────────
os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS","1")
os.environ.setdefault("NUMEXPR_NUM_THREADS",  "1")
# ───────────────────────────────────────────────────────────────────────────

import itertools
import time as tm

import numpy as np
import scipy.sparse as sp
from scipy.special import comb

# ---------------------------------------------------------------------------
# Simulation parameters  (identical to bosonic_fast.py)
# ---------------------------------------------------------------------------
N                  = 20
k                  = 2
N_INTERVALS        = 20
T_INTERVAL         = 1.0
STEPS_PER_INTERVAL = 100
SEED               = 42

T_TOTAL = N_INTERVALS * T_INTERVAL
N_STEPS = N_INTERVALS * STEPS_PER_INTERVAL
dt      = T_INTERVAL / STEPS_PER_INTERVAL


# ---------------------------------------------------------------------------
# Fock basis
# ---------------------------------------------------------------------------
def make_basis(N, k):
    V = []
    for subset in itertools.combinations(range(N + k - 1), k - 1):
        occ = [subset[0]]
        for j in range(k - 2):
            occ.append(subset[j + 1] - subset[j] - 1)
        occ.append(N + k - 2 - subset[-1])
        V.append(occ)
    Basis = V[::-1]
    Ind   = {str(b): i for i, b in enumerate(Basis)}
    return Basis, Ind


# ---------------------------------------------------------------------------
# Sparse operator construction
# ---------------------------------------------------------------------------
def build_operators(N, k):
    Basis, Ind = make_basis(N, k)
    d = len(Basis)

    weights          = np.arange(k, dtype=float) - (k - 1) / 2.0
    Sz_diag          = np.array([np.dot(weights, b) for b in Basis])
    interaction_diag = np.array(
        [sum(b[l] * (b[l] - 1) for l in range(k)) for b in Basis],
        dtype=float,
    )

    rows, cols, vals = [], [], []
    for i, occ in enumerate(Basis):
        for site in range(k - 1):
            if occ[site] == 0:
                continue
            occ2          = occ[:]
            occ2[site]   -= 1
            occ2[site+1] += 1
            j   = Ind[str(occ2)]
            amp = np.sqrt(occ[site] * occ2[site + 1])
            rows += [i, j]
            cols += [j, i]
            vals += [amp, amp]

    H_hop = sp.csr_matrix(
        (vals, (rows, cols)), shape=(d, d), dtype=np.float64
    )
    H_hop.sum_duplicates()
    return d, Sz_diag, interaction_diag, H_hop


# ---------------------------------------------------------------------------
# QFI
# ---------------------------------------------------------------------------
def qfi_pure(psi, Sz_diag, Sz_diag2):
    p  = np.abs(psi) ** 2
    E1 = p @ Sz_diag
    E2 = p @ Sz_diag2
    return 4.0 * (E2 - E1 * E1)


# ---------------------------------------------------------------------------
# Taylor-series matrix exponential
# ---------------------------------------------------------------------------
def _expm_mv_taylor(A_sparse, v, order=15):
    result = v.copy()
    term   = v
    for i in range(1, order + 1):
        term   = A_sparse.dot(term) * (1.0 / i)
        result = result + term
    return result


# ---------------------------------------------------------------------------
# Time evolution
# ---------------------------------------------------------------------------
def run_simulation(psi0, Sz_diag, interaction_diag, H_hop,
                   J_vals, U_vals, Delta_vals):
    Sz_diag2 = Sz_diag ** 2
    QFI      = np.empty(N_STEPS + 1)
    QFI[0]   = qfi_pure(psi0, Sz_diag, Sz_diag2)

    psi  = psi0.copy()
    step = 1

    for iv in range(N_INTERVALS):
        diag_phase = np.exp(
            -1j * dt * (U_vals[iv] * interaction_diag
                        + Delta_vals[iv] * Sz_diag)
        )
        hop_A = (-1j * dt * float(J_vals[iv])) * H_hop

        for _ in range(STEPS_PER_INTERVAL):
            psi        = diag_phase * psi
            psi        = _expm_mv_taylor(hop_A, psi)
            QFI[step]  = qfi_pure(psi, Sz_diag, Sz_diag2)
            step      += 1

    return QFI


# ---------------------------------------------------------------------------
# Worker helpers
# ---------------------------------------------------------------------------
_worker_state: dict = {}


def _init_worker(N_val, k_val):
    d, Sz_diag, interaction_diag, H_hop = build_operators(N_val, k_val)
    psi0    = np.zeros(d, dtype=complex)
    psi0[0] = 1.0
    _worker_state.update(
        psi0=psi0, Sz_diag=Sz_diag,
        interaction_diag=interaction_diag, H_hop=H_hop,
    )


def _run_one(seed):
    rng        = np.random.default_rng(seed)
    J_vals     = rng.uniform( 0.0, 1.0, N_INTERVALS)
    U_vals     = rng.uniform(-1.0, 1.0, N_INTERVALS)
    Delta_vals = rng.uniform(-1.0, 1.0, N_INTERVALS)
    QFI = run_simulation(
        _worker_state['psi0'], _worker_state['Sz_diag'],
        _worker_state['interaction_diag'], _worker_state['H_hop'],
        J_vals, U_vals, Delta_vals,
    )
    return QFI[::STEPS_PER_INTERVAL]


# ---------------------------------------------------------------------------
# SLURM-aware CPU count
# ---------------------------------------------------------------------------
def _allocated_cpus():
    """Return the number of CPUs allocated to this job.

    Checks SLURM environment variables first so the pool uses only the cores
    the scheduler assigned, not every core on the shared node.
    """
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_NTASKS"):
        val = os.environ.get(var)
        if val is not None:
            try:
                return int(val)
            except ValueError:
                pass
    return os.cpu_count()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
N_RUNS   = 100
K_VALUES = [4, 6]
# K_VALUES = [4, 6, 8]


def main():
    from concurrent.futures import ProcessPoolExecutor, as_completed

    n_workers = min(N_RUNS, _allocated_cpus())
    slurm_info = (
        f"SLURM_CPUS_PER_TASK={os.environ.get('SLURM_CPUS_PER_TASK', 'unset')}  "
        f"SLURM_NTASKS={os.environ.get('SLURM_NTASKS', 'unset')}"
    )
    print(f"N={N}  N_RUNS={N_RUNS}  workers={n_workers}  ({slurm_info})\n")

    for k_val in K_VALUES:
        out_dir = f"data/{k_val}-site"
        os.makedirs(out_dir, exist_ok=True)

        print(f"── k={k_val} ──")
        t0      = tm.time()
        results = [None] * N_RUNS
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            initargs=(N, k_val),
        ) as pool:
            futures = {pool.submit(_run_one, seed): seed for seed in range(N_RUNS)}
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()
        print(f"  Done in {tm.time()-t0:.1f} s")

        QFI_all = np.array(results)
        np.savetxt(
            os.path.join(out_dir, "QFI.txt"), QFI_all,
            header=(f"N={N} k={k_val}  "
                    f"rows=runs(0..{N_RUNS-1})  cols=QFI at t=0,1,...,{N_INTERVALS}"),
            comments="# ",
        )
        print(f"  Saved QFI.txt → {out_dir}/  (shape {QFI_all.shape})\n")


if __name__ == "__main__":
    main()
