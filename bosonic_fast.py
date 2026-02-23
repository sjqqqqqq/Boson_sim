# -*- coding: utf-8 -*-
"""
bosonic_fast.py — Memory- and time-efficient QFI computation.

WHY bosonic_MC.py IS SLOW
──────────────────────────
The Hopping matrix is stored dense (d × d) and expm is dense too:
  k=5, N=20 → d=10,626  → dense matrix = 1.8 GB,  expm = O(d³) per interval
  k=6, N=20 → d=53,130  → dense matrix = 45 GB   (infeasible)
  k=8, N=20 → d=888,030 → dense matrix = 12.6 TB  (infeasible)

But Hopping is SPARSE — at most (k-1) hops per basis state:
  k=5: nnz ≈ 85 K   vs  d² = 113 M   (×1,330 sparser)
  k=6: nnz ≈ 265 K  vs  d² = 2.8 G   (×10,600 sparser)
  k=8: nnz ≈ 6.2 M  vs  d² = 788 G   (×127,000 sparser)

THREE CHANGES
─────────────
1. Sparse H_hop (CSR) — O(nnz) memory, never forms the dense matrix.

2. Per Trotter step, apply exp(-i·dt·J·H_hop)|ψ⟩ via a Taylor series.
   Since ‖J·H_hop‖·dt ≤ 0.4, 15 terms suffice for double precision.
   No scipy call overhead — just 15 sparse matvecs per step.

3. QFI = 4(⟨Sz²⟩ − ⟨Sz⟩²) via dot products with Sz_diag² and Sz_diag
   — O(d) per step, not O(d²) as with the dense Sz matrix.

SCALING (N=20, same physics/accuracy as bosonic_MC.py)
───────────────────────────────────────────────────────
  k   d          old mem      new mem    old time  new time (est.)
  5   10,626     1.8  GB    ~1.4 MB     >2 h      ~0.1 s
  6   53,130     45   GB    ~7   MB     infeasible ~0.5 s
  8   888,030    12.6 TB    ~114 MB     infeasible ~10  s
"""

import itertools
import time as tm

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.special import comb

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
N                  = 20
k                  = 2
N_INTERVALS        = 20    # number of piecewise-constant intervals
T_INTERVAL         = 1.0   # duration of each interval  (total T = 20)
STEPS_PER_INTERVAL = 100   # Trotter steps per interval
SEED               = 42

T_TOTAL = N_INTERVALS * T_INTERVAL
N_STEPS = N_INTERVALS * STEPS_PER_INTERVAL
dt      = T_INTERVAL / STEPS_PER_INTERVAL


# ---------------------------------------------------------------------------
# Fock basis  (identical enumeration to general_functions.make_basis)
# ---------------------------------------------------------------------------
def make_basis(N, k):
    """Return (Basis, Ind).

    Basis : list of occupation vectors [n_0, …, n_{k-1}]
    Ind   : dict  str(occ) → basis index
    """
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
    """Build diagonal observables and the sparse hopping matrix.

    Returns
    -------
    d                : int    — Hilbert-space dimension C(N+k-1, k-1)
    Sz_diag          : (d,) float64 — diagonal of Sz
    interaction_diag : (d,) float64 — Σ_l n_l(n_l−1) per basis state
    H_hop            : (d,d) CSR float64 — nearest-neighbor hopping
    """
    Basis, Ind = make_basis(N, k)
    d = len(Basis)

    # Sz diagonal:  Σ_l n_l · (l − (k−1)/2)
    weights = np.arange(k, dtype=float) - (k - 1) / 2.0
    Sz_diag = np.array([np.dot(weights, b) for b in Basis])

    # Interaction diagonal:  Σ_l n_l(n_l − 1)
    interaction_diag = np.array(
        [sum(b[l] * (b[l] - 1) for l in range(k)) for b in Basis],
        dtype=float,
    )

    # Sparse hopping:  H_{i,j} = sqrt(n_site · (n_{site+1} + 1))
    # for each nearest-neighbor pair (site, site+1) and each basis state i.
    rows, cols, vals = [], [], []
    for i, occ in enumerate(Basis):
        for site in range(k - 1):
            if occ[site] == 0:          # no boson to hop
                continue
            occ2          = occ[:]      # copy list of Python ints
            occ2[site]   -= 1
            occ2[site+1] += 1
            j   = Ind[str(occ2)]        # index of connected state
            amp = np.sqrt(occ[site] * occ2[site + 1])
            rows += [i, j]              # H is Hermitian → add both entries
            cols += [j, i]
            vals += [amp, amp]

    H_hop = sp.csr_matrix(
        (vals, (rows, cols)), shape=(d, d), dtype=np.float64
    )
    H_hop.sum_duplicates()   # consolidate any repeated (row, col) pairs
    return d, Sz_diag, interaction_diag, H_hop


# ---------------------------------------------------------------------------
# QFI  — O(d) per state, no dense matrix required
# ---------------------------------------------------------------------------
def qfi_pure(psi, Sz_diag, Sz_diag2):
    """QFI = 4·Var(Sz) = 4(⟨Sz²⟩ − ⟨Sz⟩²) for a pure state."""
    p  = np.abs(psi) ** 2       # probability vector  (d,)
    E1 = p @ Sz_diag            # ⟨Sz⟩
    E2 = p @ Sz_diag2           # ⟨Sz²⟩
    return 4.0 * (E2 - E1 * E1)


# ---------------------------------------------------------------------------
# Taylor-series matrix exponential for small-norm operators
# ---------------------------------------------------------------------------
def _expm_mv_taylor(A_sparse, v, order=15):
    """Compute exp(A)·v via Taylor series.

    Safe for ‖A‖ ≤ 0.5 with order=15:  error ≈ 0.5^16/16! ≈ 8×10⁻¹⁹.
    For our hopping step, ‖A‖ = J·‖H_hop‖·dt.  Since ‖H_hop‖ grows
    slowly with k (≈28 for k=3, ≈37 for k=6), dt=0.01 keeps ‖A‖ ≤ 0.4
    throughout, well within the convergence regime.

    Cost: `order` sparse matrix-vector products.  No scipy overhead.
    """
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
    """Trotter evolution with sparse Taylor-series hopping steps.

    Within each piecewise-constant interval:
      1. Precompute:
           diag_phase = exp(-i·dt·(U·Σn(n-1) + Δ·Sz))   [O(d), elementwise]
           hop_A      = -i·dt·J·H_hop                    [O(nnz), sparse]
      2. Repeat STEPS_PER_INTERVAL times:
           ψ ← diag_phase * ψ                            [O(d)]
           ψ ← Taylor_exp(hop_A) · ψ                    [O(order·nnz)]
           QFI ← 4(⟨Sz²⟩ - ⟨Sz⟩²)                     [O(d)]

    Why Taylor instead of scipy.expm_multiply:
      ‖hop_A‖ = J·‖H_hop‖·dt ≤ 1 × ~20 × 0.01 = 0.2
      At this norm, expm_multiply needs ~5 matvecs but carries ~5 ms of
      scipy/Python setup overhead per call.  With 1000 calls, that is
      ~5 s of overhead alone.  The Taylor series uses 15 matvecs per call
      but has negligible overhead — only NumPy sparse-dot invocations.
    """
    Sz_diag2 = Sz_diag ** 2
    QFI      = np.empty(N_STEPS + 1)
    QFI[0]   = qfi_pure(psi0, Sz_diag, Sz_diag2)

    psi  = psi0.copy()
    step = 1

    for iv in range(N_INTERVALS):
        # Diagonal Trotter factor: exp(-i·dt·(U·n(n-1) + Δ·Sz)) [once per interval]
        diag_phase = np.exp(
            -1j * dt * (U_vals[iv] * interaction_diag
                        + Delta_vals[iv] * Sz_diag)
        )
        # Hopping generator -i·dt·J·H_hop  [sparse scalar-multiply, O(nnz)]
        hop_A = (-1j * dt * float(J_vals[iv])) * H_hop

        for _ in range(STEPS_PER_INTERVAL):
            psi        = diag_phase * psi               # diagonal step   O(d)
            psi        = _expm_mv_taylor(hop_A, psi)   # hopping step   O(15·nnz)
            QFI[step]  = qfi_pure(psi, Sz_diag, Sz_diag2)
            step      += 1

    return QFI


# ---------------------------------------------------------------------------
# SQL reference state
# ---------------------------------------------------------------------------
def sql_qfi(N, k, Sz_diag):
    """QFI of the uniform coherent state (shot-noise limit)."""
    from math import factorial
    Basis, _ = make_basis(N, k)
    d    = len(Basis)
    psi  = np.zeros(d, dtype=complex)
    fN   = float(factorial(N))
    kN   = float(k) ** N
    for i, b in enumerate(Basis):
        denom = kN * float(np.prod([factorial(nl) for nl in b]))
        psi[i] = np.sqrt(fN / denom)
    psi /= np.linalg.norm(psi)
    return qfi_pure(psi, Sz_diag, Sz_diag ** 2)


# ---------------------------------------------------------------------------
# Worker helpers for ProcessPoolExecutor
# (must be module-level so they are picklable)
# ---------------------------------------------------------------------------
_worker_state: dict = {}


def _init_worker(N_val, k_val):
    """Called once per worker process; builds operators into process-local state."""
    d, Sz_diag, interaction_diag, H_hop = build_operators(N_val, k_val)
    psi0    = np.zeros(d, dtype=complex)
    psi0[0] = 1.0
    _worker_state.update(
        psi0=psi0, Sz_diag=Sz_diag,
        interaction_diag=interaction_diag, H_hop=H_hop,
    )


def _run_one(seed):
    """Single simulation run using worker-local operators."""
    rng        = np.random.default_rng(seed)
    J_vals     = rng.uniform( 0.0, 1.0, N_INTERVALS)
    U_vals     = rng.uniform(-1.0, 1.0, N_INTERVALS)
    Delta_vals = rng.uniform(-1.0, 1.0, N_INTERVALS)
    QFI = run_simulation(
        _worker_state['psi0'], _worker_state['Sz_diag'],
        _worker_state['interaction_diag'], _worker_state['H_hop'],
        J_vals, U_vals, Delta_vals,
    )
    return QFI[::STEPS_PER_INTERVAL]   # QFI at integer t=0,1,...,N_INTERVALS


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
N_RUNS   = 100
K_VALUES = [4, 6]
# K_VALUES = [4, 6, 8]


def main():
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    n_workers = os.cpu_count()
    print(f"N={N}  N_RUNS={N_RUNS}  workers={n_workers}\n")

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

        QFI_all = np.array(results)   # shape (N_RUNS, N_INTERVALS+1)
        np.savetxt(
            os.path.join(out_dir, "QFI.txt"), QFI_all,
            header=(f"N={N} k={k_val}  "
                    f"rows=runs(0..{N_RUNS-1})  cols=QFI at t=0,1,...,{N_INTERVALS}"),
            comments="# ",
        )
        print(f"  Saved QFI.txt → {out_dir}/  (shape {QFI_all.shape})\n")


if __name__ == "__main__":
    main()
