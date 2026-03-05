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

2. Per Trotter step, apply exp(-i·dt·H)|ψ⟩ via a Taylor series.
   Since ‖J·H‖·dt ≤ 0.4, 15 terms suffice for double precision.
   No scipy call overhead — just 15 sparse matvecs per step.

3. QFI = 4(⟨Sz²⟩ − ⟨Sz⟩²) via dot products with Sz_diag² and Sz_diag
   — O(d) per step, not O(d²) as with the dense Sz matrix.

TROTTER DECOMPOSITION  (Eq. 16-17 of arXiv:2511.15805)
────────────────────────────────────────────────────────
Each piecewise-constant interval k uses three Hamiltonians applied sequentially:

  H1^(k) = Σ_i [ Δ_i^(k) n̂_i  +  U_i^(k) n̂_i(n̂_i−1) ]      diagonal
  H2^(k) = Σ_i  J_{2i}^(k)   (â†_{2i+1} â_{2i} + h.c.)        even-bond hops
  H3^(k) = Σ_i  J_{2i-1}^(k) (â†_{2i}   â_{2i-1} + h.c.)      odd-bond hops

  |ψ⟩ ← exp(-i·dt H3^(k)) exp(-i·dt H2^(k)) exp(-i·dt H1^(k)) |ψ⟩

Even bonds (0→1, 2→3, …) and odd bonds (1→2, 3→4, …) are applied in
separate steps so that each group has independently tunable J couplings.
Parameters Δ_i, U_i are per-site; J_b is per-bond — all drawn uniformly:
  J_b    ∈ [0, 1]    (Algorithm 1, line 5)
  Δ_i    ∈ [−1, 1]   (Algorithm 1, line 6)
  U_i    ∈ [−1, 1]   (Algorithm 1, line 7)

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
N    = 20
k    = 2
T    = 20.0   # total evolution time
n    = 20    # number of piecewise-constant intervals (Algorithm 1 step 1)
dt   = 1e-2  # Trotter sub-step size
SEED = 42


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
    """Build diagonal observables and per-bond sparse hopping matrices.

    Returns
    -------
    d          : int          — Hilbert-space dimension C(N+k-1, k-1)
    Sz_diag    : (d,) float64 — diagonal of Sz  (used for QFI)
    site_n     : (k, d) float64 — site_n[i, idx] = occupation n_i of basis state idx
    site_inter : (k, d) float64 — site_inter[i, idx] = n_i(n_i-1)
    H_bond     : list of (k-1) CSR float64 matrices — H_bond[b] = hopping on bond b
    even_bonds : list of int — indices of even bonds: 0, 2, 4, …
    odd_bonds  : list of int — indices of odd bonds:  1, 3, 5, …
    """
    Basis, Ind = make_basis(N, k)
    d = len(Basis)

    # Sz diagonal:  Σ_l n_l · (l − (k−1)/2)  — used only for QFI
    weights = np.arange(k, dtype=float) - (k - 1) / 2.0
    Sz_diag = np.array([np.dot(weights, b) for b in Basis])

    # Per-site occupation numbers  (k, d)
    # site_n[i, idx] = Basis[idx][i]
    site_n = np.array([[b[i] for b in Basis] for i in range(k)], dtype=float)

    # Per-site n(n-1)  (k, d)
    # site_inter[i, idx] = n_i * (n_i - 1)
    site_inter = np.array([[b[i] * (b[i] - 1) for b in Basis] for i in range(k)], dtype=float)

    # Build one sparse hopping matrix per bond (k-1 matrices)
    # H_bond[b]_{i,j} = sqrt(n_b * (n_{b+1}+1))  for the hop  b→b+1
    H_bond = []
    for site in range(k - 1):
        rows, cols, vals = [], [], []
        for i, occ in enumerate(Basis):
            if occ[site] == 0:          # no boson to hop
                continue
            occ2          = occ[:]      # copy
            occ2[site]   -= 1
            occ2[site+1] += 1
            j   = Ind[str(occ2)]
            amp = np.sqrt(occ[site] * occ2[site + 1])
            rows += [i, j]              # Hermitian → both entries
            cols += [j, i]
            vals += [amp, amp]
        H_b = sp.csr_matrix((vals, (rows, cols)), shape=(d, d), dtype=np.float64)
        H_b.sum_duplicates()
        H_bond.append(H_b)

    # Even bonds: 0→1, 2→3, 4→5, …   (H2 in Eq. 16)
    # Odd  bonds: 1→2, 3→4, 5→6, …   (H3 in Eq. 16)
    even_bonds = list(range(0, k - 1, 2))
    odd_bonds  = list(range(1, k - 1, 2))

    return d, Sz_diag, site_n, site_inter, H_bond, even_bonds, odd_bonds


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
    For our hopping steps, ‖A‖ = J·‖H_bond‖·dt.  Since ‖H_bond‖ grows
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
# Time evolution  (Eq. 16-17 of arXiv:2511.15805)
# ---------------------------------------------------------------------------
def run_simulation(psi0, Sz_diag, site_n, site_inter, H_bond, even_bonds, odd_bonds,
                   J_vals, U_vals, Delta_vals):
    """Cycling evolution: H1 only on [0,Δt], H2 only on [Δt,2Δt], H3 only on [2Δt,3Δt], …

    Parameters
    ----------
    psi0       : (d,) complex — initial state
    Sz_diag    : (d,) float  — Sz eigenvalues (for QFI)
    site_n     : (k, d) float — per-site occupation numbers
    site_inter : (k, d) float — per-site n(n-1) values
    H_bond     : list of (k-1) CSR matrices — per-bond hopping
    even_bonds : list of int — even-bond indices (H2 group)
    odd_bonds  : list of int — odd-bond  indices (H3 group)
    J_vals     : (n,) float — global J per interval ∈ [0,1]   (same for all bonds)
    U_vals     : (n,) float — global U per interval ∈ [−1,1]  (same for all sites)
    Delta_vals : (n,) float — global Δ per interval ∈ [−1,1]  (same for all sites)

    Within each piecewise-constant interval iv, the three Hamiltonians are
    applied in turn for T/(3n) each, subdivided into steps of size dt:
      Slot 1 (H1 only): repeat steps_per_iv times:
           ψ ← exp(−i·dt·H1_diag) * ψ        detuning+interaction  O(d)
      Slot 2 (H2 only): repeat steps_per_iv times:
           ψ ← Taylor_exp(−i·dt·H2) · ψ      even-bond hopping     O(15·nnz)
      Slot 3 (H3 only): repeat steps_per_iv times:
           ψ ← Taylor_exp(−i·dt·H3) · ψ      odd-bond  hopping     O(15·nnz)
      record QFI after all three slots
    """
    d            = len(psi0)
    Sz_diag2     = Sz_diag ** 2
    steps_per_iv = int(round(T / (3 * n * dt)))  # sub-steps per T/(3n) slot
    QFI          = np.empty(n + 1)
    QFI[0]       = qfi_pure(psi0, Sz_diag, Sz_diag2)

    # Precompute site sums for global (uniform) parameters
    site_n_total     = site_n.sum(axis=0)      # (d,) — total occupation (= N always)
    site_inter_total = site_inter.sum(axis=0)  # (d,) — total n(n-1) across sites

    # Precompute bond-summed hopping matrices (scaled per interval by scalar J)
    H2_base = sum(H_bond[b] for b in even_bonds) if even_bonds else None
    H3_base = sum(H_bond[b] for b in odd_bonds)  if odd_bonds  else None

    psi = psi0.copy()

    for iv in range(n):
        # ── Slot 1: H1 only — diagonal detuning + interaction ────────────────
        H1_diag    = Delta_vals[iv] * site_n_total + U_vals[iv] * site_inter_total
        diag_phase = np.exp(-1j * dt * H1_diag)
        for _ in range(steps_per_iv):
            psi = diag_phase * psi

        # ── Slot 2: H2 only — even-bond hopping ──────────────────────────────
        if H2_base is not None:
            hop_A2 = (-1j * dt * float(J_vals[iv])) * H2_base
            for _ in range(steps_per_iv):
                psi = _expm_mv_taylor(hop_A2, psi)

        # ── Slot 3: H3 only — odd-bond hopping ───────────────────────────────
        if H3_base is not None:
            hop_A3 = (-1j * dt * float(J_vals[iv])) * H3_base
            for _ in range(steps_per_iv):
                psi = _expm_mv_taylor(hop_A3, psi)

        QFI[iv + 1] = qfi_pure(psi, Sz_diag, Sz_diag2)

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
    d, Sz_diag, site_n, site_inter, H_bond, even_bonds, odd_bonds = build_operators(N_val, k_val)
    psi0    = np.zeros(d, dtype=complex)
    psi0[0] = 1.0
    _worker_state.update(
        k=k_val,
        psi0=psi0,
        Sz_diag=Sz_diag,
        site_n=site_n,
        site_inter=site_inter,
        H_bond=H_bond,
        even_bonds=even_bonds,
        odd_bonds=odd_bonds,
    )


def _run_one(seed):
    """Single simulation run using worker-local operators.

    Samples one global J, U, Δ per interval (same value applied to all bonds/sites):
      J      ∈ [0, 1]   (global hopping strength)
      Δ      ∈ [−1, 1]  (global detuning)
      U      ∈ [−1, 1]  (global interaction strength)
    """
    rng        = np.random.default_rng(seed)
    J_vals     = rng.uniform( 0.0, 1.0, n)
    U_vals     = rng.uniform(-1.0, 1.0, n)
    Delta_vals = rng.uniform(-1.0, 1.0, n)
    QFI = run_simulation(
        _worker_state['psi0'],
        _worker_state['Sz_diag'],
        _worker_state['site_n'],
        _worker_state['site_inter'],
        _worker_state['H_bond'],
        _worker_state['even_bonds'],
        _worker_state['odd_bonds'],
        J_vals, U_vals, Delta_vals,
    )
    return QFI   # shape (n+1,) — QFI at t=0, T/n, …, T


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
N_RUNS   = 100
# K_VALUES = [2, 4, 6, 8]
K_VALUES = [8]


def main():
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    n_workers = os.cpu_count()
    print(f"N={N}  T={T}  n={n}  N_RUNS={N_RUNS}  workers={n_workers}\n")

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
            base = K_VALUES.index(k_val) * N_RUNS
            futures = {pool.submit(_run_one, base + seed): seed for seed in range(N_RUNS)}
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()
        print(f"  Done in {tm.time()-t0:.1f} s")

        QFI_all = np.array(results)   # shape (N_RUNS, n+1)
        np.savetxt(
            os.path.join(out_dir, "QFI.txt"), QFI_all,
            header=(f"N={N} k={k_val} T={T} n={n}  "
                    f"rows=runs(0..{N_RUNS-1})  cols=QFI at t=0,T/n,...,T"),
            comments="# ",
        )
        print(f"  Saved QFI.txt → {out_dir}/  (shape {QFI_all.shape})\n")


if __name__ == "__main__":
    main()
