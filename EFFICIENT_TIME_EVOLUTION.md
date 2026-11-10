# Efficient Time Evolution

This note explains how the simulation propagates a multi-mode bosonic state in
time efficiently, following **Sec. S3 ("Code Description")** of
*Engineering multi-mode bosonic squeezed states using Monte-Carlo optimization*
(arXiv:2511.15805).

Time evolution is the computational bottleneck of the Monte-Carlo optimization:
every random control trajectory requires a full propagation of the state, so the
cost of a single `exp(-i Δt H)|ψ⟩` step dictates the scalability of the whole
method. The goal is to apply each Trotter step in (near-)linear time in the
Hilbert-space dimension `d`, instead of the `O(d²)`–`O(d³)` cost of forming or
exponentiating dense `d × d` matrices.

---

## 1. Setup: basis and state representation

We work in the Fock (occupation-number) basis for `N` bosons in `L` modes:

```
B = { |n_1, …, n_L⟩  :  Σ_i n_i = N }
```

The Hilbert-space dimension is the stars-and-bars count

```
d = C(N + L − 1, L − 1).
```

A quantum state `|ψ⟩` is stored as a **1-D complex array of length `d`**; entry
`c` is the amplitude on the `c`-th basis vector.

### The Trotter step

Each piecewise-constant control interval `k` evolves the state under three
Hamiltonians applied in sequence (Eq. 16–17 of the paper):

```
H1^(k) = Σ_i [ Δ_i^(k) n̂_i  +  U_i^(k) n̂_i(n̂_i − 1) ]      (diagonal)
H2^(k) = Σ_i  J_{2i}^(k)   (â†_{2i+1} â_{2i}   + h.c.)        (even-bond hops)
H3^(k) = Σ_i  J_{2i−1}^(k) (â†_{2i}   â_{2i−1} + h.c.)        (odd-bond hops)

|ψ⟩ ← exp(−i Δt H3^(k)) · exp(−i Δt H2^(k)) · exp(−i Δt H1^(k)) |ψ⟩
```

So the whole problem reduces to applying these three exponentials cheaply.

---

## 2. The easy part: the diagonal Hamiltonian `H1`

`H1` contains only number operators `n̂_i` and `n̂_i(n̂_i − 1)`, which are
**diagonal** in the Fock basis. Its exponential is therefore just an
element-wise phase:

```
exp(−i Δt H1)|ψ⟩   →   ψ[c] ← exp(−i Δt · h1[c]) · ψ[c]
```

where `h1[c] = Σ_i [ Δ_i n_i + U_i n_i(n_i − 1) ]` is precomputed from the
occupation numbers of basis state `c`. This costs **`O(d)`** — one pass over the
array, no matrix needed.

---

## 3. The hard part: the hopping Hamiltonians `H2`, `H3`

`H2` and `H3` are **sums of commuting terms**, each a single nearest-neighbor
hop `â†_i â_{i+1} + â†_{i+1} â_i`. Because the terms within a group commute,
applying `exp(−i Δt H2)` is exactly the product of the single-bond
exponentials. Everything therefore reduces to evolving **one bond at a time**:

```
|ψ⟩  →  exp(−i Δt J_{i,i+1} (â†_i â_{i+1} + â†_{i+1} â_i)) |ψ⟩
```

A naive implementation would build this as a dense `d × d` matrix and call
`expm` — prohibitively expensive (e.g. `k=6, N=20` gives `d = 53,130`, a 45 GB
dense matrix). The key is that **a single hop does not mix the whole Hilbert
space**.

### Block structure of a single bond

Consider the bond on modes 1–2 (general `i` is identical). The operator
`â†_1 â_2 + â†_2 â_1` maps

```
|n_1, n_2, n_3, …, n_L⟩   →   |n_1', n_2', n_3, …, n_L⟩
```

where **all other occupations `n_3, …, n_L` are untouched**, and the pair total
is conserved: `n_1 + n_2 = n_1' + n_2'`. Define the invariant subspaces

```
V_r = span{ |n_1, n_2, …, n_L⟩  :  n_1 + n_2 = r },     r = 0, 1, …, N.
```

Each `V_r` is closed under the hop, so the giant operator is **block-diagonal**:
a direct sum of independent blocks, one per value of the pair total `r`.

Within `V_r`, the only degree of freedom the operator changes is how the `r`
shared bosons split between modes 1 and 2 — i.e. `n_1 ∈ {0, …, r}` with
`n_2 = r − n_1`. The remaining `L − 2` modes (holding `N − r` bosons) just label
**independent copies** of the same `(r+1) × (r+1)` two-mode problem.

### The algorithm

To apply `exp(−i Δt J (â†_1 â_2 + â†_2 â_1))` to `|ψ⟩`:

1. **Split** `ψ` into projections `ψ_0, …, ψ_N`, where `ψ_r` is the part of
   `ψ` lying in `V_r`.

2. **Reshape** each `ψ_r` into an `(r+1) × K_r` matrix, where
   `K_r = C(N − r + L − 3, N − r)`. Row `j` holds the amplitudes of all basis
   states of the form `|j, r−j, n_3, …, n_L⟩` (the pair split fixed to
   `n_1 = j`), and the columns enumerate the `K_r` configurations of the other
   `L − 2` modes.

3. **Build the small unitary** `U_r = exp(−i Δt J · M_r)`, where `M_r` is the
   `(r+1) × (r+1)` matrix of `â†_1 â_2 + â†_2 â_1` for `r` bosons in **two**
   modes. This is a tiny tridiagonal matrix with entries
   `√(n_1 (n_2+1))`; exponentiate it once. There are only `N+1` such matrices,
   the largest being `(N+1) × (N+1)`.

4. **Apply** `U_r` to the reshaped block: `ψ_r ← U_r · (reshaped ψ_r)`. The
   same `U_r` acts simultaneously on every column (every configuration of the
   spectator modes) — a single dense matrix–matrix product. Reshape back to a
   1-D vector to recover `exp(...)ψ` restricted to `V_r`.

Concatenating the results over all `r` gives the full propagated state.

### Why this is fast

The work is dominated by step 4. Block `r` costs `O((r+1)² · K_r)` for the
matrix product, and summing over `r` (using `Σ_r (r+1) K_r = d`) gives a total
of about **`O(N² × L × d)`** per bond — effectively linear in `d`, with a
modest `N²` prefactor from the largest two-mode block. This is the crucial
speedup: we never form, store, or exponentiate a `d × d` matrix. Both
`exp(−i Δt H2)` and `exp(−i Δt H3)` are products of such single-bond maps.

---

## 4. The `pair_hop` helper

The reshaping/indexing in steps 1–2 is precomputed once per bond by a function
`pair_hop(i, B)`. It organizes the basis into the blocks described above.

- **Input:** the basis list `B` (each element a tuple `(n_0, …, n_{L−1})` with
  `Σ_j n_j = N`) and a bond index `i` with `0 ≤ i < L − 1`.
- **Output:** a list `[X_0, …, X_N]` of index arrays. `X_r` has shape
  `(r+1) × K_r`, with `K_r = C(N − r + L − 3, N − r)` (stars-and-bars for
  distributing `N − r` bosons over the `L − 2` spectator modes).
- **Meaning:** row `s` of `X_r` fixes `n_i = s`, `n_{i+1} = r − s`; the columns
  enumerate all lexicographically ordered assignments of the remaining `L − 2`
  occupations summing to `N − r`. So `X_r[s, c]` is the **index into `B`** of the
  basis state with that pair occupancy and the `c`-th spectator configuration.

`X_r` is exactly the "gather map" that turns the flat amplitude array into the
`(r+1) × K_r` matrix of step 2 (and scatters it back in step 4).

**How it is built:**

1. Infer `N` and `L` from `B[0]`, and build a dictionary mapping each occupation
   tuple to its position in `B`.
2. Scan `B` once, grouping states by `r = n_i + n_{i+1}` while recording the
   "other" coordinates (positions `i` and `i+1` removed).
3. Sort the "others" sets to fix a **consistent column order** across all rows of
   each `X_r`.
4. Reconstruct full tuples by inserting `(s, r − s)` back into each column's
   "others" tuple, and fill `X_r` with the corresponding indices via the
   dictionary.

Edge cases fall out naturally — e.g. `L = 2` yields only `X_N` with a single
column (there are no spectator modes, and all `N` bosons share the one bond).

---

## 5. Cost summary

| Step | Operator | Method | Cost per step |
|------|----------|--------|---------------|
| Detuning + interaction | `H1` (diagonal) | element-wise phase | `O(d)` |
| Even-bond hops | `H2` | block unitaries via `pair_hop` | `O(N² L d)` |
| Odd-bond hops | `H3` | block unitaries via `pair_hop` | `O(N² L d)` |

A full Trotter step is therefore **`O(N² L d)`**, dominated by the hopping
exponentials, versus `O(d²)` memory and `O(d³)` time for the naive dense
approach. The block-diagonal structure of nearest-neighbor hops is what makes
the multi-mode simulation — and hence the Monte-Carlo optimization over many
random control trajectories — tractable.

---

## 6. Note on this repository's `bosonic_fast.py`

The paper (Sec. S3) achieves the speedup via the **block-diagonalization /
`pair_hop`** route described above. The script `bosonic_fast.py` in this repo
reaches the same goal — never forming a dense `d × d` matrix — through an
equivalent but differently-implemented strategy:

- **Sparse hopping matrices (CSR):** each bond's hopping operator has only
  `O((k−1) d)` nonzeros and is stored as a `scipy.sparse` CSR matrix.
- **Taylor-series exponential:** since `‖J · H_bond‖ · dt ≤ 0.4`, the map
  `exp(−i Δt J H_bond)|ψ⟩` is evaluated with a 15-term Taylor series
  (15 sparse matrix–vector products, double-precision accurate), avoiding the
  overhead of `scipy.expm`.
- **`O(d)` QFI:** because `Sz` is diagonal,
  `QFI = 4(p·Sz² − (p·Sz)²)` with `p = |ψ|²` — two dot products instead of a
  dense matrix–vector product.

Both approaches share the same physics and Trotter structure (Sec. 1 above);
they differ only in *how* the single-bond exponential is applied — exact small
block unitaries (`pair_hop`) versus sparse Taylor series (`bosonic_fast.py`).

### 6.1 How the Taylor-series exponential works

The key insight is that the Taylor series **never builds `exp(A)` as a matrix at
all** — it only ever computes the *action* of the exponential on the state
vector. Sparsity is what makes that cheap.

**Action on a vector, not a matrix.** We want `|ψ'⟩ = exp(A)|ψ⟩` where
`A = −i·dt·J·H_hop`. The matrix exponential is *defined* by its series:

```
exp(A) = I + A + A²/2! + A³/3! + …
```

Apply both sides to `|ψ⟩`:

```
exp(A)|ψ⟩ = |ψ⟩ + A|ψ⟩ + A²|ψ⟩/2! + A³|ψ⟩/3! + …
```

Every term on the right is **a vector**, and each is obtained from the previous
one by *one more multiplication by `A`*. We never form `A²`, `A³`, … as matrices
— we only ever do `A·(vector)`, which is a sparse matrix–vector product (matvec).

**The recurrence in the code.** That is exactly what `_expm_mv_taylor` does
(`bosonic_fast.py:165`):

```python
result = v.copy()      # the i=0 term:  A⁰v/0! = v
term   = v
for i in range(1, order + 1):
    term   = A_sparse.dot(term) * (1.0 / i)   # term_i = A · term_{i-1} / i
    result = result + term
```

Successive terms are related by a simple recurrence:

```
        Aⁱ v        A    Aⁱ⁻¹ v
term_i = ──── = ( ─── )·( ──────── ) = (A · term_{i-1}) / i
         i!        i      (i−1)!
```

So each loop iteration is **one sparse matvec plus a scalar divide plus a vector
add**. Tracing it:

| `i` | `term` after the line | `result` |
|-----|----------------------|----------|
| start | `v` | `v` |
| 1 | `Av` | `v + Av` |
| 2 | `A²v/2` | `v + Av + A²v/2` |
| 3 | `A³v/6` | `… + A³v/6` |

After 15 iterations, `result ≈ exp(A)v`. The cost is `order` (=15) sparse
matvecs, each `O(nnz)`. **The sparsity of `H_hop` enters here:** every matvec
touches only the nonzeros, so a single step is `O(15·nnz)` instead of `O(d²)`.

**Why only 15 terms?** The series is truncated, so accuracy depends on how fast
the tail decays. The error after `order` terms is bounded by

```
‖exp(A)v − result‖  ≤  ‖A‖^(order+1) / (order+1)!  · ‖v‖
```

This is why the **norm bound matters**. The step is sized so that

```
‖A‖ = ‖−i·dt·J·H_hop‖ = dt·J·‖H_hop‖ ≤ 0.4
```

(with `dt = 0.01`, `J ≤ 1`, and `‖H_hop‖ ≈ 28–37` over the `k` used). Plugging
in:

```
0.4¹⁶ / 16!  ≈  8 × 10⁻¹⁹
```

— below double-precision round-off. So 15 terms is already exact to machine
precision; the factorial in the denominator crushes the tail extremely fast
*because `‖A‖ < 1`*. If `‖A‖` were large (a big `dt`), the series would converge
slowly and you would need many terms or a smaller step — which is the whole
reason the timestep is kept small.

**Why a series at all, instead of exponentiating the diagonal?** `H1` is
diagonal, so its exponential is a trivial element-wise phase (`O(d)`). But
`H_hop` is **purely off-diagonal** — exponentiating it genuinely mixes
amplitudes between connected Fock states, so there is no element-wise shortcut.
The two honest options are (1) a dense `expm` (forms a `d×d` matrix —
infeasible), or (2) this series, which exploits the fact that you only ever need
`exp(A)|ψ⟩`, never `exp(A)` itself.

> **Sparsity per row.** A *single bond* connects each basis state to at most one
> other state (one off-diagonal pair per row). But the code applies the
> **summed** even/odd bond groups `H2_base`/`H3_base` (`bosonic_fast.py:224`), so
> each row has up to ⌈(k−1)/2⌉ off-diagonal entries — still very sparse, just not
> literally two.
